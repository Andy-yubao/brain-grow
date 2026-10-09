"""固定20+5轮逐边门控实验；3实际前缀、4最终分支，旧训练源码不改。"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .datasets import load_bundle
from .l0_gates import BETA, GAMMA, ZETA, UNIFORM_EPS, L0GatedMLP, freeze_gated
from .models import StructuredMLP
from .pruning_experiment import (encoded, evaluate, metrics, pid_alive, restore_rng, rng_state,
                                save_checkpoint, sha, structure_stats, synchronize, within, write_json)

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ("src/braingrow/l0_gates.py", "src/braingrow/l0_experiment.py", "tools/train_l0_gates.py",
           "tools/report_l0_gates.py", "tests/test_l0_gates.py", "src/braingrow/models.py",
           "src/braingrow/codec.py", "src/braingrow/datasets.py", "src/braingrow/pruning_experiment.py",
           "src/braingrow/structural.py", "configs/h1_l0_gates_v1.json", "docs/h1_l0_gates_v1_execution.md",
           "docs/h1_l0_gates_v1_proposal.md", "tools/report_pruning.py", "tools/report_nodes_nonzero.py")
COUNT_KEYS = ("optimization_forward_samples", "optimization_backward_samples", "attempted_updates",
              "candidate_updates", "rolled_back_updates", "fallback_updates", "evaluation_forward_samples")


def new_ledger():
    return {**dict.fromkeys(COUNT_KEYS, 0), "evaluations": [], "adam_reset_epochs": []}


def reconcile(ledger, purpose, samples):
    if not any(item["purpose"] == purpose for item in ledger["evaluations"]):
        ledger["evaluation_forward_samples"] += samples
        ledger["evaluations"].append(dict(purpose=purpose, samples=samples, recovered_from_saved_predictions=True))


def persist_cost(path, past, began, config):
    write_json(path, dict(cumulative_wall_seconds=past["cumulative_wall_seconds"] + time.perf_counter() - began,
        sessions=past["sessions"] + 1, stage_limit_seconds=config["maximum_wall_seconds"],
        checked_at_complete_epoch_boundary=True))


def fresh_base(widths, state, device):
    with torch.random.fork_rng(devices=[]):
        model = StructuredMLP(widths)
    model = model.to(device)
    model.load_state_dict(state)
    return model


def fresh_optimizer(model, config, epoch):
    lr = config["weight_learning_rate"] * (.5 if epoch >= config["weight_lr_half_from_epoch"] else 1.)
    groups = ([dict(params=model.base.parameters(), lr=lr, role="weight"),
               dict(params=model.log_alpha.parameters(), lr=config["gate_learning_rate"], role="gate")]
              if isinstance(model, L0GatedMLP) else [dict(params=model.parameters(), lr=lr, role="weight")])
    return torch.optim.AdamW(groups, betas=tuple(config["adam_betas"]), eps=config["adam_epsilon"],
                            weight_decay=config["weight_decay"])


def gate_generator(device, config):
    seed = int(np.random.SeedSequence([config["model_seed"], 810]).generate_state(1, np.uint64)[0]) % (2 ** 63)
    return torch.Generator(device=device).manual_seed(seed)


def model_from_saved(saved, device, config):
    # 构造恢复对象不推进任何训练RNG，门generator在checkpoint中单独恢复。
    state = saved["model"]
    if saved["model_kind"] == "gated":
        base_state = {key.removeprefix("base."): value for key, value in state.items() if key.startswith("base.")}
        model = L0GatedMLP(fresh_base(saved["widths"], base_state, device), config["initial_log_alpha"])
        model.load_state_dict(state)
        model.gate_generator = gate_generator(device, config)
        model.gate_generator.set_state(saved["gate_rng"].cpu())
    else:
        model = fresh_base(saved["widths"], state, device)
    model.train(saved["training"])
    for parameter, grad in zip(model.parameters(), saved["parameter_grad"]):
        parameter.grad = None if grad is None else grad.clone()
    layers = model.base.layers if isinstance(model, L0GatedMLP) else model.layers
    for layer, grad in zip(layers, saved["edge_grad"]):
        layer.last_edge_grad = None if grad is None else grad.clone()
    return model


def checkpoint(model, optimizer, phase, arm, next_epoch, rows, ledger, wall, encoding_wall, freeze=None):
    layers = model.base.layers if isinstance(model, L0GatedMLP) else model.layers
    return dict(version="h1_l0_gates_v1", phase=phase, arm=arm, next_epoch=next_epoch, widths=list(model.widths),
        model_kind="gated" if isinstance(model, L0GatedMLP) else "structured", model=model.state_dict(),
        training=model.training, optimizer=optimizer.state_dict(), rng=rng_state(),
        gate_rng=model.gate_generator.get_state() if isinstance(model, L0GatedMLP) else None,
        parameter_grad=[None if p.grad is None else p.grad.detach().clone() for p in model.parameters()],
        edge_grad=[None if layer.last_edge_grad is None else layer.last_edge_grad.clone() for layer in layers],
        rows=rows, ledger=ledger, wall_seconds=wall, encoding_wall_seconds=encoding_wall, freeze=freeze,
        scheduler=dict(epoch=next_epoch-1, group_learning_rates=[g["lr"] for g in optimizer.param_groups]))


def train_step(model, optimizer, x, y, ledger, lam, config, candidate):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    ce = F.cross_entropy(model(x), y)
    penalty = (lam * model.expected_active_edges() / config["l0_normalizer"]
               if isinstance(model, L0GatedMLP) else ce.new_zeros(()))
    loss = ce + penalty
    if not torch.isfinite(loss):
        raise FloatingPointError("门控训练loss非有限")
    loss.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise FloatingPointError("门控训练梯度非有限")
    optimizer.step()
    if isinstance(model, L0GatedMLP):
        model.constrain()
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise FloatingPointError("门控参数非有限")
    ledger["optimization_forward_samples"] += len(x)
    ledger["optimization_backward_samples"] += len(x)
    ledger["attempted_updates"] += 1
    ledger["candidate_updates"] += int(candidate)
    return float(ce.detach()), float(penalty.detach())


def deployment(model):
    return freeze_gated(model)[0] if isinstance(model, L0GatedMLP) else model


def save_npz(path, **arrays):
    # 预测保存可先于账本持久化，resume按唯一purpose幂等补记；原子替换防半文件。
    temp = path.with_suffix(".tmp.npz")
    np.savez_compressed(temp, **arrays)
    os.replace(temp, path)


def prediction(model, x, y, labels, path, ledger, purpose, config):
    if path.exists():
        with np.load(path, allow_pickle=False) as values:
            if not np.array_equal(values["labels"], labels):
                raise ValueError("预测缓存标签与固定样本不符")
            result = metrics(values["logits"], values["labels"])
        reconcile(ledger, purpose, len(x))
        return result
    result, logits = evaluate(model, x, y, config["evaluation_batch_size"], ledger, purpose)
    save_npz(path, logits=logits, labels=labels)
    return result


def train_phase(phase, arm, start_model, start_rng, start_rows, folder, config, device,
                fx, fy, sx, sy, labels, parent_freeze, persist, stop):
    resume = folder / "resume.pt"
    folder.mkdir(parents=True, exist_ok=True)
    first = 1 if phase == "prefix" else config["gate_epochs"] + 1
    last = config["gate_epochs"] if phase == "prefix" else config["gate_epochs"] + config["adaptation_epochs"]
    if resume.exists():
        saved = torch.load(resume, map_location=device, weights_only=False)
        if saved["version"] != config["version"] or saved["phase"] != phase or saved["arm"] != arm:
            raise ValueError("phase/arm恢复状态不匹配")
        model = model_from_saved(saved, device, config)
        optimizer = fresh_optimizer(model, config, first)
        optimizer.load_state_dict(saved["optimizer"])
        restore_rng(saved["rng"])
        rows, ledger = saved["rows"], saved["ledger"]
        next_epoch, wall, encoding_wall = saved["next_epoch"], saved["wall_seconds"], saved["encoding_wall_seconds"]
        parent_freeze = saved["freeze"]
    else:
        model = start_model()
        restore_rng(start_rng)
        optimizer = fresh_optimizer(model, config, first)
        rows, ledger, next_epoch, wall, encoding_wall = copy.deepcopy(start_rows), new_ledger(), first, 0., 0.
        if phase == "branch":
            ledger["adam_reset_epochs"].append(first)
        saved = checkpoint(model, optimizer, phase, arm, first, rows, ledger, wall, encoding_wall, parent_freeze)
        save_checkpoint(resume, saved)
    for epoch in range(next_epoch, last + 1):
        # 构造/写证据或resume启动也可能耗尽余量；不能只在上一轮末检查。
        if stop():
            persist()
            write_json(folder / "budget_stop.json", dict(next_epoch=epoch,
                reason="request_or_cumulative_budget_before_new_epoch"))
            return None
        synchronize(device)
        began = time.perf_counter()
        weight_lr = config["weight_learning_rate"] * (.5 if epoch >= config["weight_lr_half_from_epoch"] else 1)
        for group in optimizer.param_groups:
            if group["role"] == "weight":
                group["lr"] = weight_lr
        lam = min(epoch / config["lambda_ramp_epochs"], 1.) * config["lambda_max"] if phase == "prefix" and arm == "learned" else 0.
        order = np.random.default_rng(np.random.SeedSequence([config["data_seed"], 25 + epoch])).permutation(len(fx))
        losses, penalties = [], []
        for start in range(0, len(order), config["batch_size"]):
            ids = torch.as_tensor(order[start:start + config["batch_size"]], device=device)
            ce, penalty = train_step(model, optimizer, fx[ids], fy[ids], ledger, lam, config,
                                     candidate=phase == "branch" and arm != "fixed")
            losses.append(ce)
            penalties.append(penalty)
        selected, scores = evaluate(model, sx, sy, config["evaluation_batch_size"], ledger, f"epoch_select_{epoch:03d}")
        synchronize(device)
        encode_started = time.perf_counter()
        compact = deployment(model)
        payload = encoded(compact)
        structure = structure_stats(compact, payload)
        synchronize(device)
        encoding_elapsed = time.perf_counter() - encode_started
        encoding_wall += encoding_elapsed
        if compact is not model:
            del compact
        if ledger["attempted_updates"] != (epoch - first + 1) * 50:
            raise RuntimeError("完整轮必须正好50更新")
        synchronize(device)
        elapsed = time.perf_counter() - began
        wall += elapsed
        row = dict(epoch=epoch, phase=phase, arm=arm, select=selected, structure=structure,
                   gates=model.gate_stats() if isinstance(model, L0GatedMLP) else None,
                   mean_fit_batch_ce=float(np.mean(losses)), mean_l0_penalty=float(np.mean(penalties)),
                   lambda_epoch=lam, weight_lr=weight_lr,
                   gate_lr=config["gate_learning_rate"] if isinstance(model, L0GatedMLP) else None,
                   wall_seconds=elapsed, construction_encoding_wall_seconds=encoding_elapsed,
                   attempted_updates_in_this_phase=ledger["attempted_updates"])
        rows.append(row)
        save_npz(folder / f"select_{epoch:03d}.npz", logits=scores, labels=labels)
        saved = checkpoint(model, optimizer, phase, arm, epoch+1, rows, ledger, wall, encoding_wall, parent_freeze)
        save_checkpoint(resume, saved)
        write_json(folder / "progress.json", dict(rows=rows, ledger=ledger, wall_seconds=wall, encoding_wall_seconds=encoding_wall))
        persist()
        print(json.dumps(dict(phase=phase, arm=arm, epoch=epoch, accuracy=selected["accuracy"],
            loss=selected["loss"], active_edges=structure["model_total"]["active_edges"],
            bytes=len(payload), expected_edges=row["gates"] and row["gates"]["expected_active_edges"])), flush=True)
        if stop():
            write_json(folder / "budget_stop.json", dict(completed_epoch=epoch, reason="request_or_cumulative_budget"))
            return None
    return dict(model=model, optimizer=optimizer, saved=saved, rows=rows, ledger=ledger, freeze=parent_freeze,
                wall_seconds=wall, encoding_wall_seconds=encoding_wall)


def verify_config(config):
    expected = json.loads((ROOT / "configs/h1_l0_gates_v1.json").read_text(encoding="utf-8"))
    if config != expected or config["version"] != "h1_l0_gates_v1" or config["model_seed"] != 11 or config["data_seed"] != 20261004:
        raise ValueError("只允许固定登记配置，不允许门参数或种子扫描")
    if (config["beta"], config["gamma"], config["zeta"], config["uniform_epsilon"]) != (BETA, GAMMA, ZETA, UNIFORM_EPS):
        raise ValueError("配置与门公式常量不符")


def run(args):
    began = time.perf_counter()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    verify_config(config)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    source = ROOT / config["source_run"]
    source_registration = json.loads((source / "registration.json").read_text(encoding="utf-8"))
    for path, expected in source_registration["source_sha256"].items():
        if sha(ROOT / path) != expected:
            raise ValueError(f"父实现来源不匹配：{path}")
    parent_summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    if not all(result["frozen"] for result in parent_summary["arms"].values()):
        raise ValueError("父实验未全部冻结")
    parent_path = source / "fc/resume.pt"
    widths = source_registration["config"]["tasks"]["mnist_static"]["widths"]
    folder = ROOT / "experiments/runs" / config["version"] / "mnist_static"
    folder.mkdir(parents=True, exist_ok=True)
    lock = folder / "running.lock"
    if lock.exists():
        if pid_alive(int(lock.read_text())):
            raise RuntimeError("门控运行锁仍活动")
        lock.unlink()
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    cost_path = folder / "session_cost.json"
    past = json.loads(cost_path.read_text(encoding="utf-8")) if cost_path.exists() else dict(cumulative_wall_seconds=0., sessions=0)
    active_session = False
    try:
        registration = dict(config=config, task="mnist_static", device=str(device), torch=torch.__version__,
            source_sha256={file: sha(ROOT / file) for file in SOURCES}, initial_widths=widths,
            source_checkpoint_sha256=sha(parent_path), source_registration_sha256=sha(source / "registration.json"),
            parent_run=config["source_run"], parent_arm="fc", source_epoch=25,
            hardware=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            precision="float32", execution="dense_gate_matmul_then_dense_gather_matmul", cpu_threads=2)
        reg_path = folder / "registration.json"
        if reg_path.exists():
            if json.loads(reg_path.read_text(encoding="utf-8")) != registration or not args.resume:
                raise ValueError("运行存在或源/配置/设备变化，只允许同登记--resume")
        else:
            if args.resume:
                raise ValueError("无已登记实验可恢复")
            write_json(reg_path, registration)
        provenance = json.loads((folder / "training_commit.json").read_text(encoding="utf-8"))
        if not provenance["registeredBeforeTraining"] or provenance["source_sha256"] != registration["source_sha256"]:
            raise ValueError("必须启动前记录实际实现提交与锁定源哈希")
        if (folder / "summary.json").exists():
            print(json.dumps(dict(already_complete=True)), flush=True)
            return
        if past["cumulative_wall_seconds"] >= config["maximum_wall_seconds"]:
            raise RuntimeError("累计1800秒阶段预算已用完，resume不重置")
        active_session = True
        write_json(folder / "running.json", dict(pid=os.getpid(), status="running"))
        startup = json.loads((source / "startup.json").read_text(encoding="utf-8"))
        if not startup["ok"] or startup["widths"] != widths:
            raise ValueError("既有GPU同尺寸启动证据无效")
        write_json(folder / "startup.json", dict(reused_from=str(source / "startup.json"),
            forward_samples=0, backward_samples=0, updates=0, additional_gpu_checks=0,
            note="逐边门用单项真实CPU检查；旧模型/编码/事务/GPU启动证据复用，训练有非有限保护"))
        bundle = load_bundle("mnist", include_confirm=False)
        if bundle.fit_x.shape != (3200, 784) or bundle.select_x.shape != (800, 784) or len(bundle.confirm_x):
            raise ValueError("仅允许3200fit/800select，冻结前不加载control")
        write_json(folder / "data_manifest.json", bundle.manifest)
        fx, fy, sx, sy = [torch.from_numpy(array).to(device) for array in (bundle.fit_x, bundle.fit_y, bundle.select_x, bundle.select_y)]
        parent = torch.load(parent_path, map_location=device, weights_only=False)
        if parent["next_epoch"] != 26:
            raise ValueError("父必须FC25末态")
        persist = lambda: persist_cost(cost_path, past, began, config)
        stop = lambda: (folder / "request_stop").exists() or past["cumulative_wall_seconds"] + time.perf_counter() - began >= config["maximum_wall_seconds"]
        for arm in config["prefix_arms"]:
            def start_model(arm=arm):
                base = fresh_base(widths, parent["model"], device)
                if arm == "fixed":
                    return base
                gated = L0GatedMLP(base, config["initial_log_alpha"])
                gated.gate_generator = gate_generator(device, config)
                return gated
            state = train_phase("prefix", arm, start_model, parent["rng"], [], folder / "prefixes" / arm,
                config, device, fx, fy, sx, sy, bundle.select_y, None, persist, stop)
            if state is None:
                return
            if stop():
                persist()
                write_json(folder / "budget_stop.json", dict(reason="cumulative_budget_before_prefix_deployment"))
                return
            (folder / "prefixes" / arm / "deterministic_parent_model.bgs").write_bytes(encoded(deployment(state["model"])))
            del state
        results = {}
        for arm in config["arms"]:
            if stop():
                persist()
                write_json(folder / "budget_stop.json", dict(reason="cumulative_budget_before_branch_freeze", arm=arm))
                return
            arm_folder = folder / arm
            arm_folder.mkdir(exist_ok=True)
            source_arm = "learned" if arm == "random" else arm
            source_prefix = folder / "prefixes" / source_arm
            mother = torch.load(source_prefix / "resume.pt", map_location=device, weights_only=False)
            if mother["next_epoch"] != 21:
                raise ValueError("分叉母态必须第20轮完整状态")
            mother_model = model_from_saved(mother, device, config)
            restore_rng(mother["rng"])
            synchronize(device)
            freeze_started = time.perf_counter()
            if isinstance(mother_model, L0GatedMLP):
                initial, details, arrays = freeze_gated(mother_model, permutation_seed=11 if arm == "random" else None)
                values_path = arm_folder / "freeze_gate_values.npz"
                if not values_path.exists():
                    save_npz(values_path, **arrays)
                details["gate_values_file_sha256"] = sha(values_path)
            else:
                initial = copy.deepcopy(mother_model)
                details = dict(rule="same_graph_fixed_reference_adam_reset", layers=[])
            synchronize(device)
            freeze_wall = time.perf_counter() - freeze_started
            details.update(source_prefix=source_arm, source_checkpoint_sha256=sha(source_prefix / "resume.pt"),
                           source_epoch=20, source_training_state_unchanged=True,
                           additional_model_forward_samples=0, training_gate_parameters_removed=True)
            freeze_path = arm_folder / "freeze_boundary.json"
            if freeze_path.exists():
                if json.loads(freeze_path.read_text(encoding="utf-8")) != details:
                    raise ValueError("分叉冻图证据变化")
            else:
                write_json(freeze_path, details)
                write_json(arm_folder / "freeze_cost.json", dict(wall_seconds=freeze_wall, model_forward_samples=0))
                (arm_folder / "initial_frozen_model.bgs").write_bytes(encoded(initial))
            state = train_phase("branch", arm, lambda: initial, mother["rng"], mother["rows"], arm_folder,
                config, device, fx, fy, sx, sy, bundle.select_y, details, persist, stop)
            del initial, mother_model, mother
            if state is None:
                return
            if stop():
                persist()
                write_json(folder / "budget_stop.json", dict(reason="cumulative_budget_before_final_candidate_evaluation", arm=arm))
                return
            model, optimizer, ledger = state["model"], state["optimizer"], state["ledger"]
            # 边界random只在初始化snapshot上一次评价。恢复不重复已有预测。
            if arm == "random":
                # 若首次到末态才评价边界，必须重建原冻结模型，不能误用已适应末态。
                source_state = torch.load(source_prefix / "resume.pt", map_location=device, weights_only=False)
                initial_gate = model_from_saved(source_state, device, config)
                boundary_model = freeze_gated(initial_gate, permutation_seed=11)[0]
                prediction(boundary_model, sx, sy, bundle.select_y, arm_folder / "select_freeze_boundary.npz",
                           ledger, "random_freeze_boundary", config)
                del initial_gate, boundary_model, source_state
            payload = encoded(model)
            (arm_folder / "final_model.bgs").write_bytes(payload)
            fit = prediction(model, fx, fy, bundle.fit_y, arm_folder / "fit_final.npz", ledger, "frozen_fit", config)
            saved = checkpoint(model, optimizer, "branch", arm, 26, state["rows"], ledger,
                               state["wall_seconds"], state["encoding_wall_seconds"], details)
            save_checkpoint(arm_folder / "resume.pt", saved)
            result = dict(arm=arm, frozen=True, final_epoch=25, rows=state["rows"], fit=fit, ledger=ledger,
                structure=structure_stats(model, payload), freeze=details, wall_seconds=state["wall_seconds"],
                encoding_wall_seconds=state["encoding_wall_seconds"],
                prefix_actual_cost_included_here=False, inherited_prefix=source_arm, shared_prefix=arm == "random")
            write_json(arm_folder / "result.json", result)
            results[arm] = result
            del state, model, optimizer, saved
            persist()
            if stop():
                return
        learned_layers, random_layers = results["learned"]["freeze"]["layers"], results["random"]["freeze"]["layers"]
        if [row["active_gate_positions"] for row in learned_layers] != [row["active_gate_positions"] for row in random_layers]:
            raise RuntimeError("随机分叉有效位置数不匹配")
        if not all(row["same_gate_multiset"] for row in random_layers):
            raise RuntimeError("完整门值多重集不匹配")
        fixed, learned = results["fixed"], results["learned"]
        task_ok = within(learned["rows"][-1]["select"], fixed["rows"][-1]["select"], config) and learned["fit"]["accuracy"] >= config["fit_accuracy_floor"]
        reduction = 1 - learned["structure"]["export_bytes"] / fixed["structure"]["export_bytes"]
        accepted = task_ok and reduction > 0
        acceptance = dict(accepted=accepted, task_preserved=task_ok, storage_reduction=reduction,
            storage_signal_at_least_20_percent=reduction >= config["stage_storage_reduction"], candidate_epoch=25,
            rule="one_final_full_select_vs_same_epoch_fixed_and_fit_floor_and_true_bytes",
            recovery_only_epoch=20 if not accepted else None,
            recovery_checkpoint="prefixes/learned/resume.pt" if not accepted else None,
            candidate_updates_retained_in_cost=250, additional_fallback_training=0)
        if not accepted:
            # 候选从独立母态分叉，参数/Adam/调度/梯度/门流均未触碰原文件；
            # 这里实际恢复对象及RNG，但不再训练/评价，不冒充公平25轮恢复结果。
            mother = torch.load(folder / "prefixes/learned/resume.pt", map_location=device, weights_only=False)
            recovered = model_from_saved(mother, device, config)
            recovered_adam = fresh_optimizer(recovered, config, 20)
            recovered_adam.load_state_dict(mother["optimizer"])
            restore_rng(mother["rng"])
            acceptance["restored_phase_a_full_training_state"] = True
            del mother, recovered, recovered_adam
        write_json(folder / "acceptance.json", acceptance)
        write_json(folder / "freeze.json", dict(arms={arm: 25 for arm in config["arms"]},
            before_repeated_exploration=True, acceptance_decided_without_control=True, acceptance=acceptance))
        if stop():
            persist()
            write_json(folder / "budget_stop.json", dict(reason="cumulative_budget_before_control_loading"))
            return
        confirmed = load_bundle("mnist", include_confirm=True)
        cx, cy = torch.from_numpy(confirmed.confirm_x).to(device), torch.from_numpy(confirmed.confirm_y).to(device)
        for arm, result in results.items():
            if stop():
                persist()
                write_json(folder / "budget_stop.json", dict(reason="cumulative_budget_before_repeated_exploration", arm=arm))
                return
            arm_folder = folder / arm
            saved = torch.load(arm_folder / "resume.pt", map_location=device, weights_only=False)
            result["ledger"] = saved["ledger"]
            model = model_from_saved(saved, device, config)
            result["confirmation"] = prediction(model, cx, cy, confirmed.confirm_y,
                arm_folder / "confirm_final.npz", result["ledger"], "repeated_exploration", config)
            saved["ledger"] = result["ledger"]
            save_checkpoint(arm_folder / "resume.pt", saved)
            write_json(arm_folder / "result.json", result)
            del model, saved
        prefixes = {}
        for arm in config["prefix_arms"]:
            saved = torch.load(folder / "prefixes" / arm / "resume.pt", map_location="cpu", weights_only=False)
            prefixes[arm] = {key: saved[key] for key in ("rows", "ledger", "wall_seconds", "encoding_wall_seconds")}
            prefixes[arm]["checkpoint_bytes"] = (folder / "prefixes" / arm / "resume.pt").stat().st_size
            del saved
        all_ledgers = [result["ledger"] for result in results.values()] + [result["ledger"] for result in prefixes.values()]
        total = {key: sum(ledger[key] for ledger in all_ledgers) for key in COUNT_KEYS}
        total["evaluation_calls"] = sum(len(ledger["evaluations"]) for ledger in all_ledgers)
        if total["attempted_updates"] != config["maximum_updates"] or total["optimization_forward_samples"] != config["maximum_optimization_samples"] or total["evaluation_forward_samples"] > config["maximum_evaluation_samples"]:
            raise RuntimeError("最终实际预算账本不符合4000/256000/81600")
        summary = dict(task="mnist_static", registration=registration, prefixes=prefixes, arms=results, acceptance=acceptance,
            actual_cost_total=total, random_complete_gate_multiset_and_active_positions_matched=True,
            random_identity_scope="conditional_on_shared_theta; full_gate_position_and_attenuation_assignment",
            no_extra_gpu_checks=True, execution_note="dense_gate_then_dense_gather; no_measured_inference_speed_or_energy",
            current_session_wall_seconds=time.perf_counter()-began,
            cumulative_wall_seconds=past["cumulative_wall_seconds"]+time.perf_counter()-began,
            peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            recovery_cost_scope="完整轮及已保存末态预测可恢复；未落盘的中断中间计算不可完整追溯",
            official_test_accessed=False, remaining_sealed_ids_extracted=0)
        write_json(folder / "summary.json", summary)
        print(json.dumps(dict(complete=True, acceptance=acceptance, actual_cost_total=total)), flush=True)
    finally:
        if active_session:
            persist_cost(cost_path, past, began, config)
        lock.unlink(missing_ok=True)
        write_json(folder / "running.json", dict(pid=None, status="exited"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/h1_l0_gates_v1.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
