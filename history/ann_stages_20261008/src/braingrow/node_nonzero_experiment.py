"""未剪枝FC末态延续的非零多节点简化三臂实验，保留旧节点轮。"""
from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from .codec import decode_model
from .datasets import load_bundle
from .models import StructuredMLP
from .node_reduction_nonzero import (MIN_COEFFICIENT_ABS, MIN_RECONSTRUCTION_COEFFICIENTS,
                             NodeReductionProposal, _fit_activations,
                             apply_node_reduction, propose_node_reduction)
from .pruning_experiment import (encoded, evaluate, metrics, pid_alive, restore_rng,
    rng_state, save_checkpoint, select_screen, sha, structure_stats, synchronize,
    train_batch, within, write_json)

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ("tools/train_nodes_nonzero.py", "src/braingrow/node_nonzero_experiment.py",
           "src/braingrow/node_reduction_nonzero.py", "src/braingrow/pruning_experiment.py",
           "src/braingrow/models.py", "src/braingrow/codec.py", "src/braingrow/structural.py",
           "src/braingrow/datasets.py")
UPDATE_KEYS = ("attempted_updates", "candidate_updates", "rolled_back_updates", "fallback_updates")


def new_ledger():
    return dict(optimization_forward_samples=0, optimization_backward_samples=0,
                attempted_updates=0, candidate_updates=0, rolled_back_updates=0,
                fallback_updates=0, evaluation_forward_samples=0, evaluations=[],
                activation_forward_samples=0, activation_extractions=[], adam_reset_epochs=[])


def reconcile_cached_evaluation(ledger, purpose, samples):
    """末态预测已保存而账本未落盘时幂等补记；不产生新的模型前向。"""
    if not any(item["purpose"] == purpose for item in ledger["evaluations"]):
        ledger["evaluation_forward_samples"] += samples
        ledger["evaluations"].append(dict(purpose=purpose, samples=samples,
                                           recovered_from_saved_predictions=True))


def persist_session_cost(path, past_cost, began, maximum_wall_seconds):
    # 本会话多次覆盖仍只加一次；硬中断时至少保留最近完整轮成本。
    write_json(path, dict(cumulative_wall_seconds=past_cost["cumulative_wall_seconds"]+time.perf_counter()-began,
        sessions=past_cost["sessions"]+1, checked_at_complete_epoch_boundary=True,
        stage_limit_seconds=maximum_wall_seconds))


def fit_partition(labels):
    """同一授权拟合集内固定分层；标签只决定样本集合，不进入节点求解。"""
    fit_indices, holdout_indices = [], []
    for digit in range(10):
        indices = np.flatnonzero(labels == digit)
        if len(indices) != 320:
            raise ValueError("节点实验拟合集必须每数字320个原ID")
        fit_indices.extend(indices[:256].tolist())
        holdout_indices.extend(indices[256:].tolist())
    if set(fit_indices) & set(holdout_indices):
        raise ValueError("内部拟合/留出索引重叠")
    return dict(rule="each_digit_first_256_fit_last_64_holdout_in_existing_fixed_fit_order",
                fit_indices=fit_indices, holdout_indices=holdout_indices,
                fit_per_digit=256, holdout_per_digit=64,
                labels_role="仅确定集合分层，不传给激活基组/最小二乘求解")


def fresh_adam(model, lr, config):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config["weight_decay"])


def restore_model(saved, device, initial_widths):
    with torch.random.fork_rng(devices=[]):
        model = StructuredMLP(saved.get("widths", initial_widths))
    model = model.to(device)
    model.load_state_dict(saved["model"])
    model.train(saved.get("model_training", True))
    for parameter, grad in zip(model.parameters(), saved["parameter_grad"]):
        parameter.grad = None if grad is None else grad.clone()
    for layer, grad in zip(model.layers, saved["edge_grad"]):
        layer.last_edge_grad = None if grad is None else grad.clone()
    if saved.get("node_reduction_metadata") is not None:
        model.node_reduction_metadata = copy.deepcopy(saved["node_reduction_metadata"])
    return model


def record_extraction(ledger, samples, purpose, layer_index):
    ledger["activation_forward_samples"] += samples
    ledger["activation_extractions"].append(dict(samples=samples, purpose=purpose, layer_index=layer_index))


def relative_residuals(hidden, kept, dropped, coefficients):
    error = hidden[:, list(dropped)] - hidden[:, list(kept)] @ coefficients
    absolute = torch.linalg.vector_norm(error, dim=0)
    baseline = torch.linalg.vector_norm(hidden[:, list(dropped)], dim=0)
    relative = torch.where(baseline > 0, absolute / baseline,
                           torch.where(absolute == 0, torch.zeros_like(absolute),
                                       torch.full_like(absolute, float("inf"))))
    return absolute, relative


def json_residual(values):
    return [float(value) if np.isfinite(float(value)) else "inf" for value in values]


def residual_record(proposal, holdout_hidden, fixed_coefficients):
    absolute, relative = relative_residuals(holdout_hidden, proposal.kept_indices,
                                           proposal.dropped_indices, fixed_coefficients)
    values = relative.tolist()
    return dict(fit_relative_residuals_float64_lstsq=json_residual(proposal.relative_residuals),
                fit_absolute_residuals_float64_lstsq=list(proposal.absolute_residuals),
                holdout_relative_residuals_fixed_float32_coefficients=json_residual(values),
                holdout_absolute_residuals=absolute.tolist(),
                max_holdout_relative_residual=float(max(values)) if np.isfinite(max(values)) else "inf",
                coefficients_absolute_max=float(fixed_coefficients.abs().max()),
                coefficients_frobenius_norm=float(torch.linalg.vector_norm(fixed_coefficients)),
                coefficient_rule="float64拟合后一次固定为C32；留出只验证，不再拟合",
                zero_denominator_rule="分母为0且误差为0记0，误差非零记inf"), bool(
                    torch.isfinite(relative).all() and torch.all(relative <= proposal.max_relative_residual))


def make_random_proposal(model, layer_index, fit_x, remove_count, epoch, source_proposal, config, ledger):
    width = model.widths[layer_index + 1]
    if not 0 < remove_count < width:
        raise ValueError("匹配随机缩减数量必须小于当前隐藏宽度")
    rng = np.random.default_rng(np.random.SeedSequence([11, epoch, source_proposal, layer_index]))
    dropped = tuple(sorted(rng.choice(width, size=remove_count, replace=False).tolist()))
    dropped_set = set(dropped)
    kept = tuple(index for index in range(width) if index not in dropped_set)
    hidden = _fit_activations(model, layer_index, fit_x)
    record_extraction(ledger, len(fit_x), "random_coefficient_fit", layer_index)
    coefficients = torch.linalg.lstsq(hidden[:, list(kept)], hidden[:, list(dropped)], driver="gelsd").solution
    absolute, relative = relative_residuals(hidden, kept, dropped, coefficients)
    return NodeReductionProposal(layer_index, tuple(model.widths), kept, dropped, kept, coefficients,
        tuple(relative.tolist()), tuple(absolute.tolist()), len(fit_x), config["max_relative_residual"], width - 1,
        coefficient_source="同规模均匀随机删除身份；当前网络拟合激活float64无截距lstsq，全保留节点基组，无选择反馈",
        candidate_filter_enforced=False,
        fit_activation_norms=tuple(torch.linalg.vector_norm(hidden[:, list(dropped)], dim=0).tolist()))


def reject_trial_on_original(model, optimizer, x, y, ledger):
    """随机匹配主臂已尝试而拒绝的适应成本；恢复所有训练状态后正常更新。"""
    state = copy.deepcopy(model.state_dict())
    adam = copy.deepcopy(optimizer.state_dict())
    gradients = [None if p.grad is None else p.grad.clone() for p in model.parameters()]
    edge_gradients = [None if layer.last_edge_grad is None else layer.last_edge_grad.clone() for layer in model.layers]
    before_rng, training = rng_state(), model.training
    try:
        train_batch(model, optimizer, x, y, ledger, candidate=True)
        ledger["rolled_back_updates"] += 1
    finally:
        model.load_state_dict(state)
        optimizer.load_state_dict(adam)
        model.train(training)
        for parameter, grad in zip(model.parameters(), gradients):
            parameter.grad = grad
        for layer, grad in zip(model.layers, edge_gradients):
            layer.last_edge_grad = grad
        restore_rng(before_rng)
    train_batch(model, optimizer, x, y, ledger)
    ledger["fallback_updates"] += 1


def reduction_slot(model, optimizer, fit_x, holdout_x, x, y, sx, sy, screen,
                   reference, ledger, config, epoch, proposal_index, layer_index):
    # apply 返回独立新图，旧图/参数/Adam/梯度保留不动，拒绝只丢弃候选并恢复RNG。
    before_rng = rng_state()
    event = dict(epoch=epoch, proposal=proposal_index, layer_index=layer_index,
                 accepted=False, trial_adapted=False, removed_nodes=0,
                 widths_before=list(model.widths), reason="no_candidate")
    synchronize(x.device)
    started = time.perf_counter()
    cpu_started = time.process_time()
    proposal = propose_node_reduction(model, layer_index, fit_x, max_remove=config["max_remove"],
        max_basis=model.widths[layer_index + 1] - 1, max_relative_residual=config["max_relative_residual"])
    record_extraction(ledger, len(fit_x), "task_basis_fit", layer_index)
    event["search_wall_seconds"] = time.perf_counter() - started
    event["search_cpu_process_seconds"] = time.process_time() - cpu_started
    event["search_timing_scope"] = "wall含拟合激活提取/传输与CPU基组/lstsq；CPU process另列，不冒称纯CPU墙钟"
    if proposal is None:
        restore_rng(before_rng)
        train_batch(model, optimizer, x, y, ledger)
        return model, optimizer, event
    event["proposed_removed_nodes"] = len(proposal.dropped_indices)
    coefficients = proposal.coefficients.to(dtype=torch.float32).to(dtype=torch.float64)
    supports = (coefficients.abs() > MIN_COEFFICIENT_ABS).sum(dim=0)
    event.update(coefficient_support_counts_float64=(proposal.coefficients.abs() > MIN_COEFFICIENT_ABS).sum(dim=0).tolist(),
                 coefficient_support_counts_fixed_C32=supports.tolist(),
                 minimum_coefficient_abs=MIN_COEFFICIENT_ABS,
                 minimum_reconstruction_coefficients=MIN_RECONSTRUCTION_COEFFICIENTS)
    if not torch.isfinite(coefficients).all():
        event["reason"] = "float32_coefficients_nonfinite"
    elif bool((supports < MIN_RECONSTRUCTION_COEFFICIENTS).any()):
        event["reason"] = "fixed_C32_fewer_than_two_coefficients_above_1e_8"
    else:
        started = time.perf_counter()
        holdout_hidden = _fit_activations(model, layer_index, holdout_x)
        record_extraction(ledger, len(holdout_x), "task_fixed_C32_residual_holdout", layer_index)
        details, residual_ok = residual_record(proposal, holdout_hidden, coefficients)
        event.update(details)
        event["holdout_wall_seconds"] = time.perf_counter() - started
        proposal = replace(proposal, coefficients=coefficients)
        event["node_proposal"] = proposal.to_dict()
        event["node_proposal"]["coefficients_fixed_precision"] = "float32_values_stored_in_float64_tensor"
        if not residual_ok:
            event["reason"] = "holdout_fixed_C32_residual_exceeds_threshold"
        else:
            synchronize(x.device)
            encoding_started = time.perf_counter()
            before_payload = encoded(model)
            compact = apply_node_reduction(model, proposal)
            payload = encoded(compact)
            decoded = decode_model(payload)
            event.update(previous_export_bytes=len(before_payload), proposed_export_bytes=len(payload),
                         proposed_widths=list(compact.widths), decoded_widths=list(decoded.widths),
                         export_format=structure_stats(decoded, payload)["format"],
                         compensation=copy.deepcopy(compact.node_reduction_metadata))
            width_ok = (compact.widths[layer_index+1] == model.widths[layer_index+1] - len(proposal.dropped_indices)
                        and decoded.widths == compact.widths)
            del decoded
            synchronize(x.device)
            event["construction_encoding_decode_wall_seconds"] = time.perf_counter() - encoding_started
            if not width_ok or len(payload) >= len(before_payload):
                event["reason"] = "no_physical_width_or_real_storage_reduction"
            else:
                new_optimizer = fresh_adam(compact, optimizer.param_groups[0]["lr"], config)
                event["trial_adapted"] = True
                synchronize(x.device)
                trial_started = time.perf_counter()
                train_batch(compact, new_optimizer, x, y, ledger, candidate=True)
                screened, _ = evaluate(compact, sx[screen], sy[screen], config["evaluation_batch_size"], ledger,
                                       "node_candidate_screen")
                event["screen"] = screened
                if within(screened, reference["screen"], config):
                    full, _ = evaluate(compact, sx, sy, config["evaluation_batch_size"], ledger, "node_candidate_full")
                    event["select"] = full
                    adapted_payload = encoded(compact)
                    event["adapted_export_bytes"] = len(adapted_payload)
                    if within(full, reference["select"], config) and len(adapted_payload) < len(before_payload):
                        event.update(accepted=True, removed_nodes=len(proposal.dropped_indices),
                                     reason="heldout_residual_storage_width_and_fixed_task_tolerance")
                        synchronize(x.device)
                        event["candidate_adaptation_evaluation_wall_seconds"] = time.perf_counter() - trial_started
                        return compact, new_optimizer, event
                    event["reason"] = "full_task_tolerance_or_adapted_storage_failed"
                else:
                    event["reason"] = "screen_exceeds_same_epoch_fixed_tolerance"
                ledger["rolled_back_updates"] += 1
                synchronize(x.device)
                event["candidate_adaptation_evaluation_wall_seconds"] = time.perf_counter() - trial_started
    restore_rng(before_rng)
    train_batch(model, optimizer, x, y, ledger)
    if event["trial_adapted"]:
        ledger["fallback_updates"] += 1
    return model, optimizer, event


def random_slot(model, optimizer, fit_x, holdout_x, x, y, ledger, config, source, epoch):
    event = dict(epoch=epoch, proposal=source["proposal"], matched_source_proposal=source["proposal"],
                 layer_index=source["layer_index"], accepted=source["accepted"],
                 trial_adapted=source["trial_adapted"], removed_nodes=source["removed_nodes"],
                 widths_before=list(model.widths),
                 acceptance_rule="主臂接受及删节点数匹配；随机身份和训练拟合补偿；不依据选择/残差接受")
    if not source["accepted"]:
        if source["trial_adapted"]:
            reject_trial_on_original(model, optimizer, x, y, ledger)
            event["reason"] = "matched_trial_rollback_then_original_model_update"
        else:
            train_batch(model, optimizer, x, y, ledger)
            event["reason"] = "matched_no_candidate_or_pre_adaptation_rejection_normal_update"
        return model, optimizer, event
    layer_index = source["layer_index"]
    synchronize(x.device)
    started = time.perf_counter()
    cpu_started = time.process_time()
    proposal = make_random_proposal(model, layer_index, fit_x, source["removed_nodes"], epoch,
                                    source["proposal"], config, ledger)
    event["search_wall_seconds"] = time.perf_counter() - started
    event["search_cpu_process_seconds"] = time.process_time() - cpu_started
    event["search_timing_scope"] = "wall含拟合激活提取/传输与CPU随机同型lstsq；CPU process另列"
    coefficients = proposal.coefficients.to(dtype=torch.float32).to(dtype=torch.float64)
    if not torch.isfinite(coefficients).all():
        raise FloatingPointError("匹配随机补偿C32非有限；不能改用选择反馈或重新挑随机身份")
    holdout_hidden = _fit_activations(model, layer_index, holdout_x)
    record_extraction(ledger, len(holdout_x), "random_fixed_C32_residual_holdout_report_only", layer_index)
    details, _ = residual_record(proposal, holdout_hidden, coefficients)
    event.update(details)
    proposal = replace(proposal, coefficients=coefficients)
    event["node_proposal"] = proposal.to_dict()
    event["node_proposal"]["relative_residuals"] = json_residual(proposal.relative_residuals)
    event["node_proposal"]["coefficients_fixed_precision"] = "float32_values_stored_in_float64_tensor"
    before_payload = encoded(model)
    compact = apply_node_reduction(model, proposal)
    new_optimizer = fresh_adam(compact, optimizer.param_groups[0]["lr"], config)
    train_batch(compact, new_optimizer, x, y, ledger, candidate=True)
    event.update(reason="unconditional_matched_random_nodes_with_fit_activation_compensation",
                 widths_after=list(compact.widths), compensation=copy.deepcopy(compact.node_reduction_metadata),
                 previous_export_bytes=len(before_payload), adapted_export_bytes=len(encoded(compact)))
    return compact, new_optimizer, event


def checkpoint_state(model, optimizer, rows, events, ledger, next_epoch, wall_seconds, arm):
    return dict(node_experiment_version=1, arm=arm, widths=list(model.widths), model=model.state_dict(),
                model_training=model.training, optimizer=optimizer.state_dict(),
                parameter_grad=[None if p.grad is None else p.grad.detach().clone() for p in model.parameters()],
                edge_grad=[None if layer.last_edge_grad is None else layer.last_edge_grad.clone() for layer in model.layers],
                node_reduction_metadata=copy.deepcopy(getattr(model, "node_reduction_metadata", None)),
                rng=rng_state(), rows=rows, events=events, ledger=ledger, next_epoch=next_epoch,
                candidate_cursor=len(events), current_lr=optimizer.param_groups[0]["lr"], wall_seconds=wall_seconds)


def run(args):
    began = time.perf_counter()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    if (config["model_seed"] != 11 or config["data_seed"] != 20261004
            or config["arms"] != ["fixed", "reduce", "random"] or config["source_arm"] != "fc"
            or config["version"] != "h1_nodes_nonzero_v1"
            or config.get("minimum_coefficient_abs") != MIN_COEFFICIENT_ABS
            or config.get("minimum_reconstruction_coefficients") != MIN_RECONSTRUCTION_COEFFICIENTS
            or config.get("nonzero_fit_activation_rule") != "L2_norm_strictly_positive"):
        raise ValueError("非零节点轮必须用唯一根、未剪枝FC来源及固定1e-8/至少2系数规则")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    source_run = ROOT / config["source_run"]
    source_registration = json.loads((source_run / "registration.json").read_text(encoding="utf-8"))
    for file, expected in source_registration["source_sha256"].items():
        if sha(ROOT / file) != expected:
            raise ValueError(f"父实验源文件不匹配：{file}")
    parent_summary = json.loads((source_run / "summary.json").read_text(encoding="utf-8"))
    if not all(result["frozen"] for result in parent_summary["arms"].values()):
        raise ValueError("父实验尚未全部冻结")
    initial_widths = source_registration["config"]["tasks"]["mnist_static"]["widths"]
    source_checkpoint = source_run / "fc" / "resume.pt"
    folder = ROOT / "experiments/runs" / config["version"] / "mnist_static"
    folder.mkdir(parents=True, exist_ok=True)
    lock = folder / "running.lock"
    if lock.exists():
        if pid_alive(int(lock.read_text())):
            raise RuntimeError("运行锁仍活动，拒绝重复启动")
        lock.unlink()
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    cost_path = folder / "session_cost.json"
    past_cost = json.loads(cost_path.read_text(encoding="utf-8")) if cost_path.exists() else dict(cumulative_wall_seconds=0, sessions=0)
    try:
        bundle = load_bundle("mnist", include_confirm=False)
        if (bundle.fit_x.shape != (3200, 784) or bundle.select_x.shape != (800, 784) or len(bundle.confirm_x)):
            raise ValueError("节点实验仅允许3200/800，冻结前不加载control数组")
        partition = fit_partition(bundle.fit_y)
        registration = dict(config=config, task="mnist_static", device=str(device), torch=torch.__version__,
            source_sha256={file: sha(ROOT / file) for file in SOURCES},
            source_checkpoint_sha256=sha(source_checkpoint), source_registration_sha256=sha(source_run / "registration.json"),
            parent_run=config["source_run"], parent_arm="fc", initial_widths=initial_widths, activation_partition=partition,
            provenance_note="父未剪枝FC第25轮末态；隔离首个节点缩减，不使用稀疏图作补偿起点")
        registration_path = folder / "registration.json"
        if registration_path.exists():
            if json.loads(registration_path.read_text(encoding="utf-8")) != registration or not args.resume:
                raise ValueError("运行已存在或实现/配置/父状态不同；只允许相同实现--resume")
        else:
            write_json(registration_path, registration)
        if (folder / "summary.json").exists():
            print(json.dumps(dict(already_complete=True, run=str(folder))), flush=True)
            return
        if past_cost["cumulative_wall_seconds"] >= config["maximum_wall_seconds"]:
            raise RuntimeError("节点阶段跨会话累计墙钟预算用完，--resume不重置预算")
        write_json(folder / "running.json", dict(pid=os.getpid(), status="running"))
        startup = json.loads((source_run / "startup.json").read_text(encoding="utf-8"))
        if not startup["ok"] or startup["widths"] != initial_widths:
            raise ValueError("既有同尺寸启动证据无效")
        write_json(folder / "startup.json", dict(reused_from=str(source_run / "startup.json"),
            forward_samples=0, backward_samples=0, updates=0, additional_gpu_checks=0,
            note="复用同尺寸GPU与新增节点CPU检查，不重复诊断；实际训练非有限保护"))
        write_json(folder / "data_manifest.json", dict(bundle.manifest, node_activation_partition=partition))
        fx, fy, sx, sy = [torch.from_numpy(array).to(device) for array in
                         (bundle.fit_x, bundle.fit_y, bundle.select_x, bundle.select_y)]
        proposal_x = fx[torch.as_tensor(partition["fit_indices"], device=device)]
        holdout_x = fx[torch.as_tensor(partition["holdout_indices"], device=device)]
        screen = select_screen(sy, config["screen_samples"])
        parent = torch.load(source_checkpoint, map_location=device, weights_only=False)
        if parent["next_epoch"] != 26:
            raise ValueError("父FC恢复点必须是第25轮末态")
        results = {}
        candidate_layers = {slot["epoch"]: slot["layer_index"] for slot in config["candidate_schedule"]}
        for arm in config["arms"]:
            arm_folder = folder / arm
            arm_folder.mkdir(exist_ok=True)
            if (arm_folder / "result.json").exists():
                result = json.loads((arm_folder / "result.json").read_text(encoding="utf-8"))
                saved = torch.load(arm_folder / "resume.pt", map_location=device, weights_only=False)
                result["ledger"] = saved["ledger"]
                if (arm_folder / "fit_final.npz").exists():
                    with np.load(arm_folder / "fit_final.npz", allow_pickle=False) as predictions:
                        result["fit"] = metrics(predictions["logits"], predictions["labels"])
                    reconcile_cached_evaluation(result["ledger"], "frozen_fit", result["fit"]["samples"])
                else:
                    model = restore_model(saved, device, initial_widths)
                    result["fit"], predictions = evaluate(model, fx, fy, config["evaluation_batch_size"], result["ledger"], "frozen_fit")
                    np.savez_compressed(arm_folder / "fit_final.npz", logits=predictions, labels=bundle.fit_y)
                    del model
                saved["ledger"] = result["ledger"]
                save_checkpoint(arm_folder / "resume.pt", saved)
                write_json(arm_folder / "result.json", result)
                results[arm] = result
                continue
            resume_path = arm_folder / "resume.pt"
            resumed = resume_path.exists()
            saved = torch.load(resume_path, map_location=device, weights_only=False) if resumed else parent
            model = restore_model(saved, device, initial_widths)
            optimizer = fresh_adam(model, config["learning_rate"], config)
            optimizer.load_state_dict(copy.deepcopy(saved["optimizer"]))
            restore_rng(saved["rng"])
            rows, events = (saved["rows"], saved["events"]) if resumed else ([], [])
            ledger = saved["ledger"] if resumed else new_ledger()
            next_epoch = saved["next_epoch"] if resumed else 1
            arm_wall = saved["wall_seconds"] if resumed else 0.
            checkpoint = saved if resumed else None  # next_epoch=7时也可直接完成冻结。
            if not resumed:
                (arm_folder / "initial_model.bgs").write_bytes(encoded(model))
            for epoch in range(next_epoch, config["epochs"] + 1):
                synchronize(device)
                started = time.perf_counter()
                lr = config["learning_rate"] * .5 ** sum(epoch >= k for k in config["lr_milestones"])
                for group in optimizer.param_groups:
                    group["lr"] = lr
                order = np.random.default_rng(np.random.SeedSequence([20261004, 25 + epoch])).permutation(len(fx))
                batches = [torch.as_tensor(order[start:start+config["batch_size"]], device=device)
                           for start in range(0, len(fx), config["batch_size"])]
                losses = [train_batch(model, optimizer, fx[ids], fy[ids], ledger) for ids in batches[:-1]]
                ids, event = batches[-1], None
                if epoch in candidate_layers:
                    optimizer = fresh_adam(model, lr, config)
                    ledger["adam_reset_epochs"].append(epoch)
                    if arm == "reduce":
                        model, optimizer, event = reduction_slot(model, optimizer, proposal_x, holdout_x,
                            fx[ids], fy[ids], sx, sy, screen, results["fixed"]["rows"][epoch-1], ledger,
                            config, epoch, len(events)+1, candidate_layers[epoch])
                    elif arm == "random":
                        sources = [event for event in results["reduce"]["events"] if event["epoch"] == epoch]
                        if len(sources) != 1:
                            raise RuntimeError("随机节点臂缺少唯一同轮主候选记录")
                        model, optimizer, event = random_slot(model, optimizer, proposal_x, holdout_x,
                            fx[ids], fy[ids], ledger, config, sources[0], epoch)
                    else:
                        train_batch(model, optimizer, fx[ids], fy[ids], ledger)
                else:
                    train_batch(model, optimizer, fx[ids], fy[ids], ledger)
                if event is not None:
                    events.append(event)
                selected, scores = evaluate(model, sx, sy, config["evaluation_batch_size"], ledger, "epoch_select")
                payload = encoded(model)
                structure = structure_stats(model, payload)
                net_updates = ledger["attempted_updates"] - ledger["rolled_back_updates"]
                if net_updates != epoch * 50:
                    raise RuntimeError("节点候选占用了正常50次/轮净保留更新")
                if arm == "random":
                    reference = results["reduce"]["rows"][epoch-1]
                    if (list(model.widths) != reference["widths"] or
                            {key: ledger[key] for key in UPDATE_KEYS} != reference["updates"] or
                            ledger["adam_reset_epochs"] != reference["adam_reset_epochs"]):
                        raise RuntimeError("随机节点宽度/适应更新/重置时序与主臂失配")
                synchronize(device)
                elapsed = time.perf_counter() - started
                arm_wall += elapsed
                row = dict(epoch=epoch, select=selected,
                    screen=metrics(scores[screen.cpu().numpy()], bundle.select_y[screen.cpu().numpy()]),
                    widths=list(model.widths), structure=structure, mean_ordinary_fit_loss=float(np.mean(losses)),
                    wall_seconds=elapsed, learning_rate=lr, adam_reset_epochs=list(ledger["adam_reset_epochs"]),
                    updates={key: ledger[key] for key in UPDATE_KEYS}, net_updates=net_updates)
                rows.append(row)
                np.savez_compressed(arm_folder / f"select_{epoch:03d}.npz", logits=scores, labels=bundle.select_y)
                checkpoint = checkpoint_state(model, optimizer, rows, events, ledger, epoch+1, arm_wall, arm)
                save_checkpoint(resume_path, checkpoint)
                persist_session_cost(cost_path, past_cost, began, config["maximum_wall_seconds"])
                write_json(arm_folder / "progress.json", dict(rows=rows, events=events, ledger=ledger, wall_seconds=arm_wall))
                print(json.dumps(dict(arm=arm, epoch=epoch, accuracy=selected["accuracy"], loss=selected["loss"],
                    widths=list(model.widths), export_bytes=len(payload), event=event and event["reason"])), flush=True)
                if ((folder / "request_stop").exists() or
                        past_cost["cumulative_wall_seconds"] + time.perf_counter() - began >= config["maximum_wall_seconds"]):
                    write_json(folder / "budget_stop.json", dict(reason="request_or_cumulative_stage_wall_budget",
                        completed_arm=arm, completed_epoch=epoch))
                    return
            payload = encoded(model)
            (arm_folder / "final_model.bgs").write_bytes(payload)
            if (arm_folder / "fit_final.npz").exists():
                with np.load(arm_folder / "fit_final.npz", allow_pickle=False) as predictions:
                    fit = metrics(predictions["logits"], predictions["labels"])
                reconcile_cached_evaluation(ledger, "frozen_fit", fit["samples"])
            else:
                fit, predictions = evaluate(model, fx, fy, config["evaluation_batch_size"], ledger, "frozen_fit")
                np.savez_compressed(arm_folder / "fit_final.npz", logits=predictions, labels=bundle.fit_y)
            checkpoint.update(ledger=ledger, rng=rng_state())
            save_checkpoint(resume_path, checkpoint)
            result = dict(arm=arm, frozen=True, final_epoch=config["epochs"], fit=fit, rows=rows, events=events,
                structure=structure_stats(model, payload), widths=list(model.widths), wall_seconds=arm_wall, ledger=ledger)
            results[arm] = result
            write_json(arm_folder / "result.json", result)
            del model, optimizer
        write_json(folder / "freeze.json", dict(arms={arm: result["final_epoch"] for arm, result in results.items()},
            before_repeated_exploratory_evaluation=True, source_sha256=registration["source_sha256"]))
        confirmed = load_bundle("mnist", include_confirm=True)
        cx, cy = torch.from_numpy(confirmed.confirm_x).to(device), torch.from_numpy(confirmed.confirm_y).to(device)
        for arm, result in results.items():
            arm_folder = folder / arm
            saved = torch.load(arm_folder / "resume.pt", map_location=device, weights_only=False)
            result["ledger"] = saved["ledger"]
            if (arm_folder / "confirm_final.npz").exists():
                with np.load(arm_folder / "confirm_final.npz", allow_pickle=False) as predictions:
                    confirmation = metrics(predictions["logits"], predictions["labels"])
                reconcile_cached_evaluation(result["ledger"], "repeated_exploration", confirmation["samples"])
            else:
                model = restore_model(saved, device, initial_widths)
                confirmation, scores = evaluate(model, cx, cy, config["evaluation_batch_size"], result["ledger"], "repeated_exploration")
                np.savez_compressed(arm_folder / "confirm_final.npz", logits=scores, labels=confirmed.confirm_y)
                del model
            result["confirmation"] = confirmation
            saved["ledger"] = result["ledger"]
            save_checkpoint(arm_folder / "resume.pt", saved)
            write_json(arm_folder / "result.json", result)
        comparison = {arm: dict(
            select_accuracy_delta_vs_fixed=result["rows"][-1]["select"]["accuracy"]-results["fixed"]["rows"][-1]["select"]["accuracy"],
            select_loss_delta_vs_fixed=result["rows"][-1]["select"]["loss"]-results["fixed"]["rows"][-1]["select"]["loss"],
            accepted=sum(event["accepted"] for event in result["events"]),
            hidden_nodes_removed=sum(initial_widths[1:-1])-sum(result["widths"][1:-1]),
            storage_reduction_vs_fixed=1-result["structure"]["export_bytes"]/results["fixed"]["structure"]["export_bytes"])
            for arm, result in results.items()}
        summary = dict(task="mnist_static", registration=registration, arms=results, comparison=comparison,
            runtime_ratio_is_gate=False, random_widths_and_updates_matched=True,
            confirmation_status=config["confirmation_role"], current_session_wall_seconds=time.perf_counter()-began,
            cumulative_wall_seconds=past_cost["cumulative_wall_seconds"]+time.perf_counter()-began,
            recovery_accounting_scope="保证完整轮及已保存末态预测成本；评价已发生但预测/账本均未落盘的中断成本不能完整追溯",
            peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            execution_note="真实隐藏宽度/参数槽缩小，但仍稠密gather执行；未实测推断加速或能耗")
        write_json(folder / "summary.json", summary)
        print(json.dumps(dict(complete=True, comparison=comparison)), flush=True)
    finally:
        persist_session_cost(cost_path, past_cost, began, config["maximum_wall_seconds"])
        lock.unlink(missing_ok=True)
        write_json(folder / "running.json", dict(pid=None, status="exited"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/h1_nodes_nonzero_v1.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
