"""H1 独立剪枝比较；随机臂无条件匹配主剪枝接受序列及每层删边数。"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import random
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.nn import functional as F

from braingrow.codec import decode_model, encode_model, export_stats
from braingrow.datasets import load_bundle
from braingrow.models import StructuredMLP, ConvReference1D, ConvReference2D
from braingrow.structural import CandidateTransaction, Proposal, StructureController

ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILES = ("tools/train_pruning.py", "src/braingrow/pruning_experiment.py",
                "src/braingrow/models.py", "src/braingrow/codec.py",
                "src/braingrow/datasets.py", "src/braingrow/structural.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    os.replace(temporary, path)


def save_checkpoint(path, value):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def pid_alive(pid):
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            if ctypes.get_last_error() == 87:
                return False
            raise PermissionError("无法只读核实运行锁进程，拒绝猜测或重复启动")
        try:
            code = wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
                raise OSError("读取运行锁进程状态失败")
            return code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def metrics(logits, labels):
    labels = np.asarray(labels)
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits-logits.max(axis=1, keepdims=True)
    loss = np.log(np.exp(shifted).sum(axis=1))-shifted[np.arange(len(labels)), labels]
    predictions = logits.argmax(axis=1)
    confusion = np.zeros((logits.shape[1], logits.shape[1]), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    return {"samples": len(labels), "accuracy": float(np.mean(predictions == labels)),
            "loss": float(loss.mean()), "confusion_matrix": confusion.tolist()}


def evaluate(model, x, y, batch_size, ledger, purpose):
    before = model.training
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            scores = model(x[start:start+batch_size])
            if not torch.isfinite(scores).all():
                raise FloatingPointError(f"非有限输出: {purpose}")
            output.append(scores.cpu().numpy())
    model.train(before)
    logits = np.concatenate(output)
    ledger["evaluation_forward_samples"] += len(x)
    ledger["evaluations"].append({"purpose": purpose, "samples": len(x)})
    return metrics(logits, y.cpu().numpy()), logits


def train_batch(model, optimizer, x, y, ledger, controller=None, candidate=False):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(x), y)
    if not torch.isfinite(loss):
        raise FloatingPointError("训练损失非有限")
    loss.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise FloatingPointError("训练梯度非有限")
    if controller is not None and not candidate:
        controller.observe_gradients(model)
    optimizer.step()
    ledger["optimization_forward_samples"] += len(x)
    ledger["optimization_backward_samples"] += len(x)
    ledger["attempted_updates"] += 1
    ledger["candidate_updates"] += int(candidate)
    return float(loss.detach())


def encoded(model):
    before = model.training
    model.eval()
    if isinstance(model, StructuredMLP):
        payload = encode_model(model)
    else:
        # CNN参照保留架构说明与全部state；不与共享编码混淆为相同格式。
        buffer = io.BytesIO()
        torch.save({"architecture": type(model).__name__, "state": model.state_dict()}, buffer)
        payload = buffer.getvalue()
    model.train(before)
    return payload


def structure_stats(model, payload):
    if isinstance(model, StructuredMLP):
        result = export_stats(model, payload)
        result["hidden_units"] = sum(model.widths[1:-1])
        result["output_units"] = model.widths[-1]
        result["total_computing_units"] = sum(model.widths[1:])
        result["execution"] = "dense_gather_matmul_no_compute_reuse"
        return result
    return {"format": "torch_state_with_architecture", "export_bytes": len(payload),
            "independent_parameters": sum(p.numel() for p in model.parameters()),
            "execution": "predefined_convolution_reference", "learned_structure": False}


def within(candidate, reference, config):
    return (candidate["accuracy"] >= reference["accuracy"]-config["accuracy_tolerance"]
            and candidate["loss"] <= reference["loss"]+config["loss_tolerance"])


def select_screen(y, count):
    labels = y.cpu().numpy()
    classes = int(labels.max())+1
    selections = []
    for label in range(classes):
        n = count//classes + int(label < count % classes)
        selections.extend(np.flatnonzero(labels == label)[:n].tolist())
    return torch.as_tensor(selections, device=y.device, dtype=torch.long)


def perform_candidate(model, optimizer, controller, train_x, train_y, select_x, select_y,
                      screen, reference, reference_screen, ledger, config, epoch):
    proposal = controller.propose(model)
    event = {"proposal": controller.proposals, "epoch": epoch, "kind": proposal.kind,
             "edited_pairs_or_edges": proposal.count, "accepted": False}
    counts = [0] * len(model.shared_layers())
    for layer_index, indices in proposal.edits:
        counts[layer_index] = len(indices)
    event["layer_edit_counts"] = counts
    if not proposal.count:
        event["reason"] = "no_candidate"
        train_batch(model, optimizer, train_x, train_y, ledger, controller)
        return event
    previous_bytes = len(encoded(model))
    # 候选提议游标已前进，scheduler由epoch固定决定且此事务不改变它。
    with CandidateTransaction(model, optimizer, controller) as transaction:
        controller.apply(model, optimizer, proposal)
        train_batch(model, optimizer, train_x, train_y, ledger, candidate=True)
        new_payload = encoded(model)
        event.update(previous_export_bytes=previous_bytes, proposed_export_bytes=len(new_payload))
        if len(new_payload) >= previous_bytes:
            event["reason"] = "no_real_storage_gain"
        else:
            screened, _ = evaluate(model, select_x[screen], select_y[screen], config["evaluation_batch_size"],
                                   ledger, "candidate_screen")
            event["screen"] = screened
            if not within(screened, reference_screen, config):
                event["reason"] = "screen_exceeds_fc_tolerance"
            else:
                full, _ = evaluate(model, select_x, select_y, config["evaluation_batch_size"], ledger,
                                   "candidate_full_select")
                event["select"] = full
                if within(full, reference, config):
                    event["accepted"] = True
                    event["reason"] = "storage_gain_and_task_preservation"
                    controller.accepted += 1
                    transaction.commit()
                else:
                    event["reason"] = "full_select_exceeds_fc_tolerance"
        if not event["accepted"]:
            ledger["rolled_back_updates"] += 1
    return event


def perform_random_match(model, optimizer, controller, train_x, train_y,
                         select_x, select_y, screen, source, ledger, config, epoch):
    """主臂拒绝则适应后回退，主臂接受则强制随机删除；评价不作接受依据。"""
    controller.proposals += 1
    event = {"proposal": source["proposal"], "epoch": epoch, "kind": "random_prune",
             "matched_source_proposal": source["proposal"],
             "source_prune_accepted": source["accepted"], "accepted": source["accepted"],
             "layer_edit_counts": source["layer_edit_counts"],
             "edited_pairs_or_edges": source["edited_pairs_or_edges"] if source["accepted"] else 0,
             "acceptance_rule": "unconditional_matching_of_prune_acceptance_no_select_feedback"}
    if not source["accepted"]:
        if source.get("reason") == "no_candidate":
            train_batch(model, optimizer, train_x, train_y, ledger, controller)
            event["reason"] = "source_no_candidate_ordinary_gradient_update"
        else:
            with CandidateTransaction(model, optimizer, controller):
                train_batch(model, optimizer, train_x, train_y, ledger, candidate=True)
                ledger["rolled_back_updates"] += 1
            event["reason"] = "prune_rejected_slot_one_batch_update_then_rollback"
        return event
    edits = []
    for layer_index, (layer, count) in enumerate(zip(model.shared_layers(), source["layer_edit_counts"])):
        live = np.flatnonzero(layer.group_index.detach().cpu().numpy() >= 0)
        expected = int(len(live) * config["edit_fraction"])
        if count != expected:
            raise RuntimeError("随机对照与主剪枝每层 5% 提议规模不一致")
        stream = np.random.SeedSequence([config["model_seed"], epoch, source["proposal"], layer_index])
        chosen = np.random.default_rng(stream).choice(live, size=count, replace=False)
        edits.append((layer_index, chosen))
    before, _ = evaluate(model, select_x[screen], select_y[screen], config["evaluation_batch_size"],
                         ledger, "random_before_delete_screen")
    previous_bytes = len(encoded(model))
    with CandidateTransaction(model, optimizer, controller) as transaction:
        controller.apply(model, optimizer, Proposal("prune", edits, sum(source["layer_edit_counts"])))
        immediate, _ = evaluate(model, select_x[screen], select_y[screen], config["evaluation_batch_size"],
                                ledger, "random_immediate_delete_screen")
        train_batch(model, optimizer, train_x, train_y, ledger, candidate=True)
        adapted, _ = evaluate(model, select_x[screen], select_y[screen], config["evaluation_batch_size"],
                              ledger, "random_after_adaptation_screen")
        controller.accepted += 1
        event.update(reason="unconditional_random_delete_and_one_batch_adaptation",
                     before_screen=before, immediate_screen=immediate, adapted_screen=adapted,
                     previous_export_bytes=previous_bytes, proposed_export_bytes=len(encoded(model)))
        transaction.commit()
    return event


def initialize(arm, spec, seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if arm == "cnn":
        model = ConvReference1D() if spec["widths"][0] == 64 else ConvReference2D()
    else:
        model = StructuredMLP(spec["widths"])
    return model.to(device)


def run(args):
    session_started = time.perf_counter()
    config = json.loads((ROOT/args.config).read_text(encoding="utf-8"))
    spec = config["tasks"][args.task]
    if not config["version"].startswith("pruning_"):
        raise ValueError("剪枝独立入口只允许 pruning_ 版本目录，保护旧研究恢复文件")
    if config["arms"] != ["fc", "prune", "random", "cnn"]:
        raise ValueError("独立剪枝实验必须按 fc/prune/random/cnn 顺序完成")
    if config["model_seed"] != 11 or config["data_seed"] != 20261004:
        raise ValueError("当前阶段唯一模型根 11、数据根 20261004")
    folder = ROOT/"experiments"/"runs"/config["version"]/args.task
    folder.mkdir(parents=True, exist_ok=True)
    lock = folder/"running.lock"
    if lock.exists():
        pid = int(lock.read_text())
        if pid_alive(pid):
            raise RuntimeError(f"运行锁仍活动，拒绝重复启动PID={pid}")
        lock.unlink()
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    try:
        device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                              "cpu" if args.device == "auto" else args.device)
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        provenance = {"config": config, "task": args.task,
                      "source_sha256": {p: sha(ROOT/p) for p in SOURCE_FILES},
                      "device": str(device), "torch": torch.__version__}
        registration = folder/"registration.json"
        if registration.exists():
            saved = json.loads(registration.read_text(encoding="utf-8"))
            if saved != provenance:
                raise RuntimeError("实现或配置发生变化，拒绝混入原实验；请登记新的版本")
            if not args.resume:
                raise RuntimeError("已存在运行，需--resume，不能覆盖")
        else:
            write_json(registration, provenance)
        if (folder/"summary.json").exists():
            finished = json.loads((folder/"summary.json").read_text(encoding="utf-8"))
            print(json.dumps({"already_complete": True, "gate": finished["gate"]}, ensure_ascii=False), flush=True)
            return
        write_json(folder/"running.json", {"pid": os.getpid(), "device": str(device), "task": args.task})
        task = "mnist" if args.task == "mnist_static" else args.task
        bundle = load_bundle(task, include_confirm=False)
        if (bundle.fit_x.shape != (3200, 784) or bundle.select_x.shape != (800, 784)
                or len(bundle.confirm_x) != 0):
            raise ValueError("MNIST 当前授权阶段必须是 3200/800 且确认数据尚未加载")
        write_json(folder/"data_manifest.json", bundle.manifest)
        fit_x = torch.from_numpy(bundle.fit_x).to(device)
        fit_y = torch.from_numpy(bundle.fit_y).to(device)
        select_x = torch.from_numpy(bundle.select_x).to(device)
        select_y = torch.from_numpy(bundle.select_y).to(device)
        screen = select_screen(select_y, config["screen_samples"])
        if not (folder/"startup.json").exists():
            startup_model = initialize("fc", spec, config["model_seed"], device)
            loss = F.cross_entropy(startup_model(fit_x[:8]), fit_y[:8])
            loss.backward()
            valid = bool(torch.isfinite(loss)) and all(p.grad is not None and torch.isfinite(p.grad).all()
                                                     for p in startup_model.parameters())
            write_json(folder/"startup.json", {"ok": valid, "forward_samples": 8,
                       "backward_samples": 8, "updates": 0, "model_seed": 11,
                       "device": str(device), "loss": float(loss.detach()),
                       "widths": spec["widths"], "precision": "float32",
                       "allocated_parameter_slots": sum(p.numel() for p in startup_model.parameters())})
            del startup_model
            if not valid:
                raise FloatingPointError("8样本启动检查失败")
        elif not json.loads((folder/"startup.json").read_text())["ok"]:
            raise RuntimeError("此前启动检查失败")
        completed = {}
        for arm in config["arms"]:
            arm_folder = folder/arm
            arm_folder.mkdir(exist_ok=True)
            checkpoint_path = arm_folder/"resume.pt"
            model = initialize(arm, spec, config["model_seed"], device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                          weight_decay=config["weight_decay"])
            controller = StructureController(model, "prune", config["edit_fraction"], config["gradient_decay"]) \
                if arm in ("prune", "random") else None
            rows, events, next_epoch = [], [], 1
            ledger = {"optimization_forward_samples": 0, "optimization_backward_samples": 0,
                      "attempted_updates": 0, "candidate_updates": 0, "rolled_back_updates": 0,
                      "evaluation_forward_samples": 0, "evaluations": []}
            arm_wall = 0.0
            if checkpoint_path.exists():
                saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
                state = saved
                model.load_state_dict(saved["model"])
                optimizer.load_state_dict(saved["optimizer"])
                for parameter, grad in zip(model.parameters(), saved["parameter_grad"]):
                    parameter.grad = grad
                if isinstance(model, StructuredMLP):
                    for layer, grad in zip(model.shared_layers(), saved["edge_grad"]):
                        layer.last_edge_grad = grad
                if controller is not None:
                    controller.load_state_dict(saved["controller"])
                rows, events, ledger = saved["rows"], saved["events"], saved["ledger"]
                next_epoch, arm_wall = saved["next_epoch"], saved["wall_seconds"]
                restore_rng(saved["rng"])
            initial_payload = encoded(model) if next_epoch == 1 else None
            if initial_payload is not None:
                (arm_folder/"initial_model.bgs").write_bytes(initial_payload)
            reference_rows = completed["fc"]["rows"] if arm != "fc" else None
            for epoch in range(next_epoch, spec["epochs"]+1):
                synchronize(device)
                started = time.perf_counter()
                for group in optimizer.param_groups:
                    group["lr"] = config["learning_rate"] * .5**sum(epoch >= point for point in spec["lr_milestones"])
                order = np.random.default_rng(np.random.SeedSequence([config["data_seed"], epoch])).permutation(len(fit_x))
                batches = [order[start:start+config["batch_size"]]
                           for start in range(0, len(order), config["batch_size"])]
                matched_events = ([event for event in completed["prune"]["events"] if event["epoch"] == epoch]
                                  if arm == "random" else [])
                proposals_this_epoch = (len(matched_events) if arm == "random" else
                    min(2, spec["max_proposals"]-controller.proposals)
                    if controller is not None and epoch > spec["warmup_epochs"] else 0)
                for ids in batches[:len(batches)-proposals_this_epoch]:
                    index = torch.as_tensor(ids, device=device)
                    train_batch(model, optimizer, fit_x[index], fit_y[index], ledger, controller)
                if proposals_this_epoch:
                    reference = reference_rows[epoch-1]["select"]
                    reference_screen = reference_rows[epoch-1]["screen"]
                    for candidate_index, ids in enumerate(batches[-proposals_this_epoch:]):
                        index = torch.as_tensor(ids, device=device)
                        if arm == "random":
                            event = perform_random_match(model, optimizer, controller, fit_x[index], fit_y[index],
                                                         select_x, select_y, screen, matched_events[candidate_index],
                                                         ledger, config, epoch)
                        else:
                            event = perform_candidate(model, optimizer, controller, fit_x[index], fit_y[index],
                                                      select_x, select_y, screen, reference, reference_screen,
                                                      ledger, config, epoch)
                        events.append(event)
                selected, scores = evaluate(model, select_x, select_y, config["evaluation_batch_size"], ledger,
                                            "epoch_select")
                payload = encoded(model)
                if arm == "random":
                    actual = [layer["active_edges"] for layer in structure_stats(model, payload)["layers"]]
                    expected = [layer["active_edges"] for layer in completed["prune"]["rows"][epoch-1]["structure"]["layers"]]
                    if actual != expected:
                        raise RuntimeError("随机对照与主剪枝本轮各层活跃边数失配")
                synchronize(device)
                elapsed = time.perf_counter()-started
                arm_wall += elapsed
                row = {"epoch": epoch, "select": selected,
                       "screen": metrics(scores[screen.cpu().numpy()], bundle.select_y[screen.cpu().numpy()]),
                       "structure": structure_stats(model, payload), "wall_seconds": elapsed,
                       "learning_rate": optimizer.param_groups[0]["lr"]}
                ledger["net_updates"] = ledger["attempted_updates"]-ledger["rolled_back_updates"]
                row["updates"] = {key: ledger[key] for key in
                                  ("attempted_updates", "candidate_updates", "rolled_back_updates", "net_updates")}
                if arm == "random" and row["updates"] != completed["prune"]["rows"][epoch-1]["updates"]:
                    raise RuntimeError("随机臂与主剪枝本轮尝试/候选/回退/净保留更新数量失配")
                rows.append(row)
                np.savez_compressed(arm_folder/f"select_{epoch:03d}.npz", logits=scores, labels=bundle.select_y)
                state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                         "controller": controller.state_dict() if controller is not None else None,
                         "parameter_grad": [None if p.grad is None else p.grad.detach().clone()
                                            for p in model.parameters()],
                         "edge_grad": [None if layer.last_edge_grad is None else layer.last_edge_grad.clone()
                                       for layer in model.shared_layers()] if isinstance(model, StructuredMLP) else [],
                         "rows": rows, "events": events, "ledger": ledger, "rng": rng_state(),
                         "next_epoch": epoch+1, "wall_seconds": arm_wall}
                save_checkpoint(checkpoint_path, state)
                write_json(arm_folder/"progress.json", {"rows": rows, "events": events, "ledger": ledger,
                                                       "wall_seconds": arm_wall})
                print(json.dumps({"arm": arm, "epoch": epoch, "select_accuracy": selected["accuracy"],
                      "select_loss": selected["loss"], "export_bytes": len(payload)}, ensure_ascii=False), flush=True)
                if (folder/"request_stop").exists():
                    print("安全暂停于完整轮边界；使用--resume继续", flush=True)
                    return
            # 冻结最终轮，不用控制集择优挑一轮。
            payload = encoded(model)
            (arm_folder/"final_model.bgs").write_bytes(payload)
            if not (arm_folder/"fit_final.npz").exists():
                fit_metrics, fit_scores = evaluate(model, fit_x, fit_y, config["evaluation_batch_size"], ledger,
                                                  "frozen_fit")
                np.savez_compressed(arm_folder/"fit_final.npz", logits=fit_scores, labels=bundle.fit_y)
            else:
                with np.load(arm_folder/"fit_final.npz") as cache:
                    fit_metrics = metrics(cache["logits"], cache["labels"])
            result = {"arm": arm, "rows": rows, "events": events, "ledger": ledger,
                      "wall_seconds": arm_wall, "fit": fit_metrics, "structure": structure_stats(model, payload),
                      "final_epoch": spec["epochs"], "frozen": True}
            # 最终评价账本也写回恢复检查点，完成臂不重训或重复前向。
            state.update(ledger=ledger, rng=rng_state())
            save_checkpoint(checkpoint_path, state)
            write_json(arm_folder/"result.json", result)
            completed[arm] = result
            del model, optimizer, controller
        # 所有臂先冻结，才解包确认样本；此后不修改方法/候选。
        write_json(folder/"freeze.json", {"arms": list(completed), "epochs": spec["epochs"],
                                          "source_sha256": provenance["source_sha256"]})
        confirmed = load_bundle(task, include_confirm=True)
        confirm_x = torch.from_numpy(confirmed.confirm_x).to(device)
        confirm_y = torch.from_numpy(confirmed.confirm_y).to(device)
        for arm, result in completed.items():
            arm_folder = folder/arm
            if not (arm_folder/"confirm_final.npz").exists():
                model = initialize(arm, spec, config["model_seed"], device)
                saved = torch.load(arm_folder/"resume.pt", map_location=device, weights_only=False)
                model.load_state_dict(saved["model"])
                confirmation, scores = evaluate(model, confirm_x, confirm_y, config["evaluation_batch_size"],
                                                result["ledger"], "frozen_confirmation")
                np.savez_compressed(arm_folder/"confirm_final.npz", logits=scores, labels=confirmed.confirm_y)
                saved["ledger"] = result["ledger"]
                save_checkpoint(arm_folder/"resume.pt", saved)
                del model
            else:
                with np.load(arm_folder/"confirm_final.npz") as cache:
                    confirmation = metrics(cache["logits"], cache["labels"])
            result["confirmation"] = confirmation
            write_json(arm_folder/"result.json", result)
        fc, prune = completed["fc"], completed["prune"]
        reduction = 1-prune["structure"]["export_bytes"]/fc["structure"]["export_bytes"]
        ratio = prune["wall_seconds"]/fc["wall_seconds"]
        gate = {"primary": "prune", "hypothesis": "H1",
                "fc_learned": fc["rows"][-1]["select"]["accuracy"] >= spec["learning_floor"],
                "task_preserved": within(prune["rows"][-1]["select"], fc["rows"][-1]["select"], config),
                "storage_reduction": reduction, "storage_passed": reduction >= config["minimum_storage_reduction"],
                "runtime_ratio": ratio, "runtime_passed": ratio <= config["maximum_runtime_ratio"]}
        gate["continue"] = bool(gate["fc_learned"] and gate["task_preserved"] and gate["storage_passed"]
                                and gate["runtime_passed"])
        summary = {"task": args.task, "registration": provenance, "arms": completed, "gate": gate,
                   "random_comparison": {"rule": "unconditional_matching_of_prune_acceptance_per_layer_counts_and_net_updates",
                       "matched_edges_each_epoch": True,
                       "matched_attempted_and_net_updates_each_epoch": True,
                       "rng_rule": "SeedSequence([11, epoch, matched_source_proposal, layer_index])",
                       "prune_minus_random_select_accuracy": prune["rows"][-1]["select"]["accuracy"]-completed["random"]["rows"][-1]["select"]["accuracy"],
                       "prune_minus_random_select_loss": prune["rows"][-1]["select"]["loss"]-completed["random"]["rows"][-1]["select"]["loss"],
                       "significance_claim": False},
                   "execution_note": "FC/剪枝/随机臂仍分配535818参数槽并以稠密图执行；存储缩小不代表计算或能耗减少",
                   "runtime_measurement": {"device": str(device), "batch_size": config["batch_size"],
                       "precision": "float32", "cpu_threads": 2,
                       "epoch_wall_includes": "参数更新、候选提议/适应/接受/回退、选择集评价及本轮导出编码",
                       "epoch_wall_excludes": "模型初始化、启动检查、轮后文件保存、冻结拟合/确认评价",
                       "search_extra_cost": "events 与 ledger 单列候选更新、回退更新及评价样本"},
                   "confirmation_status": "synthetic_frozen" if args.task.startswith("synthetic") else "historically_explored_ids",
                   "current_session_wall_seconds": time.perf_counter()-session_started,
                   "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}
        write_json(folder/"summary.json", summary)
        print(json.dumps({"complete": True, "task": args.task, "gate": gate}, ensure_ascii=False), flush=True)
    finally:
        lock.unlink(missing_ok=True)
        write_json(folder/"running.json", {"pid": None, "status": "exited", "task": args.task})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["mnist_static"], default="mnist_static")
    parser.add_argument("--config", default="configs/pruning_v1.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
