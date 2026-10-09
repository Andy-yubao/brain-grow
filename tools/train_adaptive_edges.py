"""固定节点的训练内输出闭合与有限自适应剪边；不启动归档入口。"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F

from braingrow.datasets import load_bundle
from tools.train_capacity_baseline import ROOT, digest, evaluate, save_json
from tools.train_capacity_pruning import MaskedFC, capture, restore, export_model, load_export, sync
from tools.train_aligned_edges import seed, edge_counts, propose, function_ok, empty_ledger


def atomic_checkpoint(path, model, optimizer, info):
    temporary = path.with_suffix(".tmp")
    torch.save({"state": capture(model, optimizer, []), "info": info}, temporary)
    for attempt in range(21):
        try:
            temporary.replace(path)
            return
        except OSError as exc:
            if getattr(exc, "winerror", None) not in (5, 32) or attempt == 20:
                raise
            time.sleep(.05)


def closed_edges(model, optimizer):
    """实际非零图的输出反向闭合；保留全部物理节点及bias。"""
    before = edge_counts(model)
    reachable = torch.ones(model.layers[-1].out_features, dtype=torch.bool,
                           device=model.layers[-1].weight.device)
    with torch.no_grad():
        for layer, mask in reversed(list(zip(model.layers, model.masks))):
            mask.logical_and_(layer.weight != 0)
            mask.logical_and_(reachable[:, None])
            reachable = mask.any(0)
        model.enforce(optimizer)
    after = edge_counts(model)
    return {"before": before, "after": after,
            "removed_nonzero_per_layer": [a-b for a, b in zip(before["nonzero_edges"], after["nonzero_edges"])],
            "removed_mask_per_layer": [a-b for a, b in zip(before["mask_edges"], after["mask_edges"])],
            "physical_nodes_changed": False, "bias_changed": False}


def shifts(x, offsets):
    offsets = offsets.to(x.device)
    rows = (torch.arange(28, device=x.device)[None, :, None] - offsets[:, 0, None, None]) % 28
    cols = (torch.arange(28, device=x.device)[None, None, :] - offsets[:, 1, None, None]) % 28
    return x.flatten(1).gather(1, (rows * 28 + cols).reshape(len(x), 784))


def calibration(model, x, batch):
    model.eval()
    squares = [torch.zeros(l.in_features, device=x.device) for l in model.layers]
    with torch.no_grad():
        for inputs in x.split(batch):
            h = inputs
            for i, layer in enumerate(model.layers):
                squares[i] += h.square().sum(0)
                z = layer(h)
                h = F.relu(z) if i < len(model.layers)-1 else z
    return [(s / len(x)).sqrt() for s in squares]


def node_proposal(model, input_rms, fraction):
    deleted = []
    with torch.no_grad():
        for layer, mask, rms in zip(model.layers, model.masks, input_rms):
            energy = layer.weight.square() * rms.square()[None, :]
            ranked, order = energy.masked_fill(~mask, float("inf")).sort(dim=1, stable=True)
            alive = mask.sum(1)
            cumulative = ranked.masked_fill(torch.isinf(ranked), 0).cumsum(1)
            total = energy.masked_fill(~mask, 0).sum(1)
            positions = torch.arange(mask.shape[1], device=mask.device)[None, :]
            eligible = (cumulative <= fraction*total[:, None]) & (positions < (alive-1).clamp_min(0)[:, None])
            chosen = torch.zeros_like(mask).scatter_(1, order, eligible)
            strongest = layer.weight.abs().masked_fill(~mask, -1).argmax(1)
            chosen[torch.arange(len(mask), device=mask.device), strongest] = False
            mask.logical_and_(~chosen)
            deleted.append(int(chosen.sum()))
        model.enforce()
    return deleted


def learning_rate(step, adaptive, cfg, slow_restart=None):
    if not adaptive or step <= cfg["first_epochs"]:
        peak, period, index = cfg["learning_rate"], cfg["first_epochs"], (step-1) % cfg["first_epochs"]
    elif slow_restart is not None and step > slow_restart:
        peak, period, index = cfg["oscillation_learning_rate"], cfg["maximum_epochs"]-slow_restart, step-slow_restart-1
    else:
        peak, period, index = cfg["extension_learning_rate"], cfg["maximum_epochs"]-cfg["first_epochs"], step-cfg["first_epochs"]-1
    return cfg["minimum_learning_rate"] + .5*(peak-cfg["minimum_learning_rate"])*(1+math.cos(math.pi*index/period))


def train_epoch(model, optimizer, x, y, cfg, seed_epoch, lr, ledger):
    torch.manual_seed(cfg["model_seed"]+2000+seed_epoch)
    stream = torch.Generator(device=x.device).manual_seed(cfg["data_seed"]+3000+seed_epoch)
    order = torch.randperm(len(x), device=x.device, generator=stream)
    offsets = torch.randint(28, (len(x), 2), device=x.device, generator=stream)
    histogram = np.bincount((offsets[:, 0]*28+offsets[:, 1]).cpu().numpy(), minlength=784)
    ledger["training_position_histogram"] = (np.asarray(ledger.get("training_position_histogram", [0]*784))+histogram).tolist()
    optimizer.param_groups[0]["lr"] = lr
    model.train()
    loss_sum = 0.
    for ids in order.split(cfg["batch_size"]):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(shifts(x[ids], offsets[ids])), y[ids])
        if not torch.isfinite(loss):
            raise FloatingPointError("训练损失非有限")
        loss.backward()
        with torch.no_grad():
            for layer, mask in zip(model.layers, model.masks):
                layer.weight.grad.mul_(mask)
        optimizer.step()
        model.enforce(optimizer)
        ledger["attempted_updates"] += 1
        ledger["optimization_samples"] += len(ids)
        loss_sum += float(loss.detach())*len(ids)
    optimizer.zero_grad(set_to_none=True)
    return {"hard_label_ce": loss_sum/len(x), "learning_rate": lr}


def measured(model, x, y, cfg, ledger, validation=False):
    result, logits = evaluate(model, x, y, cfg["evaluation_batch_size"])
    ledger["evaluation_calls"] += 1
    ledger["evaluation_samples"] += len(x)
    if validation:
        predictions = logits.argmax(1).reshape(-1, cfg["validation_positions_per_id"])
        labels = y.cpu().numpy().reshape(predictions.shape)
        result.update({"original_ids": len(predictions), "positions_per_id": cfg["validation_positions_per_id"],
                       "all_positions_correct_fraction": float((predictions == labels).all(1).mean()),
                       "prediction_consistency_fraction": float((predictions == predictions[:, :1]).all(1).mean()),
                       "sample_independence": "同ID多个位置有关联，非独立样本"})
    return result, logits


def adaptation_status(calibration_rows, cfg):
    if len(calibration_rows) < 3:
        return "insufficient_observations"
    values = [r["cross_entropy"] for r in calibration_rows[-3:]]
    tolerance = max(cfg["plateau_absolute_ce"], cfg["plateau_relative_ce"]*min(values))
    if max(values)-min(values) <= tolerance:
        return "plateau"
    if values[-1] < values[0]-tolerance:
        return "improving"
    return "oscillating_or_worsening"


def decision(adaptive, step, qualified, status, cfg, validation_improving=False):
    if qualified:
        return "accept"
    if not adaptive:
        return "old_reject"
    if status == "plateau" and not validation_improving:
        return "completed_failure"
    if step >= cfg["maximum_epochs"]:
        return "optimization_incomplete"
    return "extend"


def new_model(cfg, device):
    seed(cfg["model_seed"])
    model = MaskedFC(cfg["fc_widths"], cfg["dropout"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    return model, optimizer


def structure(model, path):
    return {**export_model(model, path), **edge_counts(model)}


def run_arm(name, cfg, tensors, reference, folder, device):
    arm = folder/name
    arm.mkdir(exist_ok=True)
    if (arm/"result.json").exists():
        return json.loads((arm/"result.json").read_text(encoding="utf-8"))
    mechanism, controller_kind = name.split("_")
    adaptive = controller_kind == "adaptive"
    source = ROOT/cfg["source_directory"]/mechanism/cfg["source_checkpoint"]
    model, optimizer = new_model(cfg, device)
    saved = torch.load(source, map_location="cpu", weights_only=False)
    restore(model, optimizer, [], saved["state"])
    x, y, sx, sy, cx = tensors
    started = time.perf_counter()
    ledger = empty_ledger()
    rows, events, stages = [], [], []
    costs = {key: 0. for key in ("training_seconds", "validation_seconds", "calibration_seconds", "proposal_seconds", "closure_seconds", "snapshot_seconds", "export_seconds")}
    c = {"slot": 1, "attempt": 1, "step": 0, "mode": "new", "strength": cfg["delete_fraction"] if mechanism == "magnitude" else cfg["node_energy_fraction"],
         "before": None, "event": None, "calibration": [], "slow_restart": None, "retained_epoch": cfg["source_epoch_offset"], "stop": None, "best_path": "initial.pt", "best_key": None}
    prior_wall = 0.
    initial_closure = None
    initial_validation = None
    if (arm/"resume.pt").exists():
        resume = torch.load(arm/"resume.pt", map_location="cpu", weights_only=False)
        restore(model, optimizer, [], resume["state"])
        info = resume["info"]
        ledger, rows, events, stages, costs, c = [info[key] for key in ("ledger", "rows", "events", "stages", "costs", "controller")]
        initial_closure, initial_validation, prior_wall = [info[key] for key in ("initial_closure", "initial_validation", "wall_seconds")]
    else:
        t = time.perf_counter()
        initial_closure = closed_edges(model, optimizer)
        costs["closure_seconds"] += time.perf_counter()-t
        initial_validation, logits = measured(model, sx, sy, cfg, ledger, True)
        initial_structure = structure(model, arm/"initial.bgm")
        stages.append({"slot": 0, "validation": initial_validation, "structure": initial_structure})
        c["best_key"] = [sum(initial_structure["nonzero_edges"]), -initial_validation["accuracy"], initial_validation["cross_entropy"]]
        assert function_ok(initial_validation, reference, cfg), "起点不满足累计功能门槛"
        atomic_checkpoint(arm/"initial.pt", model, optimizer, {"controller": copy.deepcopy(c), "validation": initial_validation, "source_checkpoint": str(source.relative_to(ROOT))})
        np.savez_compressed(arm/"initial_validation.npz", logits=logits, labels=sy.cpu().numpy())

    def elapsed():
        return prior_wall+time.perf_counter()-started

    def save():
        t = time.perf_counter()
        atomic_checkpoint(arm/"resume.pt", model, optimizer, {"ledger": ledger, "rows": rows, "events": events, "stages": stages, "costs": costs,
                          "controller": c, "initial_closure": initial_closure, "initial_validation": initial_validation, "wall_seconds": elapsed()})
        costs["snapshot_seconds"] += time.perf_counter()-t
        save_json(arm/"progress.json", {"ledger": ledger, "rows": rows, "events": events, "stages": stages, "costs": costs,
                  "controller": {k:v for k,v in c.items() if k not in ("before",)}, "wall_seconds": elapsed()})

    def rollback():
        restore(model, optimizer, [], c["before"])
        ledger["rollbacks"] += 1

    save()
    while c["slot"] <= cfg["parent_slots"] and c["stop"] is None:
        if elapsed() >= cfg["wall_seconds_per_arm"]:
            if c["before"] is not None:
                c["event"]["status"] = "arm_budget_exhausted_optimization_incomplete"
                c["event"]["adaptation_epochs"] = c["step"]
                events.append(copy.deepcopy(c["event"]))
                atomic_checkpoint(arm/f"slot_{c['slot']}_attempt_{c['attempt']}_budget_incomplete.pt", model, optimizer,
                                  {"controller": copy.deepcopy(c), "event": copy.deepcopy(c["event"])})
                rollback()
            c["stop"] = "arm_budget_exhausted_optimization_incomplete"
            break
        if c["mode"] == "new":
            optimizer.zero_grad(set_to_none=True)
            c["before"] = capture(model, optimizer, [])
            before_counts = edge_counts(model)
            t = time.perf_counter()
            if mechanism == "magnitude":
                deleted = propose(model, c["strength"])
            else:
                rms = calibration(model, cx, cfg["evaluation_batch_size"])
                deleted = node_proposal(model, rms, c["strength"])
            model.enforce(optimizer)
            sync(device)
            costs["proposal_seconds"] += time.perf_counter()-t
            t = time.perf_counter()
            closure = closed_edges(model, optimizer)
            sync(device)
            costs["closure_seconds"] += time.perf_counter()-t
            c["event"] = {"slot": c["slot"], "attempt": c["attempt"], "strength": c["strength"], "before": before_counts,
                          "direct_deleted_per_layer": deleted, "closure": closure, "candidate_structure": edge_counts(model),
                          "validation_points": [], "status": "training", "accepted": False}
            c["mode"], c["step"], c["calibration"], c["slow_restart"] = "candidate", 0, [], None
            save()
            if sum(deleted)+sum(closure["removed_nonzero_per_layer"]) == 0:
                rollback()
                c["event"]["status"] = "no_edges_proposed"
                events.append(copy.deepcopy(c["event"]))
                c["stop"] = "no_edges_proposed"
                break
        maximum = cfg["maximum_epochs"] if adaptive and c["mode"] == "candidate" else cfg["first_epochs"]
        next_step = c["step"]+1
        t = time.perf_counter()
        trained = train_epoch(model, optimizer, x, y, cfg, c["retained_epoch"]+next_step,
                              learning_rate(next_step, adaptive and c["mode"] == "candidate", cfg, c["slow_restart"]), ledger)
        sync(device)
        costs["training_seconds"] += time.perf_counter()-t
        c["step"] = next_step
        row = {"slot": c["slot"], "attempt": c["attempt"], "step": next_step, "mode": c["mode"], **trained}
        if c["mode"] == "candidate" and adaptive and next_step % cfg["calibration_interval"] == 0:
            t = time.perf_counter()
            calibrated, _ = measured(model, cx, y, cfg, ledger)
            sync(device)
            costs["calibration_seconds"] += time.perf_counter()-t
            c["calibration"].append({"step": next_step, "cross_entropy": calibrated["cross_entropy"], "accuracy": calibrated["accuracy"]})
            row["fixed_training_calibration"] = calibrated
        rows.append(row)
        validation_due = next_step in cfg["validation_epochs"] if adaptive and c["mode"] == "candidate" else next_step == maximum
        if validation_due:
            t = time.perf_counter()
            selected, logits = measured(model, sx, sy, cfg, ledger, True)
            sync(device)
            costs["validation_seconds"] += time.perf_counter()-t
            status = adaptation_status(c["calibration"], cfg)
            prior_points = c["event"]["validation_points"]
            validation_improving = False
            if prior_points and c["mode"] == "candidate":
                previous = prior_points[-1]["validation"]
                validation_improving = (previous["cross_entropy"]-selected["cross_entropy"] > cfg["validation_improvement_ce"]
                                        or selected["accuracy"]-previous["accuracy"] > cfg["validation_improvement_accuracy"])
            action = decision(adaptive, next_step, function_ok(selected, reference, cfg), status, cfg, validation_improving) if c["mode"] == "candidate" else "fallback_complete"
            point = {"step": next_step, "validation": selected, "training_status": status, "validation_improving": validation_improving, "decision": action}
            c["event"]["validation_points"].append(point)
            np.savez_compressed(arm/f"slot_{c['slot']}_attempt_{c['attempt']}_{c['mode']}_{next_step}.npz", logits=logits, labels=sy.cpu().numpy())
            print(json.dumps({"arm": name, "slot": c["slot"], "attempt": c["attempt"], "step": next_step,
                              "accuracy": selected["accuracy"], "ce": selected["cross_entropy"], "decision": action, "training_status": status}), flush=True)
            if action == "extend":
                if status == "oscillating_or_worsening" and c["slow_restart"] is None and next_step >= 16:
                    c["slow_restart"] = next_step
                    point["learning_rate_action"] = "one_lower_peak_restart_then_slow_decay"
            else:
                event = c["event"]
                event.update({"status": action, "adaptation_epochs": next_step, "fixed_training_calibration": copy.deepcopy(c["calibration"]),
                              "validation": selected, "accepted": action == "accept"})
                atomic_checkpoint(arm/f"slot_{c['slot']}_attempt_{c['attempt']}_{c['mode']}.pt", model, optimizer,
                                  {"controller": copy.deepcopy(c), "event": copy.deepcopy(event), "ledger": copy.deepcopy(ledger)})
                if action in ("accept", "fallback_complete"):
                    exported = structure(model, arm/f"slot_{c['slot']}.bgm")
                    stage = {"slot": c["slot"], "validation": selected, "structure": exported, "event": copy.deepcopy(event)}
                    stages.append(stage)
                    ledger["net_updates"] += next_step*math.ceil(len(x)/cfg["batch_size"])
                    c["retained_epoch"] += next_step
                    key = [sum(exported["nonzero_edges"]), -selected["accuracy"], selected["cross_entropy"]]
                    if function_ok(selected, reference, cfg) and key < c["best_key"]:
                        c["best_key"], c["best_path"] = key, f"slot_{c['slot']}.pt"
                    events.append(copy.deepcopy(event))
                    c.update({"slot": c["slot"]+1, "attempt": 1, "step": 0, "mode": "new", "before": None, "event": None,
                              "calibration": [], "slow_restart": None})
                    atomic_checkpoint(arm/f"slot_{stage['slot']}.pt", model, optimizer, {"controller": copy.deepcopy(c), "stage": stage, "ledger": copy.deepcopy(ledger)})
                elif action == "old_reject":
                    events.append(copy.deepcopy(event))
                    rollback()
                    c["mode"], c["step"] = "fallback", 0
                elif action == "completed_failure":
                    rollback()
                    events.append(copy.deepcopy(event))
                    if c["attempt"] >= cfg["maximum_attempts_per_parent"]:
                        c["stop"] = "completed_failure_retry_limit"
                    else:
                        c["strength"] *= .5
                        c["attempt"] += 1
                        c["mode"], c["step"] = "new", 0
                else:
                    rollback()
                    events.append(copy.deepcopy(event))
                    c["stop"] = "optimization_incomplete_no_halving"
        save()
        if device.type == "cuda" and torch.cuda.max_memory_allocated() > cfg["peak_memory_limit_bytes"]:
            raise MemoryError("峰值显存超过预登记8GiB预算")
    save()
    best = torch.load(arm/c["best_path"], map_location="cpu", weights_only=False)
    restore(model, optimizer, [], best["state"])
    selected, logits = measured(model, sx, sy, cfg, ledger, True)
    fit, fit_logits = measured(model, cx, y, cfg, ledger)
    exported = structure(model, arm/"model.bgm")
    atomic_checkpoint(arm/"model_full.pt", model, optimizer, {"controller": best["info"]["controller"], "validation": selected, "ledger": ledger, "selected_full_checkpoint": c["best_path"]})
    np.savez_compressed(arm/"validation_final.npz", logits=logits, labels=sy.cpu().numpy())
    np.savez_compressed(arm/"training_fixed_one_position.npz", logits=fit_logits, labels=y.cpu().numpy())
    result = {"arm": name, "source_checkpoint": str(source.relative_to(ROOT)), "initial_closure": initial_closure, "initial_validation": initial_validation,
              "fixed_reference": reference, "stages": stages, "events": events, "rows": rows, "structure": exported, "validation": selected,
              "training_fixed_one_position": fit, "ledger": ledger, "costs": costs, "wall_seconds": elapsed(), "stop": c["stop"] or "parent_slots_completed",
              "selected_full_checkpoint": c["best_path"], "full_checkpoint_sha256": digest(arm/"model_full.pt"),
              "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
              "official_test_accessed": False, "historical_test_accessed": False, "sealed_remaining_ids_accessed": False}
    save_json(arm/"result.json", result)
    return result


def tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(tree_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(tree_equal(x, y) for x, y in zip(a, b))
    return a == b


def check(cfg):
    torch.set_num_threads(2)
    toy_cfg = {**cfg, "fc_widths": [2, 3, 2, 1], "dropout": 0.}
    model, optimizer = new_model(toy_cfg, torch.device("cpu"))
    inputs = torch.randn(7, 2)
    model(inputs).square().mean().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.layers[-1].weight[0, 1] = 0
        model.masks[-1][0, 1] = False
        model.layers[1].weight[0, 1:] = 0
        model.masks[1][0, 1:] = False
    model.enforce(optimizer)
    saved = capture(model, optimizer, [])
    original_bias = [l.bias.clone() for l in model.layers]
    expected = model.eval()(inputs).detach().clone()
    closed = closed_edges(model, optimizer)
    assert sum(closed["removed_nonzero_per_layer"]) > 0
    torch.testing.assert_close(model(inputs), expected, rtol=0, atol=0)
    assert all(torch.equal(a, l.bias) for a, l in zip(original_bias, model.layers))
    assert sum(closed_edges(model, optimizer)["removed_nonzero_per_layer"]) == 0
    for layer, mask in zip(model.layers, model.masks):
        assert not torch.count_nonzero(layer.weight[~mask])
        for k in ("exp_avg", "exp_avg_sq"):
            assert not torch.count_nonzero(optimizer.state[layer.weight][k][~mask])
    restore(model, optimizer, [], saved)
    assert tree_equal(capture(model, optimizer, []), saved)
    assert decision(True, 48, False, "improving", cfg) == "optimization_incomplete"
    assert decision(True, 16, False, "plateau", cfg) == "completed_failure"
    assert decision(True, 16, False, "plateau", cfg, True) == "extend"
    assert decision(True, 48, False, "plateau", cfg, True) == "optimization_incomplete"
    assert decision(True, 8, True, "insufficient_observations", cfg) == "accept"
    strength = cfg["delete_fraction"]
    if decision(True, 16, False, "plateau", cfg) == "completed_failure":
        strength *= .5
    assert strength == .25
    import tempfile
    with tempfile.TemporaryDirectory(dir=ROOT/"tmp") as directory:
        path = Path(directory)/"full.pt"
        closed_edges(model, optimizer)
        chosen = capture(model, optimizer, [])
        atomic_checkpoint(path, model, optimizer, {"controller": {"step": 16, "mode": "candidate", "strength": .25}})
        with torch.no_grad():
            model.layers[0].weight.add_(1)
        best = torch.load(path, map_location="cpu", weights_only=False)
        restore(model, optimizer, [], best["state"])
        assert tree_equal(capture(model, optimizer, []), chosen)
        assert best["info"]["controller"]["step"] == 16
    source_dir = ROOT/cfg["source_directory"]
    sources = {}
    for mechanism in ("magnitude", "node"):
        path = source_dir/mechanism/cfg["source_checkpoint"]
        original = torch.load(path, map_location="cpu", weights_only=False)["state"]
        frozen = load_export(source_dir/mechanism/"model.bgm")
        assert tree_equal(frozen.state_dict(), original["model"])
        assert original["optimizer"]["state"] and original["cuda_rng"]
        sources[mechanism] = {"checkpoint_sha256": digest(path), "frozen_model_sha256": digest(source_dir/mechanism/"model.bgm"), "exact_model_match": True}
    report = {"ok": True, "device": "CPU", "new_gpu_check": False, "dataset_access": False,
              "closure_output_equal": True, "closure_idempotent": True, "masked_adam_zero": True,
              "full_rollback_exact": True, "incomplete_does_not_halve": True, "completed_failure_halves": True,
              "best_full_state_restore": True, "sources": sources, "source_sha256": digest(Path(__file__)), "config_sha256": digest(ROOT/"configs/h1_adaptive_edges_v1.json")}
    target = ROOT/"docs/reports/2026-10-08/h1_adaptive_edges_v1"
    target.mkdir(parents=True, exist_ok=True)
    save_json(target/"check.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg_path = ROOT/"configs/h1_adaptive_edges_v1.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if args.check:
        check(cfg)
        return
    assert cfg["model_seed"] == 11 and cfg["data_seed"] == 20261004
    assert cfg["fc_widths"] == [784, 3136, 1568, 128, 10]
    folder = ROOT/"experiments/runs"/cfg["version"]/"mnist_circular"
    folder.mkdir(parents=True, exist_ok=True)
    if any(folder.iterdir()) and not args.resume:
        raise FileExistsError("已有证据，只允许--resume接续")
    if (folder/"summary.json").exists():
        print((folder/"summary.json").read_text(encoding="utf-8"), flush=True)
        return
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("完整实验需要既有GPU；不静默改成CPU超预算")
    source_dir = ROOT/cfg["source_directory"]
    source_paths = [source_dir/mechanism/cfg["source_checkpoint"] for mechanism in ("magnitude", "node")]
    source_paths += [source_dir/"position_manifest.npz", source_dir/"baseline/result.json"]
    reused = [ROOT/"tools/train_capacity_pruning.py", ROOT/"tools/train_capacity_baseline.py", ROOT/"tools/train_aligned_edges.py", ROOT/"src/braingrow/datasets.py"]
    registration = {"config": cfg, "config_sha256": digest(cfg_path), "source_sha256": digest(Path(__file__)),
                    "reused_sha256": {str(p.relative_to(ROOT)): digest(p) for p in reused}, "parent_sha256": {str(p.relative_to(ROOT)): digest(p) for p in source_paths},
                    "training_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    "device": str(device), "device_name": torch.cuda.get_device_name(), "torch": torch.__version__, "precision": "FP32", "batch_size": cfg["batch_size"],
                    "execution": "dense_linear_zero_mask", "cpu_threads": 2, "official_test_accessed": False, "historical_test_accessed": False, "sealed_remaining_ids_accessed": False}
    if (folder/"registration.json").exists():
        old = json.loads((folder/"registration.json").read_text(encoding="utf-8"))
        for key in ("config_sha256", "source_sha256", "reused_sha256", "parent_sha256"):
            assert old[key] == registration[key], f"{key}变化，不允许混源恢复"
        registration = old
    else:
        save_json(folder/"registration.json", registration)
    data = load_bundle("mnist", include_confirm=False)
    assert data.fit_x.shape == (3200, 784) and data.select_x.shape == (800, 784) and not len(data.confirm_x)
    positions = np.load(source_dir/"position_manifest.npz")
    assert np.array_equal(positions["train_original_ids"], data.manifest["splits"]["fit"])
    assert np.array_equal(positions["validation_original_ids"], data.manifest["splits"]["select"])
    x, y = torch.from_numpy(data.fit_x).to(device), torch.from_numpy(data.fit_y).to(device)
    sx = shifts(torch.from_numpy(data.select_x).to(device).repeat_interleave(8, 0), torch.from_numpy(positions["validation_offsets"].reshape(-1, 2)))
    sy = torch.from_numpy(data.select_y).to(device).repeat_interleave(8)
    cx = shifts(x, torch.from_numpy(positions["calibration_offsets"][:, 0]))
    reference = json.loads((source_dir/"baseline/result.json").read_text(encoding="utf-8"))["validation"]
    save_json(folder/"data_manifest.json", {"original": data.manifest, "source_positions_sha256": digest(source_dir/"position_manifest.npz"),
              "validation_samples": len(sx), "training_calibration_samples": len(cx), "fixed_reference": reference,
              "positions_rule": "完全复用旧验证8位置和校准1位置", "historical_test_accessed": False})
    results = {}
    try:
        for name in cfg["arms"]:
            if sum(r["wall_seconds"] for r in results.values()) >= cfg["wall_seconds_total"]:
                raise TimeoutError("总预算3600秒达到")
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            results[name] = run_arm(name, cfg, (x, y, sx, sy, cx), reference, folder, device)
            print(json.dumps({"arm_complete": name, "accuracy": results[name]["validation"]["accuracy"], "edges": sum(results[name]["structure"]["nonzero_edges"]), "stop": results[name]["stop"]}), flush=True)
        summary = {"registration": registration, "results": results, "gate": "completed", "total_arm_wall_seconds": sum(r["wall_seconds"] for r in results.values()),
                   "official_test_accessed": False, "historical_test_accessed": False, "sealed_remaining_ids_accessed": False}
        save_json(folder/"summary.json", summary)
        print(json.dumps({"gate": "completed", "wall_seconds": summary["total_arm_wall_seconds"]}), flush=True)
    except Exception as exc:
        save_json(folder/"failure.json", {"error": repr(exc), "completed_arms": list(results), "resume": "python -X utf8 -B -m tools.train_adaptive_edges --resume"})
        raise


if __name__ == "__main__":
    main()
