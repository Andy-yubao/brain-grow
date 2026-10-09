"""有效非零边数接近CNN的固定分层多档曲线，准确率不作逐档接受门槛。"""
from pathlib import Path
import json
import math
import subprocess
import time

import numpy as np
import torch

from tools.train_capacity_baseline import ROOT, digest, evaluate, save_json
from tools.train_capacity_pruning import MaskedFC, capture, restore, train_epoch, export_model, choose_delete, sync, load_export
from braingrow.datasets import load_bundle


def allocation(initial, target):
    total = sum(initial)
    keep = [target * count // total for count in initial]
    rank = sorted(range(len(initial)), key=lambda i: (-(target * initial[i] % total), i))
    for i in rank[:target - sum(keep)]:
        keep[i] += 1
    if sum(keep) != target or any(k > n for k, n in zip(keep, initial)):
        raise ValueError("配额异常")
    return keep


def run_arm(name, cfg, base, data, device, folder, source, quotas, reference=None):
    arm = folder / name
    arm.mkdir()
    saved = torch.load(source, map_location="cpu", weights_only=False)
    model = MaskedFC(base["fc_widths"], base["dropout"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=base["weight_decay"])
    ema = [torch.zeros_like(layer.weight) for layer in model.layers]
    restore(model, optimizer, ema, saved["state"])
    del saved
    x, y = torch.from_numpy(data.fit_x).to(device), torch.from_numpy(data.fit_y).to(device)
    sx, sy = torch.from_numpy(data.select_x).to(device), torch.from_numpy(data.select_y).to(device)
    random_stream = torch.Generator(device=device).manual_seed(cfg["model_seed"] + 5000)
    ledger = {"attempted_updates": 0, "net_updates": 0, "optimization_samples": 0, "evaluation_samples": 0, "evaluation_calls": 0,
              "structure_edits": 0, "rollbacks": 0}
    costs = {"training_seconds": 0., "evaluation_seconds": 0., "proposal_seconds": 0., "export_seconds": 0., "checkpoint_seconds": 0.}
    limit = cfg["epochs"] if name == "task" else len(reference["rows"])
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    rows, stages, events = [], [], []
    collapse_count = 0
    stop_reason = "registered_curve_complete"
    for additional in range(1, limit + 1):
        epoch = cfg["parent_epoch"] + additional
        stage = (additional - 1) // cfg["epochs_per_stage"]
        if (additional - 1) % cfg["epochs_per_stage"] == 0 and name != "fixed":
            before = [int(mask.sum()) for mask in model.masks]
            counts = [n - k for n, k in zip(before, quotas[stage])]
            if any(count < 0 for count in counts):
                raise ValueError("配额反向增长")
            if name == "random" and counts != reference["events"][stage]["delete_counts"]:
                raise ValueError("主/随机逐层删量不匹配")
            t = time.perf_counter()
            deleted = choose_delete(model, ema, counts, random_stream if name == "random" else None)
            model.enforce(optimizer)
            sync(device)
            costs["proposal_seconds"] += time.perf_counter() - t
            np.savez_compressed(arm / f"stage_{stage+1}_deleted.npz", **{f"layer_{i}": a for i, a in enumerate(deleted)})
            del deleted
            events.append({"stage": stage + 1, "additional_epoch": additional, "target_edges": cfg["target_edge_counts"][stage],
                           "before_edges": before, "keep_counts": quotas[stage], "delete_counts": counts})
            ledger["structure_edits"] += 1
        t = time.perf_counter()
        loss = train_epoch(model, optimizer, ema, epoch, x, y, cfg, base, device, ledger)
        sync(device)
        costs["training_seconds"] += time.perf_counter() - t
        ledger["net_updates"] += math.ceil(len(x) / cfg["batch_size"])
        t = time.perf_counter()
        selected, validation_logits = evaluate(model, sx, sy, cfg["evaluation_batch_size"])
        fitted, fit_logits = evaluate(model, x, y, cfg["evaluation_batch_size"])
        sync(device)
        costs["evaluation_seconds"] += time.perf_counter() - t
        ledger["evaluation_calls"] += 2
        ledger["evaluation_samples"] += len(x) + len(sx)
        collapsed = fitted["accuracy"] < cfg["collapse_fit_accuracy"] and selected["accuracy"] < cfg["collapse_validation_accuracy"]
        collapse_count = collapse_count + 1 if collapsed else 0
        stop = name == "task" and collapse_count >= cfg["collapse_consecutive_epochs"]
        if stop:
            stop_reason = "severe_task_collapse"
        counts = [int(mask.sum()) for mask in model.masks]
        nonzero = [int(torch.count_nonzero(layer.weight)) for layer in model.layers]
        if name != "fixed" and (counts != quotas[stage] or sum(nonzero) > cfg["target_edge_counts"][stage]):
            raise ValueError("实际mask/非零边未达到预登记目标")
        row = {"epoch": epoch, "additional_epoch": additional, "stage": stage + 1, "select": selected, "fit": fitted,
               "fit_augmented_cross_entropy": loss, "active_edges": counts, "nonzero_edges": nonzero, "collapse_condition": collapsed,
               "collapse_streak": collapse_count, "learning_rate": optimizer.param_groups[0]["lr"]}
        rows.append(row)
        if additional % cfg["epochs_per_stage"] == 0 or stop or additional == limit:
            stage_dir = arm / f"stage_{stage+1}"
            stage_dir.mkdir()
            t = time.perf_counter()
            exported = export_model(model, stage_dir / "model.bgm")
            costs["export_seconds"] += time.perf_counter() - t
            stage_result = {"stage": stage + 1, "epoch": epoch, "additional_epoch": additional, "target_edges": cfg["target_edge_counts"][stage],
                            "select": selected, "fit": fitted, "structure": exported, "nonzero_edges": nonzero,
                            "active_edge_target_error": sum(counts) - cfg["target_edge_counts"][stage] if name != "fixed" else None,
                            "nonzero_edge_target_error": sum(nonzero) - cfg["target_edge_counts"][stage] if name != "fixed" else None,
                            "stage_epochs_run": (additional - 1) % cfg["epochs_per_stage"] + 1}
            stages.append(stage_result)
            np.savez_compressed(stage_dir / "validation.npz", logits=validation_logits, labels=data.select_y)
            np.savez_compressed(stage_dir / "fit.npz", logits=fit_logits, labels=data.fit_y)
            t = time.perf_counter()
            torch.save({"epoch": epoch, "state": capture(model, optimizer, ema), "ledger": ledger, "stages": stages, "events": events}, stage_dir / "resume.pt")
            costs["checkpoint_seconds"] += time.perf_counter() - t
            save_json(stage_dir / "result.json", stage_result)
        save_json(arm / "progress.json", {"rows": rows, "stages": stages, "events": events, "ledger": ledger, "costs": costs})
        print(json.dumps({"arm": name, "additional_epoch": additional, "stage": stage+1, "validation_accuracy": selected["accuracy"],
                          "fit_accuracy": fitted["accuracy"], "nonzero_edges": sum(nonzero), "collapse_streak": collapse_count}), flush=True)
        if time.perf_counter() - started > cfg["wall_seconds_per_arm"]:
            raise TimeoutError("臂工程预算达到")
        if device.type == "cuda" and torch.cuda.max_memory_allocated() > cfg["peak_memory_limit_bytes"]:
            raise MemoryError("CUDA峰上限达到")
        if stop:
            break
    result = {"rows": rows, "stages": stages, "events": events, "select": selected, "fit": fitted,
              "structure": exported, "nonzero_edges": nonzero, "ledger": ledger, "costs": costs,
              "stop_reason": stop_reason if name == "task" else "matched_main_budget", "wall_seconds": time.perf_counter() - started,
              "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None}
    save_json(arm / "result.json", result)
    del model, optimizer, ema
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main():
    cfg_path = ROOT / "configs/h1_capacity_edge_curve_v1.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    parent = ROOT / "experiments/runs/h1_capacity_adaptation_v1/mnist_static"
    parent_summary = json.loads((parent / "summary.json").read_text(encoding="utf-8"))
    base = json.loads((ROOT / "configs/capacity_baseline_v1.json").read_text(encoding="utf-8"))
    source = parent / "task/resume.pt"
    saved = torch.load(source, map_location="cpu", weights_only=False)
    if saved["epoch"] != cfg["parent_epoch"] or cfg["model_seed"] != 11 or cfg["data_seed"] != 20261004:
        raise ValueError("父全局轮/固定种子不一致")
    initial = [int(saved["state"]["model"][f"mask_{i}"].sum()) for i in range(4)]
    del saved
    quotas = [allocation(initial, target) for target in cfg["target_edge_counts"]]
    folder = ROOT / "experiments/runs" / cfg["version"] / "mnist_static"
    folder.mkdir(parents=True, exist_ok=True)
    if any(folder.iterdir()):
        raise FileExistsError("拒绝覆盖/重训")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    registration = {"config": cfg, "config_sha256": digest(cfg_path), "source_sha256": digest(Path(__file__)), "parent_checkpoint_sha256": digest(source),
                    "parent_source_sha256": digest(ROOT / "tools/train_capacity_pruning.py"), "base_config": base,
                    "training_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    "initial_edges": initial, "layer_quotas": quotas, "quota_rule": "初始比例整数最大余数；并列层号优先",
                    "device": str(device), "device_name": torch.cuda.get_device_name() if device.type == "cuda" else "CPU",
                    "torch": torch.__version__, "precision": "FP32", "execution": "dense_linear_zero_mask", "cpu_threads": 2,
                    "parent_validation_accuracy": parent_summary["results"]["task"]["select"]["accuracy"]}
    save_json(folder / "registration.json", registration)
    data = load_bundle("mnist", include_confirm=False)
    if data.fit_x.shape != (3200, 784) or data.select_x.shape != (800, 784) or len(data.confirm_x):
        raise ValueError("数据权限/形状异常")
    save_json(folder / "data_manifest.json", data.manifest)
    results = {}
    session_start = time.perf_counter()
    try:
        results["task"] = run_arm("task", cfg, base, data, device, folder, source, quotas)
        for name in ("fixed", "random"):
            results[name] = run_arm(name, cfg, base, data, device, folder, source, quotas, results["task"])
        if results["task"]["events"] != results["random"]["events"] or any(r["ledger"]["net_updates"] != results["task"]["ledger"]["net_updates"] for r in results.values()):
            raise ValueError("身份对照规模/时序/更新未匹配")
        confirmed = load_bundle("mnist", include_confirm=True)
        control_start = time.perf_counter()
        for name, result in results.items():
            model = load_export(folder / name / f"stage_{result['stages'][-1]['stage']}" / "model.bgm").to(device)
            measured, logits = evaluate(model, torch.from_numpy(confirmed.confirm_x).to(device), torch.from_numpy(confirmed.confirm_y).to(device), cfg["evaluation_batch_size"])
            result["control"] = measured
            result["ledger"]["evaluation_calls"] += 1
            result["ledger"]["evaluation_samples"] += len(confirmed.confirm_x)
            np.savez_compressed(folder / name / "control_final.npz", logits=logits, labels=confirmed.confirm_y)
            save_json(folder / name / "result.json", result)
            del model
        summary = {"registration": registration, "results": results, "control_phase_seconds": time.perf_counter() - control_start,
                   "session_wall_seconds": time.perf_counter() - session_start, "official_test_accessed": False,
                   "sealed_remaining_ids_accessed": False, "all_planned_stages_reached": len(results["task"]["stages"]) == len(quotas),
                   "matched_random_schedule_and_net_updates": True, "accuracy_is_hard_acceptance_gate": False}
        save_json(folder / "summary.json", summary)
        print(json.dumps({"stages_reached": len(results["task"]["stages"]), "main_stop": results["task"]["stop_reason"],
                          "main_nonzero_edges": sum(results["task"]["nonzero_edges"]), "control": {n:r["control"]["accuracy"] for n,r in results.items()}}),flush=True)
    except Exception as exc:
        save_json(folder / "failure.json", {"error":repr(exc),"results":results,"session_wall_seconds":time.perf_counter()-session_start})
        raise


if __name__ == "__main__":
    main()
