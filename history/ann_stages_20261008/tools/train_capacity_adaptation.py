"""父第8轮完整状态延续；冻结结构的有限等预算诊断。"""
from pathlib import Path
import json
import math
import subprocess
import time

import numpy as np
import torch

from tools.train_capacity_baseline import ROOT, digest, evaluate, save_json
from tools.train_capacity_pruning import MaskedFC, capture, restore, train_epoch, export_model, sync, load_export
from braingrow.datasets import load_bundle


def run_arm(name, cfg, base, data, device, folder, parent):
    arm = folder / name
    arm.mkdir()
    saved = torch.load(parent / name / "resume.pt", map_location="cpu", weights_only=False)
    if saved["epoch"] != cfg["parent_epoch"]:
        raise ValueError("父checkpoint轮数不一致")
    model = MaskedFC(base["fc_widths"], base["dropout"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=base["weight_decay"])
    ema = [torch.zeros_like(layer.weight) for layer in model.layers]
    restore(model, optimizer, ema, saved["state"])
    initial_masks = [mask.cpu().clone() for mask in model.masks]
    initial = export_model(model, arm / "initial.bgm")
    ledger = {"attempted_updates": 0, "net_updates": 0, "optimization_samples": 0, "evaluation_samples": 0,
              "evaluation_calls": 0, "new_structure_edits": 0, "rollbacks": 0}
    costs = {"training_seconds": 0., "selection_seconds": 0., "export_seconds": 0., "checkpoint_seconds": 0., "fit_seconds": 0.}
    x, y = torch.from_numpy(data.fit_x).to(device), torch.from_numpy(data.fit_y).to(device)
    sx, sy = torch.from_numpy(data.select_x).to(device), torch.from_numpy(data.select_y).to(device)
    started = time.perf_counter()
    rows = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for additional in range(1, cfg["epochs"] + 1):
        epoch = cfg["parent_epoch"] + additional
        t = time.perf_counter()
        loss = train_epoch(model, optimizer, ema, epoch, x, y, cfg, base, device, ledger)
        sync(device)
        costs["training_seconds"] += time.perf_counter() - t
        ledger["net_updates"] += math.ceil(len(x) / cfg["batch_size"])
        t = time.perf_counter()
        selected, logits = evaluate(model, sx, sy, cfg["evaluation_batch_size"])
        sync(device)
        costs["selection_seconds"] += time.perf_counter() - t
        ledger["evaluation_calls"] += 1
        ledger["evaluation_samples"] += len(sx)
        np.savez_compressed(arm / f"select_epoch_{epoch}.npz", logits=logits, labels=data.select_y)
        t = time.perf_counter()
        exported = export_model(model, arm / "model.bgm")
        costs["export_seconds"] += time.perf_counter() - t
        rows.append({"epoch": epoch, "select": selected, "fit_augmented_cross_entropy": loss,
                     "structure": exported, "learning_rate": optimizer.param_groups[0]["lr"]})
        t = time.perf_counter()
        torch.save({"epoch": epoch, "state": capture(model, optimizer, ema), "ledger": ledger}, arm / "resume.pt")
        costs["checkpoint_seconds"] += time.perf_counter() - t
        save_json(arm / "progress.json", {"rows": rows, "ledger": ledger, "costs": costs})
        print(json.dumps({"arm": name, "global_epoch": epoch, "select_accuracy": selected["accuracy"], "select_ce": selected["cross_entropy"]}), flush=True)
        if time.perf_counter() - started > cfg["wall_seconds_per_arm"]:
            raise TimeoutError("完整轮工程墙钟预算达到")
        if device.type == "cuda" and torch.cuda.max_memory_allocated() > cfg["peak_memory_limit_bytes"]:
            raise MemoryError("登记峰内存上限达到")
    masks_unchanged = all(torch.equal(before, after.cpu()) for before, after in zip(initial_masks, model.masks))
    if not masks_unchanged or initial["bytes"] != exported["bytes"] or initial["active_edges"] != exported["active_edges"]:
        raise ValueError("冻结结构发生改变")
    t = time.perf_counter()
    fit, logits = evaluate(model, x, y, cfg["evaluation_batch_size"])
    sync(device)
    costs["fit_seconds"] += time.perf_counter() - t
    ledger["evaluation_calls"] += 1
    ledger["evaluation_samples"] += len(x)
    np.savez_compressed(arm / "fit_final.npz", logits=logits, labels=data.fit_y)
    result = {"rows": rows, "select": rows[-1]["select"], "fit": fit, "structure": exported,
              "initial_structure": initial, "masks_unchanged": masks_unchanged, "ledger": ledger, "costs": costs,
              "wall_seconds": time.perf_counter() - started, "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None}
    save_json(arm / "result.json", result)
    del model, optimizer, ema, saved
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main():
    cfg_path = ROOT / "configs/h1_capacity_adaptation_v1.json"
    cfg = json.loads(cfg_path.read_text())
    parent = ROOT / "experiments/runs/h1_capacity_pruning_v1/mnist_static"
    parent_summary = json.loads((parent / "summary.json").read_text())
    base = parent_summary["registration"]["base_config"]
    if cfg["model_seed"] != 11 or cfg["data_seed"] != 20261004 or parent_summary["gate"] != "pass":
        raise ValueError("父完成状态/种子不一致")
    if digest(ROOT / "tools/train_capacity_pruning.py") != parent_summary["registration"]["source_sha256"]:
        raise ValueError("父源文件变化，停止")
    if any(abs(parent_summary["results"][n]["rows"][-1]["learning_rate"] - cfg["learning_rate"]) > 1e-15 for n in ("fixed", "task", "random")):
        raise ValueError("延续学习率不一致")
    folder = ROOT / "experiments/runs" / cfg["version"] / "mnist_static"
    folder.mkdir(parents=True, exist_ok=True)
    if any(folder.iterdir()):
        raise FileExistsError("已有内容拒绝覆盖/重跑")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    registration = {"config": cfg, "config_sha256": digest(cfg_path), "source_sha256": digest(Path(__file__)),
                    "parent_source_sha256": digest(ROOT / "tools/train_capacity_pruning.py"),
                    "parent_checkpoint_sha256": {n: digest(parent / n / "resume.pt") for n in ("fixed", "task", "random")},
                    "training_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    "device_name": torch.cuda.get_device_name() if device.type == "cuda" else "CPU", "device": str(device),
                    "precision": "FP32", "torch": torch.__version__, "execution": "dense_linear_zero_mask", "cpu_threads": 2}
    save_json(folder / "registration.json", registration)
    data = load_bundle("mnist", include_confirm=False)
    if data.fit_x.shape != (3200, 784) or data.select_x.shape != (800, 784) or len(data.confirm_x):
        raise ValueError("数据形状/权限错误")
    save_json(folder / "data_manifest.json", data.manifest)
    results = {}
    session_start = time.perf_counter()
    try:
        for name in ("fixed", "task", "random"):
            results[name] = run_arm(name, cfg, base, data, device, folder, parent)
        confirmed = load_bundle("mnist", include_confirm=True)
        control_start = time.perf_counter()
        for name, result in results.items():
            model = load_export(folder / name / "model.bgm").to(device)
            control, logits = evaluate(model, torch.from_numpy(confirmed.confirm_x).to(device), torch.from_numpy(confirmed.confirm_y).to(device), cfg["evaluation_batch_size"])
            result["control"] = control
            result["ledger"]["evaluation_calls"] += 1
            result["ledger"]["evaluation_samples"] += len(confirmed.confirm_x)
            np.savez_compressed(folder / name / "control_final.npz", logits=logits, labels=confirmed.confirm_y)
            result["branch_accuracy_pass"] = all(result[s]["accuracy"] + 1e-7 >= cfg["accuracy_floor"] for s in ("select", "control"))
            save_json(folder / name / "result.json", result)
            del model
        task, fixed, rand = (results[n] for n in ("task", "fixed", "random"))
        catchup = (rand["select"]["accuracy"] + cfg["accuracy_drop"] + 1e-7 >= task["select"]["accuracy"]
                   and rand["select"]["cross_entropy"] <= task["select"]["cross_entropy"] + cfg["loss_increase"])
        retention = (task["branch_accuracy_pass"] and fixed["branch_accuracy_pass"] and task["select"]["accuracy"] + cfg["accuracy_drop"] + 1e-7 >= fixed["select"]["accuracy"]
                     and task["select"]["cross_entropy"] <= fixed["select"]["cross_entropy"] + cfg["loss_increase"])
        summary = {"registration": registration, "results": results, "random_caught_up": catchup, "task_retention_pass": retention,
                   "control_phase_seconds": time.perf_counter() - control_start, "session_wall_seconds": time.perf_counter() - session_start,
                   "official_test_accessed": False, "sealed_remaining_ids_accessed": False,
                   "interpretation_scope": "固定父末轮学习率的额外8轮；未追上不能判断长期容量劣势"}
        save_json(folder / "summary.json", summary)
        print(json.dumps({"random_caught_up": catchup, "task_retention_pass": retention,
                          "control": {n: r["control"]["accuracy"] for n, r in results.items()}}), flush=True)
    except Exception as exc:
        save_json(folder / "failure.json", {"error": repr(exc), "results": results, "session_wall_seconds": time.perf_counter() - session_start})
        raise


if __name__ == "__main__":
    main()
