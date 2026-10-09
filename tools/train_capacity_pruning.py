"""扩容FC三臂剪枝；独立有限预算，不覆盖基线。"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import struct
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch.nn import functional as F

from tools.train_capacity_baseline import ROOT, FourLayerFC, augment, digest, evaluate, save_json
from braingrow.datasets import load_bundle


class MaskedFC(FourLayerFC):
    def __init__(self, widths, dropout):
        super().__init__(widths, dropout)
        for i, layer in enumerate(self.layers):
            self.register_buffer(f"mask_{i}", torch.ones_like(layer.weight, dtype=torch.bool))

    @property
    def masks(self):
        return [getattr(self, f"mask_{i}") for i in range(len(self.layers))]

    def enforce(self, optimizer=None):
        with torch.no_grad():
            for layer, mask in zip(self.layers, self.masks):
                layer.weight.mul_(mask)
                if optimizer is not None:
                    state = optimizer.state.get(layer.weight, {})
                    for key in ("exp_avg", "exp_avg_sq"):
                        if key in state:
                            state[key].mul_(mask)


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    return copy.deepcopy(value)


def capture(model, optimizer, ema):
    return {"model": cpu_tree(model.state_dict()), "optimizer": cpu_tree(optimizer.state_dict()),
            "ema": cpu_tree(ema), "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else []}


def restore(model, optimizer, ema, saved):
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    for target, source in zip(ema, saved["ema"]):
        target.copy_(source)
    random.setstate(saved["python_rng"])
    np.random.set_state(saved["numpy_rng"])
    torch.set_rng_state(saved["torch_rng"])
    if saved["cuda_rng"]:
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
    optimizer.zero_grad(set_to_none=True)


def export_model(model, path):
    header = {"format": "bgm_mask_fp32_v1", "widths": [model.layers[0].in_features] + [l.out_features for l in model.layers],
              "dropout": model.dropout, "sharing": "none", "layers": []}
    arrays = []
    for layer, mask in zip(model.layers, model.masks):
        alive = mask.cpu().numpy().flatten()
        values = layer.weight.detach().cpu().numpy().flatten()[alive].astype("<f4")
        bias = layer.bias.detach().cpu().numpy().astype("<f4")
        bitmap = np.packbits(alive, bitorder="little")
        header["layers"].append({"shape": list(layer.weight.shape), "edges": int(alive.sum()), "bitmap_bytes": len(bitmap), "bias_count": len(bias)})
        arrays.extend((bitmap, values, bias))
    raw = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    with path.open("wb") as target:
        target.write(b"BGM1" + struct.pack("<I", len(raw)) + raw)
        for array in arrays:
            target.write(array.tobytes())
    return {"bytes": path.stat().st_size, "sha256": digest(path), "active_edges": [r["edges"] for r in header["layers"]],
            "independent_parameters": sum(r["edges"] + r["bias_count"] for r in header["layers"]),
            "allocated_parameters": sum(p.numel() for p in model.parameters()), "widths": header["widths"]}


def load_export(path):
    with path.open("rb") as source:
        if source.read(4) != b"BGM1":
            raise ValueError("格式错误")
        header = json.loads(source.read(struct.unpack("<I", source.read(4))[0]))
        model = MaskedFC(header["widths"], header["dropout"])
        with torch.no_grad():
            for layer, mask, row in zip(model.layers, model.masks, header["layers"]):
                alive = np.unpackbits(np.frombuffer(source.read(row["bitmap_bytes"]), dtype=np.uint8), bitorder="little")[:layer.weight.numel()].astype(bool)
                values = np.frombuffer(source.read(4 * row["edges"]), dtype="<f4").copy()
                weights = np.zeros(layer.weight.numel(), dtype=np.float32)
                weights[alive] = values
                layer.weight.copy_(torch.from_numpy(weights.reshape(row["shape"])))
                mask.copy_(torch.from_numpy(alive.reshape(row["shape"])))
                layer.bias.copy_(torch.from_numpy(np.frombuffer(source.read(4 * row["bias_count"]), dtype="<f4").copy()))
            if source.read(1):
                raise ValueError("尾部冗余字节")
    return model


def train_epoch(model, optimizer, ema, epoch, x, y, cfg, base, device, ledger):
    torch.manual_seed(cfg["model_seed"] + 2000 + epoch)
    stream = torch.Generator(device=device).manual_seed(cfg["model_seed"] + 3000 + epoch)
    order = torch.randperm(len(x), device=device, generator=stream)
    model.train()
    optimizer.param_groups[0]["lr"] = cfg["minimum_learning_rate"] + .5 * (cfg["learning_rate"] - cfg["minimum_learning_rate"]) * (1 + math.cos(math.pi * (epoch - 1) / cfg["epochs"]))
    loss_sum = 0.0
    for ids in order.split(cfg["batch_size"]):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(augment(x[ids], stream, base)), y[ids])
        if not torch.isfinite(loss):
            raise FloatingPointError("训练损失非有限")
        loss.backward()
        with torch.no_grad():
            for layer, mask, score in zip(model.layers, model.masks, ema):
                layer.weight.grad.mul_(mask)
                score.mul_(cfg["sensitivity_ema"]).add_((layer.weight * layer.weight.grad).abs(), alpha=1 - cfg["sensitivity_ema"])
        optimizer.step()
        model.enforce(optimizer)
        ledger["attempted_updates"] += 1
        ledger["optimization_samples"] += len(ids)
        loss_sum += float(loss.detach()) * len(ids)
    return loss_sum / len(x)


def choose_delete(model, ema, counts, random_stream=None):
    deleted = []
    with torch.no_grad():
        for mask, score, count in zip(model.masks, ema, counts):
            if random_stream is None:
                rank = score.flatten().masked_fill(~mask.flatten(), float("inf"))
                indices = torch.topk(rank, count, largest=False, sorted=False).indices
            else:
                alive = torch.nonzero(mask.flatten(), as_tuple=False).flatten()
                indices = alive[torch.randperm(len(alive), generator=random_stream, device=alive.device)[:count]]
            mask.flatten()[indices] = False
            deleted.append(indices.cpu().numpy().astype(np.int32))
        model.enforce()
    return deleted


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_arm(name, cfg, base, data, device, folder, parent_state, reference, events):
    arm = folder / name
    arm.mkdir()
    torch.manual_seed(cfg["model_seed"])
    model = MaskedFC(base["fc_widths"], base["dropout"]).to(device)
    model.load_state_dict(parent_state, strict=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    ema = [torch.zeros_like(l.weight) for l in model.layers]
    random_stream = torch.Generator(device=device).manual_seed(cfg["model_seed"] + 4000)
    x, y = torch.from_numpy(data.fit_x).to(device), torch.from_numpy(data.fit_y).to(device)
    sx, sy = torch.from_numpy(data.select_x).to(device), torch.from_numpy(data.select_y).to(device)
    ledger = {"attempted_updates": 0, "net_updates": 0, "optimization_samples": 0, "evaluation_samples": 0, "evaluation_calls": 0,
              "candidate_adaptations": 0, "rollbacks": 0, "fallback_epochs": 0}
    costs = {"training_seconds": 0., "selection_seconds": 0., "snapshot_seconds": 0., "proposal_seconds": 0., "export_seconds": 0.}
    rows = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, cfg["epochs"] + 1):
        candidate = name == "task" and epoch in cfg["candidate_epochs"]
        event = None
        if candidate:
            t = time.perf_counter()
            saved = capture(model, optimizer, ema)
            costs["snapshot_seconds"] += time.perf_counter() - t
            counts = [math.floor(int(mask.sum()) * cfg["prune_fraction"]) for mask in model.masks]
            t = time.perf_counter()
            before = export_model(model, arm / "before.bgm")
            costs["export_seconds"] += time.perf_counter() - t
            t = time.perf_counter()
            deletion = choose_delete(model, ema, counts)
            sync(device)
            costs["proposal_seconds"] += time.perf_counter() - t
            event = {"epoch": epoch, "counts": counts, "before": before}
            ledger["candidate_adaptations"] += 1
        elif name == "random" and epoch in events and events[epoch]["accepted"]:
            event = events[epoch]
            t = time.perf_counter()
            deletion = choose_delete(model, ema, event["counts"], random_stream)
            sync(device)
            costs["proposal_seconds"] += time.perf_counter() - t
            np.savez_compressed(arm / f"deleted_epoch_{epoch}.npz", **{f"layer_{i}": a for i, a in enumerate(deletion)})
            ledger["candidate_adaptations"] += 1
        t = time.perf_counter()
        train_loss = train_epoch(model, optimizer, ema, epoch, x, y, cfg, base, device, ledger)
        sync(device)
        costs["training_seconds"] += time.perf_counter() - t
        t = time.perf_counter()
        selected, scores = evaluate(model, sx, sy, cfg["evaluation_batch_size"])
        sync(device)
        costs["selection_seconds"] += time.perf_counter() - t
        ledger["evaluation_calls"] += 1
        ledger["evaluation_samples"] += len(sx)
        if candidate:
            t = time.perf_counter()
            after = export_model(model, arm / "candidate.bgm")
            costs["export_seconds"] += time.perf_counter() - t
            ref = reference[epoch - 1]["select"]
            accepted = (selected["accuracy"] + 1e-7 >= cfg["accuracy_floor"] and selected["accuracy"] + cfg["accuracy_drop"] + 1e-7 >= ref["accuracy"]
                        and selected["cross_entropy"] <= ref["cross_entropy"] + cfg["loss_increase"] and after["bytes"] < before["bytes"])
            event.update({"accepted": accepted, "candidate_select": selected, "after": after})
            events[epoch] = event
            if accepted:
                np.savez_compressed(arm / f"deleted_epoch_{epoch}.npz", **{f"layer_{i}": a for i, a in enumerate(deletion)})
            else:
                t = time.perf_counter()
                restore(model, optimizer, ema, saved)
                costs["snapshot_seconds"] += time.perf_counter() - t
                ledger["rollbacks"] += 1
                ledger["fallback_epochs"] += 1
                t = time.perf_counter()
                train_loss = train_epoch(model, optimizer, ema, epoch, x, y, cfg, base, device, ledger)
                sync(device)
                costs["training_seconds"] += time.perf_counter() - t
                t = time.perf_counter()
                selected, scores = evaluate(model, sx, sy, cfg["evaluation_batch_size"])
                sync(device)
                costs["selection_seconds"] += time.perf_counter() - t
                ledger["evaluation_calls"] += 1
                ledger["evaluation_samples"] += len(sx)
            del saved, deletion
        ledger["net_updates"] += math.ceil(len(x) / cfg["batch_size"])
        t = time.perf_counter()
        exported = export_model(model, arm / "model.bgm")
        costs["export_seconds"] += time.perf_counter() - t
        row = {"epoch": epoch, "select": selected, "fit_augmented_cross_entropy": train_loss, "structure": exported,
               "event": event if candidate else None, "learning_rate": optimizer.param_groups[0]["lr"]}
        rows.append(row)
        np.savez_compressed(arm / f"select_epoch_{epoch}.npz", logits=scores, labels=data.select_y)
        torch.save({"epoch": epoch, "state": capture(model, optimizer, ema), "ledger": ledger, "costs": costs, "events": events}, arm / "resume.pt")
        save_json(arm / "progress.json", {"rows": rows, "ledger": ledger, "costs": costs})
        print(json.dumps({"arm": name, "epoch": epoch, "select_accuracy": selected["accuracy"], "active_edges": exported["active_edges"],
                          "accepted": event["accepted"] if candidate else None}, ensure_ascii=False), flush=True)
        if time.perf_counter() - started > cfg["wall_seconds_per_arm"]:
            raise TimeoutError(f"{name}完整轮墙钟预算达到")
        if device.type == "cuda" and torch.cuda.max_memory_allocated() > cfg["peak_memory_limit_bytes"]:
            raise MemoryError("CUDA峰超过登记8GiB")
    fit, logits = evaluate(model, x, y, cfg["evaluation_batch_size"])
    ledger["evaluation_calls"] += 1
    ledger["evaluation_samples"] += len(x)
    np.savez_compressed(arm / "fit_final.npz", logits=logits, labels=data.fit_y)
    output = {"rows": rows, "select": rows[-1]["select"], "fit": fit, "structure": exported, "ledger": ledger, "costs": costs,
              "wall_seconds": time.perf_counter() - started, "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None}
    save_json(arm / "result.json", output)
    del model, optimizer, ema
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output


def main():
    cfg_path = ROOT / "configs/h1_capacity_pruning_v1.json"
    base_path = ROOT / "configs/capacity_baseline_v1.json"
    cfg, base = json.loads(cfg_path.read_text()), json.loads(base_path.read_text())
    parent = ROOT / "experiments/runs/capacity_baseline_v1/mnist_static"
    baseline = json.loads((parent / "summary.json").read_text())
    if baseline["gate"] != "pass" or cfg["model_seed"] != 11 or cfg["data_seed"] != 20261004 or baseline["results"]["fc"]["best_select"]["epoch"] != 28:
        raise ValueError("基线或固定登记不满足")
    folder = ROOT / "experiments/runs" / cfg["version"] / "mnist_static"
    folder.mkdir(parents=True, exist_ok=True)
    if any(folder.iterdir()):
        raise FileExistsError("已有运行目录，拒绝重复或覆盖")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import subprocess
    registration = {"config": cfg, "base_config": base, "config_sha256": digest(cfg_path), "source_sha256": digest(Path(__file__)),
                    "baseline_source_sha256": digest(ROOT / "tools/train_capacity_baseline.py"), "parent_sha256": digest(parent / "fc/best.pt"),
                    "training_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    "device": str(device), "device_name": torch.cuda.get_device_name() if device.type == "cuda" else "CPU",
                    "precision": "FP32", "torch": torch.__version__, "execution": "dense_linear_zero_mask", "cpu_threads": 2}
    save_json(folder / "registration.json", registration)
    data = load_bundle("mnist", include_confirm=False)
    if data.fit_x.shape != (3200, 784) or data.select_x.shape != (800, 784) or len(data.confirm_x):
        raise ValueError("数据权限/形状异常")
    save_json(folder / "data_manifest.json", data.manifest)
    state = torch.load(parent / "fc/best.pt", map_location="cpu", weights_only=False)["state"]
    results, events = {}, {}
    session_start = time.perf_counter()
    try:
        for name in ("fixed", "task", "random"):
            results[name] = run_arm(name, cfg, base, data, device, folder, state, results.get("fixed", {}).get("rows"), events)
        confirmed = load_bundle("mnist", include_confirm=True)
        for name, result in results.items():
            model = load_export(folder / name / "model.bgm").to(device)
            control, logits = evaluate(model, torch.from_numpy(confirmed.confirm_x).to(device), torch.from_numpy(confirmed.confirm_y).to(device), cfg["evaluation_batch_size"])
            result["control"] = control
            result["ledger"]["evaluation_calls"] += 1
            result["ledger"]["evaluation_samples"] += len(confirmed.confirm_x)
            np.savez_compressed(folder / name / "control_final.npz", logits=logits, labels=confirmed.confirm_y)
            save_json(folder / name / "result.json", result)
            del model
        reduction = 1 - results["task"]["structure"]["bytes"] / results["fixed"]["structure"]["bytes"]
        task, fixed = results["task"], results["fixed"]
        performance = (all(results[n][s]["accuracy"] + 1e-7 >= cfg["accuracy_floor"] for n in ("task", "fixed") for s in ("select", "control"))
                       and task["select"]["accuracy"] + cfg["accuracy_drop"] + 1e-7 >= fixed["select"]["accuracy"]
                       and task["select"]["cross_entropy"] <= fixed["select"]["cross_entropy"] + cfg["loss_increase"])
        summary = {"registration": registration, "results": results, "events": events, "storage_reduction": reduction,
                   "performance_pass": performance, "storage_pass": reduction >= cfg["storage_reduction"], "gate": "pass" if performance and reduction >= cfg["storage_reduction"] else "stop_this_expansion",
                   "session_wall_seconds": time.perf_counter() - session_start, "official_test_accessed": False, "sealed_remaining_ids_accessed": False}
        save_json(folder / "summary.json", summary)
        print(json.dumps({"gate": summary["gate"], "task_control": task["control"]["accuracy"], "storage_reduction": reduction}), flush=True)
    except Exception as exc:
        save_json(folder / "failure.json", {"error": repr(exc), "results": results, "events": events, "session_wall_seconds": time.perf_counter() - session_start})
        raise


if __name__ == "__main__":
    main()
