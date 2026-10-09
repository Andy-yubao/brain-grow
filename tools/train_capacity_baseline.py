"""扩容 CNN/FC 的独立准确率门槛；从项目根以 module 启动。"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from braingrow.datasets import load_bundle


class FourLayerFC(nn.Module):
    def __init__(self, widths: list[int], dropout: float):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(widths, widths[1:]))
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(1)
        for layer in self.layers[:-1]:
            x = F.dropout(F.relu(layer(x)), p=self.dropout, training=self.training)
        return self.layers[-1](x)


class TwoConvTwoFC(nn.Module):
    def __init__(self, channels: list[int], hidden: int, dropout: float):
        super().__init__()
        self.conv1 = nn.Conv2d(1, channels[0], 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, padding=1)
        self.fc1 = nn.Linear(7 * 7 * channels[1], hidden)
        self.fc2 = nn.Linear(hidden, 10)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1, 1, 28, 28)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x)).flatten(1)
        x = F.dropout(F.relu(self.fc1(x)), p=self.dropout, training=self.training)
        return self.fc2(x)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch: int) -> tuple[dict, np.ndarray]:
    model.eval()
    logits = []
    with torch.no_grad():
        for start in range(0, len(x), batch):
            output = model(x[start:start + batch])
            if not torch.isfinite(output).all():
                raise FloatingPointError("评价输出非有限")
            logits.append(output.cpu())
    scores = torch.cat(logits)
    prediction = scores.argmax(1)
    labels = y.cpu()
    matrix = torch.bincount(labels * 10 + prediction, minlength=100).reshape(10, 10)
    return {"samples": len(labels), "accuracy": float((prediction == labels).float().mean()),
            "cross_entropy": float(F.cross_entropy(scores, labels)),
            "confusion_matrix": matrix.tolist()}, scores.numpy()


def augment(x: torch.Tensor, stream: torch.Generator, cfg: dict) -> torch.Tensor:
    images = x.reshape(-1, 1, 28, 28)
    count = len(images)
    radians = (torch.rand(count, device=x.device, generator=stream) * 2 - 1) * math.radians(cfg["augmentation_degrees"])
    shifts = (torch.rand(count, 2, device=x.device, generator=stream) * 2 - 1) * (2 * cfg["augmentation_translation_fraction"])
    cosine, sine = radians.cos(), radians.sin()
    transform = torch.stack((cosine, -sine, shifts[:, 0], sine, cosine, shifts[:, 1]), dim=1).reshape(-1, 2, 3)
    grid = F.affine_grid(transform, images.shape, align_corners=False)
    return F.grid_sample(images, grid, padding_mode="zeros", align_corners=False).flatten(1)


def make_model(name: str, cfg: dict, device: torch.device) -> nn.Module:
    random.seed(cfg["model_seed"])
    np.random.seed(cfg["model_seed"])
    torch.manual_seed(cfg["model_seed"])
    if name == "cnn":
        model = TwoConvTwoFC(cfg["cnn_channels"], cfg["cnn_hidden"], cfg["dropout"])
    else:
        model = FourLayerFC(cfg["fc_widths"], cfg["dropout"])
    return model.to(device)


def run_arm(name: str, cfg: dict, data, device: torch.device, folder: Path) -> dict:
    model = make_model(name, cfg, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    stream = torch.Generator(device=device).manual_seed(cfg["model_seed"] + 1000)
    x = torch.from_numpy(data.fit_x).to(device)
    y = torch.from_numpy(data.fit_y).to(device)
    select_x = torch.from_numpy(data.select_x).to(device)
    select_y = torch.from_numpy(data.select_y).to(device)
    arm_dir = folder / name
    arm_dir.mkdir(exist_ok=False)
    rows = []
    best = None
    wall_start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        optimizer.param_groups[0]["lr"] = cfg["minimum_learning_rate"] + 0.5 * (cfg["learning_rate"] - cfg["minimum_learning_rate"]) * (1 + math.cos(math.pi * (epoch - 1) / cfg["epochs"]))
        order = torch.randperm(len(x), device=device, generator=stream)
        epoch_loss = 0.0
        started = time.perf_counter()
        for indices in order.split(cfg["batch_size"]):
            inputs = augment(x[indices], stream, cfg)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(inputs), y[indices])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{name} 第{epoch}轮损失非有限")
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach()) * len(indices)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        training_seconds = time.perf_counter() - started
        started = time.perf_counter()
        selected, scores = evaluate(model, select_x, select_y, cfg["evaluation_batch_size"])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        evaluation_seconds = time.perf_counter() - started
        row = {"epoch": epoch, "fit_augmented_cross_entropy": epoch_loss / len(x), "select": selected,
               "learning_rate": optimizer.param_groups[0]["lr"], "training_seconds": training_seconds,
               "evaluation_seconds": evaluation_seconds}
        rows.append(row)
        if best is None or (selected["accuracy"], -selected["cross_entropy"]) > (best["accuracy"], -best["cross_entropy"]):
            best = {"epoch": epoch, **selected}
            torch.save({"architecture": name, "state": {key: value.detach().cpu() for key, value in model.state_dict().items()}}, arm_dir / "best.pt")
            np.savez_compressed(arm_dir / "select_best.npz", logits=scores, labels=data.select_y)
        save_json(arm_dir / "progress.json", {"rows": rows, "best_select": best})
        print(json.dumps({"arm": name, "epoch": epoch, "select_accuracy": selected["accuracy"],
                          "best_accuracy": best["accuracy"]}, ensure_ascii=False), flush=True)
        if epoch >= cfg["minimum_epochs"] and best["accuracy"] >= 0.97:
            break
        if time.perf_counter() - wall_start > 1800:
            break
    stop_reason = "target_reached" if best["accuracy"] >= 0.97 else ("epoch_limit" if epoch == cfg["epochs"] else "wall_budget")
    output = {"arm": name, "best_select": best, "epochs_run": epoch, "stop_reason": stop_reason,
              "rows": rows, "parameters": sum(parameter.numel() for parameter in model.parameters()),
              "checkpoint_bytes": (arm_dir / "best.pt").stat().st_size,
              "wall_seconds": time.perf_counter() - wall_start,
              "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}
    save_json(arm_dir / "result.json", output)
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output


def main() -> None:
    cfg_path = ROOT / "configs" / "capacity_baseline_v1.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if cfg["model_seed"] != 11 or cfg["data_seed"] != 20261004:
        raise ValueError("固定种子不匹配")
    if cfg["fc_widths"] != [784, 2 * 14 * 14 * cfg["cnn_channels"][0],
                            2 * 7 * 7 * cfg["cnn_channels"][1], 2 * cfg["cnn_hidden"], 10]:
        raise ValueError("FC 宽度未满足对应层两倍约束")
    folder = ROOT / "experiments" / "runs" / cfg["version"] / "mnist_static"
    folder.mkdir(parents=True, exist_ok=True)
    if any(folder.iterdir()):
        raise FileExistsError("本轮运行目录已有内容，拒绝覆盖或重复运行")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    registration = {"config": cfg, "config_sha256": digest(cfg_path), "source_sha256": digest(Path(__file__)),
                    "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                    "torch": torch.__version__, "precision": "FP32", "execution": "dense_pytorch"}
    save_json(folder / "registration.json", registration)
    data = load_bundle("mnist", include_confirm=False)
    if data.fit_x.shape != (3200, 784) or data.select_x.shape != (800, 784) or len(data.confirm_x):
        raise ValueError("数据划分或确认权限异常")
    save_json(folder / "data_manifest.json", data.manifest)
    results = {}
    for name in ("cnn", "fc"):
        result = run_arm(name, cfg, data, device, folder)
        results[name] = result
        if result["best_select"]["accuracy"] < cfg["accuracy_floor"] or result["stop_reason"] == "wall_budget":
            save_json(folder / "summary.json", {"registration": registration, "results": results,
                                                 "gate": "stop_below_select_target_or_budget", "official_test_accessed": False})
            print("基线未达到预登记门槛，本轮实验立即停止", flush=True)
            return
    confirmed = load_bundle("mnist", include_confirm=True)
    control_x = torch.from_numpy(confirmed.confirm_x).to(device)
    control_y = torch.from_numpy(confirmed.confirm_y).to(device)
    for name in ("cnn", "fc"):
        model = make_model(name, cfg, device)
        saved = torch.load(folder / name / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(saved["state"])
        measured, logits = evaluate(model, control_x, control_y, cfg["evaluation_batch_size"])
        results[name]["exploratory_control"] = measured
        np.savez_compressed(folder / name / "control_final.npz", logits=logits, labels=confirmed.confirm_y)
        save_json(folder / name / "result.json", results[name])
    passed = all(results[name]["exploratory_control"]["accuracy"] >= cfg["accuracy_floor"] for name in results)
    save_json(folder / "summary.json", {"registration": registration, "results": results,
                                         "gate": "pass" if passed else "stop_below_control_target",
                                         "official_test_accessed": False, "sealed_remaining_ids_accessed": False})
    print(json.dumps({"gate": "pass" if passed else "stop_below_control_target",
                      "cnn_control": results["cnn"]["exploratory_control"]["accuracy"],
                      "fc_control": results["fc"]["exploratory_control"]["accuracy"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
