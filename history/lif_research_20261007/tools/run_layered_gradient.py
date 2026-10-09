"""v5分层LIF代理梯度基线：固定神经动力学，只训练真实边权。"""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import ctypes
import hashlib
import json
import platform
import shutil
import subprocess
import time
from pathlib import Path
from ctypes import wintypes

import numpy as np
import torch
from torch.nn import functional as F

from braingrow.data.moving_mnist_v5 import CONFIG
from braingrow.evaluation.metrics import evaluate
from braingrow.network.layered_gradient import Dynamics, LayeredGradientSNN


RUN_DIR = Path("experiments/runs/layered_gradient_v1")


def peak_working_set_bytes() -> int:
    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.windll.kernel32
    psapi = ctypes.windll.psapi
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), wintypes.DWORD]
    ok = psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
    if not ok:
        raise OSError("无法读取当前训练进程峰值工作集")
    return int(counters.PeakWorkingSetSize)


def write_json(path: Path, value: dict) -> None:
    pending = path.with_name(path.name + ".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    pending.replace(path)


def record_cost(outdir: Path, costs: dict, kind: str, videos: int, epoch: int) -> None:
    row = {"kind": kind, "videos": int(videos), "epoch": epoch,
           "cumulative_costs": costs.copy(), "timestamp": time.time()}
    with (outdir / "cost_journal.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def training_data():
    path = Path("data/dynamic_v5.npz")
    manifest_path = Path("data/manifest_v5.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["config"] != CONFIG or hashlib.sha256(path.read_bytes()).hexdigest() != manifest["cache_sha256"]:
        raise ValueError("v5协议或哈希不符，拒绝覆盖数据")
    # 只解包训练成员；validation和sealed_test不参与逐轮结构决策。
    with np.load(path) as z:
        x, y, ids = (z[k] for k in ("train_x", "train_y", "train_ids"))
    selected = np.zeros(len(ids), bool)
    for digit in range(10):
        ordered_ids = list(dict.fromkeys(ids[y[:, 0] == digit].tolist()))
        if len(ordered_ids) != 20:
            raise ValueError("每数字训练原始ID必须为20")
        selected |= np.isin(ids, ordered_ids[:16])
    fit = (x[selected], y[selected], ids[selected])
    control = (x[~selected], y[~selected], ids[~selected])
    if len(set(fit[2]) & set(control[2])) or len(fit[0]) != 4480 or len(control[0]) != 1120:
        raise ValueError("训练/结构控制原始ID隔离失败")
    return fit, control, {"cache_sha256": manifest["cache_sha256"],
                          "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                          "fit_original_ids": int(len(set(fit[2]))),
                          "control_original_ids": int(len(set(control[2])))}


def loss_for(potential: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(potential[:, :10], target[:, 0]) + F.cross_entropy(potential[:, 10:], target[:, 1])


def smoke(model: LayeredGradientSNN, x: torch.Tensor, y: torch.Tensor) -> dict:
    model.train()
    p, trace = model(x[:16], trace=True)
    loss = loss_for(p, y[:16])
    loss.backward()
    gradients = [float(w.grad.abs().sum().item()) for w in model.weights]
    if not np.isfinite(gradients).all() or min(gradients) <= 0:
        raise RuntimeError(f"梯度失效: {gradients}")
    rates = [float(f.sum().item() / (16 * 32 * n))
             for f, n in zip(trace["per_unit_fire"], model.widths)]
    if min(rates) < 0.001 or max(rates) >= 0.99:
        raise RuntimeError(f"隐藏层发放严重失效: {rates}")
    rebuilt = trace["last_layer_counts"] @ (model.weights[2] * model.mask_output)
    if not torch.allclose(p, rebuilt, rtol=1e-5, atol=1e-4):
        raise RuntimeError("固定累积输出不能由真实末层Fire重建")
    model.zero_grad(set_to_none=True)
    # 检查任一层全连接出生不会立即改变旧网络的输出。
    checks = []
    for layer in (0, 1):
        probe = LayeredGradientSNN(model.widths, device=p.device)
        probe.load_state_dict(model.state_dict())
        probe.grow(layer, 4, layer + 1)
        with torch.no_grad():
            checks.append(bool(torch.allclose(p, probe(x[:16]), rtol=1e-6, atol=1e-5)))
    if not all(checks):
        raise RuntimeError("全连接出生改变了已有前向功能")
    return {"loss": float(loss.item()), "gradient_abs_sum": gradients,
            "spikes_per_lif_step_by_layer": rates,
            "growth_preserved_output_by_layer": checks,
            "output_reconstruction_max_error": float((p - rebuilt).abs().max().item()),
            "videos": 16, "forward_and_backward_videos": 16,
            "note": "仅诊断，不计为训练准确率"}


@torch.no_grad()
def evaluate_control(model: LayeredGradientSNN, x: torch.Tensor, y: torch.Tensor,
                     batch_size: int) -> tuple[dict, dict[str, np.ndarray]]:
    model.eval()
    potentials, frame_counts, video_counts = [], [], []
    units = [torch.zeros(n, device=x.device) for n in model.widths]
    maximum_reconstruction_error = 0.0
    for start in range(0, len(x), batch_size):
        p, activity = model(x[start:start + batch_size], trace=True)
        rebuilt = activity["last_layer_counts"] @ (model.weights[2] * model.mask_output)
        maximum_reconstruction_error = max(maximum_reconstruction_error,
                                           float((p - rebuilt).abs().max().item()))
        potentials.append(p.cpu().numpy())
        frame_counts.append(activity["frame_fire"].cpu().numpy())
        video_counts.append(activity["per_video_fire"].cpu().numpy())
        for layer in (0, 1):
            units[layer] += activity["per_unit_fire"][layer]
    potential = np.concatenate(potentials)
    frames = np.concatenate(frame_counts)
    videos = np.concatenate(video_counts)
    metrics = evaluate(potential, y.cpu().numpy())
    total = float(videos.sum())
    metrics.update({"videos": int(len(x)), "hidden_spikes": int(total), "output_spikes": 0,
                    "mean_spikes_per_video": total / len(x),
                    "mean_spikes_per_frame": total / (len(x) * 8),
                    "spikes_per_lif_step": total / (len(x) * 32 * sum(model.widths)),
                    "spikes_per_lif_step_by_layer": [float(u.sum().item() / (len(x) * 32 * n))
                                                     for u, n in zip(units, model.widths)],
                    "silent_lif_by_layer": [int((u == 0).sum().item()) for u in units],
                    "per_frame_mean_by_layer": frames.mean(axis=0).tolist(),
                    "output_reconstruction_max_error": maximum_reconstruction_error,
                    "resources": model.resources()})
    arrays = {"potential": potential, "y": y.cpu().numpy(),
              "frame_fire_by_layer": frames, "video_fire_by_layer": videos,
              "per_unit_fire_layer1": units[0].cpu().numpy(),
              "per_unit_fire_layer2": units[1].cpu().numpy()}
    return metrics, arrays


def save_checkpoint(path: Path, model: LayeredGradientSNN, optimizer: torch.optim.Optimizer,
                    epoch: int, rng: np.random.Generator, best_score: float,
                    stale: int, costs: dict) -> None:
    pending = path.with_name(path.name + ".pending")
    torch.save({"model": model.state_dict(), "widths": model.widths,
                "optimizer": optimizer.state_dict(), "epoch": epoch,
                "numpy_rng": rng.bit_generator.state,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
                "best_score": best_score, "stale": stale, "costs": costs}, pending)
    pending.replace(path)


def record_start(outdir: Path, metadata: dict) -> None:
    outdir.mkdir(parents=True, exist_ok=False)
    metadata.update({"start_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "start_status": subprocess.check_output(["git", "status", "--short"], text=True),
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "root_model_seed": 11, "data_seed": 20261004,
        "validation_arrays_unpacked": False, "validation_forward_evaluations": 0,
        "sealed_test_reads": 0, "neural_time_ms": 160,
        "video_time": "8帧，实际播放时长未规定",
        "trainable": "仅三类真实边权；LIF泄露和阈值固定",
        "surrogate": "forward: hard Fire; backward: 1/(1+5*abs(v-threshold))^2"})
    write_json(outdir / "config.json", metadata)
    (outdir / "start_diff.patch").write_text(
        subprocess.check_output(["git", "diff"], text=True, encoding="utf-8"), encoding="utf-8")
    snapshot = outdir / "source_snapshot"
    snapshot.mkdir()
    for name in ("src/braingrow/network/layered_gradient.py", "tools/run_layered_gradient.py",
                 "src/braingrow/evaluation/metrics.py", "src/braingrow/data/moving_mnist_v5.py",
                 "docs/layered_gradient_execution.md", "pyproject.toml"):
        dest = snapshot / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(name, dest)


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用；本实验不退回CPU训练")
    previous_summary_path = args.outdir / "baseline_latest_summary.json"
    completed_path = previous_summary_path if previous_summary_path.exists() else args.outdir / "baseline_summary.json"
    if args.resume and completed_path.exists():
        previous = json.loads(completed_path.read_text(encoding="utf-8"))
        if args.epochs <= previous["last_epoch"]:
            print("已完成当前登记预算；恢复入口不重复加载数据或训练", flush=True)
            return
    torch.manual_seed(11)
    torch.cuda.manual_seed_all(11)
    torch.use_deterministic_algorithms(True)
    fit, control, data_info = training_data()
    device = torch.device("cuda")
    x_fit = torch.as_tensor(fit[0], device=device)
    y_fit = torch.as_tensor(fit[1], device=device)
    x_control = torch.as_tensor(control[0], device=device)
    y_control = torch.as_tensor(control[1], device=device)
    widths = (args.width if args.width_layer1 is None else args.width_layer1,
              args.width if args.width_layer2 is None else args.width_layer2)
    model = LayeredGradientSNN(widths, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    rng = np.random.default_rng(11)
    costs = {"train_forward_videos": 0, "train_backward_videos": 0,
             "control_forward_videos": 0, "metric_evaluations": 0,
             "smoke_forward_videos": 0}
    outdir = args.outdir
    metadata = {"data": data_info, "initial_widths": list(model.widths),
                "batch_size": args.batch_size, "learning_rate": args.learning_rate,
                "baseline_epoch_budget": args.epochs, "minimum_epochs": args.min_epochs,
                "plateau_patience": args.patience, "plateau_min_delta": 0.002,
                "fit_videos": len(fit[0]), "control_videos": len(control[0]),
                "device": "cuda:0", "steps_per_frame": 4, "step_ms": 5,
                "gradient_clip_norm": 1.0,
                "stop_signal": "至少最少轮数后，控制分数连续patience轮无改善；低功能时先诊断训练，不触发缩网或盲扩容"}
    if args.resume:
        checkpoint = torch.load(outdir / "baseline_resume.pt", map_location="cpu", weights_only=False)
        if tuple(checkpoint["widths"]) != model.widths:
            raise ValueError("恢复模型宽度不符")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        rng.bit_generator.state = checkpoint["numpy_rng"]
        torch.set_rng_state(checkpoint["torch_rng"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint["best_score"])
        stale = int(checkpoint["stale"])
        costs = checkpoint["costs"]
        journal = outdir / "cost_journal.jsonl"
        if journal.exists():
            last = json.loads(journal.read_text(encoding="utf-8").splitlines()[-1])
            for key, count in last["cumulative_costs"].items():
                costs[key] = max(costs.get(key, 0), int(count))
        if args.epochs > int(checkpoint["epoch"]):
            write_json(outdir / f"extension_epoch{checkpoint['epoch']}_to{args.epochs}.json",
                       {"from_epoch": int(checkpoint["epoch"]), "to_epoch": args.epochs,
                        "reason": "前阶段到达工程预算时控制指标与拟合损失仍持续改善；保持同一模型、优化器与唯一根种子继续训练",
                        "additional_train_forward_video_budget": (args.epochs - int(checkpoint["epoch"])) * len(x_fit),
                        "additional_control_forward_video_budget": (args.epochs - int(checkpoint["epoch"])) * len(x_control),
                        "training_complete_claim": False})
    else:
        record_start(outdir, metadata)
        diagnostic = smoke(model, x_fit, y_fit)
        costs["smoke_forward_videos"] += diagnostic["videos"] * 3
        record_cost(outdir, costs, "smoke", diagnostic["videos"] * 3, 0)
        write_json(outdir / "smoke.json", diagnostic)
        if args.smoke_only:
            write_json(outdir / "smoke_summary.json", {"completed": True, "costs": costs,
                "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "process_peak_working_set_bytes": peak_working_set_bytes()})
            print(json.dumps(diagnostic, ensure_ascii=False), flush=True)
            return
        start_epoch, best_score, stale = 1, float("-inf"), 0
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        order = rng.permutation(len(x_fit))
        train_loss_sum = 0.0
        for begin in range(0, len(order), args.batch_size):
            positions = torch.as_tensor(order[begin:begin + args.batch_size], device=device)
            target = y_fit[positions]
            optimizer.zero_grad(set_to_none=True)
            potential = model(x_fit[positions])
            loss = loss_for(potential, target)
            if not torch.isfinite(loss):
                raise RuntimeError("训练损失非有限值；保留上一个epoch检查点")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss_sum += float(loss.item()) * len(positions)
            costs["train_forward_videos"] += len(positions)
            costs["train_backward_videos"] += len(positions)
            record_cost(outdir, costs, "train_forward_backward", len(positions), epoch)
        metrics, arrays = evaluate_control(model, x_control, y_control, args.batch_size)
        costs["control_forward_videos"] += len(x_control)
        costs["metric_evaluations"] += 1
        record_cost(outdir, costs, "control_evaluation", len(x_control), epoch)
        score = metrics["digit_accuracy"] + metrics["motion_accuracy"] + metrics["joint_accuracy"]
        improved = score > best_score + 0.002
        if improved:
            best_score, stale = score, 0
            np.savez_compressed(outdir / "best_control_outputs.npz", **arrays)
            write_json(outdir / "best_control_metrics.json", {"epoch": epoch, **metrics})
        else:
            stale += 1
        row = {"epoch": epoch, "fit_loss": train_loss_sum / len(x_fit),
               "control": metrics, "control_score": score, "best_score": best_score,
               "stale_epochs": stale, "cumulative_costs": costs.copy(),
               "wall_seconds_this_session": time.perf_counter() - started,
               "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
               "process_peak_working_set_bytes": peak_working_set_bytes()}
        with (outdir / "epochs.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
        save_checkpoint(outdir / "baseline_resume.pt", model, optimizer, epoch,
                        rng, best_score, stale, costs)
        if improved:
            save_checkpoint(outdir / "baseline_best.pt", model, optimizer, epoch,
                            rng, best_score, stale, costs)
        print(json.dumps({"epoch": epoch, "fit_loss": row["fit_loss"],
              "digit": metrics["digit_accuracy"], "motion": metrics["motion_accuracy"],
              "joint": metrics["joint_accuracy"], "fire_rates": metrics["spikes_per_lif_step_by_layer"],
              "wall_s": round(row["wall_seconds_this_session"], 1)}, ensure_ascii=False), flush=True)
        if epoch >= args.min_epochs and stale >= args.patience:
            break
    summary = {"phase_budget_completed": True, "training_converged": stale >= args.patience,
        "last_epoch": epoch, "best_control_score": best_score,
        "stop_reason": "control_plateau" if stale >= args.patience else "registered_epoch_budget",
        "costs": costs, "wall_seconds_this_session": time.perf_counter() - started,
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "process_peak_working_set_bytes": peak_working_set_bytes(),
        "resume_command": ".venv/Scripts/python.exe -B -m tools.run_layered_gradient --resume",
        "validation_forward_evaluations": 0, "sealed_test_reads": 0}
    if not args.resume:
        write_json(outdir / "baseline_summary.json", summary)
    else:
        write_json(outdir / f"baseline_summary_epoch{epoch}.json", summary)
    write_json(previous_summary_path, summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=RUN_DIR)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--width-layer1", type=int)
    parser.add_argument("--width-layer2", type=int)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--min-epochs", type=int, default=6)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
