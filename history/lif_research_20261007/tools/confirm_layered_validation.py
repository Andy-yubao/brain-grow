"""冻结已选固定LIF图，在原始ID隔离的v5验证池仅确认一次。"""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from braingrow.data.moving_mnist_v5 import CONFIG
from braingrow.network.layered_gradient import LayeredGradientSNN
from tools.run_layered_gradient import evaluate_control, peak_working_set_bytes, write_json


CHECKPOINT = Path("experiments/runs/layered_growth_d1/fixed_best.pt")
CHECKPOINT_SHA256 = "d55d766001448c42dec714817d3102bf43bfa42ab2b5f4de601591907477e0f4"
OUTDIR = Path("experiments/runs/layered_validation_c1")


def run(checkpoint_path: Path, outdir: Path, batch_size: int) -> None:
    if batch_size <= 0:
        raise ValueError("batch_size必须为正")
    if outdir.exists():
        raise FileExistsError("验证确认目录已存在；拒绝重复查看或覆盖")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用；验证不退回CPU")
    torch.manual_seed(11)
    torch.cuda.manual_seed_all(11)
    torch.use_deterministic_algorithms(True)
    path = Path("data/dynamic_v5.npz")
    manifest_path = Path("data/manifest_v5.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["config"] != CONFIG or hashlib.sha256(path.read_bytes()).hexdigest() != manifest["cache_sha256"]:
        raise ValueError("v5协议或缓存哈希不符")
    # 此入口只解包validation成员；不读取封存的其他原始ID或官方测试文件。
    with np.load(path) as z:
        x, y, ids = (z[k] for k in ("validation_x", "validation_y", "validation_ids"))
    if len(x) != 1400 or len(set(ids.tolist())) != 100:
        raise ValueError("验证池数量或原始ID不符")
    if set(ids.tolist()) != set(manifest["splits"]["validation"]) or set(ids.tolist()) & set(manifest["splits"]["train"]):
        raise ValueError("验证原始ID与manifest不符或与训练池重叠")
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if checkpoint_sha256 != CHECKPOINT_SHA256:
        raise ValueError("冻结检查点哈希与预登记不符")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (checkpoint["arm"] != "fixed" or tuple(checkpoint["widths"]) != (512, 512)
            or checkpoint["epoch_from_base"] != 2):
        raise ValueError("只确认已冻结的D1固定图最佳检查点")
    device = torch.device("cuda")
    model = LayeredGradientSNN((512, 512), device=device)
    model.load_state_dict(checkpoint["model"])
    outdir.mkdir(parents=True)
    write_json(outdir / "config.json", {"selected_checkpoint": str(checkpoint_path),
        "selected_checkpoint_sha256": checkpoint_sha256,
        "selection": "D1固定臂四遍内结构控制分数最高的第2遍；加点臂均未通过预设收益门槛",
        "start_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "data_cache_sha256": manifest["cache_sha256"], "data_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "model_seed": 11, "data_seed": 20261004, "device": torch.cuda.get_device_name(0),
        "validation_videos": 1400, "validation_original_ids": 100,
        "validation_forward_budget": 1400, "metric_evaluation_budget": 1,
        "validation_role": "本梯度路线首次查看该池；旧研究曾重复使用，结果仍属探索性跨ID确认",
        "sealed_test_reads": 0, "steps_per_frame": 4, "step_ms": 5,
        "neural_time_ms": 160, "video_time": "8帧，实际播放时长未规定"})
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    metrics, arrays = evaluate_control(model, torch.as_tensor(x, device=device),
                                       torch.as_tensor(y, device=device), batch_size)
    np.savez_compressed(outdir / "outputs.npz", **arrays, raw_ids=ids)
    write_json(outdir / "result.json", {"metrics": metrics, "validation_forward_videos": len(x),
        "metric_evaluations": 1, "unique_raw_ids": int(len(set(ids.tolist()))),
        "model_parameters_updated": 0, "wall_seconds": time.perf_counter() - started,
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "process_peak_working_set_bytes": peak_working_set_bytes(), "sealed_test_reads": 0})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--outdir", type=Path, default=OUTDIR)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    run(args.checkpoint, args.outdir, args.batch_size)


if __name__ == "__main__":
    main()
