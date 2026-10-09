"""冻结后评价实际当轮优化视频；不训练、不重新生成、不读取验证或测试。"""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from tools import train_generalization_tuning as tuning
from tools import train_large_baseline as base


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    root = tuning.ROOT
    suite, config = tuning.read(root / "suite.json"), tuning.read(root / "config.json")
    assert suite["status"] == "complete_registered_round" and suite["winner"] is not None
    assert config["source_sha256"] == tuning.hashes()
    out = root / "online_fit_diagnostic"
    if (out / "result.json").exists():
        print("动态拟合诊断已完成，保留结果；不重复评价。")
        return
    if out.exists():
        raise RuntimeError("诊断目录不完整，保留已观察成本，拒绝静默重跑")
    name = suite["winner"]
    frozen = suite["frozen"][name]
    checkpoint = Path(frozen["path"])
    assert sha(checkpoint) == frozen["sha256"]
    torch.manual_seed(11)
    torch.cuda.manual_seed_all(11)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    epoch = payload["state"]["best_control"]["epoch"]
    model = base.FixedLayeredSNN(device="cuda")
    model.load_state_dict(payload["model"])
    del payload
    cache = root / "augmentation" / f"epoch_{epoch:03d}.npz"
    metadata = tuning.read(cache.with_suffix(".json"))
    assert sha(cache) == metadata["sha256"]
    fit, control, data = base.training_data()
    assert data == config["data"]
    with np.load(cache, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in ("x", "y", "raw_ids", "variants")}
    assert len(arrays["x"]) == 4480 and len(set(arrays["raw_ids"])) == 160
    assert np.array_equal(arrays["raw_ids"], fit[2]) and np.array_equal(arrays["y"], fit[1])
    assert not set(arrays["raw_ids"]) & set(control[2])
    out.mkdir()
    declaration = dict(arm=name, epoch=epoch, frozen=frozen, cache=str(cache.resolve()),
        cache_sha256=metadata["sha256"], variants=metadata["variants"], videos=4480,
        purpose="冻结后描述性动态拟合诊断，不参与选参或延长训练", source_sha256=tuning.hashes(),
        diagnostic_source_sha256=sha(Path(__file__)), device=torch.cuda.get_device_name(0), torch=torch.__version__)
    base.write_json(out / "config.json", declaration)
    cost = base.new_costs()
    ledger = base.Ledger(out, cost)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    metrics, predictions = base.evaluate_pool(model, torch.as_tensor(arrays["x"], device="cuda"),
        torch.as_tensor(arrays["y"], device="cuda"), 32, ledger, "fit", epoch, base.StopRequested())
    tuning.add_joint_confusion(metrics, predictions)
    np.savez_compressed(out / "predictions.npz", **predictions, raw_ids=arrays["raw_ids"], variants=arrays["variants"])
    base.write_json(out / "result.json", dict(config=declaration, metrics=metrics, costs=cost,
        wall_seconds=time.perf_counter()-started, gpu_peak_bytes=torch.cuda.max_memory_allocated(),
        optimizer_updates=0, backward_videos=0, new_render_videos=0, validation_forward_videos=0,
        sealed_test_reads=0, control=suite["summaries"][name]["best_control"],
        matched_epoch_gap={key: metrics[key]-suite["summaries"][name]["best_control"][key] for key in tuning.TASKS}))
    print(f"冻结第{epoch}轮动态拟合：数字{metrics['digit_accuracy']:.2%} 运动{metrics['motion_accuracy']:.2%} 联合{metrics['joint_accuracy']:.2%}；0更新。")


if __name__ == "__main__":
    run()
