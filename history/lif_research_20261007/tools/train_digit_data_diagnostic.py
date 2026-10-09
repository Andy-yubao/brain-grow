"""固定预算的尺寸×初始角四组CNN诊断；不修改v5与已有模型。"""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import copy
import gc
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from braingrow.data.online_v5 import selected_original_images
from braingrow.data.rendering import foreground, pose_bounds, render_pose
from braingrow.network.ann_controls import make_model, resources
from tools.run_layered_gradient import loss_for, peak_working_set_bytes, write_json
from tools.train_ann_comparison import (Ledger, atomic_save, costs_zero, diagnostic, evaluate_pool,
    load_resume, nested_equal, optimizer_for, payload, restore, save_resume, seed, sha, update)


ROOT = Path("experiments/runs/digit_data_diagnostic_v1")
ARMS = ("small_full", "small_narrow", "large_full", "large_narrow")
SOURCES = ("tools/train_digit_data_diagnostic.py", "src/braingrow/data/rendering.py",
           "src/braingrow/data/online_v5.py", "src/braingrow/data/moving_mnist_v5.py",
           "src/braingrow/network/ann_controls.py", "tools/train_ann_comparison.py",
           "tools/run_layered_gradient.py", "src/braingrow/evaluation/metrics.py")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def paired_geometry(record):
    scale = record["scale"]
    bbox = record["foreground_bbox_xyxy"]
    radius = scale * np.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
    original = np.asarray(record["poses_xy_clockwise_degrees"], dtype=np.float64)
    low, high = np.full(2, radius + 1), np.full(2, 27 - radius)
    shared_low, shared_high = np.full(2, 1.8 * radius + 1), np.full(2, 27 - 1.8 * radius)
    poses = original.copy()
    for motion in range(7):
        old_delta = original[motion, -1, :2] - original[motion, 0, :2]
        delta = np.zeros(2)
        if motion in (1, 2, 3, 4):
            axis = 1 if motion < 3 else 0
            length = abs(old_delta[axis])
            delta[axis] = np.sign(old_delta[axis]) * (4.1 + (length - 14.1))
            assert 4.1 - 1e-10 <= abs(delta[axis]) <= 4.4 + 1e-10
        quantile = (original[motion, 0, :2] - (low - np.minimum(old_delta, 0))) / (high - low - np.abs(old_delta))
        assert np.all((quantile >= -1e-10) & (quantile <= 1 + 1e-10))
        start = shared_low - np.minimum(delta, 0) + quantile * (shared_high - shared_low - np.abs(delta))
        poses[motion, :, :2] = start + np.linspace(0, 1, 8)[:, None] * delta
    result = {}
    for arm in ARMS:
        arm_poses = poses.copy()
        if arm.endswith("narrow"):
            arm_poses[:, :, 2] = original[:, 0, 2, None] / 12 + (original[:, :, 2] - original[:, 0, 2, None])
        result[arm] = (scale * (1.8 if arm.startswith("large") else 1), arm_poses)
        bounds = pose_bounds(bbox, result[arm][0], arm_poses)
        assert bounds[..., :2].min() >= 1 - 1e-10 and bounds[..., 2:].max() <= 27 + 1e-10
    np.testing.assert_array_equal(result["small_full"][1], result["large_full"][1])
    np.testing.assert_array_equal(result["small_narrow"][1], result["large_narrow"][1])
    np.testing.assert_array_equal(result["small_full"][1][..., :2], result["small_narrow"][1][..., :2])
    return result


def prepare_data():
    started = time.perf_counter()
    manifest_path = Path("data/manifest_v5.json")
    manifest = read(manifest_path)
    assert sha("data/dynamic_v5.npz") == manifest["cache_sha256"]
    with np.load("reports/ann_comparison_20261005_evidence/split_evidence.npz") as split:
        fit_ids, control_ids = split["fit_ids"], split["control_ids"]
        fit_y, control_y = split["fit_y"], split["control_y"]
    fit_set, control_set = set(fit_ids.tolist()), set(control_ids.tolist())
    assert len(fit_set) == 160 and len(control_set) == 40 and not fit_set & control_set
    with np.load("data/dynamic_v5.npz") as original:
        y, ids, variants = (original[key] for key in ("train_y", "train_ids", "train_variants"))
    fit_mask = np.isin(ids, list(fit_set))
    np.testing.assert_array_equal(ids[fit_mask], fit_ids)
    np.testing.assert_array_equal(y[fit_mask], fit_y)
    np.testing.assert_array_equal(ids[~fit_mask], control_ids)
    np.testing.assert_array_equal(y[~fit_mask], control_y)
    images = selected_original_images(list(dict.fromkeys(ids.tolist())))
    records = manifest["samples"]["train"]
    # 小样本审计：按每数字第一笔迹取variant0；覆盖7类、8帧和四个组。
    pilot = [next(record for record in records if record["digit"] == digit and record["variant"] == 0)
             for digit in range(10)]
    with np.load("data/dynamic_v5.npz") as original:
        original_x = original["train_x"]
    pilot_margins, compared_frames = [], 0
    for record in pilot:
        tile, bbox = foreground(images[record["raw_id"]])
        positions = np.flatnonzero((ids == record["raw_id"]) & (variants == 0))
        original_poses = np.asarray(record["poses_xy_clockwise_degrees"])
        reproduced = np.stack([render_pose(tile, record["scale"], pose)
                               for pose in original_poses.reshape(-1, 3)]).reshape(7, 8, 784).astype(np.float32) / 255
        np.testing.assert_array_equal(reproduced, original_x[positions])
        for scale, poses in paired_geometry(record).values():
            bounds = pose_bounds(bbox, scale, poses)
            pilot_margins.append(float(min(bounds[..., :2].min(), 28 - bounds[..., 2:].max())))
            for pose in poses.reshape(-1, 3):
                normal, reference = render_pose(tile, scale, pose), render_pose(tile, scale, pose, padding=14)
                np.testing.assert_array_equal(normal, reference[14:42, 14:42])
                outside = reference.copy()
                outside[14:42, 14:42] = 0
                assert not outside.any() and normal.any()
                compared_frames += 1
    del original_x
    write_json(ROOT / "data_pilot.json", dict(ok=True, raw_ids=[r["raw_id"] for r in pilot],
        original_v5_frames_reproduced=560, independent_reference_frames=compared_frames,
        cropped_frames=0, empty_frames=0, minimum_geometry_margin=min(pilot_margins),
        pair_geometry_checks=True, validation_or_sealed_ids=0))
    print("小样本渲染审计通过：原v5逐像素复现、四组配对和独立大画布裁切检查有效。", flush=True)
    arrays = {arm: np.empty((5600, 8, 784), dtype=np.uint8) for arm in ARMS}
    pose_arrays = {arm: np.empty((5600, 8, 3), dtype=np.float64) for arm in ARMS}
    scales = {arm: np.empty(5600, dtype=np.float64) for arm in ARMS}
    minimum_margin, empty_frames = 100., 0
    for index, record in enumerate(records):
        tile, bbox = foreground(images[record["raw_id"]])
        target_slice = slice(7 * index, 7 * (index + 1))
        assert np.all(ids[target_slice] == record["raw_id"]) and np.all(variants[target_slice] == record["variant"])
        for arm, (scale, poses) in paired_geometry(record).items():
            pixels = np.stack([render_pose(tile, scale, pose) for pose in poses.reshape(-1, 3)]).reshape(7, 8, 784)
            assert pixels.any(2).all()
            arrays[arm][target_slice], pose_arrays[arm][target_slice], scales[arm][target_slice] = pixels, poses, scale
            bounds = pose_bounds(bbox, scale, poses)
            minimum_margin = min(minimum_margin, float(min(bounds[..., :2].min(), 28 - bounds[..., 2:].max())))
        if (index + 1) % 200 == 0:
            print(f"四组配对数据已生成{index + 1}/{len(records)}条原始轨迹。", flush=True)
    np.testing.assert_array_equal(pose_arrays["small_full"], pose_arrays["large_full"])
    np.testing.assert_array_equal(pose_arrays["small_narrow"], pose_arrays["large_narrow"])
    np.testing.assert_array_equal(pose_arrays["small_full"][..., :2], pose_arrays["small_narrow"][..., :2])
    np.testing.assert_allclose(scales["large_full"], 1.8 * scales["small_full"], rtol=0, atol=0)
    hashes = {}
    for arm in ARMS:
        path = ROOT / f"{arm}_data.npz"
        np.savez_compressed(path, x=arrays[arm], y=y, ids=ids, fit_mask=fit_mask,
                            variants=variants, poses=pose_arrays[arm], scales=scales[arm])
        hashes[arm] = sha(path)
    data = dict(manifest_sha256=sha(manifest_path), source_cache_sha256=manifest["cache_sha256"],
        data_seed=20261004, fit_original_ids=160, control_original_ids=40,
        fit_videos=4480, control_videos=1120, raw_ids_extracted=200,
        generated_videos=4 * 5600, generated_frames=4 * 5600 * 8,
        full_generation_pair_checks=True, minimum_geometry_margin=minimum_margin,
        empty_frames=empty_frames, cache_sha256=hashes, generation_wall_seconds=time.perf_counter() - started,
        validation_or_sealed_generation=0,
        caveat="共同缩短位移和收窄位置；纯因素比较适用于这四组，不是原v5绝对成绩。")
    write_json(ROOT / "data_audit.json", data)
    return data


def prepare():
    if (ROOT / "config.json").exists():
        raise ValueError("运行目录已登记；使用--resume，禁止覆盖证据")
    ROOT.mkdir(parents=True, exist_ok=True)
    data = prepare_data()
    source_hashes = {path: sha(path) for path in SOURCES}
    for source in SOURCES:
        path = ROOT / "source_snapshot" / source
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, path)
    seed()
    model = make_model("stacked_cnn")
    torch.save(dict(model=model.state_dict(), model_seed=11), ROOT / "initial_model.pt")
    config = dict(protocol="diagnostic_size_angle_v1", model_seed=11, maximum_epochs=20,
        arms=list(ARMS), batch_size=128, learning_rate=0.001, weight_decay=0.0001,
        training_objective="数字CE+运动CE", precision="FP32 highest deterministic TF32 off",
        optimizer="AdamW foreach=False; clip_grad_norm=1", model_resources=resources(model, "stacked_cnn"),
        source_sha256=source_hashes, data=data, initial_model_sha256=sha(ROOT / "initial_model.pt"),
        start_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        workspace_status=subprocess.check_output(["git", "status", "--short"], text=True),
        gpu_wall_budget_seconds=900, control_epochs=[5, 10, 15, 20], fit_epochs=[10, 20],
        primary_metric="控制第15/20遍数字均值；不按最佳检查点选择",
        practical_support_threshold_percentage_points=5.0, validation_forward_evaluations=0, sealed_test_reads=0)
    write_json(ROOT / "config.json", config)
    print("四组数据和共同种子11初始模型已登记；正式训练尚未开始。", flush=True)
    return config


def verify_config():
    config = read(ROOT / "config.json")
    for path, expected in config["source_sha256"].items():
        if sha(path) != expected:
            raise ValueError(f"源码哈希改变：{path}，拒绝无记录续训")
    assert sha(ROOT / "initial_model.pt") == config["initial_model_sha256"]
    assert sha("data/manifest_v5.json") == config["data"]["manifest_sha256"]
    for arm, expected in config["data"]["cache_sha256"].items():
        assert sha(ROOT / f"{arm}_data.npz") == expected
    return config


def load_data(arm):
    with np.load(ROOT / f"{arm}_data.npz") as arrays:
        mask = arrays["fit_mask"]
        x, y = arrays["x"].astype(np.float32) / 255, arrays["y"]
        ids = arrays["ids"]
    fit = tuple(torch.as_tensor(a, device="cuda") for a in (x[mask], y[mask]))
    control = tuple(torch.as_tensor(a, device="cuda") for a in (x[~mask], y[~mask]))
    return fit, control, ids[mask], ids[~mask]


def train(config):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用；保留准备结果")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    initial = torch.load(ROOT / "initial_model.pt", map_location="cpu", weights_only=False)
    budget = read(ROOT / "gpu_budget.json") if (ROOT / "gpu_budget.json").exists() else dict(wall_seconds=0.)
    gpu_session_start, previous_wall = time.perf_counter(), budget["wall_seconds"]

    def save_budget():
        budget["wall_seconds"] = previous_wall + time.perf_counter() - gpu_session_start
        write_json(ROOT / "gpu_budget.json", budget)
        return budget["wall_seconds"]

    if not (ROOT / "diagnostic/diagnostic.json").exists():
        directory = ROOT / "diagnostic"
        directory.mkdir(exist_ok=True)
        seed()
        model = make_model("stacked_cnn").cuda()
        model.load_state_dict(initial["model"])
        optimizer, rng = optimizer_for(model), np.random.default_rng(11)
        fit, control, _, _ = load_data("large_narrow")
        positions = np.linspace(0, len(fit[0]) - 1, 32, dtype=int)
        ledger = Ledger(directory, costs_zero())
        ledger.recover()
        before = copy.deepcopy(payload(model, optimizer, rng, {}))
        diagnostic(directory, model, optimizer, rng, fit[0][positions], fit[1][positions], ledger)
        restore(before, model, optimizer, rng)
        assert nested_equal(before, payload(model, optimizer, rng, {}))
        evidence = read(directory / "diagnostic.json")
        evidence["restored_to_original_model_optimizer_rng"] = True
        write_json(directory / "diagnostic.json", evidence)
        del before, model, optimizer, fit, control
        gc.collect()
        torch.cuda.empty_cache()
        print("真实GPU梯度/优化器/RNG重放诊断通过，全部诊断更新已撤回。", flush=True)
    for arm in ARMS:
        directory = ROOT / arm
        directory.mkdir(exist_ok=True)
        (directory / "evaluations").mkdir(exist_ok=True)
        if (directory / "result.json").exists() and read(directory / "result.json")["status"] == "completed":
            print(f"{arm}已完成20遍，跳过模型运行。", flush=True)
            continue
        seed()
        model = make_model("stacked_cnn").cuda()
        model.load_state_dict(initial["model"])
        optimizer, rng = optimizer_for(model), np.random.default_rng(11)
        state = dict(completed_epochs=0, costs=costs_zero(), model_initialization_equal=True,
                     wall_seconds=0., optimizer_wall_seconds=0.)
        if (directory / "resume.pt").exists():
            state = restore(load_resume(directory), model, optimizer, rng)
        else:
            assert nested_equal(initial["model"], model.state_dict())
            save_resume(directory, model, optimizer, rng, state)
        ledger = Ledger(directory, state["costs"])
        ledger.recover()
        fit, control, fit_ids, control_ids = load_data(arm)
        np.savez_compressed(directory / "split_evidence.npz", fit_ids=fit_ids,
                            control_ids=control_ids, fit_y=fit[1].cpu().numpy(), control_y=control[1].cpu().numpy())
        torch.cuda.reset_peak_memory_stats()
        arm_start, prior_wall = time.perf_counter(), state["wall_seconds"]
        print(f"{arm}开始：同初始权重、4480拟合/1120控制、最多20遍。", flush=True)
        try:
            for epoch in range(state["completed_epochs"] + 1, 21):
                if save_budget() >= 900 or (ROOT / "STOP").exists():
                    state["wall_seconds"] = prior_wall + time.perf_counter() - arm_start
                    save_resume(directory, model, optimizer, rng, state)
                    write_json(directory / "result.json", dict(status="engineering_stop", **state))
                    print("工程停止：保存完整遍检查点，不扩大预算。", flush=True)
                    return False
                torch.cuda.synchronize()
                began, loss_sum, norm_max = time.perf_counter(), 0., 0.
                order = rng.permutation(len(fit[0]))
                for offset in range(0, len(order), 128):
                    positions = torch.as_tensor(order[offset:offset + 128], device="cuda")
                    loss, norm = update(model, optimizer, fit[0][positions], fit[1][positions])
                    ledger.add("optimization", epoch, train_forward_videos=len(positions),
                               train_backward_videos=len(positions), optimizer_updates=1)
                    loss_sum += loss * len(positions)
                    norm_max = max(norm_max, norm)
                torch.cuda.synchronize()
                train_wall = time.perf_counter() - began
                state["optimizer_wall_seconds"] += train_wall
                control_metrics = evaluate_pool(model, *control, directory, "control", epoch, ledger, 128) if epoch % 5 == 0 else None
                fit_metrics = evaluate_pool(model, *fit, directory, "fit", epoch, ledger, 128) if epoch % 10 == 0 else None
                row = dict(epoch=epoch, learning_rate=0.001, optimizer_mean_loss=loss_sum / len(fit[0]),
                    gradient_norm_max=norm_max, optimizer_wall_seconds=train_wall,
                    control=control_metrics, fit=fit_metrics, costs=ledger.costs.copy())
                write_json(directory / f"epoch_{epoch:03d}.json", row)
                state["completed_epochs"] = epoch
                state["wall_seconds"] = prior_wall + time.perf_counter() - arm_start
                save_resume(directory, model, optimizer, rng, state)
                save_budget()
                if control_metrics is not None:
                    print(f"{arm}第{epoch}遍 控制数字{control_metrics['digit_accuracy']:.2%} 运动{control_metrics['motion_accuracy']:.2%} 联合{control_metrics['joint_accuracy']:.2%}；损失{row['optimizer_mean_loss']:.4f}", flush=True)
                if fit_metrics is not None:
                    print(f"{arm}第{epoch}遍 拟合数字{fit_metrics['digit_accuracy']:.2%} 运动{fit_metrics['motion_accuracy']:.2%} 联合{fit_metrics['joint_accuracy']:.2%}", flush=True)
            result = dict(status="completed", **state, resources=resources(model, "stacked_cnn"),
                gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                process_peak_working_set_bytes=peak_working_set_bytes(),
                model_seed=11, device=torch.cuda.get_device_name(0), torch_version=torch.__version__,
                resume_sha256=sha(directory / "resume.pt"))
            write_json(directory / "result.json", result)
        except (Exception, KeyboardInterrupt) as exc:
            write_json(directory / "failure.json", dict(error=repr(exc), last_completed_epoch=state["completed_epochs"],
                costs=ledger.costs.copy(), note="在途工作保留账目；只能从完整遍检查点续训并另计重放成本。"))
            save_budget()
            raise
        finally:
            del model, optimizer, fit, control
            gc.collect()
            torch.cuda.empty_cache()
    write_json(ROOT / "summary.json", dict(completed=True, arms={arm: read(ROOT / arm / "result.json") for arm in ARMS},
        common_diagnostic=read(ROOT / "diagnostic/diagnostic.json"), gpu_wall_seconds=save_budget(),
        validation_forward_evaluations=0, sealed_test_reads=0,
        resume_command=".venv\\Scripts\\python.exe -u -B -m tools.train_digit_data_diagnostic --resume"))
    print("四组均完成登记20遍预算；没有追加训练，验证及封存测试未读取。", flush=True)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    config = verify_config() if args.resume else prepare()
    if not args.prepare_only:
        train(config)


if __name__ == "__main__":
    main()
