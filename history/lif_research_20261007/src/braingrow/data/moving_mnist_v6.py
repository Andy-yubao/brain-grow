"""v6大数字、小幅运动、有限出画；原始ID划分继承v5，不读取封存像素。"""
from __future__ import annotations

import gzip
import hashlib
import json
import struct
import time
from pathlib import Path

import numpy as np

from braingrow.data.rendering import foreground, pose_bounds, render_pose


CONFIG = {
    "renderer_revision": "large_narrow_partial_crop_v6_1", "frames": 8,
    "output": 28, "input_dimensions": 784, "seed": 20261004,
    "foreground_diagonal_pixels": [18., 20.7], "start_angle_all_motions": [-15., 15.],
    "translation_total_pixels": [8., 10.], "rotation_total_degrees": [60., 90.],
    "trajectory_midpoint_xy_range": [8., 20.], "maximum_ink_mass_cropped_fraction": .15,
    "reference_padding_pixels": 12, "maximum_candidate_attempts_per_motion": 64,
    "train_variants": 4, "validation_variants": 2,
    "seed_rule": "root20261004_raw_id_variant_revision6; independent_motion_substream",
    "digit_identity": "original_MNIST_label", "interpolation": "original_foreground_one_affine_bilinear_4x_then_BOX",
    "visibility_measure": "sum_uint8_gray_in_28x28/sum_uint8_gray_in_reference",
    "remaining_shortcut": "finite_motion_window_and_visibility_rejection_may_constrain_positions",
}
MOTIONS = ("静止", "上移", "下移", "左移", "右移", "顺时针", "逆时针")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render(image, raw_id, variant=0, seed=20261004):
    if seed != CONFIG["seed"]:
        raise ValueError("数据根固定20261004，不换种子")
    if int(raw_id) != raw_id or int(variant) != variant or raw_id < 0 or variant < 0:
        raise ValueError("原始ID和variant须为非负整数")
    tile, bbox = foreground(image)
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(raw_id), int(variant), 6]))
    diagonal = float(rng.uniform(*CONFIG["foreground_diagonal_pixels"]))
    scale = diagonal / np.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1])
    pixels, all_poses, all_visible, attempts = [], [], [], []
    generation_reference_calls = 0
    t = np.linspace(-.5, .5, 8)
    for motion in range(7):
        motion_rng = np.random.default_rng(np.random.SeedSequence([seed, int(raw_id), int(variant), 6, motion]))
        for attempt in range(1, CONFIG["maximum_candidate_attempts_per_motion"] + 1):
            angle = motion_rng.uniform(*CONFIG["start_angle_all_motions"])
            midpoint = motion_rng.uniform(*CONFIG["trajectory_midpoint_xy_range"], size=2)
            delta = np.zeros(2)
            if motion in (1, 2, 3, 4):
                axis = 1 if motion < 3 else 0
                delta[axis] = (-1 if motion in (1, 3) else 1) * motion_rng.uniform(*CONFIG["translation_total_pixels"])
            poses = np.zeros((8, 3), dtype=np.float64)
            poses[:, :2] = midpoint + t[:, None] * delta
            # 初始角为第1帧的角；轨迹中心只用于平移对称定位。
            rotation = motion_rng.uniform(*CONFIG["rotation_total_degrees"]) * (1 if motion == 5 else -1) if motion in (5, 6) else 0.
            poses[:, 2] = angle + (t + .5) * rotation
            padding = CONFIG["reference_padding_pixels"]
            references = np.stack([render_pose(tile, scale, pose, padding=padding) for pose in poses])
            generation_reference_calls += 8
            frame_pixels = references[:, padding:padding + 28, padding:padding + 28]
            full_mass = references.sum((1, 2), dtype=np.uint64)
            visible_mass = frame_pixels.sum((1, 2), dtype=np.uint64)
            visibility = visible_mass / full_mass
            if np.all(full_mass > 0) and frame_pixels.any((1, 2)).all() and np.all(visibility >= 1 - CONFIG["maximum_ink_mass_cropped_fraction"]):
                pixels.append(frame_pixels.reshape(8, 784))
                all_poses.append(poses)
                all_visible.append(visibility)
                attempts.append(attempt)
                break
        else:
            raise ValueError(f"v6可见度采样失败，保留现场：ID={raw_id},variant={variant},motion={motion}")
    poses = np.stack(all_poses)
    metadata = dict(raw_id=int(raw_id), variant=int(variant), scale=float(scale),
        foreground_diagonal_pixels=diagonal, foreground_bbox_xyxy=list(bbox),
        poses_xy_clockwise_degrees=poses.tolist(), ink_visible_fraction=np.stack(all_visible).tolist(),
        foreground_bounds_xyxy=pose_bounds(bbox, scale, poses).tolist(),
        candidate_attempts_by_motion=attempts, generation_reference_render_calls=generation_reference_calls)
    return np.stack(pixels).astype(np.float32) / 255, metadata


def selected_images(ids, root, original_manifest):
    permitted = set(original_manifest["splits"]["train"]) | set(original_manifest["splits"]["validation"])
    ids = [int(v) for v in ids]
    if len(set(ids)) != len(ids) or not set(ids).issubset(permitted):
        raise ValueError("仅允许原200训练ID和100验证ID；不得提取封存像素")
    path = root / "mnist/train-images-idx3-ubyte.gz"
    if sha(path) != original_manifest["sources"][path.name]["sha256"]:
        raise ValueError("原始MNIST文件哈希不符")
    raw = gzip.decompress(path.read_bytes())
    if struct.unpack(">IIII", raw[:16]) != (2051, 60000, 28, 28):
        raise ValueError("原始MNIST头部不符")
    return {raw_id: np.frombuffer(raw, np.uint8, offset=16 + raw_id * 784, count=784).reshape(28, 28).copy()
            for raw_id in ids}


def load_cache(root=Path("data"), members=None):
    root = Path(root)
    path, manifest_path = root / "dynamic_v6.npz", root / "manifest_v6.json"
    if not path.exists() or not manifest_path.exists():
        raise ValueError("v6缓存不完整；拒绝覆盖现场")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["config"] != CONFIG or manifest["cache_sha256"] != sha(path):
        raise ValueError("v6协议/缓存哈希不符；拒绝覆盖")
    with np.load(path) as arrays:
        return {key: arrays[key] for key in (arrays.files if members is None else members)}


def training_data(root=Path("data")):
    root = Path(root)
    manifest = json.loads((root / "manifest_v6.json").read_text(encoding="utf-8"))
    data = load_cache(root, members=("train_x", "train_y", "train_ids"))
    x, y, ids = (data[key] for key in ("train_x", "train_y", "train_ids"))
    selected = np.isin(ids, manifest["optimization_splits"]["fit"])
    if not set(ids[selected]).isdisjoint(ids[~selected]) or selected.sum() != 4480 or (~selected).sum() != 1120:
        raise ValueError("v6拟合/控制隔离不符")
    return (x[selected], y[selected], ids[selected]), (x[~selected], y[~selected], ids[~selected])


def prepare(root=Path("data")):
    root = Path(root)
    path, manifest_path = root / "dynamic_v6.npz", root / "manifest_v6.json"
    if path.exists() or manifest_path.exists():
        return load_cache(root)
    if path.with_suffix(".pending.npz").exists() or manifest_path.with_suffix(".pending.json").exists():
        raise ValueError("存在未完成v6生成物；保留现场，拒绝覆盖")
    started = time.perf_counter()
    original_path = root / "manifest_v5.json"
    original = json.loads(original_path.read_text(encoding="utf-8"))
    if original["seed"] != CONFIG["seed"] or sha(root / "dynamic_v5.npz") != original["cache_sha256"]:
        raise ValueError("原v5根或哈希不符；拒绝改动")
    selected_ids = original["splits"]["train"] + original["splits"]["validation"]
    if len(set(selected_ids)) != 300 or not set(selected_ids).isdisjoint(original["splits"]["sealed_test"]):
        raise ValueError("源ID划分不符")
    labels = {row["raw_id"]: row["digit"] for split in ("train", "validation") for row in original["samples"][split]}
    images = selected_images(selected_ids, root, original)
    data, records = {}, {}
    for split, variants in (("train", 4), ("validation", 2)):
        videos, targets, ids, poses, variant_ids, visible = [], [], [], [], [], []
        records[split] = []
        for index, raw_id in enumerate(original["splits"][split]):
            for variant in range(variants):
                x, metadata = render(images[raw_id], raw_id, variant)
                metadata["digit"] = int(labels[raw_id])
                records[split].append(metadata)
                videos.append(x)
                targets.append(np.column_stack((np.full(7, labels[raw_id]), np.arange(7))))
                ids.extend([raw_id] * 7)
                variant_ids.extend([variant] * 7)
                poses.append(np.asarray(metadata["poses_xy_clockwise_degrees"]))
                visible.append(np.asarray(metadata["ink_visible_fraction"]))
            if (index + 1) % 50 == 0:
                print(f"v6 {split}：已生成{index + 1}/{len(original['splits'][split])}个原始笔迹。", flush=True)
        data.update({split + "_x": np.concatenate(videos), split + "_y": np.concatenate(targets).astype(np.int64),
                     split + "_ids": np.asarray(ids), split + "_poses": np.concatenate(poses),
                     split + "_variants": np.asarray(variant_ids), split + "_visible": np.concatenate(visible)})
    fit_ids, control_ids = [], []
    for digit in range(10):
        ordered = [raw_id for raw_id in original["splits"]["train"] if labels[raw_id] == digit]
        assert len(ordered) == 20
        fit_ids.extend(ordered[:16])
        control_ids.extend(ordered[16:])
    pending = path.with_suffix(".pending.npz")
    np.savez_compressed(pending, **data)
    manifest = dict(version="v6", seed=CONFIG["seed"], config=CONFIG, splits=original["splits"],
        optimization_splits=dict(fit=fit_ids, control=control_ids), samples=records,
        sources=original["sources"], parent_v5_manifest_sha256=sha(original_path),
        parent_v5_cache_sha256=original["cache_sha256"], cache_sha256=sha(pending),
        renderer_source_sha256=sha(__file__), shared_renderer_sha256=sha(Path(__file__).with_name("rendering.py")),
        motion_labels=list(MOTIONS), test_policy="只生成原200训练/100验证ID；最终测试封存且官方测试未下载",
        generation_wall_seconds=time.perf_counter() - started,
        generation_reference_render_calls=sum(r["generation_reference_render_calls"] for group in records.values() for r in group),
        original_ids_extracted=300, sealed_test_pixels_extracted=0, model_forward_videos=0, optimizer_updates=0)
    mp = manifest_path.with_suffix(".pending.json")
    mp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    pending.replace(path)
    mp.replace(manifest_path)
    return data
