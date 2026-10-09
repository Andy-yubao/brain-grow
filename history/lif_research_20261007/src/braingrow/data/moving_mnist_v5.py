"""v5增强：所有运动共享全圆姿态分布，固定ID内增加独立轨迹；不提供特征。"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from braingrow.data.rendering import foreground, render_pose, pose_bounds

CONFIG = {
    "renderer_revision": "full_circle_shared_pose_v5_1", "frames": 8,
    "output": 28, "input_dimensions": 784, "seed": 20261004,
    "start_angle_all_motions": [-180., 180.], "rotation_total_degrees": 105.,
    "translation_total_pixels": [14.1, 14.4], "foreground_diagonal_pixels": [10., 11.5],
    "geometry_margin_pixels": 1., "train_variants": 4, "validation_variants": 2,
    "interpolation": "original_foreground_one_affine_bilinear_4x_then_BOX",
    "seed_rule": "single_root_20261004_SeedSequence_raw_id_variant",
    "digit_identity": "original_MNIST_label_including_rotated_6_9",
    "remaining_shortcut": "large_net_translation_constrains_endpoint_position",
}


def render(image, raw_id, variant=0, seed=20261004):
    if seed != 20261004:
        raise ValueError("早期数据根种子固定20261004")
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(raw_id), int(variant), 5]))
    tile, bbox = foreground(image)
    scale = rng.uniform(*CONFIG["foreground_diagonal_pixels"]) / np.hypot(bbox[2]-bbox[0], bbox[3]-bbox[1])
    # 全圆的共同安全区使位置/尺度不再由旋转方向或角窗决定。
    radius = scale * np.hypot(bbox[2]-bbox[0], bbox[3]-bbox[1]) / 2
    low, high = np.full(2, radius + 1), np.full(2, 27 - radius)
    poses = np.zeros((7, 8, 3), np.float64)
    t = np.linspace(0, 1, 8)
    for motion in range(7):
        angle = rng.uniform(-180., 180.)
        delta = np.zeros(2)
        if motion in (1, 2, 3, 4):
            axis = 1 if motion < 3 else 0
            sign = -1 if motion in (1, 3) else 1
            delta[axis] = sign * rng.uniform(14.1, 14.4)
        start = rng.uniform(low - np.minimum(delta, 0), high - np.maximum(delta, 0))
        poses[motion, :, :2] = start + t[:, None] * delta
        poses[motion, :, 2] = angle + t * (105. if motion == 5 else -105. if motion == 6 else 0.)
    bounds = pose_bounds(bbox, scale, poses)
    if bounds[..., :2].min() < 1 or bounds[..., 2:].max() > 27:
        raise ValueError("增强轨迹越界")
    pixels = np.stack([render_pose(tile, scale, pose) for pose in poses.reshape(-1, 3)])
    metadata = {"raw_id": int(raw_id), "variant": int(variant), "scale": float(scale),
                "foreground_bbox_xyxy": list(bbox), "poses_xy_clockwise_degrees": poses.tolist()}
    return pixels.reshape(7, 8, 784).astype(np.float32) / 255, metadata


def prepare(root=Path("data")):
    from braingrow.data.dynamic import load_mnist, split_ids
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path, manifest_path = root / "dynamic_v5.npz", root / "manifest_v5.json"
    if path.exists() or manifest_path.exists():
        if not path.exists() or not manifest_path.exists():
            raise ValueError("v5缓存不完整；保留现场")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["config"] != CONFIG or hashlib.sha256(path.read_bytes()).hexdigest() != manifest["cache_sha256"]:
            raise ValueError("v5协议/哈希不匹配；禁止覆盖")
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    images, labels, sources = load_mnist(root / "mnist")
    splits = split_ids(labels, 20261004)
    data, records = {}, {}
    for split, variants in (("train", 4), ("validation", 2)):
        videos, targets, ids, poses, variant_ids = [], [], [], [], []
        records[split] = []
        for raw_id in splits[split]:
            for variant in range(variants):
                x, m = render(images[raw_id], raw_id, variant)
                m["digit"] = int(labels[raw_id])
                records[split].append(m)
                videos.append(x)
                targets.append(np.column_stack((np.full(7, labels[raw_id]), np.arange(7))))
                ids.extend([raw_id]*7)
                variant_ids.extend([variant]*7)
                poses.append(np.asarray(m["poses_xy_clockwise_degrees"]))
        data.update({split+"_x": np.concatenate(videos), split+"_y": np.concatenate(targets).astype(np.int64),
                     split+"_ids": np.asarray(ids), split+"_poses": np.concatenate(poses),
                     split+"_variants": np.asarray(variant_ids)})
    pending = path.with_suffix(".pending.npz")
    np.savez_compressed(pending, **data)
    manifest = {"version": "v5", "seed": 20261004, "config": CONFIG, "splits": splits,
                "samples": records, "sources": sources, "cache_sha256": hashlib.sha256(pending.read_bytes()).hexdigest(),
                "renderer_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "test_policy": "原始训练池余下ID继续封存；官方测试未下载",
                "motion_labels": ["静止", "上移", "下移", "左移", "右移", "顺时针", "逆时针"]}
    mp = manifest_path.with_suffix(".pending.json")
    mp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    pending.replace(path)
    mp.replace(manifest_path)
    return data
