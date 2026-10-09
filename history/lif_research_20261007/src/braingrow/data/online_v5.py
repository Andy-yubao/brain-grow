"""在原拟合ID内每轮重新渲染v5轨迹，不扩展ID，不改变输入表征。"""
from __future__ import annotations

import gzip
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from braingrow.data.moving_mnist_v5 import render


def selected_original_images(ids, root=Path("data")):
    """原压缩文件保持原样；只提取已列出的拟合ID像素，不返回其他ID。"""
    ids = [int(value) for value in ids]
    manifest = json.loads((root / "manifest_v5.json").read_text(encoding="utf-8"))
    permitted = set(manifest["splits"]["train"])
    if len(ids) != len(set(ids)) or not set(ids).issubset(permitted):
        raise ValueError("增强只能提取原训练划分中的唯一原始ID")
    path = root / "mnist" / "train-images-idx3-ubyte.gz"
    compressed = path.read_bytes()
    expected = manifest["sources"][path.name]["sha256"]
    if hashlib.sha256(compressed).hexdigest() != expected:
        raise ValueError("原始MNIST文件SHA256不符")
    raw = gzip.decompress(compressed)
    images = {raw_id: np.frombuffer(raw, dtype=np.uint8, offset=16 + raw_id * 784,
                                  count=784).reshape(28, 28).copy() for raw_id in ids}
    return images


def render_epoch(images, ids, digits, epoch):
    if epoch < 1 or int(epoch) != epoch:
        raise ValueError("调优遍数须为正整数")
    videos, targets, raw_ids, variants, bounds_metadata = [], [], [], [], []
    for raw_id, digit in zip(ids, digits, strict=True):
        for offset in range(4):
            variant = 4 + (int(epoch) - 1) * 4 + offset
            x, metadata = render(images[int(raw_id)], int(raw_id), variant)
            videos.append(x)
            targets.append(np.column_stack((np.full(7, digit), np.arange(7))))
            raw_ids.extend([int(raw_id)] * 7)
            variants.extend([variant] * 7)
            bounds_metadata.append(metadata)
    return dict(x=np.concatenate(videos), y=np.concatenate(targets).astype(np.int64),
                raw_ids=np.asarray(raw_ids), variants=np.asarray(variants)), bounds_metadata


def epoch_cache(root, images, ids, digits, epoch):
    """两增强臂共享完全相同的每轮视频；缓存校验失败时拒绝覆盖。"""
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"epoch_{epoch:03d}.npz"
    metadata_path = root / f"epoch_{epoch:03d}.json"
    if path.exists() or metadata_path.exists():
        if not path.exists() or not metadata_path.exists():
            raise ValueError("增强缓存不完整，保留证据")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (metadata["raw_ids"] != list(map(int, ids)) or metadata["digits"] != list(map(int, digits))
                or metadata["epoch"] != epoch
                or metadata["sha256"] != hashlib.sha256(path.read_bytes()).hexdigest()):
            raise ValueError("增强缓存ID/遍数/哈希不符")
        with np.load(path, allow_pickle=False) as saved:
            return {key: saved[key] for key in saved.files}, dict(generated=False, **metadata)
    started = time.perf_counter()
    arrays, trajectories = render_epoch(images, ids, digits, epoch)
    pending = path.with_name(path.name + ".pending.npz")
    np.savez_compressed(pending, **arrays)
    metadata = dict(epoch=epoch, data_seed=20261004, raw_ids=list(map(int, ids)),
                    digits=list(map(int, digits)), videos=len(arrays["x"]),
                    variants=[4 + (epoch - 1) * 4, 7 + (epoch - 1) * 4],
                    sha256=hashlib.sha256(pending.read_bytes()).hexdigest(),
                    renderer="原v5 render；全圆姿态、原灰度、8帧784维；几何检查由原渲染器执行",
                    wall_seconds=time.perf_counter() - started, trajectories=trajectories)
    temporary_metadata = metadata_path.with_name(metadata_path.name + ".pending")
    temporary_metadata.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
    pending.replace(path)
    temporary_metadata.replace(metadata_path)
    return arrays, dict(generated=True, **metadata)
