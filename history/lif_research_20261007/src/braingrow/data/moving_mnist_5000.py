"""用户授权的5000原笔迹扩展；沿用v6渲染，4000拟合/1000控制。"""
import gzip
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

from braingrow.data.moving_mnist_v6 import CONFIG, render


ROOT = Path("data/cnn_5000_v6")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_ids(parent, labels):
    """保留旧拟合/控制，排除旧验证；新增ID只用于此次明确授权实验。"""
    old_fit = parent["optimization_splits"]["fit"]
    old_control = parent["optimization_splits"]["control"]
    excluded = set(parent["splits"]["train"]) | set(parent["splits"]["validation"])
    fit, control = [], []
    rng = np.random.default_rng(20261004)
    for digit in range(10):
        fit_digit = [i for i in old_fit if labels[i] == digit]
        control_digit = [i for i in old_control if labels[i] == digit]
        assert len(fit_digit) == 16 and len(control_digit) == 4
        candidates = np.array([i for i in parent["splits"]["sealed_test"]
                               if i not in excluded and labels[i] == digit], dtype=np.int64)
        chosen = rng.permutation(candidates)[:480].tolist()
        fit.extend(fit_digit + chosen[:384])
        control.extend(control_digit + chosen[384:])
    assert len(fit) == 4000 and len(control) == 1000 and not set(fit) & set(control)
    assert not (set(fit) | set(control)) & set(parent["splits"]["validation"])
    return dict(fit=fit, control=control)


def raw_sources(parent):
    for name in ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"):
        if sha(Path("data/mnist") / name) != parent["sources"][name]["sha256"]:
            raise ValueError("原始MNIST哈希不符")
    label_bytes = gzip.decompress(Path("data/mnist/train-labels-idx1-ubyte.gz").read_bytes())
    if struct.unpack(">II", label_bytes[:8]) != (2049, 60000):
        raise ValueError("原标签头部错误")
    return np.frombuffer(label_bytes, np.uint8, offset=8).copy()


def extract_images(splits, parent):
    ids = splits["fit"] + splits["control"]
    if len(set(ids)) != 5000 or set(ids) & set(parent["splits"]["validation"]):
        raise ValueError("仅许可此次5000原始笔迹；禁止提取其它ID")
    path = Path("data/mnist/train-images-idx3-ubyte.gz")
    if sha(path) != parent["sources"][path.name]["sha256"]:
        raise ValueError("原始MNIST像素哈希不符")
    raw = gzip.decompress(path.read_bytes())
    if struct.unpack(">IIII", raw[:16]) != (2051, 60000, 28, 28):
        raise ValueError("MNIST像素头部错误")
    return {i: np.frombuffer(raw, np.uint8, offset=16 + i * 784, count=784).reshape(28, 28).copy()
            for i in ids}


def load(root=ROOT):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["config"] != CONFIG or manifest["version"] != "v6_5000":
        raise ValueError("5000笔迹协议不符")
    for name, digest in manifest["file_sha256"].items():
        if sha(root / name) != digest:
            raise ValueError(f"扩展数据哈希不符：{name}")
    pools = []
    for pool in ("fit", "control"):
        x = np.load(root / f"{pool}_x.npy", mmap_mode="r")
        y = np.load(root / f"{pool}_y.npy", mmap_mode="r")
        ids = np.load(root / f"{pool}_ids.npy", mmap_mode="r")
        expected = 112000 if pool == "fit" else 28000
        if x.shape != (expected, 8, 784) or x.dtype != np.uint8 or len(y) != expected:
            raise ValueError("扩展数据尺寸不符")
        if set(ids.tolist()) != set(manifest["splits"][pool]):
            raise ValueError("扩展ID不符")
        pools.append((x, y, ids))
    return *pools, manifest
