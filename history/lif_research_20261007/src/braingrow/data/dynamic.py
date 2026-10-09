"""MNIST原始文件校验、固定ID划分及当前v5数据入口。"""
from __future__ import annotations
import gzip
import hashlib
import struct
import urllib.request
from pathlib import Path
import numpy as np

MOTIONS = ("静止", "上移", "下移", "左移", "右移", "顺时针", "逆时针")
BASE = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {"train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
         "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432"}

def load_mnist(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    sources = {}
    for name, md5 in FILES.items():
        path = root / name
        if not path.exists():
            with urllib.request.urlopen(BASE + name, timeout=60) as response:
                raw = response.read()
            if hashlib.md5(raw).hexdigest() != md5:
                raise ValueError("MNIST 下载校验失败")
            path.write_bytes(raw)
        raw = path.read_bytes()
        if hashlib.md5(raw).hexdigest() != md5:
            raise ValueError("MNIST 文件校验失败")
        sources[name] = {"url": BASE + name, "md5": md5, "sha256": hashlib.sha256(raw).hexdigest()}
    images = gzip.decompress((root / "train-images-idx3-ubyte.gz").read_bytes())
    labels = gzip.decompress((root / "train-labels-idx1-ubyte.gz").read_bytes())
    assert struct.unpack(">IIII", images[:16]) == (2051, 60000, 28, 28)
    assert struct.unpack(">II", labels[:8]) == (2049, 60000)
    return np.frombuffer(images, np.uint8, offset=16).reshape(-1, 28, 28), np.frombuffer(labels, np.uint8, offset=8), sources

def split_ids(labels, seed=20261004, train_per_digit=20, val_per_digit=10):
    rng = np.random.default_rng(seed)
    splits = {"train": [], "validation": [], "sealed_test": []}
    for digit in range(10):
        ids = rng.permutation(np.flatnonzero(labels == digit))
        splits["train"].extend(ids[:train_per_digit].tolist())
        splits["validation"].extend(ids[train_per_digit:train_per_digit + val_per_digit].tolist())
        splits["sealed_test"].extend(ids[train_per_digit + val_per_digit:].tolist())
    return splits


def prepare(root=Path("data"), seed=20261004, version="v5"):
    if version != "v5" or seed != 20261004:
        raise ValueError("当前工作区仅支持v5、固定数据种子20261004；旧版本见Git历史")
    from braingrow.data.moving_mnist_v5 import prepare as prepare_v5
    return prepare_v5(root)
