"""新结构研究的数据边界；教师信息与原始 ID 只进入分析清单。"""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile

import numpy as np


DATA_SEED = 20261004
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_VERSION = "structure_synthetic_v1"
MNIST_VERSION = "structure_mnist_v1"


@dataclass(frozen=True)
class DatasetBundle:
    fit_x: np.ndarray
    fit_y: np.ndarray
    select_x: np.ndarray
    select_y: np.ndarray
    confirm_x: np.ndarray
    confirm_y: np.ndarray
    manifest: dict


def _hash_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     suffix=".tmp", delete=False) as target:
        temporary = Path(target.name)
        json.dump(value, target, ensure_ascii=False, indent=2)
        target.write("\n")
    os.replace(temporary, path)


def _save_npz(path: Path, **arrays: np.ndarray) -> str:
    if path.exists():
        raise ValueError(f"拒绝覆盖已有缓存：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as target:
        temporary = Path(target.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return _hash_file(path)


def _read_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_array(x: np.ndarray, y: np.ndarray, count: int, dimensions: int,
                    classes: int) -> None:
    if (x.dtype != np.float32 or y.dtype != np.int64 or x.shape != (count, dimensions)
            or y.shape != (count,) or not np.isfinite(x).all()
            or (count and (y.min() < 0 or y.max() >= classes))):
        raise ValueError("数据缓存维度、类型或数值非法")


def _empty(dimensions: int) -> tuple[np.ndarray, np.ndarray]:
    return np.empty((0, dimensions), dtype=np.float32), np.empty(0, dtype=np.int64)


def _verify_cache(path: Path, expected_hash: str) -> None:
    if not path.is_file() or _hash_file(path) != expected_hash:
        raise ValueError(f"缓存缺失或哈希不一致，拒绝自动覆盖：{path}")


def load_synthetic(condition: str = "translated", seed: int = DATA_SEED,
                   root: str | Path | None = None,
                   include_confirm: bool = False) -> DatasetBundle:
    """64 维环面模板分类；默认不加载冻结确认数组。root 是项目根目录。"""
    if condition not in {"translated", "fixed"}:
        raise ValueError("condition 必须为 translated 或 fixed")
    if seed != DATA_SEED:
        raise ValueError("当前早期阶段固定数据根 20261004，不允许挑选数据种子")
    project = Path(root) if root is not None else PROJECT_ROOT
    cache = project / "data" / SYNTHETIC_VERSION
    manifest_path = cache / "manifest.json"
    array_path = cache / f"{condition}.npz"
    config = dict(version=SYNTHETIC_VERSION, seed=seed, condition=condition,
                  input_dimensions=64, classes=2, split_counts=dict(fit=10000, select=2000,
                  confirm=2000), templates=[[-1, -.5, 0, .5, 1], [-1, .5, 0, .5, -1]],
                  normalization="subtract_mean_then_unit_l2", amplitude_range=[1.5, 2.0], noise_std=.15,
                  boundary="circular", position_rule="uniform_0_63" if condition == "translated"
                  else "fixed_32", split_rng="SeedSequence(root).spawn(3)",
                  label_rule="exact_balance_then_rng_shuffle")
    parameter_hash = _hash_json(config)
    manifest = (_read_manifest(manifest_path) if manifest_path.exists() else
                dict(version=SYNTHETIC_VERSION, seed=seed, conditions={}))
    if manifest.get("version") != SYNTHETIC_VERSION or manifest.get("seed") != seed:
        raise ValueError("合成缓存版本或数据种子不一致，拒绝覆盖")
    entries = manifest.get("conditions")
    if not isinstance(entries, dict):
        raise ValueError("合成数据清单缺少 conditions")
    if condition in entries:
        entry = entries[condition]
        if entry.get("parameter_sha256") != parameter_hash or entry.get("config") != config:
            raise ValueError("合成缓存参数哈希不一致，拒绝覆盖")
        _verify_cache(array_path, entry["cache_sha256"])
    else:
        if array_path.exists():
            raise ValueError("合成缓存已有数组却没有对应清单，拒绝覆盖")
        arrays, positions = {}, {}
        templates = np.asarray(config["templates"], dtype=np.float32)
        templates -= templates.mean(axis=1, keepdims=True)
        templates /= np.linalg.norm(templates, axis=1, keepdims=True)
        for (split, count), stream in zip(config["split_counts"].items(),
                                         np.random.SeedSequence(seed).spawn(3)):
            rng = np.random.default_rng(stream)
            labels = np.repeat(np.arange(2, dtype=np.int64), count // 2)
            rng.shuffle(labels)
            sampled_positions = rng.integers(0, 64, size=count)
            starts = sampled_positions if condition == "translated" else np.full(count, 32)
            amplitudes = rng.uniform(1.5, 2., size=count).astype(np.float32)
            pixels = rng.normal(0., .15, size=(count, 64)).astype(np.float32)
            indices = (starts[:, None] + np.arange(5)[None, :]) % 64
            pixels[np.arange(count)[:, None], indices] += templates[labels] * amplitudes[:, None]
            arrays[f"{split}_x"], arrays[f"{split}_y"] = pixels, labels
            positions[split] = starts.tolist()
        entry = dict(config=config, parameter_sha256=parameter_hash,
                     cache_sha256=_save_npz(array_path, **arrays),
                     analysis_truth=dict(normalized_templates=templates.tolist(),
                                         template_start_positions=positions),
                     model_input="仅 float32 输入与 int64 类别；模板与位置不得传给优化器",
                     split_isolation="每集合独立随机子流，无样本跨集合复用")
        entries[condition] = entry
        _write_json(manifest_path, manifest)
    loaded = {}
    with np.load(array_path, allow_pickle=False) as archive:
        for split in ("fit", "select", "confirm") if include_confirm else ("fit", "select"):
            loaded[split] = (archive[f"{split}_x"], archive[f"{split}_y"])
            _validate_array(*loaded[split], config["split_counts"][split], 64, 2)
    confirm_x, confirm_y = loaded.get("confirm", _empty(64))
    return DatasetBundle(*loaded["fit"], *loaded["select"], confirm_x, confirm_y,
                         dict(version=SYNTHETIC_VERSION, task=f"synthetic_{condition}",
                              seed=seed, **entry, confirm_loaded=include_confirm))


def _skip(stream, length: int) -> None:
    """有界丢弃字节，不将跳过的封存样本解析或保留为像素数组。"""
    while length:
        chunk = stream.read(min(length, 1024 * 1024))
        if not chunk:
            raise ValueError("IDX 文件提前结束")
        length -= len(chunk)


def _selected_idx(path: Path, ids: list[int], images: bool) -> np.ndarray:
    # gzip 必须连续解压经过中间字节；这不构成封存 ID 的提取/访问授权。
    expected_magic, header_size = (2051, 16) if images else (2049, 8)
    with gzip.open(path, "rb") as source:
        header = source.read(header_size)
        if len(header) != header_size:
            raise ValueError("IDX 文件头不完整")
        fields = struct.unpack(">" + "I" * (header_size // 4), header)
        if fields[0] != expected_magic or fields[1] != 60000:
            raise ValueError("仅允许本地 MNIST 60000 条训练来源")
        if images and fields[2:] != (28, 28):
            raise ValueError("MNIST 图像应为 28×28")
        width = 784 if images else 1
        selected = np.empty((len(ids), width), dtype=np.uint8)
        cursor = 0
        for output, raw_id in sorted(enumerate(ids), key=lambda pair: pair[1]):
            _skip(source, (raw_id - cursor) * width)
            payload = source.read(width)
            if len(payload) != width:
                raise ValueError("IDX 允许样本不完整")
            selected[output] = np.frombuffer(payload, dtype=np.uint8)
            cursor = raw_id + 1
    return selected if images else selected[:, 0].astype(np.int64)


def _authorized_ids(manifest: dict) -> tuple[list[int], list[int]]:
    if manifest.get("seed") != DATA_SEED:
        raise ValueError("母清单数据根与 20261004 不一致")
    splits = manifest.get("splits", {})
    fit, confirm = splits.get("fit", []), splits.get("control", [])
    if len(fit) != 4000 or len(confirm) != 1000:
        raise ValueError("授权母清单必须包含 4000 fit 与 1000 control 原 ID")
    combined = fit + confirm
    if (any(type(raw_id) is not int or raw_id < 0 or raw_id >= 60000 for raw_id in combined)
            or len(set(combined)) != 5000):
        raise ValueError("授权原 ID 重复、重叠或超出训练来源范围")
    return fit, confirm


def _load_mnist_split(cache: Path, manifest: dict, split: str) -> tuple[np.ndarray, np.ndarray]:
    file = cache / f"{split}.npz"
    _verify_cache(file, manifest["cache_sha256"][split])
    with np.load(file, allow_pickle=False) as archive:
        x, y = archive["x"], archive["y"]
    _validate_array(x, y, len(manifest["splits"][split]), 784, 10)
    if np.any(x < 0) or np.any(x > 1):
        raise ValueError("MNIST 像素超出 [0,1]")
    return x, y


def load_mnist(root: str | Path | None = None,
               include_confirm: bool = False) -> DatasetBundle:
    """仅授权 5000 原 ID；默认只提取 4000 fit 范围用于 3200/800 划分。"""
    project = Path(root) if root is not None else PROJECT_ROOT
    mother_path = project / "data" / "cnn_5000_v6" / "manifest.json"
    image_path = project / "data" / "mnist" / "train-images-idx3-ubyte.gz"
    label_path = project / "data" / "mnist" / "train-labels-idx1-ubyte.gz"
    mother = _read_manifest(mother_path)
    authorized_fit, authorized_confirm = _authorized_ids(mother)
    sources = dict(mother_manifest_sha256=_hash_file(mother_path),
                   train_images_gzip_sha256=_hash_file(image_path),
                   train_labels_gzip_sha256=_hash_file(label_path))
    config = dict(version=MNIST_VERSION, seed=DATA_SEED, input_dimensions=784,
                  pixel_scale="uint8/255_float32", split_rule="sort_ids_per_digit_then_root_rng_permute",
                  fit_per_digit=320, select_per_digit=80, confirm_per_digit=100,
                  sources=sources)
    parameter_hash = _hash_json(config)
    cache = project / "data" / MNIST_VERSION
    manifest_path = cache / "manifest.json"
    if manifest_path.exists():
        manifest = _read_manifest(manifest_path)
        if (manifest.get("version") != MNIST_VERSION
                or manifest.get("parameter_sha256") != parameter_hash
                or manifest.get("config") != config):
            raise ValueError("MNIST 缓存版本、来源或参数哈希不一致，拒绝覆盖")
        splits = manifest.get("splits", {})
        fit_ids, select_ids, confirm_ids = [splits.get(name, []) for name in ("fit", "select", "confirm")]
        if (len(fit_ids) != 3200 or len(select_ids) != 800 or len(set(fit_ids + select_ids)) != 4000
                or set(fit_ids + select_ids) != set(authorized_fit)
                or confirm_ids != authorized_confirm):
            raise ValueError("缓存 ID 划分与母清单授权不一致")
    else:
        if cache.exists() and any(cache.glob("*.npz")):
            raise ValueError("MNIST 缓存数组缺少清单，拒绝覆盖")
        labels = _selected_idx(label_path, authorized_fit, images=False)
        pixels = _selected_idx(image_path, authorized_fit, images=True).astype(np.float32) / np.float32(255)
        raw_ids = np.asarray(authorized_fit, dtype=np.int64)
        fit_indices, select_indices = [], []
        for digit, stream in enumerate(np.random.SeedSequence(DATA_SEED).spawn(10)):
            members = np.flatnonzero(labels == digit)
            if len(members) != 400:
                raise ValueError("母清单 fit 必须每数字 400 原 ID")
            members = members[np.argsort(raw_ids[members])]
            members = np.random.default_rng(stream).permutation(members)
            fit_indices.extend(members[:320].tolist())
            select_indices.extend(members[320:].tolist())
        manifest = dict(version=MNIST_VERSION, task="mnist", seed=DATA_SEED,
                        config=config, parameter_sha256=parameter_hash,
                        splits=dict(fit=raw_ids[fit_indices].tolist(), select=raw_ids[select_indices].tolist(),
                                    confirm=authorized_confirm), cache_sha256={},
                        label_counts=dict(fit=[320] * 10, select=[80] * 10),
                        permissions_used=dict(authorized_original_ids=5000, extracted_fit_original_ids=4000,
                                              extracted_confirm_original_ids=0,
                                              sealed_original_ids_extracted=0, official_test_accessed=False),
                        extraction_note="gzip 连续解压中间字节仅为到达授权 ID；跳过字节不解析、不缓存为样本，封存 ID 未解封",
                        confirmation_note="原 control 曾用于历史探索，冻结后评价仍为探索性确认，非全新最终测试",
                        model_input="仅 784 维原静态灰度 float32[0,1] 与 int64 数字类别，无 ID、位置、旧姿态或运动标签")
        for split, indices in (("fit", fit_indices), ("select", select_indices)):
            manifest["cache_sha256"][split] = _save_npz(cache / f"{split}.npz", x=pixels[indices], y=labels[indices])
        _write_json(manifest_path, manifest)
    fit_x, fit_y = _load_mnist_split(cache, manifest, "fit")
    select_x, select_y = _load_mnist_split(cache, manifest, "select")
    if include_confirm:
        if "confirm" not in manifest["cache_sha256"]:
            if (cache / "confirm.npz").exists():
                raise ValueError("确认缓存没有对应哈希，拒绝覆盖")
            confirm_ids = manifest["splits"]["confirm"]
            confirm_y = _selected_idx(label_path, confirm_ids, images=False)
            confirm_x = _selected_idx(image_path, confirm_ids, images=True).astype(np.float32) / np.float32(255)
            if np.bincount(confirm_y, minlength=10).tolist() != [100] * 10:
                raise ValueError("授权 control 必须每数字 100 原 ID")
            manifest["cache_sha256"]["confirm"] = _save_npz(cache / "confirm.npz", x=confirm_x, y=confirm_y)
            manifest["label_counts"]["confirm"] = [100] * 10
            manifest["permissions_used"]["extracted_confirm_original_ids"] = 1000
            _write_json(manifest_path, manifest)
        confirm_x, confirm_y = _load_mnist_split(cache, manifest, "confirm")
    else:
        confirm_x, confirm_y = _empty(784)
    return DatasetBundle(fit_x, fit_y, select_x, select_y, confirm_x, confirm_y,
                         dict(manifest, confirm_loaded=include_confirm))


def load_bundle(task: str, root: str | Path | None = None,
                include_confirm: bool = False) -> DatasetBundle:
    """统一入口；确认数据只在候选全部冻结后显式开启。"""
    if task == "mnist":
        return load_mnist(root, include_confirm=include_confirm)
    conditions = {"synthetic_translated": "translated", "synthetic_fixed": "fixed"}
    if task not in conditions:
        raise ValueError(f"未知数据任务：{task}")
    return load_synthetic(conditions[task], root=root, include_confirm=include_confirm)
