"""FP32 严格共享推断导出：保存图及活跃代表，不量化，不保存优化器。

    BGS1 | uint32 JSON长度 | UTF-8元数据 | 各层zlib(npz)字节流。
    层内选择实测字节最短布局；identity无映射，exceptions相对identity，
    packed使用紧凑组编号，sparse显式活跃边，masked_identity压位掩码。
    每种布局都包含偏置、代表值及解码所需的完整结构描述。
"""

import io
import json
import struct
import zlib

import numpy as np
import torch

from .models import StrictSharedLinear, StructuredMLP

MAGIC = b"BGS1"
FORMAT = "braingrow-lossless-npz-zlib-v1"


def _uint_dtype(maximum: int):
    if maximum <= np.iinfo(np.uint8).max:
        return np.uint8
    if maximum <= np.iinfo(np.uint16).max:
        return np.uint16
    if maximum <= np.iinfo(np.uint32).max:
        return np.uint32
    return np.uint64


def _pack(codes: np.ndarray, bits: int) -> np.ndarray:
    flags = ((codes.astype(np.uint64)[:, None] >> np.arange(bits, dtype=np.uint64)) & 1).astype(np.uint8)
    return np.packbits(flags.reshape(-1), bitorder="little")


def _unpack(payload: np.ndarray, count: int, bits: int) -> np.ndarray:
    if bits < 1 or bits > 63 or payload.size != (count * bits + 7) // 8:
        raise ValueError("压位映射长度或位宽不合法")
    flags = np.unpackbits(payload, bitorder="little", count=count * bits).reshape(count, bits)
    return (flags.astype(np.uint64) << np.arange(bits, dtype=np.uint64)).sum(axis=1).astype(np.int64)


def _npz_blob(layout: str, weights: np.ndarray, bias: np.ndarray, **mapping) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, layout=np.array(layout), weights=weights, bias=bias, **mapping)
    return zlib.compress(buffer.getvalue(), level=9)


def _encode_layer(layer: StrictSharedLinear) -> tuple[bytes, str]:
    if layer.theta.dtype != torch.float32 or layer.bias.dtype != torch.float32:
        raise ValueError("当前导出协议只接受FP32，不隐式转换或量化")
    groups = layer.group_index.detach().cpu().numpy()
    edges = groups.size
    if groups.shape != (layer.in_features * layer.out_features,) or ((groups < -1) | (groups >= edges)).any():
        raise ValueError("group_index不满足展平固定slot协议")
    representatives = np.unique(groups[groups >= 0])
    weights = layer.theta.detach().cpu().numpy()[representatives].astype("<f4", copy=False)
    bias = layer.bias.detach().cpu().numpy().astype("<f4", copy=False)
    identity = np.arange(edges, dtype=np.int64)
    dtype = _uint_dtype(edges)
    candidates: list[tuple[bytes, str]] = []
    if np.array_equal(groups, identity):
        return _npz_blob("identity", weights, bias), "identity"
    changed = np.flatnonzero(groups != identity)
    candidates.append((_npz_blob("exceptions", weights, bias,
                                positions=changed.astype(dtype), codes=(groups[changed] + 1).astype(dtype)), "exceptions"))
    active_positions = np.flatnonzero(groups >= 0)
    candidates.append((_npz_blob("sparse", weights, bias,
                                positions=active_positions.astype(dtype), codes=(groups[active_positions] + 1).astype(dtype)), "sparse"))
    codes = np.zeros(edges, dtype=np.int64)
    codes[active_positions] = np.searchsorted(representatives, groups[active_positions]) + 1
    bits = max(1, int(representatives.size).bit_length())
    candidates.append((_npz_blob("packed", weights, bias,
                                representatives=representatives.astype(dtype), bits=np.array(bits, dtype=np.uint8),
                                packed=_pack(codes, bits)), "packed"))
    if np.all((groups == identity) | (groups == -1)):
        candidates.append((_npz_blob("masked_identity", weights, bias,
                                    packed=np.packbits(groups >= 0, bitorder="little")), "masked_identity"))
    return min(candidates, key=lambda candidate: len(candidate[0]))


def encode_model(model: StructuredMLP) -> bytes:
    """有效结构及活跃FP32参数无损保存，所有元数据计入真实文件长度。"""
    if not isinstance(model, StructuredMLP):
        raise TypeError("当前协议只导出StructuredMLP，CNN是单独参照")
    encoded = [_encode_layer(layer) for layer in model.shared_layers()]
    header = json.dumps({"version": 1, "widths": model.widths, "training": model.training,
                         "layer_sizes": [len(blob) for blob, _ in encoded]},
                        sort_keys=True, separators=(",", ":")).encode("utf-8")
    return MAGIC + struct.pack("<I", len(header)) + header + b"".join(blob for blob, _ in encoded)


def _decode_layer(blob: bytes, layer: StrictSharedLinear) -> None:
    edges = layer.theta.numel()
    with np.load(io.BytesIO(zlib.decompress(blob)), allow_pickle=False) as arrays:
        layout = str(arrays["layout"].item())
        if layout == "identity":
            groups = np.arange(edges, dtype=np.int64)
        elif layout in ("exceptions", "sparse"):
            groups = np.arange(edges, dtype=np.int64) if layout == "exceptions" else np.full(edges, -1, dtype=np.int64)
            positions = arrays["positions"].astype(np.int64)
            codes = arrays["codes"].astype(np.int64)
            if positions.ndim != 1 or codes.shape != positions.shape or ((positions < 0) | (positions >= edges)).any():
                raise ValueError("连接索引不合法")
            if np.unique(positions).size != positions.size:
                raise ValueError("连接索引重复")
            groups[positions] = codes - 1
        elif layout == "masked_identity":
            packed = arrays["packed"]
            if packed.size != (edges + 7) // 8:
                raise ValueError("掩码长度不合法")
            alive = np.unpackbits(packed, bitorder="little", count=edges).astype(bool)
            groups = np.where(alive, np.arange(edges, dtype=np.int64), -1)
        elif layout == "packed":
            representatives = arrays["representatives"].astype(np.int64)
            codes = _unpack(arrays["packed"], edges, int(arrays["bits"].item()))
            if (codes > representatives.size).any():
                raise ValueError("组编号不合法")
            groups = np.full(edges, -1, dtype=np.int64)
            active = codes > 0
            groups[active] = representatives[codes[active] - 1]
        else:
            raise ValueError(f"未知布局: {layout}")
        if ((groups < -1) | (groups >= edges)).any():
            raise ValueError("组slot不合法")
        representatives = np.unique(groups[groups >= 0])
        weights, bias = arrays["weights"], arrays["bias"]
        if weights.dtype != np.dtype("<f4") or bias.dtype != np.dtype("<f4"):
            raise ValueError("参数必须为FP32")
        if weights.shape != representatives.shape or bias.shape != (layer.out_features,):
            raise ValueError("参数维度不匹配")
        with torch.no_grad():
            layer.group_index.copy_(torch.from_numpy(groups))
            layer.theta.zero_()
            layer.theta[torch.from_numpy(representatives)] = torch.from_numpy(weights.copy())
            layer.bias.copy_(torch.from_numpy(bias.copy()))


def decode_model(payload: bytes, device="cpu") -> StructuredMLP:
    """恢复严格共享推断图；未引用slot置零，不用于恢复训练事务。"""
    if len(payload) < 8 or payload[:4] != MAGIC:
        raise ValueError("不是有效的BGS1模型文件")
    header_size = struct.unpack("<I", payload[4:8])[0]
    if header_size > len(payload) - 8:
        raise ValueError("模型头被截断")
    header = json.loads(payload[8:8 + header_size].decode("utf-8"))
    if header.get("version") != 1:
        raise ValueError("不支持此导出版本")
    widths, sizes = header["widths"], header["layer_sizes"]
    if len(sizes) != len(widths) - 1 or any(not isinstance(size, int) or size <= 0 for size in sizes):
        raise ValueError("层字节长度不合法")
    if sum(sizes) != len(payload) - 8 - header_size:
        raise ValueError("层字节长度与实际文件不符")
    # 解码不改变调用方随机事件；新模型仅在CPU初始化，然后移到指定设备。
    with torch.random.fork_rng(devices=[]):
        model = StructuredMLP(widths)
    offset = 8 + header_size
    for layer, size in zip(model.shared_layers(), sizes):
        _decode_layer(payload[offset:offset + size], layer)
        offset += size
    model.train(bool(header.get("training", False)))
    return model.to(device)


def export_stats(model: StructuredMLP, encoded: bytes | None = None) -> dict:
    """自由参数计数、零组排除统计与包含所有元数据的实际导出字节数。"""
    if encoded is None:
        encoded = encode_model(model)
    layers = [layer.stats() for layer in model.shared_layers()]
    summed = ("total_edges", "active_edges", "nonzero_edges", "independent_weight_parameters",
              "bias_parameters", "independent_parameters", "allocated_parameter_slots", "shared_groups",
              "nonzero_shared_groups", "zero_groups", "shared_edges", "nonzero_shared_edges")
    total = {key: sum(layer[key] for layer in layers) for key in summed}
    total["max_group_size"] = max(layer["max_group_size"] for layer in layers)
    return {"format": FORMAT, "layers": layers, "model_total": total, "export_bytes": len(encoded)}
