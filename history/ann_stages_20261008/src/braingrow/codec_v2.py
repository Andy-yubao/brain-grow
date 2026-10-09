"""BGS2无损FP32推断编码；仅修改表示，不引入新结构学习机制。

文件为 BGS2 + zlib(二进制负载)。负载含版本、训练标志、宽度清单，
各层依次保存掩码标志、alias数、可选压位活跃掩码、delta-varint
alias位置、按slot位宽压位的alias目标、升序活跃代表FP32、独立偏置。
删除和共享分开：删除边只在mask出现，不重复进入alias索引表。
代表slot可以对应已删除的边；group_index本身完全保留。
不保存未引用theta slot或优化器，因此仍不是训练事务检查点。
"""

import struct
import zlib

import numpy as np
import torch

from . import codec as v1
from .models import StrictSharedLinear, StructuredMLP

MAGIC = b"BGS2"
FORMAT = "braingrow-lossless-mask-alias-zlib-v2"


def _put_varint(value: int, stream: bytearray) -> None:
    if value < 0:
        raise ValueError("unsigned varint不能为负")
    while value >= 128:
        stream.append((value & 127) | 128)
        value >>= 7
    stream.append(value)


class _Reader:
    def __init__(self, payload: bytes):
        self.payload = payload
        self.offset = 0

    def take(self, count: int) -> bytes:
        if count < 0 or self.offset + count > len(self.payload):
            raise ValueError("BGS2负载被截断")
        result = self.payload[self.offset:self.offset + count]
        self.offset += count
        return result

    def varint(self) -> int:
        result = 0
        for shift in range(0, 70, 7):
            value = self.take(1)[0]
            result |= (value & 127) << shift
            if value < 128:
                return result
        raise ValueError("BGS2 varint过长")


def _layer_bytes(layer: StrictSharedLinear) -> tuple[bytes, dict]:
    if layer.theta.dtype != torch.float32 or layer.bias.dtype != torch.float32:
        raise ValueError("BGS2只接受FP32，不转换精度或量化")
    groups = layer.group_index.detach().cpu().numpy()
    edges = layer.theta.numel()
    if groups.shape != (edges,) or ((groups < -1) | (groups >= edges)).any():
        raise ValueError("group_index不满足固定slot协议")
    identity = np.arange(edges, dtype=np.int64)
    alive = groups >= 0
    has_mask = not bool(alive.all())
    aliases = np.flatnonzero(alive & (groups != identity))
    representatives = np.unique(groups[alive])
    stream = bytearray([int(has_mask)])
    _put_varint(int(aliases.size), stream)
    metadata_bytes = len(stream)
    mask = np.packbits(alive, bitorder="little").tobytes() if has_mask else b""
    stream.extend(mask)
    previous = -1
    position_bytes = bytearray()
    for position in aliases:
        _put_varint(int(position) - previous, position_bytes)
        previous = int(position)
    stream.extend(position_bytes)
    target_bits = max(1, (edges - 1).bit_length())
    targets = v1._pack(groups[aliases], target_bits).tobytes()
    stream.extend(targets)
    weights = layer.theta.detach().cpu().numpy()[representatives].astype("<f4", copy=False).tobytes()
    bias = layer.bias.detach().cpu().numpy().astype("<f4", copy=False).tobytes()
    stream.extend(weights)
    stream.extend(bias)
    details = {"active_edges": int(alive.sum()), "alias_edges": int(aliases.size),
               "independent_weight_parameters": int(representatives.size),
               "metadata_bytes_uncompressed": metadata_bytes,
               "mask_bytes_uncompressed": len(mask),
               "alias_position_bytes_uncompressed": len(position_bytes),
               "alias_target_bytes_uncompressed": len(targets),
               "weight_bytes_uncompressed": len(weights), "bias_bytes_uncompressed": len(bias),
               "layer_bytes_uncompressed": len(stream), "target_bits": target_bits}
    return bytes(stream), details


def _payload(model: StructuredMLP) -> tuple[bytes, dict]:
    if not isinstance(model, StructuredMLP):
        raise TypeError("BGS2只支持StructuredMLP")
    stream = bytearray([1, int(model.training)])
    _put_varint(len(model.widths), stream)
    for width in model.widths:
        _put_varint(width, stream)
    header_bytes = len(stream)
    layer_details = []
    for layer in model.shared_layers():
        blob, details = _layer_bytes(layer)
        stream.extend(blob)
        layer_details.append(details)
    return bytes(stream), {"header_bytes_uncompressed": header_bytes, "layers": layer_details,
                           "payload_bytes_uncompressed": len(stream), "magic_bytes": len(MAGIC)}


def encode_model(model: StructuredMLP) -> bytes:
    """包括全部元数据、掩码和alias成本的真实BGS2文件字节。"""
    raw, _ = _payload(model)
    return MAGIC + zlib.compress(raw, level=9)


def encoding_details(model: StructuredMLP) -> dict:
    """未压缩组成仅用于定位；接受门槛仍须使用完整encode_model长度。"""
    _, details = _payload(model)
    return details


def decode_model(payload: bytes, device="cpu") -> StructuredMLP:
    if payload[:4] != MAGIC:
        raise ValueError("不是BGS2模型文件")
    decompressor = zlib.decompressobj()
    raw = decompressor.decompress(payload[4:]) + decompressor.flush()
    if not decompressor.eof or decompressor.unused_data:
        raise ValueError("BGS2压缩负载被截断或有额外尾部")
    reader = _Reader(raw)
    version, training = reader.take(2)
    if version != 1 or training not in (0, 1):
        raise ValueError("不支持BGS2版本或训练标志")
    width_count = reader.varint()
    if width_count < 2 or width_count > len(raw):
        raise ValueError("宽度清单长度不合法")
    widths = [reader.varint() for _ in range(width_count)]
    if any(width <= 0 for width in widths):
        raise ValueError("层宽度必须为正")
    # 构造CPU模型不消费调用方RNG；目标设备只在恢复完成后转换。
    with torch.random.fork_rng(devices=[]):
        model = StructuredMLP(widths)
    for layer in model.shared_layers():
        edges = layer.theta.numel()
        has_mask = reader.take(1)[0]
        if has_mask not in (0, 1):
            raise ValueError("活跃掩码标志不合法")
        aliases = reader.varint()
        if aliases > edges:
            raise ValueError("alias数超过连接数")
        if has_mask:
            packed = np.frombuffer(reader.take((edges + 7) // 8), dtype=np.uint8)
            alive = np.unpackbits(packed, count=edges, bitorder="little").astype(bool)
        else:
            alive = np.ones(edges, dtype=bool)
        positions = np.empty(aliases, dtype=np.int64)
        previous = -1
        for index in range(aliases):
            delta = reader.varint()
            position = previous + delta
            if delta == 0 or position >= edges or not alive[position]:
                raise ValueError("alias位置不合法、重复或已删除")
            positions[index] = position
            previous = position
        target_bits = max(1, (edges - 1).bit_length())
        packed = np.frombuffer(reader.take((aliases * target_bits + 7) // 8), dtype=np.uint8)
        targets = v1._unpack(packed, aliases, target_bits)
        if (targets >= edges).any() or np.any(targets == positions):
            raise ValueError("alias目标不合法或冗余")
        groups = np.where(alive, np.arange(edges, dtype=np.int64), -1)
        groups[positions] = targets
        representatives = np.unique(groups[alive])
        weights = np.frombuffer(reader.take(4 * representatives.size), dtype="<f4").copy()
        bias = np.frombuffer(reader.take(4 * layer.out_features), dtype="<f4").copy()
        with torch.no_grad():
            layer.group_index.copy_(torch.from_numpy(groups))
            layer.theta.zero_()
            layer.theta[torch.from_numpy(representatives)] = torch.from_numpy(weights)
            layer.bias.copy_(torch.from_numpy(bias))
    if reader.offset != len(raw):
        raise ValueError("BGS2存在额外未解释字节")
    model.train(bool(training))
    return model.to(device)


def export_stats(model: StructuredMLP, encoded: bytes | None = None) -> dict:
    encoded = encode_model(model) if encoded is None else encoded
    result = v1.export_stats(model, encoded)
    result["format"] = FORMAT
    return result


@torch.no_grad()
def _diagnose_saved_models(source, destination) -> dict:
    """一次CPU编码定位；无forward、训练、数据加载或确认集读取。"""
    import hashlib
    import io
    import json
    import platform
    import time
    from pathlib import Path

    started = time.perf_counter()
    source = Path(source)
    torch.set_num_threads(1)
    report = {"version": "structural_codec_diagnostic_20261007", "device": "cpu",
              "python": platform.python_version(), "torch": torch.__version__, "precision": "float32",
              "threads": 1, "forward_calls": 0, "training_updates": 0, "confirm_reads": 0,
              "candidate_rule": "每层活跃组按标量值稳定排序，相邻disjoint对按绝对差取前floor(5%×组数)，至少1对",
              "scope": "仅改无损编码，未引入新学习机制；一次候选仅定位真实字节增量，不构成接受或效果证据",
              "models": {}}

    def verify(before, after):
        return {"group_index_exact": all(torch.equal(a.group_index, b.group_index)
                                         for a, b in zip(before.shared_layers(), after.shared_layers())),
                "effective_weight_bits_exact": all(torch.equal(a.expanded_weight().view(torch.int32),
                                                               b.expanded_weight().view(torch.int32))
                                                   for a, b in zip(before.shared_layers(), after.shared_layers())),
                "bias_bits_exact": all(torch.equal(a.bias.view(torch.int32), b.bias.view(torch.int32))
                                       for a, b in zip(before.shared_layers(), after.shared_layers()))}

    def v1_layouts(payload):
        size = struct.unpack("<I", payload[4:8])[0]
        header = json.loads(payload[8:8 + size])
        offset = 8 + size
        layers = []
        for byte_count in header["layer_sizes"]:
            with np.load(io.BytesIO(zlib.decompress(payload[offset:offset + byte_count])), allow_pickle=False) as arrays:
                layers.append({"layout": str(arrays["layout"].item()), "compressed_npz_bytes": byte_count,
                               "array_bytes": {key: int(arrays[key].nbytes) for key in arrays.files}})
            offset += byte_count
        return {"container_header_bytes": 8 + size, "layers": layers}

    for arm in ("fc", "joint"):
        path = source / arm / "final_model.bgs"
        saved = path.read_bytes()
        model = v1.decode_model(saved, device="cpu")
        original_v1 = v1.encode_model(model)
        original_v2 = encode_model(model)
        restored = decode_model(original_v2)
        before = {"saved_path": path.as_posix(), "saved_sha256": hashlib.sha256(saved).hexdigest(),
                  "saved_bytes": len(saved), "v1_bytes": len(original_v1), "v2_bytes": len(original_v2),
                  "v2_minus_v1_bytes": len(original_v2) - len(original_v1),
                  "v1_selected_layouts": v1_layouts(original_v1), "v2_encoding": encoding_details(model),
                  "roundtrip": verify(model, restored), "structure": v1.export_stats(model, original_v1)}
        pair_details = []
        for layer in model.shared_layers():
            groups = np.unique(layer.group_index.numpy()[layer.group_index.numpy() >= 0])
            values = layer.theta.detach().numpy()[groups]
            order = groups[np.argsort(values, kind="stable")]
            pairs = order[:len(order) // 2 * 2].reshape(-1, 2)
            differences = np.abs(layer.theta.detach().numpy()[pairs[:, 0]] - layer.theta.detach().numpy()[pairs[:, 1]])
            count = min(len(pairs), max(1, int(len(groups) * 0.05)))
            chosen = pairs[np.argsort(differences, kind="stable")[:count]]
            pair_details.append({"active_groups_before": len(groups), "merged_pairs": len(chosen),
                                 "value_difference_max": float(differences[np.argsort(differences, kind="stable")[:count]].max())
                                 if count else None})
            layer.tie_pairs(torch.from_numpy(chosen.copy()))
        candidate_v1, candidate_v2 = v1.encode_model(model), encode_model(model)
        delta_v1 = len(candidate_v1) - len(original_v1)
        delta_v2 = len(candidate_v2) - len(original_v2)
        after = {"v1_bytes": len(candidate_v1), "v2_bytes": len(candidate_v2),
                 "v1_edit_delta_bytes": delta_v1, "v2_edit_delta_bytes": delta_v2,
                 "v2_storage_improves_for_this_candidate": delta_v2 < 0,
                 "roundtrip_v1": verify(model, v1.decode_model(candidate_v1)),
                 "roundtrip_v2": verify(model, decode_model(candidate_v2)),
                 "v1_selected_layouts": v1_layouts(candidate_v1), "v2_encoding": encoding_details(model),
                 "pair_details": pair_details, "structure": export_stats(model, candidate_v2)}
        report["models"][arm] = {"unmodified": before, "one_untrained_candidate": after}
    report["wall_seconds"] = time.perf_counter() - started
    report["conclusion"] = "v2只拆分删除与alias索引并降低容器开销；是否改善编辑真实字节以各臂v2_edit_delta_bytes为准，未检验任务性能"
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="一次CPU无损编码定位，不执行模型forward或训练")
    parser.add_argument("--diagnose", action="store_true", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    diagnostic = _diagnose_saved_models(args.source, args.report)
    print(json.dumps({arm: {"v1_before": item["unmodified"]["v1_bytes"],
                           "v2_before": item["unmodified"]["v2_bytes"],
                           "v1_edit_delta": item["one_untrained_candidate"]["v1_edit_delta_bytes"],
                           "v2_edit_delta": item["one_untrained_candidate"]["v2_edit_delta_bytes"]}
                      for arm, item in diagnostic["models"].items()}, ensure_ascii=False))
