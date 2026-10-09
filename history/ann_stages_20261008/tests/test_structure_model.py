"""三个小CPU行为检查：梯度、成员加权编辑、严格共享导出恢复。"""

import io
import json
import struct
import zlib

import numpy as np
import torch

from braingrow.codec import decode_model, encode_model, export_stats
from braingrow.models import StrictSharedLinear, StructuredMLP


def test_shared_gradient_is_member_sum():
    torch.manual_seed(11)
    layer = StrictSharedLinear(3, 1)
    layer.tie_groups([0, 1])
    layer.prune_edges([2])
    layer(torch.tensor([[2.0, 3.0, 7.0]])).sum().backward()
    torch.testing.assert_close(layer.theta.grad, torch.tensor([5.0, 0.0, 0.0]))
    torch.testing.assert_close(layer.bias.grad, torch.ones(1))
    torch.testing.assert_close(layer.last_edge_grad[:2], torch.tensor([2.0, 3.0]))
    remembered = layer.last_edge_grad.clone()
    with torch.no_grad():
        layer(torch.ones(1, 3))
    torch.testing.assert_close(layer.last_edge_grad, remembered)


def test_batched_merges_weight_actual_members_and_exclude_zero_sharing():
    torch.manual_seed(11)
    layer = StrictSharedLinear(6, 1)
    with torch.no_grad():
        layer.theta.copy_(torch.tensor([1.0, 3.0, 9.0, 0.0, 0.0, 5.0]))
    layer.tie_groups([0, 1])  # 2成员，均值2。
    affected = layer.tie_pairs(torch.tensor([[2, 0], [4, 3]]))
    torch.testing.assert_close(affected, torch.tensor([0, 2, 3, 4]))
    torch.testing.assert_close(layer.theta[0], torch.tensor(13.0 / 3.0))
    torch.testing.assert_close(layer.group_index, torch.tensor([0, 0, 0, 3, 3, 5]))
    layer.prune_edges([0])  # 代表所在边消失，slot仍被其他成员引用。
    stats = layer.stats()
    assert stats["active_edges"] == 5
    assert stats["shared_groups"] == 2
    assert stats["nonzero_shared_groups"] == 1
    assert stats["nonzero_shared_edges"] == 2


def test_lossless_export_restores_identity_sparse_and_strict_shared_outputs():
    torch.manual_seed(11)
    model = StructuredMLP([8, 4, 2])
    x = torch.randn(3, 8)
    initial = encode_model(model)
    header_len = struct.unpack("<I", initial[4:8])[0]
    header = json.loads(initial[8:8 + header_len])
    blob = initial[8 + header_len:8 + header_len + header["layer_sizes"][0]]
    with np.load(io.BytesIO(zlib.decompress(blob)), allow_pickle=False) as arrays:
        assert str(arrays["layout"].item()) == "identity"
        assert "positions" not in arrays and "representatives" not in arrays
    for variant in ("initial", "sparse", "shared"):
        if variant == "sparse":
            model.layers[0].prune_edges([0, 7, 14, 31])
        elif variant == "shared":
            model.layers[0].tie_pairs(torch.tensor([[1, 8], [2, 9], [3, 10]]))
            model.layers[1].tie_groups([0, 2, 4])
            with torch.no_grad():
                model.layers[1].theta[0] = 0.0
        encoded = encode_model(model)
        rng = torch.get_rng_state().clone()
        restored = decode_model(encoded)
        assert torch.equal(torch.get_rng_state(), rng)
        torch.testing.assert_close(restored(x), model(x), rtol=0, atol=0)
        for before, after in zip(model.shared_layers(), restored.shared_layers()):
            assert torch.equal(before.group_index, after.group_index)
            torch.testing.assert_close(before.expanded_weight(), after.expanded_weight(), rtol=0, atol=0)
            assert before.stats() == after.stats()
        assert export_stats(restored, encoded)["export_bytes"] == len(encoded)
