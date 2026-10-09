"""一个小CPU检查：分离掩码/alias后保留真实group状态与有效权重。"""

import torch

from braingrow.codec_v2 import decode_model, encode_model, export_stats
from braingrow.models import StructuredMLP


def test_v2_keeps_mask_alias_zero_groups_and_fp32_exact():
    torch.manual_seed(11)
    model = StructuredMLP([7, 4, 2])
    x = torch.randn(3, 7)
    for state in ("identity", "mask_alias", "fully_deleted"):
        if state == "mask_alias":
            first, second = model.shared_layers()
            first.tie_pairs(torch.tensor([[0, 7], [2, 9]]))
            first.prune_edges([0, 3, 12])  # slot0的自身边删掉，其他成员仍引用。
            second.tie_groups([0, 2, 4])
            with torch.no_grad():
                second.theta[0] = -0.0  # 零共享组仍保持，但不计有意义共享。
        elif state == "fully_deleted":
            model.layers[0].prune_edges(torch.arange(28))
        payload = encode_model(model)
        rng = torch.get_rng_state().clone()
        restored = decode_model(payload)
        assert torch.equal(rng, torch.get_rng_state())
        for a, b in zip(model.shared_layers(), restored.shared_layers()):
            assert torch.equal(a.group_index, b.group_index)
            assert torch.equal(a.expanded_weight().view(torch.int32), b.expanded_weight().view(torch.int32))
            assert torch.equal(a.bias.view(torch.int32), b.bias.view(torch.int32))
        torch.testing.assert_close(model(x), restored(x), rtol=0, atol=0)
        stats = export_stats(restored, payload)
        assert stats["export_bytes"] == len(payload)
        assert stats["format"] == "braingrow-lossless-mask-alias-zlib-v2"
