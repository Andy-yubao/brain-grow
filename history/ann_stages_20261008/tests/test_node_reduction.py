"""一个CPU行为检查：真实ReLU多节点关系、出边补偿和物理缩减。"""

import torch

from braingrow.codec import decode_model, encode_model
from braingrow.models import StructuredMLP
from braingrow.node_reduction import apply_node_reduction, propose_node_reduction


def test_multinode_relu_reduction_compensates_outputs_and_physically_shrinks():
    torch.manual_seed(11)
    model = StructuredMLP([2, 3, 2])
    with torch.no_grad():
        # 非负域上h3=h1+h2，须用两个节点重构第三个。
        model.layers[0].theta.copy_(torch.tensor([1., 0., 0., 1., 1., 1.]))
        model.layers[0].bias.zero_()
        model.layers[1].theta.copy_(torch.tensor([0., 2., 3., 4., 0., -1.]))
        model.layers[1].bias.copy_(torch.tensor([0.2, -0.3]))
        model.layers[0].prune_edges([1, 2])
        model.layers[1].prune_edges([0, 4])
    fit_x = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [2., 1.], [1., 3.]])
    proposal = propose_node_reduction(model, 0, fit_x, max_remove=1,
                                      max_relative_residual=1e-10, max_basis=2)
    assert proposal is not None
    assert proposal.dropped_indices == (2,)
    assert proposal.kept_indices == (0, 1)
    torch.testing.assert_close(proposal.coefficients, torch.ones(2, 1, dtype=torch.float64))
    source = {key: value.clone() for key, value in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    compact = apply_node_reduction(model, proposal)
    assert torch.equal(torch.get_rng_state(), rng)
    assert compact.widths == (2, 2, 2)
    assert compact.layers[0].theta.numel() == 4
    assert compact.layers[1].in_features == 2
    assert compact.node_reduction_metadata["compensation_new_nonzero_edges"] == 2
    for key, value in model.state_dict().items():
        assert torch.equal(value, source[key])
    for layer in compact.shared_layers():
        assert layer.stats()["shared_groups"] == 0
    # 测试域满足解析关系；普通候选的有限拟合残差不提供这类全域保证。
    check_x = torch.tensor([[0., 0.], [2., 7.], [8., 1.], [3., 3.]])
    torch.testing.assert_close(compact(check_x), model(check_x), rtol=0, atol=2e-6)
    restored = decode_model(encode_model(compact))
    assert restored.widths == compact.widths
    torch.testing.assert_close(restored(check_x), compact(check_x), rtol=0, atol=0)
