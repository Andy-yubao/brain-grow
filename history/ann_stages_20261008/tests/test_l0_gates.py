"""仅一项小CPU真实检查；不重复旧编码/事务/GPU诊断。"""
import copy

import torch

from braingrow.l0_gates import L0GatedMLP, freeze_gated
from braingrow.models import StructuredMLP


def test_gate_gradient_fold_and_full_permutation():
    torch.manual_seed(11)
    base = StructuredMLP([3, 4, 2])
    model = L0GatedMLP(base)
    model.gate_generator = torch.Generator().manual_seed(11)
    x = torch.tensor([[.2, .6, .4], [.9, .1, .3]])
    loss = torch.nn.functional.cross_entropy(model(x), torch.tensor([0, 1]))
    loss = loss + .01 * model.expected_active_edges()
    loss.backward()
    assert all(a.grad is not None and torch.isfinite(a.grad).all() and a.grad.abs().sum() > 0 for a in model.log_alpha)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.base.parameters())
    with torch.no_grad():
        for a in model.log_alpha:
            a.copy_(torch.linspace(-4., 4., a.numel()))
    model.eval()
    source = copy.deepcopy(model.state_dict())
    before_rng = torch.get_rng_state().clone()
    expected = model(x)
    folded, evidence, values = freeze_gated(model)
    random, random_evidence, random_values = freeze_gated(model, permutation_seed=11)
    torch.testing.assert_close(folded(x), expected, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), before_rng)
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in source.items())
    assert evidence["additional_model_forward_samples"] == 0
    assert all(layer["same_gate_multiset"] for layer in random_evidence["layers"])
    for index, (left, right) in enumerate(zip(folded.layers, random.layers)):
        assert 0 < int((left.group_index >= 0).sum()) < left.theta.numel()
        assert int((left.group_index >= 0).sum()) == int((right.group_index >= 0).sum())
        permutation = torch.from_numpy(random_values[f"permutation_{index}"].astype("int64"))
        z = torch.from_numpy(values[f"gate_{index}"])[permutation]
        torch.testing.assert_close(right.theta, model.base.layers[index].theta * z, rtol=0, atol=0)
    assert not any("log_alpha" in key for key in folded.state_dict())
