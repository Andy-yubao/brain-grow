"""新分层图的前向边界、出生保持及权重学习原语。"""
import torch

from braingrow.network.layered_gradient import LayeredGradientSNN
from tools.run_layered_growth import optimizer_after_growth


def test_output_only_accumulates_actual_last_layer_fire():
    net = LayeredGradientSNN((3, 4), device="cpu")
    with torch.no_grad():
        net.weights[0].fill_(0.1)
        net.weights[1].fill_(1.0)
        net.weights[2].fill_(0.2)
    x = torch.ones((2, 8, 784))
    with torch.no_grad():
        potential, activity = net(x, trace=True)
        again = net(x)
    assert activity["last_layer_counts"].sum() > 0
    torch.testing.assert_close(potential, activity["last_layer_counts"] @ net.weights[2])
    torch.testing.assert_close(potential, again)  # 视频边界每次归零。


def test_new_node_connects_both_adjacent_layers_without_changing_old_output():
    x = torch.ones((2, 8, 784))
    for layer in (0, 1):
        net = LayeredGradientSNN((3, 4), device="cpu")
        with torch.no_grad():
            before = net(x)
        net.grow(layer, 2, proposal_id=layer + 1)
        with torch.no_grad():
            after = net(x)
        torch.testing.assert_close(before, after)
        assert net.resources()["effective_weight_slots"] == sum(w.numel() for w in net.weights)
        assert all(w.requires_grad for w in net.weights)


def test_growth_keeps_old_optimizer_moments_and_zeros_new_moments():
    net = LayeredGradientSNN((3, 4), device="cpu")
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.001)
    potential = net(torch.ones((2, 8, 784)))
    potential.sum().backward()
    optimizer.step()
    old_moment = optimizer.state[net.weights[1]]["exp_avg"].clone()
    pairs = net.grow(0, 2, proposal_id=1)
    expanded = optimizer_after_growth(optimizer, net, pairs)
    moment = expanded.state[net.weights[1]]["exp_avg"]
    torch.testing.assert_close(moment[:3], old_moment)
    assert torch.count_nonzero(moment[3:]) == 0


def test_active_split_preserves_function_and_starts_with_identical_fire():
    x = torch.ones((2, 8, 784))
    for layer in (0, 1):
        net = LayeredGradientSNN((3, 4), device="cpu")
        with torch.no_grad():
            net.weights[0].fill_(0.1)
            net.weights[1].fill_(1.0)
            net.weights[2].fill_(0.2)
            before = net(x)
        net.split_active(layer, torch.tensor([0, 1]), proposal_id=3 + layer)
        with torch.no_grad():
            after, activity = net(x, trace=True)
        torch.testing.assert_close(before, after, atol=1e-4, rtol=1e-5)
        counts = activity["per_unit_fire"][layer]
        assert counts[-2:].sum() > 0
        torch.testing.assert_close(counts[-2:], counts[:2])
