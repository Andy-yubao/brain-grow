"""新功能代理/原单位C32补偿的一次小真实CPU风险检查，不GPU。"""
import copy

import numpy as np
import torch
from torch.nn import functional as F

from braingrow.models import StructuredMLP
from braingrow.node_reduction import apply_node_reduction
from braingrow.task_basis import (build_library, extract_snapshot, function_scores,
                                  group_delta, reduction_proposal, select_group)


def test_shared_snapshot_gradient_scaling_c32_and_zero_rms():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        model = StructuredMLP([3, 6, 4, 2])
        with torch.no_grad():
            for layer in model.layers[:2]:
                layer.theta.abs_().add_(.1)
                layer.bias.abs_().add_(.2)
        x = torch.rand(8, 3)
        y = torch.tensor([0, 1]*4)
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        for layer in model.layers:
            layer.last_edge_grad = torch.ones_like(layer.theta)
        before_rng = torch.get_rng_state().clone()
        original = extract_snapshot(model, x, y, batch_size=4)
        assert torch.equal(before_rng, torch.get_rng_state())
        assert all(torch.all(parameter.grad == 1) for parameter in model.parameters())
        assert all(torch.all(layer.last_edge_grad == 1) for layer in model.layers)
        assert all(torch.isfinite(record["gradient"]).all() for record in original)
        scaled = copy.deepcopy(model)
        with torch.no_grad():
            for index, factors in enumerate((torch.tensor([.5, 2., 4., .25, 8., 1.]), torch.tensor([2., .5, 4., .25]))):
                layer, following = scaled.layers[index:index+2]
                layer.theta.view(layer.out_features, layer.in_features).mul_(factors[:, None])
                layer.bias.mul_(factors)
                following.theta.view(following.out_features, following.in_features).div_(factors[None, :])
        transformed = extract_snapshot(scaled, x, y, batch_size=4)
        torch.testing.assert_close(original[1]["z"], transformed[1]["z"], atol=2e-6, rtol=2e-6)
        libraries = []
        for layer in (0, 1):
            first = build_library(original[layer], layer, model.widths, range(6), range(6, 8), 2)
            second = build_library(transformed[layer], layer, model.widths, range(6), range(6, 8), 2)
            assert first.basis == second.basis and first.pool == second.pool and len(first.pool) >= 1
            dropped = (first.pool[0],)
            j, q = function_scores(original[layer], first, dropped, range(8), batch_size=4)
            j2, q2 = function_scores(transformed[layer], second, dropped, range(8), batch_size=4)
            np.testing.assert_allclose([j, q], [j2, q2], rtol=1e-5, atol=1e-7)
            assert select_group(original[layer], first, "random", count=len(first.pool)+1)[0] is None
            libraries.append(first)
        proposal = reduction_proposal(libraries[0], (libraries[0].pool[0],))
        compact = apply_node_reduction(model, proposal)
        actual_z = compact.layers[1](F.relu(compact.layers[0](x))).detach().double()
        expected_delta = group_delta(original[0], libraries[0], proposal.dropped_indices, torch.arange(8))
        torch.testing.assert_close(original[0]["z"]-actual_z, expected_delta, rtol=1e-4, atol=2e-6)
        zero_sigma = copy.deepcopy(libraries[0])
        zero_sigma.sigma.zero_()
        assert function_scores(original[0], zero_sigma, proposal.dropped_indices, range(8))[1] == float("inf")
        exact_zero = copy.deepcopy(original[0])
        exact_zero["errors"].zero_()
        assert function_scores(exact_zero, zero_sigma, proposal.dropped_indices, range(8)) == (0., 0.)
