"""新增恢复机制一次真实 CPU 行为检查，不访问研究数据或 GPU。"""

import copy

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from braingrow.edge_recovery import EdgeRecoveryController
from braingrow.models import StructuredMLP
from braingrow.structural import CandidateTransaction


def _assert_same(actual, expected):
    if isinstance(actual, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_same(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            _assert_same(a, b)
    else:
        assert actual == expected


def test_inactive_gradient_zero_projection_adaptation_and_transaction_rollback():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        model = StructuredMLP([200, 2])
        layer = model.shared_layers()[0]
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=0, amsgrad=True)
        x = torch.arange(1, 201, dtype=torch.float32).reshape(1, -1) / 200
        y = torch.tensor([1])
        # 一次真实更新先建立非零 Adam 矩，避免只检查未初始化优化器。
        F.cross_entropy(model(x), y).backward()
        optimizer.step()
        inactive = np.array([0, 1, 2, 200, 201, 202], dtype=np.int64)
        layer.prune_edges(inactive)
        controller = EdgeRecoveryController(model)
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(model(x), y).backward()
        assert torch.all(layer.last_edge_grad[inactive] != 0)
        assert torch.all(layer.theta.grad[inactive] == 0)
        controller.observe_gradients(model)
        proposal = controller.propose(model)  # 原始400边的0.5%=2边。
        assert proposal.kind == "recover" and proposal.count == 2
        restored = torch.as_tensor(proposal.edits[0][1])
        with torch.no_grad():
            original_output = model(x).clone()
        original_model = copy.deepcopy(model.state_dict())
        original_optimizer = copy.deepcopy(optimizer.state_dict())
        original_controller = copy.deepcopy(controller.state_dict())
        original_edge_grad = layer.last_edge_grad.clone()
        original_parameter_grad = [p.grad.clone() for p in model.parameters()]
        with CandidateTransaction(model, optimizer, controller):
            controller.apply(model, optimizer, proposal)
            with torch.no_grad():
                assert torch.equal(model(x), original_output)
            for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                assert torch.all(optimizer.state[layer.theta][name][restored] == 0)
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(model(x), y).backward()
            optimizer.step()
            assert torch.all(layer.theta[restored] != 0)
            # 模拟runner在提交前更新接受游标；无commit应完整恢复。
            controller.accepted += 1
        _assert_same(model.state_dict(), original_model)
        _assert_same(optimizer.state_dict(), original_optimizer)
        _assert_same(controller.state_dict(), original_controller)
        assert torch.equal(layer.last_edge_grad, original_edge_grad)
        for parameter, grad in zip(model.parameters(), original_parameter_grad):
            assert torch.equal(parameter.grad, grad)
        random_controller = EdgeRecoveryController(model)
        first = random_controller.propose(model, random_match=proposal, seed_parts=[11, 1, 1])
        second = random_controller.propose(model, random_match=proposal, seed_parts=[11, 1, 1])
        assert first.count == proposal.count
        np.testing.assert_array_equal(first.edits[0][1], second.edits[0][1])
        with torch.no_grad():
            layer.group_index[4] = 3
        with pytest.raises(ValueError, match="identity-slot"):
            controller.propose(model)
