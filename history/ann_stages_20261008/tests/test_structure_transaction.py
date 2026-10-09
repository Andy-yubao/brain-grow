"""只验证新增拒绝事务的真实状态，不重跑旧恢复实验。"""
import copy
import random

import numpy as np
import torch

from braingrow.models import StructuredMLP
from braingrow.structural import CandidateTransaction, StructureController


def test_rejected_edit_restores_model_optimizer_and_rng():
    torch.manual_seed(11)
    model = StructuredMLP([4, 3, 2])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0)
    x = torch.tensor([[0.1, 0.2, 0.7, 0.4], [0.3, 0.0, 0.1, 0.2]])
    y = torch.tensor([0, 1])
    torch.nn.functional.cross_entropy(model(x), y).backward()
    optimizer.step()
    controller = StructureController(model, "joint")
    controller.observe_gradients(model)
    expected = copy.deepcopy(model.state_dict())
    opt_expected = copy.deepcopy(optimizer.state_dict())
    rng_before = torch.get_rng_state().clone()
    numpy_before = np.random.get_state()
    python_before = random.getstate()
    proposal = controller.propose(model)
    cursor = controller.proposals
    with CandidateTransaction(model, optimizer, controller):
        controller.apply(model, optimizer, proposal)
        optimizer.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(model(x), y).backward()
        optimizer.step()
        torch.rand(1)
        np.random.random()
        random.random()
    for name, tensor in expected.items():
        assert torch.equal(model.state_dict()[name], tensor), name
    for pi, state in opt_expected["state"].items():
        for name, value in state.items():
            assert torch.equal(optimizer.state_dict()["state"][pi][name], value)
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert np.array_equal(np.random.get_state()[1], numpy_before[1])
    assert random.getstate() == python_before
    assert controller.proposals == cursor
