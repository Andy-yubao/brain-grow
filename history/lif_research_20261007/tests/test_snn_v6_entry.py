"""v6入口预算与真实LIF优化器调度状态；不运行性能实验。"""
import copy

import pytest
import torch

from braingrow.network.fixed_layered import FixedLayeredSNN
from tools import train_snn_baseline_v6 as entry


def test_entry_budget_enforces_cap_and_final_epoch():
    budget = entry.budget_for(25)
    assert budget.evaluation_epochs("control")[-1] == 25
    assert budget.remaining_epochs(24, 25) == 1
    assert budget.remaining_epochs(25) == 0
    with pytest.raises(ValueError):
        entry.budget_for(26)


def test_feedback_restore_does_not_repeat_control_or_reset_learning_rate():
    net = FixedLayeredSNN((12, 9, 6), device="cpu")
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.001, foreach=False)
    scheduler = entry.adaptive.make_scheduler(optimizer)
    state = dict(tracker=dict(control_stale=0, fit_stale=0),
        lr_policy=dict(scheduler=scheduler.state_dict(), last_control_epoch=0,
            reductions=0, last_reduction_epoch=None))
    metrics = dict(loss=2., digit_accuracy=.2, motion_accuracy=.3, joint_accuracy=.1)
    for epoch in (2, 4, 6):
        entry.adaptive.control_feedback(scheduler, state, metrics, epoch)
    assert optimizer.param_groups[0]["lr"] == .0005
    saved = copy.deepcopy(state)
    restored_optimizer = torch.optim.AdamW(net.parameters(), lr=.001, foreach=False)
    restored_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    restored_scheduler = entry.adaptive.make_scheduler(restored_optimizer)
    restored_scheduler.load_state_dict(saved["lr_policy"]["scheduler"])
    assert entry.adaptive.control_feedback(restored_scheduler, saved, metrics, 6) is None
    assert restored_optimizer.param_groups[0]["lr"] == .0005
    for epoch in (8, 10, 12):
        entry.adaptive.control_feedback(scheduler, state, metrics, epoch)
        entry.adaptive.control_feedback(restored_scheduler, saved, metrics, epoch)
    assert saved == state
    assert restored_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"] == .00025
