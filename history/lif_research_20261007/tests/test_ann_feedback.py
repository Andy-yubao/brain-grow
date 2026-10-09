"""检验真实PyTorch优化器的降率、早停及序列化恢复，不运行模型训练。"""
import copy
import io

import pytest
import torch

from braingrow.ann_feedback import initial_feedback, make_scheduler, step_feedback


def setup_policy():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.001, foreach=False)
    scheduler = make_scheduler(optimizer)
    return optimizer, scheduler, initial_feedback(scheduler)


def test_plateau_reduces_before_early_stop_and_improvement_resets():
    optimizer, scheduler, state = setup_policy()
    step_feedback(scheduler, state, 2.0, 2)
    assert not step_feedback(scheduler, state, 2.0, 4)["reduced"]
    event = step_feedback(scheduler, state, 2.0, 6)
    assert event["reduced"] and optimizer.param_groups[0]["lr"] == 0.0005
    assert not event["early_stop"]
    step_feedback(scheduler, state, 1.5, 8)
    assert state["stale_checks"] == 0
    for epoch in (10, 12, 14, 16):
        assert not step_feedback(scheduler, state, 1.5, epoch)["early_stop"]
    assert step_feedback(scheduler, state, 1.5, 18)["early_stop"]
    assert state["stale_checks"] == 5


def test_feedback_and_optimizer_resume_exactly():
    optimizer, scheduler, state = setup_policy()
    for epoch in (2, 4, 6):
        step_feedback(scheduler, state, 2.0, epoch)
    buffer = io.BytesIO()
    torch.save(dict(optimizer=optimizer.state_dict(), feedback=copy.deepcopy(state)), buffer)
    buffer.seek(0)
    saved = torch.load(buffer, weights_only=False)
    restored_optimizer, restored_scheduler, _ = setup_policy()
    restored_optimizer.load_state_dict(saved["optimizer"])
    restored_scheduler.load_state_dict(saved["feedback"]["scheduler"])
    restored_state = saved["feedback"]
    for epoch, loss in ((8, 2.0), (10, 2.0), (12, 2.0)):
        assert step_feedback(scheduler, state, loss, epoch) == step_feedback(
            restored_scheduler, restored_state, loss, epoch)
    assert state == restored_state
    assert state["stale_checks"] == 5
    assert optimizer.param_groups[0]["lr"] == restored_optimizer.param_groups[0]["lr"]


def test_invalid_feedback_does_not_advance_policy():
    _, scheduler, state = setup_policy()
    step_feedback(scheduler, state, 1.0, 2)
    previous = copy.deepcopy(state)
    for loss, epoch in ((float("nan"), 4), (float("inf"), 4), (-1, 4), (1, 2), (1, 1)):
        with pytest.raises(ValueError):
            step_feedback(scheduler, state, loss, epoch)
        assert state == previous
        assert scheduler.state_dict() == previous["scheduler"]


def test_small_fluctuations_do_not_reset_patience_and_lr_has_floor():
    optimizer, scheduler, state = setup_policy()
    step_feedback(scheduler, state, 1.0, 2)
    for epoch in range(4, 54, 2):
        event = step_feedback(scheduler, state, 0.995, epoch)
    assert state["best_loss"] == 1.0
    assert event["early_stop"]
    assert optimizer.param_groups[0]["lr"] == 1e-5
