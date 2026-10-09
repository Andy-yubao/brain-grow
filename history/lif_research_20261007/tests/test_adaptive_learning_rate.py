"""实际优化器的平台调度与真实LIF检查点接续；不代表任务准确率验证。"""
import copy
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from braingrow.network.fixed_layered import FixedLayeredSNN
from tools import train_large_baseline as base
from tools.train_adaptive_baseline import (
    POLICY, FileStopRequested, RunLock, adopt_history, control_feedback,
    fork_paused_run, make_scheduler, process_alive,
)


@pytest.fixture(autouse=True)
def few_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def metrics(loss):
    return dict(loss=loss, digit_accuracy=0.3, motion_accuracy=0.7, joint_accuracy=0.2)


def training_state():
    return dict(completed_epochs=15, progress=None, maximum_epochs=100,
                micro_batch=2, costs=base.new_costs(), stop_reason="user_interrupt",
                tracker=dict(best_control=None, accepted_control=None, control_stale=2,
                             fit_stale=1, best_fit_loss=None, best_fit_score=None,
                             activity_fault_checks=0), zero_gradient_updates=0)


def network():
    model = FixedLayeredSNN((12, 9, 6), device="cpu")
    with torch.no_grad():
        model.weights[0].fill_(0.1)
        for weight in model.weights[1:-1]:
            weight.fill_(1.0)
    return model


def test_improvement_patience_cooldown_and_floor():
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=0.001)
    state = training_state()
    adopt_history(optimizer, state, [])
    scheduler = make_scheduler(optimizer)
    scheduler.load_state_dict(state["lr_policy"]["scheduler"])
    control_feedback(scheduler, state, metrics(3.0), 5)
    control_feedback(scheduler, state, metrics(2.9), 10)  # >1%改善重置平台。
    control_feedback(scheduler, state, metrics(2.88), 15)  # <1%改善不重置。
    assert optimizer.param_groups[0]["lr"] == 0.001
    event = control_feedback(scheduler, state, metrics(3.2), 20)
    assert event["reduced"] and event["effective_epoch"] == 21
    assert optimizer.param_groups[0]["lr"] == 0.0005
    control_feedback(scheduler, state, metrics(3.2), 25)  # 冷却检查。
    assert scheduler.num_bad_epochs == 0
    control_feedback(scheduler, state, metrics(3.2), 30)
    assert optimizer.param_groups[0]["lr"] == 0.0005
    control_feedback(scheduler, state, metrics(3.2), 35)
    assert optimizer.param_groups[0]["lr"] == 0.00025
    snapshot = copy.deepcopy(state["lr_policy"])
    assert control_feedback(scheduler, state, metrics(3.2), 35) is None
    assert state["lr_policy"] == snapshot
    for epoch in range(40, 205, 5):
        control_feedback(scheduler, state, metrics(3.2), epoch)
    assert optimizer.param_groups[0]["lr"] == POLICY["min_lr"]
    reductions = state["lr_policy"]["reductions"]
    control_feedback(scheduler, state, metrics(3.2), 205)
    assert state["lr_policy"]["reductions"] == reductions
    with pytest.raises(ValueError):
        control_feedback(scheduler, state, metrics(float("nan")), 210)


def test_history_causes_one_real_adoption_reduction_and_keeps_rng_and_adam_moments(tmp_path):
    model = network()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, foreach=False)
    x = torch.ones(4, 8, 784)
    y = torch.tensor([[i, i % 7] for i in range(4)])
    base.training_update(model, optimizer, x, y, torch.arange(4), 2,
                         base.Ledger(tmp_path, base.new_costs()), 1)
    adam_before, weights_before = copy.deepcopy(optimizer.state_dict()), copy.deepcopy(model.state_dict())
    random_before = torch.get_rng_state().clone()
    state = training_state()
    rows = [dict(epoch=epoch, **metrics(loss)) for epoch, loss in
            ((5, 3.25), (10, 3.95), (15, 4.96), (20, 4.9), (25, 5.0))]
    event = adopt_history(optimizer, state, rows)
    assert event["historical_bad_checks"] == 4
    assert optimizer.param_groups[0]["lr"] == 0.0005  # 只在现在降一次。
    torch.testing.assert_close(torch.get_rng_state(), random_before, rtol=0, atol=0)
    for name, weight in model.state_dict().items():
        torch.testing.assert_close(weight, weights_before[name], rtol=0, atol=0)
    for index, values in adam_before["state"].items():
        for key, value in values.items():
            torch.testing.assert_close(value, optimizer.state_dict()["state"][index][key], rtol=0, atol=0)
    assert state["tracker"]["control_stale"] == state["tracker"]["fit_stale"] == 0


def test_real_lif_resume_keeps_pending_plateau_and_next_update(tmp_path):
    model = network()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, foreach=False)
    rng = np.random.default_rng(11)
    x, y = torch.ones(4, 8, 784), torch.tensor([[i, i % 7] for i in range(4)])
    ledger = base.Ledger(tmp_path, base.new_costs())
    base.training_update(model, optimizer, x, y, torch.arange(4), 2, ledger, 1)
    state = training_state()
    adopt_history(optimizer, state, [dict(epoch=5, **metrics(3.0)), dict(epoch=10, **metrics(3.5))])
    scheduler = make_scheduler(optimizer)
    scheduler.load_state_dict(state["lr_policy"]["scheduler"])
    base.save_resume(tmp_path, model, optimizer, rng, state)
    payload, _ = base.load_resume(tmp_path, expected_widths=model.widths)
    restored = network()
    restored_opt = torch.optim.AdamW(restored.parameters(), lr=0.001, foreach=False)
    restored_rng = np.random.default_rng(11)
    base.restore_checkpoint(payload, restored, restored_opt, restored_rng)
    restored_scheduler = make_scheduler(restored_opt)
    restored_scheduler.load_state_dict(payload["state"]["lr_policy"]["scheduler"])
    event1 = control_feedback(scheduler, state, metrics(3.2), 15)
    event2 = control_feedback(restored_scheduler, payload["state"], metrics(3.2), 15)
    assert event1 == event2 and event1["reduced"]
    positions = rng.permutation(4)
    np.testing.assert_array_equal(positions, restored_rng.permutation(4))
    (tmp_path / "restored").mkdir()
    base.training_update(model, optimizer, x, y, torch.as_tensor(positions), 2, ledger, 2)
    base.training_update(restored, restored_opt, x, y, torch.as_tensor(positions), 2,
                         base.Ledger(tmp_path / "restored", base.new_costs()), 2)
    for left, right in zip(model.weights, restored.weights):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
        for key in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(optimizer.state[left][key], restored_opt.state[right][key], rtol=0, atol=0)


def test_fork_real_paused_checkpoint_preserves_original_and_incomplete_epoch(tmp_path, monkeypatch):
    model = network()
    monkeypatch.setattr(base, "WIDTHS", model.widths)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, foreach=False)
    rng = np.random.default_rng(11)
    parent, child = tmp_path / "parent", tmp_path / "adaptive"
    parent.mkdir()
    (parent / "evaluations").mkdir()
    x, y = torch.ones(4, 8, 784), torch.tensor([[i, i % 7] for i in range(4)])
    state = training_state()
    base.training_update(model, optimizer, x, y, torch.arange(4), 2,
                         base.Ledger(parent, state["costs"]), 16)
    state["progress"] = dict(epoch=16, stage="train", next_sample=128,
                            order=rng.permutation(4480), loss_sum=123.0, control=None, fit=None)
    for epoch, loss in ((5, 3.25), (10, 3.95), (15, 4.96)):
        base.write_json(parent / "evaluations" / f"control_{epoch:03d}.json", metrics(loss))
    base.write_json(parent / "config.json", dict(source_sha256=base.source_hashes(), effective_batch=128))
    base.save_resume(parent, model, optimizer, rng, state)
    original_bytes = (parent / "resume.pt").read_bytes()
    args = SimpleNamespace(batch_size=128, epochs=100, extension_reason="")
    inheritance = fork_paused_run(parent, child, args)
    assert (parent / "resume.pt").read_bytes() == original_bytes
    payload, _ = base.load_resume(child, expected_widths=model.widths)
    assert payload["optimizer"]["param_groups"][0]["lr"] == 0.0005
    assert payload["state"]["progress"]["lr_segments"] == [
        dict(start_sample=0, learning_rate=0.001), dict(start_sample=128, learning_rate=0.0005)]
    np.testing.assert_array_equal(payload["state"]["progress"]["order"], state["progress"]["order"])
    assert payload["state"]["costs"] == state["costs"] == inheritance["inherited_costs"]
    for name, weight in model.state_dict().items():
        torch.testing.assert_close(weight, payload["model"][name], rtol=0, atol=0)
    for index, values in optimizer.state_dict()["state"].items():
        for key, value in values.items():
            torch.testing.assert_close(value, payload["optimizer"]["state"][index][key], rtol=0, atol=0)
    assert payload["numpy_rng"] == rng.bit_generator.state
    torch.testing.assert_close(payload["torch_rng"], torch.get_rng_state(), rtol=0, atol=0)
    assert json.loads((child / "config.json").read_text(encoding="utf-8"))["learning_rate_schedule"] == POLICY
    with pytest.raises(FileExistsError):
        fork_paused_run(parent, child, args)


def test_background_pause_request_and_duplicate_writer_guard(tmp_path):
    stop = FileStopRequested(tmp_path)
    assert not stop.requested
    base.write_json(tmp_path / "stop_request.json", dict(reason="user_requested_pause"))
    assert stop.requested
    assert process_alive(os.getpid())
    with RunLock(tmp_path):
        with pytest.raises(RuntimeError, match="禁止重复训练"):
            with RunLock(tmp_path):
                pytest.fail("不能允许第二个写入者")
    assert not (tmp_path / "running.json").exists()
