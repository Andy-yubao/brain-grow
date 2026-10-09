"""真实v5渲染、LIF梯度/恢复和实验接受规则；不冒充任务成绩。"""
import copy

import numpy as np
import pytest
import torch

from braingrow.data.online_v5 import render_epoch, epoch_cache
from braingrow.network.fixed_layered import FixedLayeredSNN
from tools import train_large_baseline as base
from tools.train_generalization_tuning import training_update, costs, paired_comparison


@pytest.fixture(autouse=True)
def few_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def model():
    net = FixedLayeredSNN((12, 9, 6), device="cpu")
    with torch.no_grad():
        net.weights[0].fill_(0.1)
        for weight in net.weights[1:-1]:
            weight.fill_(1.0)
    return net


def test_new_variants_repeat_exactly_and_preserve_video_and_labels(tmp_path):
    glyph = np.zeros((28, 28), np.uint8)
    glyph[5:23, 12:16] = 255
    images = {123: glyph}
    first, _ = render_epoch(images, [123], [1], 1)
    repeated, _ = render_epoch(images, [123], [1], 1)
    second, _ = render_epoch(images, [123], [1], 2)
    for key in first:
        np.testing.assert_array_equal(first[key], repeated[key])
    assert first["x"].shape == (28, 8, 784)
    assert first["x"].dtype == np.float32
    assert 0 <= first["x"].min() < first["x"].max() <= 1
    assert not np.array_equal(first["x"], second["x"])
    np.testing.assert_array_equal(first["y"], second["y"])
    np.testing.assert_array_equal(first["y"][:, 1], np.tile(np.arange(7), 4))
    assert set(first["variants"]) == {4, 5, 6, 7}
    assert set(second["variants"]) == {8, 9, 10, 11}
    cached, metadata = epoch_cache(tmp_path, images, [123], [1], 1)
    shared, again = epoch_cache(tmp_path, images, [123], [1], 1)
    assert metadata["generated"] and not again["generated"]
    np.testing.assert_array_equal(cached["x"], shared["x"])
    with (tmp_path / "epoch_001.npz").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="哈希"):
        epoch_cache(tmp_path, images, [123], [1], 1)


def test_zero_smoothing_has_same_real_lif_update_as_existing_baseline(tmp_path):
    old, new = model(), model()
    opt1 = torch.optim.AdamW(old.parameters(), lr=0.0003, foreach=False)
    opt2 = torch.optim.AdamW(new.parameters(), lr=0.0003, foreach=False)
    x = torch.linspace(0.2, 1, 8).reshape(8, 1, 1).expand(8, 8, 784).clone()
    y = torch.tensor([[i % 10, i % 7] for i in range(8)])
    (tmp_path / "old").mkdir()
    (tmp_path / "new").mkdir()
    a = base.training_update(old, opt1, x, y, torch.arange(8), 3,
                             base.Ledger(tmp_path / "old", base.new_costs()), 1)
    ledger = base.Ledger(tmp_path / "new", costs())
    b = training_update(new, opt2, x, y, torch.arange(8), 3, ledger, 1, 0)
    assert a[0] == pytest.approx(b[0], rel=1e-7)
    for left, right in zip(old.weights, new.weights):
        torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-7)
    assert ledger.costs["optimizer_updates"] == 1
    assert ledger.costs["train_forward_videos"] == ledger.costs["train_backward_videos"] == 8


def test_smoothed_real_lif_accumulation_and_resume_next_update(tmp_path):
    full, micro = model(), model()
    a = torch.optim.AdamW(full.parameters(), lr=0.0003, foreach=False)
    b = torch.optim.AdamW(micro.parameters(), lr=0.0003, foreach=False)
    x, y = torch.ones(8, 8, 784), torch.tensor([[i, i % 7] for i in range(8)])
    (tmp_path / "full").mkdir()
    (tmp_path / "micro").mkdir()
    la, lb = base.Ledger(tmp_path / "full", costs()), base.Ledger(tmp_path / "micro", costs())
    training_update(full, a, x, y, torch.arange(8), 8, la, 1, 0.1)
    training_update(micro, b, x, y, torch.arange(8), 3, lb, 1, 0.1)
    for left, right in zip(full.weights, micro.weights):
        torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-7)
    rng = np.random.default_rng(11)
    base.save_resume(tmp_path, micro, b, rng, dict(costs=lb.costs))
    payload, _ = base.load_resume(tmp_path, expected_widths=micro.widths)
    resumed = model()
    opt = torch.optim.AdamW(resumed.parameters(), lr=0.0003, foreach=False)
    restored_rng = np.random.default_rng(11)
    base.restore_checkpoint(payload, resumed, opt, restored_rng)
    order = rng.permutation(8)
    np.testing.assert_array_equal(order, restored_rng.permutation(8))
    training_update(micro, b, x, y, torch.as_tensor(order), 3, lb, 2, 0.1)
    (tmp_path / "resumed").mkdir()
    training_update(resumed, opt, x, y, torch.as_tensor(order), 3,
                    base.Ledger(tmp_path / "resumed", costs()), 2, 0.1)
    for left, right in zip(micro.weights, resumed.weights):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
        torch.testing.assert_close(b.state[left]["exp_avg"], opt.state[right]["exp_avg"], rtol=0, atol=0)


def test_gate_uses_two_evaluations_and_rejects_task_regression():
    metric = dict(digit_accuracy=0.36, motion_accuracy=0.71, joint_accuracy=0.275, loss=7.0)
    summary = dict(history=[dict(epoch=4, metrics=metric), dict(epoch=6, metrics=metric)])
    rows = {name: copy.deepcopy(summary) for name in ("fixed", "refresh", "refresh_smooth")}
    for row in rows["refresh"]["history"]:
        row["metrics"] = dict(metric, digit_accuracy=0.39, motion_accuracy=0.70, joint_accuracy=0.295, loss=6.9)
    for row in rows["refresh_smooth"]["history"]:
        row["metrics"] = dict(metric, digit_accuracy=0.40, motion_accuracy=0.69, joint_accuracy=0.31, loss=6.0)
    result = paired_comparison(rows)
    assert result["winner"] == "refresh"
    assert result["passed"] == ["refresh"]  # 联合更高也不能损害运动超过门槛。
    rows["refresh"]["history"][0]["metrics"]["joint_accuracy"] = 0.275
    assert paired_comparison(rows)["winner"] is None  # 不靠挑一次最好成绩接受。
