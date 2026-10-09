"""真实三层LIF前向、梯度累积及优化器/随机状态恢复。"""
import numpy as np
import pytest
import torch

from braingrow.network.fixed_layered import FixedLayeredSNN
from tools.train_large_baseline import (
    Ledger, load_resume, new_costs, restore_checkpoint, save_resume, training_update,
)


@pytest.fixture(autouse=True)
def few_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def active_network(widths=(12, 9, 6)):
    model = FixedLayeredSNN(widths, device="cpu")
    with torch.no_grad():
        model.weights[0].fill_(0.1)
        for w in model.weights[1:-1]:
            w.fill_(1.0)
    return model


def test_three_layers_have_real_step_delay_and_no_256_fire_cap():
    model = active_network((300, 270, 260))
    x = torch.ones(2, 8, 784)
    with torch.no_grad():
        potential, activity = model(x, trace=True)
        repeated = model(x)
    # 32步中第1/2/3层分别从第1/2/3步开始Fire；每层全部节点都能发放。
    expected = torch.tensor([300 * 32, 270 * 31, 260 * 30], dtype=torch.float32)
    torch.testing.assert_close(activity["per_video_fire"], expected.expand(2, -1))
    torch.testing.assert_close(activity["frame_fire"][:, 0],
                               torch.tensor([300 * 4, 270 * 3, 260 * 2], dtype=torch.float32).expand(2, -1))
    torch.testing.assert_close(potential, activity["last_layer_counts"] @ model.weights[-1])
    torch.testing.assert_close(potential, repeated)  # 视频边界清零。
    assert model.resources()["effective_weight_slots"] == sum(w.numel() for w in model.weights)
    assert not any("bias" in name for name, _ in model.named_parameters())


def test_effective_batch_gradient_accumulation_matches_full_batch(tmp_path):
    full, micro = active_network(), active_network()
    opt1 = torch.optim.AdamW(full.parameters(), lr=0.001, foreach=False)
    opt2 = torch.optim.AdamW(micro.parameters(), lr=0.001, foreach=False)
    x = torch.linspace(0.2, 1.0, 8).reshape(8, 1, 1).expand(8, 8, 784).clone()
    y = torch.tensor([[i % 10, i % 7] for i in range(8)])
    positions = torch.arange(8)
    (tmp_path / "full").mkdir()
    (tmp_path / "micro").mkdir()
    ledger1, ledger2 = Ledger(tmp_path / "full", new_costs()), Ledger(tmp_path / "micro", new_costs())
    result1 = training_update(full, opt1, x, y, positions, 8, ledger1, 1)
    result2 = training_update(micro, opt2, x, y, positions, 3, ledger2, 1)
    assert result1[0] == pytest.approx(result2[0], rel=1e-6)
    assert result1[2] > 0 and result2[2] > 0
    for a, b in zip(full.weights, micro.weights):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-7)
    assert ledger1.costs == ledger2.costs
    assert ledger1.costs["train_forward_videos"] == 8
    assert ledger1.costs["train_backward_videos"] == 8


def test_checkpoint_restores_next_update_and_sample_random_stream(tmp_path):
    model = active_network()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, foreach=False)
    rng = np.random.default_rng(11)
    x, y = torch.ones(8, 8, 784), torch.tensor([[i, i % 7] for i in range(8)])
    ledger = Ledger(tmp_path, new_costs())
    first = torch.as_tensor(rng.permutation(8))
    training_update(model, optimizer, x, y, first, 3, ledger, 1)
    state = dict(costs=ledger.costs, progress=dict(order=first.numpy(), next_sample=8))
    save_resume(tmp_path, model, optimizer, rng, state)
    loaded, _ = load_resume(tmp_path, expected_widths=model.widths)
    restored = active_network()
    restored_opt = torch.optim.AdamW(restored.parameters(), lr=0.001, foreach=False)
    restored_rng = np.random.default_rng(11)
    restore_checkpoint(loaded, restored, restored_opt, restored_rng)
    next_order = rng.permutation(8)
    np.testing.assert_array_equal(next_order, restored_rng.permutation(8))
    (tmp_path / "restored").mkdir()
    restored_ledger = Ledger(tmp_path / "restored", new_costs())
    training_update(model, optimizer, x, y, torch.as_tensor(next_order), 3, ledger, 2)
    training_update(restored, restored_opt, x, y, torch.as_tensor(next_order), 3, restored_ledger, 2)
    for a, b in zip(model.weights, restored.weights):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(optimizer.state[a]["exp_avg"], restored_opt.state[b]["exp_avg"], rtol=0, atol=0)
        torch.testing.assert_close(optimizer.state[a]["exp_avg_sq"], restored_opt.state[b]["exp_avg_sq"], rtol=0, atol=0)


def test_previous_real_checkpoint_recovers_incomplete_checkpoint_write(tmp_path):
    model = active_network()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, foreach=False)
    rng = np.random.default_rng(11)
    save_resume(tmp_path, model, optimizer, rng, dict(marker=1))
    save_resume(tmp_path, model, optimizer, rng, dict(marker=2))
    (tmp_path / "resume.pt").write_bytes(b"")
    payload, path = load_resume(tmp_path, expected_widths=model.widths)
    assert path.name == "resume_previous.pt"
    assert payload["state"]["marker"] == 1
    restored = active_network()
    restored.load_state_dict(payload["model"])
    with torch.no_grad():
        torch.testing.assert_close(model(torch.ones(1, 8, 784)), restored(torch.ones(1, 8, 784)))
