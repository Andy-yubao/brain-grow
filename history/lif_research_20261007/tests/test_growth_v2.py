"""修正规则的真实原语检查：事件门控、均衡覆盖、保护与保留优先级。"""
from types import SimpleNamespace
import numpy as np
from braingrow.network.discrete import initial_network
from braingrow.network.discrete_v2 import EventLearner, EventWindow
from braingrow.network.communities_v2 import select_candidates
from tools.growth_v2_common import ensure_budget, record_cost, restore_cost, BudgetExhausted, save


class SelectAll:
    def random(self, size): return np.zeros(size)


def learner():
    return EventLearner(initial_network(8), [(0, 0, 1)] * 8, [1] * 17,
        stable_ids=list(range(8)), next_id=8)


def event_window(action, sign):
    w = EventWindow(8); w.count[:] = 16; w.ids = [{1, 2, 3}, {4, 5, 6}]
    w.source[0 if sign > 0 else 1] = 16
    w.previous_action[:] = action; w.previous_sign[:] = sign
    return w


def test_hidden_requires_miss_or_hit_events_and_false_alarm_events():
    a = learner(); weights = np.zeros(8, np.float32)
    w = event_window(1, 1); w.eligible[0] = 16
    a._transition(weights, w, .5, 1., SelectAll(), "input", 0)
    assert not weights.any()  # 单纯source侧偏与eligible不构成H/M事件
    w = event_window(1, 1); w.event_count[:] = [0, 16, 0, 16]; w.eligible_events[1] = 16
    a._transition(weights, w, .5, 1., SelectAll(), "input", 0)
    assert np.all(weights == .5)
    w = event_window(2, 1); w.event_count[:] = [16, 0, 5, 11]; w.causal_events[0] = 16
    a._transition(weights, w, .5, 1., SelectAll(), "input", 0)
    assert np.all(weights <= .5)  # 高误报禁止H功能巩固
    w = event_window(2, 1); w.event_count[:] = [16, 0, 0, 16]; w.causal_events[0] = 16
    weights[:] = .5; a._transition(weights, w, .5, 1., SelectAll(), "input", 0)
    assert np.all(weights == 1.)
    w = event_window(3, -1); w.event_count[:] = [0, 16, 16, 0]; w.eligible_events[2] = 16
    weights[:] = 0; a._transition(weights, w, .5, 1., SelectAll(), "input", 0)
    assert np.all(weights == -.5)


def test_inhibition_preservation_requires_correct_rejection_and_positive_hits():
    a = learner(); weights = np.full(8, -.5, np.float32)
    w = event_window(4, -1); w.event_count[:] = [8, 8, 4, 12]
    a._transition(weights, w, .5, 1., SelectAll(), "internal", 0)
    assert np.all(weights == -.5) and np.all(w.age == 1)
    w = event_window(4, -1); w.event_count[:] = [7, 9, 4, 12]
    a._transition(weights, w, .5, 1., SelectAll(), "internal", 0)
    assert not w.age.any()


def test_output_all_negative_and_nuisance_quotas_fill_with_three_ids():
    a = learner(); source = np.ones(40, bool)
    for target, group, classes in [(0, 0, 10), (12, 1, 7)]:
        w = a.output_windows[target]
        c = target if group == 0 else target - 10
        for variant in range(4):
            for motion in range(7):
                for rank in range(3):
                    for digit in range(10):
                        labels = (digit, motion); side = int(labels[group] != c)
                        a._record(w, side, 100 * digit + rank, source, source, source, True,
                            labels[group] if side else None, labels[1 - group])
        assert a._ready(w)
        assert w.negative_seen == w.negative_quota()
        assert all(counts == w.nuisance_quota() for counts in w.nuisance_seen)
        assert max(w.negative_seen.values()) - min(w.negative_seen.values()) <= 1
        assert all(max(counts.values()) <= 6 for counts in w.id_counts)
        quotas = w.negative_quota().copy()
        # 输出迁移与实际配额日志的窗号必须一致。
        a._transition(a.net.out[:, target], w, .03, .10, np.random.default_rng(11), "output", target, True)
        assert a.events[-1]["negative_class_quotas"] == quotas
        assert a.events[-1]["negative_class_exposures"] == quotas


def test_birth_extends_event_arrays_and_uses_four_qualified_window_age():
    a = learner(); a.birth((0, 0, 1), 2)
    assert a.net.valid() and a.net.n == 10
    for w in a.hidden_windows:
        assert w.source_events.shape == (4, 794) and w.previous_action.shape == (794,)
    assert all(w.source_events.shape == (4, 50) for w in a.output_windows)
    assert all(item["protected"] for item in a.protection_status())
    a.hidden_windows[-1].windows = 4
    assert not a.protection_status()[-1]["protected"]
    assert not a.protection_status()[-1]["functional_mature"]
    rejected = a.clone(); rejected.hidden_windows[0].eligible_events[:] = 9
    rejected.birth_window_start[8] = 99; rejected.rng.random()
    assert not a.hidden_windows[0].eligible_events.any() and a.birth_window_start[8] == 0


def candidate(pid, repairs, damage, ids, videos):
    return SimpleNamespace(proposal_id=pid, score={"target_repairs": repairs, "damage": damage,
        "total_repairs": repairs, "repair_ids": ids, "signature": videos})


def test_behavior_slots_prioritize_nondominated_then_ids_before_videos():
    candidates = [candidate(0, 10, 0, [1], list(range(10))), candidate(1, 11, 1, [2], [10]),
        candidate(2, 12, 2, [3], [11]), candidate(3, 13, 3, [4], [12]),
        candidate(4, 1, 9, list(range(5, 25)), list(range(20, 40))), candidate(5, 0, 20, [30], [40])]
    result = select_candidates(candidates, np.random.default_rng(11))
    assert {c.proposal_id for c in result[:4]} == {0, 1, 2, 3}
    candidates = [candidate(0, 10, 0, [1], list(range(10))), candidate(1, 10, 0, [2, 3], [20]),
        candidate(2, 10, 0, [4], list(range(30, 60))), candidate(3, 10, 0, [5], [60]),
        candidate(4, 0, 20, [6], [70]), candidate(5, 0, 20, [7], [80])]
    result = select_candidates(candidates, np.random.default_rng(11))
    assert result[0].proposal_id == 1


def test_journal_budget_counts_replayed_cost_and_stops_before_next_forward(tmp_path):
    state = {"simulation_videos": 0, "simulation_budget": 30, "metric_evaluations": 0}
    ensure_budget(state, 30, tmp_path); record_cost(state, 30, "actual_forward", tmp_path)
    restored = {"simulation_videos": 0, "simulation_budget": 30, "metric_evaluations": 0}
    restore_cost(restored, tmp_path)
    assert restored["simulation_videos"] == 30
    try: ensure_budget(restored, 1, tmp_path)
    except BudgetExhausted: pass
    else: raise AssertionError("预算耗尽必须在下一前向之前停止")
    assert restored["stopped_budget"] and (tmp_path / "budget_stop.json").exists()


def test_completed_resume_exits_before_loading_training_or_validation(tmp_path, monkeypatch):
    from tools import run_discrete_growth_v2, run_community_growth_v2
    save(tmp_path / "resume.pkl", {"completed": True, "simulation_videos": 0, "metric_evaluations": 0})
    def forbidden(): raise AssertionError("完成检查点不得加载数据")
    for module in (run_discrete_growth_v2, run_community_growth_v2):
        monkeypatch.setattr(module, "training_data", forbidden)
        module.run(SimpleNamespace(run=str(tmp_path), resume=True))
