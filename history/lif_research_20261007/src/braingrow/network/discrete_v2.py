"""A2：逐源H/M/F/C条件、全负类读出与覆盖窗口保护；不改变无标签前向。"""
from __future__ import annotations
import copy
from dataclasses import dataclass, field
import numpy as np
from .discrete import DiscreteLearner, Window, sync_masks


@dataclass
class EventWindow(Window):
    id_counts: list = field(default_factory=lambda: [{}, {}])
    negative_classes: list = field(default_factory=list)
    negative_seen: dict = field(default_factory=dict)
    nuisance_classes: list = field(default_factory=list)
    nuisance_seen: list = field(default_factory=lambda: [{}, {}])

    def __post_init__(self):
        super().__post_init__()
        self.source_events = np.zeros((4, self.width), np.uint16)
        self.eligible_events = np.zeros((4, self.width), np.uint16)
        self.causal_events = np.zeros((4, self.width), np.uint16)
        self.event_count = np.zeros(4, np.uint16)
        self.previous_action = np.zeros(self.width, np.int8)

    def clear(self):
        super().clear()
        self.id_counts = [{}, {}]; self.negative_seen = {}
        self.nuisance_seen = [{}, {}]
        self.source_events[:] = 0; self.eligible_events[:] = 0
        self.causal_events[:] = 0; self.event_count[:] = 0

    def negative_quota(self):
        n = len(self.negative_classes)
        base, extra = divmod(16, n)
        rotating = [(self.windows + j) % n for j in range(extra)]
        return {c: base + int(j in rotating) for j, c in enumerate(self.negative_classes)}

    def nuisance_quota(self):
        base, extra = divmod(16, len(self.nuisance_classes))
        rotating = [(self.windows + j) % len(self.nuisance_classes) for j in range(extra)]
        return {c: base + int(j in rotating) for j, c in enumerate(self.nuisance_classes)}


@dataclass
class EventLearner(DiscreteLearner):
    birth_window_start: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.hidden_windows:
            self.hidden_windows = [EventWindow(784 + self.net.n) for _ in range(self.net.n)]
            for agenda, w in zip(self.agendas, self.hidden_windows):
                w.nuisance_classes = list(range(7 if agenda[0] == 0 else 10))
        if not self.output_windows:
            self.output_windows = [EventWindow(self.net.n * 5) for _ in range(17)]
            for target, w in enumerate(self.output_windows):
                c = target if target < 10 else target - 10
                w.negative_classes = [j for j in range(10 if target < 10 else 7) if j != c]
                w.nuisance_classes = list(range(7 if target < 10 else 10))

    def _record(self, w, side, raw_id, source, eligible, causal, fired, negative_class=None, nuisance_class=None):
        if side < 0 or w.count[side] >= 16: return
        raw_id = int(raw_id)
        # 填满16必有至少3ID；单ID配额不改变前向或输入。
        if w.id_counts[side].get(raw_id, 0) >= 6: return
        if nuisance_class is not None and w.nuisance_seen[side].get(nuisance_class, 0) >= w.nuisance_quota()[nuisance_class]: return
        if side == 1 and negative_class is not None:
            if w.negative_seen.get(negative_class, 0) >= w.negative_quota()[negative_class]: return
            w.negative_seen[negative_class] = w.negative_seen.get(negative_class, 0) + 1
        if nuisance_class is not None:
            w.nuisance_seen[side][nuisance_class] = w.nuisance_seen[side].get(nuisance_class, 0) + 1
        self._add(w, side, raw_id, source, eligible, fired)
        w.id_counts[side][raw_id] = w.id_counts[side].get(raw_id, 0) + 1
        event = (0 if fired else 1) if side == 0 else (2 if fired else 3)  # H/M/F/C
        w.event_count[event] += 1
        w.source_events[event] += source.astype(np.uint16)
        w.eligible_events[event] += eligible.astype(np.uint16)
        w.causal_events[event] += causal.astype(np.uint16)

    def _transition(self, weights, w, weak, strong, rng, kind, identifier, output=False):
        rates = w.source / 16
        quotas = w.negative_quota() if output else {}
        nuisance_quotas = w.nuisance_quota() if w.nuisance_classes else {}
        old = weights.copy()
        if output:
            difference = rates[0] - rates[1]
            sign = np.where(difference >= .25, 1, np.where(difference <= -.25, -1, 0)).astype(np.int8)
            action = sign.copy()
        else:
            positive = (rates[0] >= .5) & (rates[1] <= .25)
            negative = (rates[1] >= .5) & (rates[0] <= .25)
            missed = positive & (w.eligible_events[1] >= 2)
            hit = positive & (w.causal_events[0] >= 2) & (w.event_count[2] <= 4)
            false_alarm = negative & (w.eligible_events[2] >= 2)
            inhibitory_keep = negative & (old < 0) & (w.event_count[3] >= 12) & (w.event_count[0] >= 8)
            action = np.zeros(len(weights), np.int8)
            action[hit] = 2; action[missed] = 1
            action[inhibitory_keep] = 4; action[false_alarm] = 3
            sign = np.where((action == 1) | (action == 2), 1,
                np.where((action == 3) | (action == 4), -1, 0)).astype(np.int8)
        repeated = (action != 0) & (action == w.previous_action) & (sign == w.previous_sign)
        if output:
            repeated_edges = repeated.reshape(self.net.n, 5)
            choices = np.argmax(repeated_edges, axis=1)
            edge_sign = sign.reshape(self.net.n, 5)[np.arange(self.net.n), choices]
            qualified = repeated_edges.any(axis=1)
            ages = w.age.reshape(self.net.n, 5).max(axis=1)
        else:
            edge_sign, qualified, ages = sign, repeated, w.age
        selected = qualified & (rng.random(len(weights)) < .1)
        opposite = selected & (old != 0) & (np.sign(old) != edge_sign)
        weights[opposite] = 0
        absent = selected & (old == 0)
        weights[absent] = weak * edge_sign[absent]
        reinforce = selected & (old != 0) & (np.sign(old) == edge_sign)
        # H/M允许兴奋巩固；抑制F只提名弱边，C/H保持记录不冒充因果成熟。
        if output:
            weights[reinforce] = strong * edge_sign[reinforce]
        else:
            excitatory = reinforce & (edge_sign > 0)
            weights[excitatory] = strong
        if output:
            no_signal = (sign.reshape(self.net.n, 5) == 0).all(axis=1) & (w.previous_sign.reshape(self.net.n, 5) == 0).all(axis=1)
        else:
            no_signal = (sign == 0) & (w.previous_sign == 0)
        retire = no_signal & (ages < 2) & (rng.random(len(weights)) < .01) & (old != 0)
        weights[retire] = np.where(np.abs(old[retire]) > weak, weak * np.sign(old[retire]), 0)
        w.age[repeated] += 1
        w.previous_sign = sign.copy(); w.previous_action = action.copy(); w.windows += 1
        self.events.append({"kind": kind, "id": identifier, "window": w.windows,
            "positive_ids": sorted(w.ids[0]), "negative_ids": sorted(w.ids[1]),
            "positive_id_exposures": dict(w.id_counts[0]), "negative_id_exposures": dict(w.id_counts[1]),
            "negative_class_exposures": dict(w.negative_seen), "negative_class_quotas": quotas,
            "nuisance_exposures": [dict(v) for v in w.nuisance_seen], "nuisance_quotas_both_sides": nuisance_quotas,
            "H_M_F_C": w.event_count.tolist(), "eligible_repeated": int(qualified.sum()),
            "actions_qualified": {str(a): int(((action == a) & repeated).sum()) for a in (1, 2, 3, 4)},
            "formed": int(((old == 0) & (weights != 0)).sum()), "deleted": int(((old != 0) & (weights == 0)).sum()),
            "changed": int((old != weights).sum()), "consolidation_records": int((w.age >= 2).sum())})
        w.clear()

    def learn_batch(self, x, y, ids, cost_callback=None):
        p, activity, trace = self.net.forward(x, record_steps=True)
        if cost_callback is not None: cost_callback(len(x))
        spikes = trace["spikes"] > 0; counts = spikes.sum(axis=1)
        near = trace["membrane_before_reset"] >= .5 * self.net.threshold
        presence = (x > .1).any(axis=1)
        arrival = np.zeros_like(spikes)
        for lag in range(1, 5): arrival[:, lag:] |= spikes[:, :-lag]
        for row in range(len(x)):
            if self.full:
                for i, w in enumerate(self.hidden_windows):
                    side = self._side(y[row], self.agendas[i])
                    if side < 0 or w.count[side] >= 16: continue
                    input_eligible = ((x[row] > .1) & near[row].reshape(8, 4, self.net.n).any(axis=1)[:, i, None]).any(axis=0)
                    input_causal = ((x[row] > .1) & spikes[row].reshape(8, 4, self.net.n).any(axis=1)[:, i, None]).any(axis=0)
                    rec_eligible = (arrival[row] & near[row, :, i, None]).any(axis=0)
                    rec_causal = (arrival[row] & spikes[row, :, i, None]).any(axis=0)
                    source = np.concatenate((presence[row], spikes[row].any(axis=0)))
                    eligible = np.concatenate((input_eligible, rec_eligible))
                    causal = np.concatenate((input_causal, rec_causal))
                    source[784 + i] = False; eligible[784 + i] = False; causal[784 + i] = False
                    self._record(w, side, ids[row], source, eligible, causal, counts[row, i] > 0,
                        nuisance_class=int(y[row, 1 - self.agendas[i][0]]))
            for target, w in enumerate(self.output_windows):
                g, c = (0, target) if target < 10 else (1, target - 10)
                side = 0 if y[row, g] == c else 1
                source = (counts[row, :, None] >= np.array([1, 2, 5, 9, 17])).reshape(-1)
                self._record(w, side, ids[row], source, source, source, True, int(y[row, g]) if side == 1 else None,
                    int(y[row, 1 - g]))
        if self.full:
            for i, w in enumerate(self.hidden_windows):
                if not self._ready(w): continue
                all_weights = np.concatenate((self.net.win[:, i], self.net.rec[:, i]))
                for kind, begin, end, weak, strong in (("input", 0, 784, .5, 1.), ("internal", 784, 784 + self.net.n, 2., 4.)):
                    sub = copy.deepcopy(w); sub.width = end - begin
                    for name in ("source", "eligible", "source_events", "eligible_events", "causal_events"):
                        setattr(sub, name, getattr(w, name)[:, begin:end].copy())
                    for name in ("previous_sign", "previous_action", "age"):
                        setattr(sub, name, getattr(w, name)[begin:end].copy())
                    self._transition(all_weights[begin:end], sub, weak, strong, self.rng, kind, self.stable_ids[i])
                    for name in ("previous_sign", "previous_action", "age"):
                        getattr(w, name)[begin:end] = getattr(sub, name)
                self.net.win[:, i] = all_weights[:784]; self.net.rec[:, i] = all_weights[784:]
                rate = counts[:, i].mean() / 32
                if rate < .005: self.net.threshold[i] = max(.25, self.net.threshold[i] / 2)
                if rate > .6: self.net.threshold[i] = min(2., self.net.threshold[i] * 2)
                if self.rng.random() < .1:
                    source = int(self.rng.integers(784 + self.net.n - 1))
                    if source >= 784 + i: source += 1
                    if source < 784: self.net.win[source, i] = self.rng.choice([0., -.5, .5])
                    else: self.net.rec[source - 784, i] = self.rng.choice([0., -2., 2.])
                w.windows += 1; w.clear()
                stable = self.stable_ids[i]
                if stable in self.birth_window_start:
                    age = w.windows - self.birth_window_start[stable]
                    self.events.append({"kind": "birth_protection", "id": stable, "qualified_window_age": age,
                        "protected": age < 4, "functional_mature": False})
        for target, w in enumerate(self.output_windows):
            if self._ready(w): self._transition(self.net.out[:, target], w, .03, .10, self.output_rng, "output", target, True)
        sync_masks(self.net); self.exposures += len(x)
        return p, activity, trace

    def birth(self, agenda, number=4):
        before = self.net.n
        super().birth(agenda, number)
        # 基类扩展所有旧窗口；新增窗口转为同等事件状态，稳定ID记录真实保护起点。
        for i in range(before):
            w = self.hidden_windows[i]
            for name in ("source_events", "eligible_events", "causal_events"):
                setattr(w, name, np.pad(getattr(w, name), ((0, 0), (0, self.net.n - before))))
            w.previous_action = np.pad(w.previous_action, (0, self.net.n - before))
        for i in range(before, self.net.n):
            self.hidden_windows[i] = EventWindow(784 + self.net.n)
            self.hidden_windows[i].nuisance_classes = list(range(7 if agenda[0] == 0 else 10))
            self.birth_window_start[self.stable_ids[i]] = 0
        for w in self.output_windows:
            for name in ("source_events", "eligible_events", "causal_events"):
                setattr(w, name, np.pad(getattr(w, name), ((0, 0), (0, 5 * (self.net.n - before)))))
            w.previous_action = np.pad(w.previous_action, (0, 5 * (self.net.n - before)))

    def protection_status(self):
        return [{"id": stable, "qualified_window_age": self.hidden_windows[i].windows - self.birth_window_start[stable],
            "protected": self.hidden_windows[i].windows - self.birth_window_start[stable] < 4,
            "functional_mature": False} for i, stable in enumerate(self.stable_ids) if stable in self.birth_window_start]
