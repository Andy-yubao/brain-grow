"""只读已完成A2/B2证据；保存重要汇总/恢复复核，不执行LIF。"""
from pathlib import Path
from collections import Counter
import json, pickle
import numpy as np
from braingrow.network.communities import born_candidate, offspring
from braingrow.network.communities_v2 import select_candidates
from tools.growth_v2_common import write_json


def main():
    adir = Path("experiments/runs/discrete_growth_d1_a2"); bdir = Path("experiments/runs/community_growth_d1_b2")
    a = json.loads((adir / "summary.json").read_text(encoding="utf-8")); b = json.loads((bdir / "summary.json").read_text(encoding="utf-8"))
    assert a["completed"] and b["completed"]
    learners = {arm: pickle.loads((adir / (arm + "_learner.pkl")).read_bytes()) for arm in ("full", "output_only")}
    coverage = {}; edge_totals = {}
    for arm, learner in learners.items():
        rows = [e for e in learner.events if e["kind"] in ("input", "internal", "output")]
        for row in rows:
            assert len(row["positive_ids"]) >= 3 and len(row["negative_ids"]) >= 3
            assert max(row["positive_id_exposures"].values()) <= 6 and max(row["negative_id_exposures"].values()) <= 6
            assert all(v == row["nuisance_quotas_both_sides"] for v in row["nuisance_exposures"])
            if row["kind"] == "output": assert row["negative_class_exposures"] == row["negative_class_quotas"]
        coverage[arm] = {"completed_event_records": len(rows), "all_completed_windows_cover_IDs_and_balanced_classes": True,
            "hidden_window_range": [min(w.windows for w in learner.hidden_windows), max(w.windows for w in learner.hidden_windows)],
            "output_each_class": [{"target": target, "completed_windows": w.windows, "unfinished_positive_negative_counts": w.count.tolist(),
                "unfinished_ids": [sorted(ids) for ids in w.ids], "unfinished_negative_class_counts": w.negative_seen,
                "unfinished_nuisance_counts": w.nuisance_seen} for target, w in enumerate(learner.output_windows)],
            "birth_protection_final": learner.protection_status(), "functional_mature_edges_confirmed": 0}
        edge_totals[arm] = {kind: {key: sum(row[key] for row in rows if row["kind"] == kind) for key in ("formed", "deleted", "changed")} for kind in ("input", "internal", "output")}
    costs = {}
    for name, directory, summary, budget in (("A2", adir, a, 40320), ("B2", bdir, b, 95200)):
        journal = [json.loads(line) for line in (directory / "cost_journal.jsonl").read_text(encoding="utf-8").splitlines()]
        assert sum(row["videos"] for row in journal) == summary["simulation_videos"] == journal[-1]["simulation_videos_cumulative"] == budget
        assert sum(row["kind"].startswith("metric_complete_") for row in journal) == summary["metric_evaluations"]
        costs[name] = {"actual_simulation_videos": summary["simulation_videos"], "journal_matches_checkpoint": True,
            "metric_evaluations": summary["metric_evaluations"], "journal_rows": len(journal), "interrupted_replayed_simulation_videos": 0,
            "wall_seconds": summary["wall_seconds"], "peak_working_set_bytes": summary["peak_working_set_bytes"]}
    bstate = pickle.loads((bdir / "resume.pkl").read_bytes()); exact_graphs = 0; pareto_violations = 0
    for arm in ("full", "output_only"):
        lineage = {row["proposal_id"]: row for row in b["lineage"] if row["arm"] == arm}; candidates = {}; rng = np.random.default_rng(np.random.SeedSequence([11, 6]))
        for pid in range(6):
            c = born_candidate(bstate["initial"], pid); c.score = lineage[pid]["score"]; candidates[pid] = c
        retained = [candidates[pid] for pid in range(6)]
        for curve in [c for c in b["curves"] if c["arm"] == arm]:
            generation = curve["generation"]; children = []
            for index in range(6):
                pid = 6 + (generation - 1) * 6 + index; row = lineage[pid]
                parent = retained[int(rng.integers(6))]; assert parent.proposal_id == row["parent_id"]
                c = offspring(parent, pid, full=arm == "full"); c.score = row["score"]; candidates[pid] = c; children.append(c)
            pool = retained + children; retained = select_candidates(pool, rng)
            assert [c.proposal_id for c in retained] == [c["id"] for c in curve["retained"]]
            nd = [c for c in pool if not any(o.score["target_repairs"] >= c.score["target_repairs"] and o.score["damage"] <= c.score["damage"] and
                (o.score["target_repairs"] > c.score["target_repairs"] or o.score["damage"] < c.score["damage"]) for o in pool if o is not c)]
            for c in retained[:4]:
                if c not in nd and any(o not in retained[:4] for o in nd): pareto_violations += 1
        for pid, c in candidates.items():
            with np.load(bdir / f"{arm}_graph{pid:03d}.npz") as z:
                for key in ("win", "rec", "out", "alpha", "bias", "threshold", "input_mask", "rec_mask", "output_mask"):
                    np.testing.assert_array_equal(getattr(c.net, key), z[key])
            assert c.stable_ids == lineage[pid]["stable_ids"] and c.actions == lineage[pid]["actions"]
            exact_graphs += 1
        for p in bdir.glob(arm + "_frozen*.pkl"):
            saved = pickle.loads(p.read_bytes()); assert saved.proposal_rng_state == candidates[saved.proposal_id].proposal_rng_state
        if arm == "output_only": assert rng.bit_generator.state == bstate["rng"].bit_generator.state
    assert pareto_violations == 0
    audit = {"ok": True, "LIF_simulation_videos": 0, "actual_graphs_exactly_reconstructed": exact_graphs,
        "parent_choices_and_retention_exactly_reconstructed": True, "private_frozen_RNG_states_match": True,
        "management_RNG_matches": True, "pareto_priority_violations": pareto_violations, "A2_completed_windows_coverage": coverage,
        "actual_cost_journals": costs}
    write_json("reports/growth_v2_recovery_audit_20261004.json", audit)
    actions = {arm: dict(Counter(action["kind"] for row in b["lineage"] if row["arm"] == arm for action in row["actions"])) for arm in ("full", "output_only")}
    evidence = {"date": "2026-10-04", "stage": "早期唯一种子11、训练池不同ID确认，非验证/最终测试", "root_model_seed": 11, "data_seed": 20261004,
        "validation_arrays_unpacked": False, "validation_forward_evaluations": 0, "sealed_test_reads": 0, "dependencies_added": [],
        "A2": {"initial_confirmation": a["initial_confirmation"], "full_confirmation": a["full_confirmation"], "output_only_confirmation": a["output_only_confirmation"],
            "curves": a["curves"], "demands": a["demands"], "interventions": a["interventions"], "continuation_gate": a["continuation_gate"],
            "coverage": coverage, "event_change_totals": edge_totals, "cost": costs["A2"], "config": json.loads((adir / "config.json").read_text(encoding="utf-8"))},
        "B2": {"initial_confirmation": b["baseline_confirmation"], "full_confirmations": b["full_confirmations"], "output_only_confirmations": b["output_only_confirmations"],
            "curves": b["curves"], "lineage": b["lineage"], "action_counts": actions, "cost": costs["B2"],
            "graduates_full": sum(c["gate"]["graduated"] for c in b["full_confirmations"]), "graduates_output_only": sum(c["gate"]["graduated"] for c in b["output_only_confirmations"]),
            "config": json.loads((bdir / "config.json").read_text(encoding="utf-8"))},
        "total_actual_simulation_videos": a["simulation_videos"] + b["simulation_videos"], "total_neural_video_steps": 32 * (a["simulation_videos"] + b["simulation_videos"]),
        "A1_B1_preserved_cost": 137340, "all_versions_actual_simulation_videos": 137340 + a["simulation_videos"] + b["simulation_videos"],
        "research_goal_achieved": False, "D2_or_C_started": False}
    write_json("reports/growth_v2_diagnostics_20261004.json", evidence)
    print({"audit_ok": True, "reconstructed_B2_graphs": exact_graphs, "A2_videos": a["simulation_videos"], "B2_videos": b["simulation_videos"], "new_total_videos": evidence["total_actual_simulation_videos"]})


if __name__ == "__main__": main()
