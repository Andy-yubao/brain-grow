"""B2：修正保留优先级与目标修复静音检查，不扩大固定代数。"""
from __future__ import annotations
import os
for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"): os.environ[name] = "1"
import argparse, pickle, time
from pathlib import Path
import numpy as np
from braingrow.network.growing import GrowingNetwork
from braingrow.network.discrete import sync_masks
from braingrow.network.communities import born_candidate, offspring, behavior_score
from braingrow.network.communities_v2 import select_candidates
from tools.run_discrete_growth import agendas_from
from tools.run_community_growth import graduation
from tools.growth_v2_common import training_data, subset, evaluate_net, restore_cost, provenance, prediction, save, write_json, peak_memory, BudgetExhausted


def evaluate_candidate(candidate, pool, state, name, outdir, baseline, agenda):
    result = evaluate_net(candidate.net, pool, state, name, outdir)
    candidate.prediction = prediction(outdir, name)
    behavior_score(candidate, baseline, pool[1], pool[2], agenda)
    return result


def repair_evidence(pred, baseline, pool, agenda):
    g, target, rival = agenda; y, ids = pool[1:]
    repaired = (y[:, g] == target) & (baseline[:, g] == rival) & (pred[:, g] == target)
    return {"videos": int(repaired.sum()), "ids": sorted(set(ids[repaired].tolist()))}


def run(args):
    outdir = Path(args.run); outdir.mkdir(parents=True, exist_ok=True); checkpoint = outdir / "resume.pkl"
    if args.resume:
        state = pickle.loads(checkpoint.read_bytes()); restore_cost(state, outdir)
        if state["completed"]: print("B2阶段已完成，未加载数据或重训", flush=True); return
        if state.get("stopped_budget"): print("B2实际预算已耗尽，未加载数据或重训", flush=True); return
    else:
        if checkpoint.exists(): raise ValueError("已有状态，请使用--resume")
        state = {"started": time.time(), "simulation_videos": 0, "metric_evaluations": 0, "arm_index": 0, "generation": 0,
            "curves": [], "lineage": [], "completed": False, "simulation_budget": 95200, "rng": np.random.default_rng(np.random.SeedSequence([11, 6]))}
        provenance(outdir, {"mechanism": "B2_Pareto_ID_diversity_target_silence", "initial_lif": 128,
            "initial_candidates": 6, "initial_new_nodes": 8, "generations": 12, "children_each_generation": 6,
            "development_videos": 560, "confirmation_videos": 560, "max_confirmed_candidates_each_arm": 3,
            "phase_max_simulation_videos": 95200, "selection_order": ["nondominated", "new_repair_IDs", "new_repair_videos", "task_score"],
            "behavior_slots": 4, "random_exploration_slots": 2, "silence_gate": "target_repair_videos_decrease"})
    data = training_data(); dev = subset(data, 2); conf = subset(data, 2, True)
    if "initial" not in state:
        initial = GrowingNetwork.load("experiments/runs/discrete_growth_d1/common_initial.npz")
        initial.save(outdir / "common_initial.npz"); state["initial"] = initial
        state["baseline_development"] = evaluate_net(initial, dev, state, "initial_development", outdir)
        state["baseline_dev_prediction"] = prediction(outdir, "initial_development")
        state["agenda"] = agendas_from(state["baseline_development"]["metrics"])[0]
        state["starting_candidates"] = [born_candidate(initial, j) for j in range(6)]; state["candidates"] = []
        save(checkpoint, state)
    for arm_index in range(state["arm_index"], 2):
        arm = "full" if arm_index == 0 else "output_only"
        if state["arm_index"] != arm_index:
            state.update(arm_index=arm_index, generation=0, candidates=[], rng=np.random.default_rng(np.random.SeedSequence([11, 6])))
            save(checkpoint, state)
        if not state["candidates"]:
            for start in state["starting_candidates"]:
                candidate = start.clone()
                result = evaluate_candidate(candidate, dev, state, f"{arm}_candidate{candidate.proposal_id:03d}", outdir, state["baseline_dev_prediction"], state["agenda"])
                state["candidates"].append(candidate); candidate.net.save(outdir / f"{arm}_graph{candidate.proposal_id:03d}.npz")
                state["lineage"].append({"arm": arm, "proposal_id": candidate.proposal_id, "parent_id": None, "actions": [],
                    "stable_ids": candidate.stable_ids, "metrics": result["metrics"], "activity": result["activity"], "score": candidate.score, "resources": candidate.net.resources()})
            save(checkpoint, state)
        for generation in range(state["generation"], 12):
            tick = time.time(); children = []
            for child_index in range(6):
                parent = state["candidates"][int(state["rng"].integers(6))]; pid = 6 + generation * 6 + child_index
                candidate = offspring(parent, pid, full=arm_index == 0)
                result = evaluate_candidate(candidate, dev, state, f"{arm}_candidate{pid:03d}", outdir, state["baseline_dev_prediction"], state["agenda"])
                children.append(candidate); candidate.net.save(outdir / f"{arm}_graph{pid:03d}.npz")
                state["lineage"].append({"arm": arm, "proposal_id": pid, "parent_id": candidate.parent_id, "actions": candidate.actions,
                    "stable_ids": candidate.stable_ids, "metrics": result["metrics"], "activity": result["activity"], "score": candidate.score, "resources": candidate.net.resources()})
            state["candidates"] = select_candidates(state["candidates"] + children, state["rng"])
            state["generation"] = generation + 1
            state["curves"].append({"arm": arm, "generation": generation + 1,
                "retained": [{"id": c.proposal_id, "n": c.net.n, "age": c.age, "slot": "behavior" if j < 4 else "exploration", "score": c.score} for j, c in enumerate(state["candidates"])],
                "seconds": time.time() - tick, "simulation_videos_cumulative": state["simulation_videos"]})
            write_json(outdir / "curves.json", state["curves"]); write_json(outdir / "lineage.json", state["lineage"]); save(checkpoint, state)
            print(arm, "代", generation + 1, "最大修复", max(c.score["target_repairs"] for c in state["candidates"]), flush=True)
        frozen = sorted(state["candidates"], key=lambda c: (-c.score["target_repairs"], c.score["damage"], -c.score["total_repairs"], c.proposal_id))[:3]
        if "baseline_confirmation" not in state:
            state["baseline_confirmation"] = evaluate_net(state["initial"], conf, state, "initial_confirmation", outdir)
            state["baseline_conf_prediction"] = prediction(outdir, "initial_confirmation")
        confirmations = []
        for original in frozen:
            c = original.clone(); name = f"{arm}_confirmation{c.proposal_id:03d}"
            result = evaluate_candidate(c, conf, state, name, outdir, state["baseline_conf_prediction"], state["agenda"])
            gate = graduation(result, state["baseline_confirmation"], c, state["baseline_conf_prediction"], conf, state["agenda"])
            silence = c.net.clone(); silence.out[128:] = 0; silence.rec[128:] = 0; sync_masks(silence)
            silence_name = f"{arm}_silenced{c.proposal_id:03d}"; silenced = evaluate_net(silence, conf, state, silence_name, outdir)
            before = repair_evidence(c.prediction, state["baseline_conf_prediction"], conf, state["agenda"])
            after = repair_evidence(prediction(outdir, silence_name), state["baseline_conf_prediction"], conf, state["agenda"])
            gate.update(repair_before_silence=before, repair_after_silence=after, silence_damages_target_repairs=after["videos"] < before["videos"])
            gate["graduated"] = bool(gate["qualified_before_silence"] and gate["silence_damages_target_repairs"])
            confirmations.append({"proposal_id": c.proposal_id, "result": result, "gate": gate, "silenced": silenced})
            save(outdir / f"{arm}_frozen{c.proposal_id:03d}.pkl", c)
        state[arm + "_confirmations"] = confirmations; save(checkpoint, state)
    state["completed"] = True; state["wall_seconds"] = time.time() - state["started"]; state["peak_working_set_bytes"] = peak_memory()
    write_json(outdir / "summary.json", {k: v for k, v in state.items() if k not in ("initial", "candidates", "starting_candidates", "rng", "baseline_dev_prediction", "baseline_conf_prediction")})
    save(checkpoint, state); print("B2完成；实际视频次", state["simulation_videos"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--run", default="experiments/runs/community_growth_d1_b2"); parser.add_argument("--resume", action="store_true")
    try: run(parser.parse_args())
    except BudgetExhausted as exc: print(str(exc), flush=True)
