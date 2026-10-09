"""A2登记后的固定预算复核；完成检查点在加载数据前退出。"""
from __future__ import annotations
import os
for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"): os.environ[name] = "1"
import argparse, pickle, time
from pathlib import Path
import numpy as np
from braingrow.network.growing import GrowingNetwork
from braingrow.network.discrete import sync_masks
from braingrow.network.discrete_v2 import EventLearner
from tools.run_discrete_growth import agendas_from
from tools.growth_v2_common import training_data, subset, evaluate_net, record_cost, restore_cost, provenance, prediction, save, write_json, peak_memory, ensure_budget, BudgetExhausted


def demand(pred, pool, agenda):
    g, target, _ = agenda; y, ids = pool[1:]
    wrong = (y[:, g] == target) & (pred[:, g] != target)
    return {"target_errors": int(wrong.sum()), "error_ids": sorted(set(ids[wrong].tolist())), "unresolved": len(set(ids[wrong].tolist())) >= 3}


def run(args):
    outdir = Path(args.run); outdir.mkdir(parents=True, exist_ok=True); checkpoint = outdir / "resume.pkl"
    if args.resume:
        state = pickle.loads(checkpoint.read_bytes()); restore_cost(state, outdir)
        if state["completed"]: print("A2阶段已完成，未加载数据或重训", flush=True); return
        if state.get("stopped_budget"): print("A2实际预算已耗尽，未加载数据或重训", flush=True); return
    else:
        if checkpoint.exists(): raise ValueError("已有检查点，请使用--resume")
        state = {"started": time.time(), "simulation_videos": 0, "metric_evaluations": 0, "arm_index": 0,
            "epoch": 0, "curves": [], "demands": [], "completed": False, "simulation_budget": 40320}
        provenance(outdir, {"mechanism": "A2_H_M_F_C_all_negative", "epochs": 10, "videos_each_pool": 840,
            "initial_lif": 128, "window_each_side": 16, "minimum_ids_each_side": 3, "maximum_records_each_id": 6,
            "repeated_windows": 2, "update_probability": .1, "event_repetition_minimum": 2,
            "birth_opportunity_epochs": [2, 4], "birth_nodes_each_batch": 4, "birth_protection_qualified_windows": 4,
            "phase_max_simulation_videos": 40320, "D0_inherited_from": "experiments/runs/discrete_growth_d1/common_initial.npz",
            "new_D0_simulation_videos": 0, "multiple_corrections_no_individual_attribution": True})
    data = training_data(); development = subset(data); confirmation = subset(data, confirm=True)
    if "initial" not in state:
        initial = GrowingNetwork.load("experiments/runs/discrete_growth_d1/common_initial.npz")
        initial.save(outdir / "common_initial.npz")
        base = evaluate_net(initial, development, state, "initial_development", outdir)
        main, agendas, competitors = agendas_from(base["metrics"])
        state.update(initial=initial, main=main, agendas=agendas, competitors=competitors, baseline_development=base,
            learner=EventLearner(initial.clone(), agendas.copy(), competitors.copy(), full=True))
        save(checkpoint, state)
    x, y, ids = development
    for arm_index in range(state["arm_index"], 2):
        arm = "full" if arm_index == 0 else "output_only"
        if state["arm_index"] != arm_index:
            state.update(arm_index=arm_index, epoch=0, learner=EventLearner(state["initial"].clone(), state["agendas"].copy(), state["competitors"].copy(), full=False))
            save(checkpoint, state)
        learner = state["learner"]
        for epoch in range(state["epoch"], 10):
            tick = time.time()
            for begin in range(0, len(x), 30):
                ensure_budget(state, len(x[begin:begin + 30]), outdir)
                learner.learn_batch(x[begin:begin + 30], y[begin:begin + 30], ids[begin:begin + 30],
                    cost_callback=lambda n: record_cost(state, n, "learning_" + arm, outdir))
            label = f"{arm}_epoch{epoch + 1:02d}"
            result = evaluate_net(learner.net, development, state, label, outdir)
            if arm_index == 0:
                current = demand(prediction(outdir, label), development, state["main"])
                state["demands"].append({"epoch": epoch + 1, **current})
                previous = state["demands"][-2] if len(state["demands"]) >= 2 else None
                eligible = any(w.windows >= 2 for i, w in enumerate(learner.hidden_windows) if tuple(learner.agendas[i]) == tuple(state["main"]))
                if epoch + 1 in (2, 4) and previous and previous["unresolved"] and current["unresolved"] and eligible:
                    g, target, _ = state["main"]
                    cm = np.array(result["metrics"]["digit_confusion" if g == 0 else "motion_confusion"])
                    cm[target, target] = -1
                    learner.birth((g, target, int(cm[target].argmax())))
            curve = {"arm": arm, "epoch": epoch + 1, "metrics": result["metrics"], "activity": result["activity"],
                "resources_after_management": learner.net.resources(), "birth_batches": learner.birth_batches,
                "protection": learner.protection_status(), "hidden_windows_range": [min(w.windows for w in learner.hidden_windows), max(w.windows for w in learner.hidden_windows)],
                "output_windows_each_class": [w.windows for w in learner.output_windows], "elapsed_epoch_seconds": time.time() - tick,
                "simulation_videos_cumulative": state["simulation_videos"]}
            state["curves"].append(curve); state["epoch"] = epoch + 1
            learner.net.save(outdir / (label + ".npz")); write_json(outdir / (arm + "_events.json"), learner.events)
            write_json(outdir / "curves.json", state["curves"]); save(checkpoint, state)
            print(arm, epoch + 1, result["metrics"]["digit_accuracy"], result["metrics"]["motion_accuracy"], result["metrics"]["joint_accuracy"], learner.net.n, flush=True)
        state[arm + "_confirmation"] = evaluate_net(learner.net, confirmation, state, arm + "_confirmation", outdir)
        learner.net.save(outdir / (arm + "_final.npz")); save(outdir / (arm + "_learner.pkl"), learner)
        if arm_index == 0:
            interventions = {}
            for kind in ("birth_silenced", "input_removed", "internal_removed", "output_removed"):
                net = learner.net.clone()
                if kind == "birth_silenced": net.out[128:] = 0; net.rec[128:] = 0
                elif kind == "input_removed": net.win[:] = 0
                elif kind == "internal_removed": net.rec[:] = 0
                else: net.out[:] = 0
                sync_masks(net); interventions[kind] = evaluate_net(net, confirmation, state, kind, outdir)
            state["interventions"] = interventions
        save(checkpoint, state)
    state["initial_confirmation"] = evaluate_net(state["initial"], confirmation, state, "initial_confirmation", outdir)
    base = state["initial_confirmation"]["metrics"]; full = state["full_confirmation"]["metrics"]; control = state["output_only_confirmation"]["metrics"]
    eligible_tasks = [key for key in ("digit_accuracy", "motion_accuracy") if full[key] - base[key] >= .05 and full[key] - control[key] >= .03]
    baseline_pred = prediction(outdir, "initial_confirmation"); full_pred = prediction(outdir, "full_confirmation")
    ids_repaired = {key: sorted(set(confirmation[2][(full_pred[:, g] == confirmation[1][:, g]) & (baseline_pred[:, g] != confirmation[1][:, g])].tolist())) for g, key in enumerate(("digit_accuracy", "motion_accuracy"))}
    state["continuation_gate"] = {"passed": bool(all(full[k] >= base[k] for k in ("digit_accuracy", "motion_accuracy")) and full["joint_accuracy"] - base["joint_accuracy"] >= .02 and any(len(ids_repaired[k]) >= 5 for k in eligible_tasks)),
        "ids_with_repair": ids_repaired, "eligible_improved_tasks": eligible_tasks}
    failed = pickle.loads((outdir / "full_learner.pkl").read_bytes())
    restored = EventLearner(state["initial"].clone(), state["agendas"].copy(), state["competitors"].copy(), full=True)
    restored.management_rng = failed.management_rng
    save(outdir / "restored_stage_start.pkl", {"learner": restored, "simulation_videos_cumulative": state["simulation_videos"],
        "failed_candidate_path": "full_learner.pkl", "activated": not state["continuation_gate"]["passed"]})
    state["completed"] = True; state["wall_seconds"] = time.time() - state["started"]; state["peak_working_set_bytes"] = peak_memory()
    write_json(outdir / "summary.json", {k: v for k, v in state.items() if k not in ("initial", "learner", "agendas", "competitors")}); save(checkpoint, state)
    print("A2完成；继续门槛", state["continuation_gate"]["passed"], "实际视频次", state["simulation_videos"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--run", default="experiments/runs/discrete_growth_d1_a2"); parser.add_argument("--resume", action="store_true")
    try: run(parser.parse_args())
    except BudgetExhausted as exc: print(str(exc), flush=True)
