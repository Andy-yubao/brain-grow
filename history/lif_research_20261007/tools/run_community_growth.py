"""B1完整有限多代发育诊断。"""
from __future__ import annotations
import os
for name in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):os.environ[name]="1"
import argparse, json, pickle, shutil, time, subprocess, hashlib
from pathlib import Path
import numpy as np
from braingrow.data.dynamic import prepare
from braingrow.network.growing import GrowingNetwork
from braingrow.network.discrete import sync_masks
from braingrow.network.communities import born_candidate, offspring, behavior_score, select_candidates
from tools.run_discrete_growth import subset, evaluate_net, agendas_from, save, write_json, peak_memory,restore_cost_progress


def evaluate_candidate(c,pool,state,name,outdir,baseline_prediction,agenda):
    result=evaluate_net(c.net,pool,state,name,outdir)
    with np.load(outdir/(name+"_outputs.npz")) as z:
        potential=z["potential"]
    c.prediction=np.column_stack((potential[:,:10].argmax(axis=1),potential[:,10:].argmax(axis=1)))
    behavior_score(c,baseline_prediction,pool[1],pool[2],agenda)
    return result


def graduation(result,baseline,c,baseline_prediction,pool,agenda):
    metrics=result["metrics"];bm=baseline["metrics"]
    nondecrease=all(metrics[k]>=bm[k] for k in ("digit_accuracy","motion_accuracy","joint_accuracy"))
    improvement=max(metrics[k]-bm[k] for k in ("digit_accuracy","motion_accuracy"))>=.05 and metrics["joint_accuracy"]-bm["joint_accuracy"]>=.02
    class_decrease=[]
    for key in ("digit_confusion","motion_confusion"):
        cm=np.array(metrics[key]);base=np.array(bm[key]);class_decrease.extend((np.diag(cm)/cm.sum(axis=1)-np.diag(base)/base.sum(axis=1)).tolist())
    g,target,r=agenda;y=pool[1];ids=pool[2]
    repair=(y[:,g]==target)&(baseline_prediction[:,g]==r)&(c.prediction[:,g]==target)
    repair_ids=sorted(set(ids[repair].tolist()))
    return {"qualified_before_silence":bool(nondecrease and improvement and len(repair_ids)>=2 and min(class_decrease)>=-.05),
        "nondecrease":nondecrease,"improvement_gate":improvement,"repair_ids":repair_ids,"minimum_class_delta":min(class_decrease)}


def run(args):
    outdir=Path(args.run);outdir.mkdir(parents=True,exist_ok=True);checkpoint=outdir/"resume.pkl"
    data=prepare();dev=subset(data,2);conf=subset(data,2,True)
    if args.resume:
        state=pickle.loads(checkpoint.read_bytes());restore_cost_progress(state,outdir)
    else:
        if checkpoint.exists():raise ValueError("目录已有状态，请使用--resume")
        state={"started":time.time(),"simulation_videos":0,"metric_evaluations":0,"arm_index":0,"generation":0,"curves":[],"lineage":[],"completed":False,
            "rng":np.random.default_rng(np.random.SeedSequence([11,6]))}
        initial=GrowingNetwork.load("experiments/runs/discrete_growth_d1/common_initial.npz")
        state["initial"]=initial
        write_json(outdir/"config.json",{"mechanism":"B1_communities","model_root_seed":11,"data_seed":20261004,"initial_lif":128,
            "initial_candidates":6,"initial_new_nodes":8,"generations":12,"children_each_generation":6,"development_videos":560,"confirmation_videos":560,
            "max_confirmed_candidates_each_arm":3,"max_simulation_videos":98560,"sampling":"same_fixed_v5_training_ID_pools","validation_reads":0,"sealed_test_reads":0,
            "environment":"local_venv_CPU","dependencies_added":[],"start_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
            "data_hashes":{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ("data/dynamic_v5.npz","data/manifest_v5.json")}})
        for source in ("src","tools"):shutil.copytree(source,outdir/"source_snapshot"/source,ignore=shutil.ignore_patterns("__pycache__"),dirs_exist_ok=True)
        (outdir/"start_diff.patch").write_text(subprocess.check_output(["git","diff"],text=True,encoding="utf-8"),encoding="utf-8")
        state["baseline_development"]=evaluate_net(initial,dev,state,"initial_development",outdir)
        with np.load(outdir/"initial_development_outputs.npz") as z:
            p=z["potential"];state["baseline_dev_prediction"]=np.column_stack((p[:,:10].argmax(axis=1),p[:,10:].argmax(axis=1)))
        state["agenda"]=agendas_from(state["baseline_development"]["metrics"])[0]
        state["starting_candidates"]=[born_candidate(initial,j) for j in range(6)]
        state["candidates"]=[]
        save(checkpoint,state)
    if state["completed"]:print("B阶段已完成，未重复训练",flush=True);return
    for arm_index in range(state["arm_index"],2):
        arm="full" if arm_index==0 else "output_only"
        if state["arm_index"]!=arm_index:
            state["arm_index"]=arm_index;state["generation"]=0;state["candidates"]=[]
            state["rng"]=np.random.default_rng(np.random.SeedSequence([11,6]))
        if not state["candidates"]:
            for start in state["starting_candidates"]:
                c=start.clone();result=evaluate_candidate(c,dev,state,f"{arm}_candidate{c.proposal_id:03d}",outdir,state["baseline_dev_prediction"],state["agenda"])
                state["candidates"].append(c);c.net.save(outdir/f"{arm}_graph{c.proposal_id:03d}.npz")
                state["lineage"].append({"arm":arm,"proposal_id":c.proposal_id,"parent_id":None,"actions":[],"stable_ids":c.stable_ids,"metrics":result["metrics"],"score":c.score,"resources":c.net.resources()})
            save(checkpoint,state)
        for generation in range(state["generation"],12):
            tick=time.time();children=[]
            for child_index in range(6):
                parent=state["candidates"][int(state["rng"].integers(6))]
                proposal_id=6+generation*6+child_index
                c=offspring(parent,proposal_id,full=arm_index==0)
                result=evaluate_candidate(c,dev,state,f"{arm}_candidate{proposal_id:03d}",outdir,state["baseline_dev_prediction"],state["agenda"])
                children.append(c);c.net.save(outdir/f"{arm}_graph{proposal_id:03d}.npz")
                state["lineage"].append({"arm":arm,"proposal_id":proposal_id,"parent_id":c.parent_id,"actions":c.actions,"stable_ids":c.stable_ids,
                    "metrics":result["metrics"],"activity":result["activity"],"score":c.score,"resources":c.net.resources()})
            state["candidates"]=select_candidates(state["candidates"]+children,state["rng"])
            state["generation"]=generation+1
            curve={"arm":arm,"generation":generation+1,"retained":[{"id":c.proposal_id,"n":c.net.n,"age":c.age,"score":c.score} for c in state["candidates"]],
                "seconds":time.time()-tick,"simulation_videos_cumulative":state["simulation_videos"]}
            state["curves"].append(curve);write_json(outdir/"curves.json",state["curves"]);write_json(outdir/"lineage.json",state["lineage"])
            save(checkpoint,state)
            print(json.dumps({"arm":arm,"generation":generation+1,"best_target_repairs":max(c.score["target_repairs"] for c in state["candidates"]),
                "best_damage":min(c.score["damage"] for c in state["candidates"]),"seconds":round(time.time()-tick,2)},ensure_ascii=False),flush=True)
        frozen=sorted(state["candidates"],key=lambda c:(-c.score["target_repairs"],c.score["damage"],-c.score["total_repairs"],c.proposal_id))[:3]
        # 冻结排序后才第一次打开确认池，初图公共基线此时计算。
        if "baseline_confirmation" not in state:
            state["baseline_confirmation"]=evaluate_net(state["initial"],conf,state,"initial_confirmation",outdir)
            with np.load(outdir/"initial_confirmation_outputs.npz") as z:
                p=z["potential"];state["baseline_conf_prediction"]=np.column_stack((p[:,:10].argmax(axis=1),p[:,10:].argmax(axis=1)))
        confirmations=[]
        for c in frozen:
            c=c.clone()
            result=evaluate_candidate(c,conf,state,f"{arm}_confirmation{c.proposal_id:03d}",outdir,state["baseline_conf_prediction"],state["agenda"])
            gate=graduation(result,state["baseline_confirmation"],c,state["baseline_conf_prediction"],conf,state["agenda"])
            silence=c.net.clone();silence.out[128:]=0;silence.rec[128:]=0;sync_masks(silence)
            silenced=evaluate_net(silence,conf,state,f"{arm}_silenced{c.proposal_id:03d}",outdir)
            gate["silence_damages_joint"]=silenced["metrics"]["joint_accuracy"]<result["metrics"]["joint_accuracy"]
            gate["graduated"]=bool(gate["qualified_before_silence"] and gate["silence_damages_joint"])
            confirmations.append({"proposal_id":c.proposal_id,"result":result,"gate":gate,"silenced":silenced})
            save(outdir/f"{arm}_frozen{c.proposal_id:03d}.pkl",c)
        state[f"{arm}_confirmations"]=confirmations;save(checkpoint,state)
    state["completed"]=True;state["wall_seconds"]=time.time()-state["started"];state["peak_working_set_bytes"]=peak_memory()
    summary={k:v for k,v in state.items() if k not in ("initial","candidates","starting_candidates","rng","baseline_dev_prediction","baseline_conf_prediction")}
    write_json(outdir/"summary.json",summary);save(checkpoint,state);print("B阶段完成，全部候选/谱系/确认已保存",flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--run",default="experiments/runs/community_growth_d1");parser.add_argument("--resume",action="store_true")
    run(parser.parse_args())


if __name__=="__main__":main()
