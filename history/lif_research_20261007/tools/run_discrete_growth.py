"""从项目根目录运行A小诊断；本地pickle只用于可信项目检查点。"""
from __future__ import annotations
import os
for _name in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):
    os.environ[_name]="1"
import argparse, hashlib, json, pickle, platform, shutil, subprocess, time
from pathlib import Path
import numpy as np
from braingrow.data.dynamic import prepare
from braingrow.evaluation.metrics import evaluate
from braingrow.network.discrete import DiscreteLearner, initial_network


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding="utf-8")


def save(path, state):
    pending=Path(str(path)+".pending")
    pending.write_bytes(pickle.dumps(state,protocol=5)); pending.replace(path)


def record_cost(state,videos,kind,outdir):
    """成本独立于模型轮次检查点提交，失败/重放不会回滚已耗前向。"""
    state["simulation_videos"]+=videos
    entry={"simulation_videos_cumulative":state["simulation_videos"],
        "metric_evaluations_cumulative":state["metric_evaluations"],"videos":videos,"kind":kind,"time":time.time()}
    with (outdir/"cost_journal.jsonl").open("a",encoding="utf-8") as handle:
        handle.write(json.dumps(entry,ensure_ascii=False)+"\n");handle.flush()


def restore_cost_progress(state,outdir):
    journal=outdir/"cost_journal.jsonl"
    if journal.exists():
        rows=journal.read_text(encoding="utf-8").splitlines()
        if rows:
            last=json.loads(rows[-1])
            state["simulation_videos"]=max(state["simulation_videos"],last["simulation_videos_cumulative"])
            state["metric_evaluations"]=max(state["metric_evaluations"],last["metric_evaluations_cumulative"])


def subset(data, count=3, confirm=False):
    manifest=json.loads(Path("data/manifest_v5.json").read_text(encoding="utf-8"))
    ids=manifest["splits"]["train"]
    selected=[value for digit in range(10) for value in ids[digit*20+(10 if confirm else 0):digit*20+(10 if confirm else 0)+count]]
    indices=np.flatnonzero(np.isin(data["train_ids"],selected))
    # 同ID的变体/运动穿插；早期16正/负事件跨三个原始ID。
    rank={value:i%count for i,value in enumerate(selected)}
    indices=sorted(indices,key=lambda k:(int(data["train_variants"][k]),int(data["train_y"][k,1]),rank[int(data["train_ids"][k])],int(data["train_y"][k,0])))
    indices=np.array(indices)
    return data["train_x"][indices], data["train_y"][indices], data["train_ids"][indices]


def evaluate_net(net, pool, state, name, outdir):
    x,y,ids=pool; ps=[]; frames=[]
    for begin in range(0,len(x),84):
        p,a,t=net.forward(x[begin:begin+84],trace=True); ps.append(p); frames.append(t)
    potential=np.concatenate(ps); trace=np.concatenate(frames)
    state["metric_evaluations"]+=1;record_cost(state,len(x),"evaluation_"+name,outdir)
    counts=trace.sum(axis=1)
    reconstructed=counts.astype(np.float64)@net.out.astype(np.float64)
    np.testing.assert_allclose(potential,reconstructed,atol=1e-4,rtol=1e-6)
    metrics=evaluate(potential,y)
    total=float(trace.sum())
    activity={"hidden_spikes":int(total),"output_spikes":0,"mean_spikes_per_frame":total/(len(x)*8),
        "mean_spikes_per_video":total/len(x),"spikes_per_lif_step":total/(len(x)*32*net.n),
        "per_frame_mean":trace.sum(axis=(0,2)).tolist(),"silent_units":int((counts.sum(axis=0)==0).sum())}
    activity["per_frame_mean"]=[v/len(x) for v in activity["per_frame_mean"]]
    result={"name":name,"metrics":metrics,"activity":activity,"resources":net.resources(),
        "unique_ids":len(set(ids.tolist())),"videos":len(x),
        "potential_reconstruction_max_absolute_error":float(np.abs(potential-reconstructed).max())}
    np.savez_compressed(outdir/(name+"_outputs.npz"),potential=potential,counts=counts,frame_counts=trace,y=y,ids=ids)
    write_json(outdir/(name+".json"),result)
    return result


def agendas_from(metrics, n=128):
    cms=[np.array(metrics["digit_confusion"]),np.array(metrics["motion_confusion"])]
    errors=[1-np.mean(np.diag(c)/c.sum(axis=1)) for c in cms]
    group=int(np.argmax(errors)); cm=cms[group].copy(); np.fill_diagonal(cm,-1)
    c,r=np.unravel_index(np.argmax(cm),cm.shape); main=(group,int(c),int(r))
    competitors=[]
    for cm in cms:
        cm=cm.copy(); np.fill_diagonal(cm,-1); competitors.extend(cm.argmax(axis=1).tolist())
    agendas=[main]*(n//2)
    for i in range(n-len(agendas)):
        target=i%17; g,c=(0,target) if target<10 else (1,target-10)
        agendas.append((g,c,competitors[target]))
    return main,agendas,competitors


def d0(data,state,outdir):
    x,y,ids=subset(data,1)
    # 第一变体，共70视频；另70零输入。
    x=x[:70]; net=initial_network(); records=[]
    for exposure in range(5):
        _,a,t=net.forward(x,record_steps=True)
        _,zero,_=net.forward(np.zeros_like(x),record_steps=True)
        record_cost(state,140,"D0_normal_and_zero",outdir)
        rates=t["spikes"].mean(axis=(0,1)); near=(t["membrane_before_reset"]>=.5*net.threshold).any(axis=(0,1))
        records.append({"exposure":exposure+1,"normal":a,"zero":zero,"silent":int((rates==0).sum()),
            "saturated":int((rates>.6).sum()),"without_plastic_opportunity":int((~near).sum()),
            "input_current":t["mean_absolute_input_current"],"recurrent_current":t["mean_absolute_recurrent_current"],
            "thresholds":np.unique(net.threshold,return_counts=True)[1].tolist()})
        if zero["hidden_spikes"]>0: raise RuntimeError("D0无输入自发循环，停止")
        if near.sum()>=net.n//2 and (rates>.6).sum()<net.n//2: break
        net.threshold[rates==0]=np.maximum(.25,net.threshold[rates==0]/2)
    write_json(outdir/"d0.json",records); net.save(outdir/"common_initial.npz")
    if (~near).sum()>net.n//2: raise RuntimeError("D0多数单元无法启动，停止")
    return net


def peak_memory():
    # Windows原生进程峰值工作集（不是模型数组/真实能耗）。
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_=[("cb",wintypes.DWORD),("PageFaultCount",wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in
            ("PeakWorkingSetSize","WorkingSetSize","QuotaPeakPagedPoolUsage","QuotaPagedPoolUsage","QuotaPeakNonPagedPoolUsage","QuotaNonPagedPoolUsage","PagefileUsage","PeakPagefileUsage")]
    c=Counters();c.cb=ctypes.sizeof(c)
    process=ctypes.windll.kernel32.GetCurrentProcess
    process.restype=wintypes.HANDLE
    measure=ctypes.windll.psapi.GetProcessMemoryInfo
    measure.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
    measure.restype=wintypes.BOOL
    if not measure(process(),ctypes.byref(c),c.cb):
        raise OSError("无法读取Windows进程工作集")
    return int(c.PeakWorkingSetSize)


def run(args):
    outdir=Path(args.run); outdir.mkdir(parents=True,exist_ok=True)
    checkpoint=outdir/"resume.pkl"
    data=prepare(); development=subset(data); confirmation=subset(data,confirm=True)
    if args.resume:
        state=pickle.loads(checkpoint.read_bytes())
        restore_cost_progress(state,outdir)
    else:
        if checkpoint.exists(): raise ValueError("目录已有检查点，请使用--resume")
        state={"simulation_videos":0,"metric_evaluations":0,"started":time.time(),"epoch":0,"arm_index":0,"curves":[],"completed":False}
        write_json(outdir/"config.json",{"mechanism":"A1_discrete_events","root_model_seed":11,"data_seed":20261004,
            "epochs":10,"development_ids":30,"confirmation_ids":30,"videos_per_pool":840,"batch":30,
            "window_each_side":16,"minimum_ids_each_side":3,"repeated_windows":2,"update_probability":.1,
            "birth_batches":2,"birth_batch_nodes":4,"birth_protection_qualified_windows":4,
            "phase_max_simulation_videos":58800,"D0_max_simulation_videos":700,
            "phase_stop":"确认任务均不降，一项+5pp且联合+2pp，完整相应任务优于仅输出3pp，否则停止A扩展",
            "data_hashes":{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ("data/dynamic_v5.npz","data/manifest_v5.json")},
            "start_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"device":"CPU","platform":platform.platform(),"numpy":np.__version__,
            "neural_time_ms":160,"video_time":"数据8帧无固定真实播放时长","dependencies_added":[]})
        (outdir/"start_diff.patch").write_text(subprocess.check_output(["git","diff"],text=True,encoding="utf-8"),encoding="utf-8")
        for source in ("src","tools"):
            shutil.copytree(source,outdir/"source_snapshot"/source,ignore=shutil.ignore_patterns("__pycache__"),dirs_exist_ok=True)
        initial=d0(data,state,outdir)
        base=evaluate_net(initial,development,state,"initial_development",outdir)
        main,agendas,competitors=agendas_from(base["metrics"])
        state.update(initial=initial,main=main,agendas=agendas,competitors=competitors,baseline_development=base)
        state["learner"]=DiscreteLearner(initial.clone(),agendas.copy(),competitors.copy(),full=True)
        save(checkpoint,state)
    if state["completed"]:
        print("阶段已完成；检查点未再次训练",flush=True); return
    x,y,ids=development
    for arm_index in range(state["arm_index"],2):
        arm="full" if arm_index==0 else "output_only"
        if state["arm_index"]!=arm_index:
            state["arm_index"]=arm_index;state["epoch"]=0
            state["learner"]=DiscreteLearner(state["initial"].clone(),state["agendas"].copy(),state["competitors"].copy(),full=False)
            save(checkpoint,state)
        learner=state["learner"]
        for epoch in range(state["epoch"],10):
            tick=time.time()
            for begin in range(0,len(x),30):
                learner.learn_batch(x[begin:begin+30],y[begin:begin+30],ids[begin:begin+30])
                record_cost(state,len(x[begin:begin+30]),"learning_"+arm,outdir)
            result=evaluate_net(learner.net,development,state,f"{arm}_epoch{epoch+1:02d}",outdir)
            main=state["main"]; eligible=[w for i,w in enumerate(learner.hidden_windows) if tuple(learner.agendas[i])==tuple(main) and w.windows>=2]
            cm=np.array(result["metrics"]["digit_confusion" if main[0]==0 else "motion_confusion"])
            if arm_index==0 and epoch+1 in (2,4) and eligible and cm[main[1],main[2]]>0:
                learner.birth(main)
            learner.net.save(outdir/f"{arm}_epoch{epoch+1:02d}.npz")
            curve={"arm":arm,"epoch":epoch+1,"metrics":result["metrics"],"activity":result["activity"],"resources_after_management":learner.net.resources(),
                "windows":sum(w.windows for w in learner.hidden_windows),"events":len(learner.events),"birth_batches":learner.birth_batches,
                "elapsed_epoch_seconds":time.time()-tick,"simulation_videos_cumulative":state["simulation_videos"]}
            state["curves"].append(curve);state["epoch"]=epoch+1
            write_json(outdir/"curves.json",state["curves"]);write_json(outdir/f"{arm}_events.json",learner.events)
            save(checkpoint,state)
            print(json.dumps({"arm":arm,"epoch":epoch+1,"digit":result["metrics"]["digit_accuracy"],"motion":result["metrics"]["motion_accuracy"],"joint":result["metrics"]["joint_accuracy"],"n":learner.net.n,"seconds":round(time.time()-tick,2)},ensure_ascii=False),flush=True)
        result=evaluate_net(learner.net,confirmation,state,f"{arm}_confirmation",outdir)
        learner.net.save(outdir/f"{arm}_final.npz")
        save(outdir/f"{arm}_learner.pkl",learner)
        state[f"{arm}_confirmation"]=result
        if arm_index==0:
            # 出生/输入/内部/输出三类功能干预；不据分差更新其它权重。
            interventions={}
            for kind in ("birth_silenced","input_removed","internal_removed","output_removed"):
                candidate=learner.net.clone()
                if kind=="birth_silenced":
                    candidate.out[128:]=0;candidate.rec[128:]=0
                elif kind=="input_removed": candidate.win[:]=0
                elif kind=="internal_removed": candidate.rec[:]=0
                else: candidate.out[:]=0
                from braingrow.network.discrete import sync_masks
                sync_masks(candidate)
                interventions[kind]=evaluate_net(candidate,confirmation,state,kind,outdir)
            state["interventions"]=interventions
        save(checkpoint,state)
    state["initial_confirmation"]=evaluate_net(state["initial"],confirmation,state,"initial_confirmation",outdir)
    state["completed"]=True;state["wall_seconds"]=time.time()-state["started"];state["peak_working_set_bytes"]=peak_memory()
    summary={k:v for k,v in state.items() if k not in ("initial","learner","agendas","competitors")}
    write_json(outdir/"summary.json",summary);save(checkpoint,state)
    print("A阶段完成，证据与完整恢复状态已保存",flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--run",default="experiments/runs/discrete_growth_d1");parser.add_argument("--resume",action="store_true")
    run(parser.parse_args())


if __name__=="__main__": main()
