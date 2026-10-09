"""自然任务删边+精确物理闭合处于同一事务；无指定连接数目标。"""
from __future__ import annotations
import gc
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from tools.train_capacity_baseline import ROOT,augment,digest,save_json
from tools.train_capacity_pruning import export_model,load_export,sync
from tools.train_capacity_natural import (make_model,widths,structure,snapshot as base_snapshot,recover as base_recover,train_batches,predict,
                                         functional_test,apply_edges,measured_eval,acceptable)
from tools.clean_capacity_reachability import clean_graph,compare_logits
from tools.measure_capacity_deployment import memory_info
from braingrow.datasets import load_bundle


def snapshot(model,optimizer,ema):
    saved=base_snapshot(model,optimizer,ema)
    saved["training_mode"]=model.training
    return saved


def recover(model,optimizer,ema,saved,cfg):
    model,optimizer,ema=base_recover(model,optimizer,ema,saved,cfg)
    model.train(saved["training_mode"])
    return model,optimizer,ema


def propose_edges(model,ema,slot,hx,hy,cfg,ledger,random_count=None):
    started=time.perf_counter()
    layer=(slot-1)%len(model.layers)
    alive=model.masks[layer].flatten() & (model.layers[layer].weight.flatten()!=0)
    event={"layer":layer,"kind":"edge_with_exact_closure","deleted":[],"trials":[],"proposal_seconds":0.}
    actual=int(alive.sum())
    if random_count is not None:
        event["matching_available"]=actual>=random_count
        if not event["matching_available"]: return event
        ids=torch.nonzero(alive,as_tuple=False).flatten()
        stream=torch.Generator(device=alive.device).manual_seed(cfg["model_seed"]+8000+slot)
        event["deleted"]=ids[torch.randperm(len(ids),generator=stream,device=alive.device)[:random_count]].cpu().tolist()
        event["proposal_seconds"]=time.perf_counter()-started
        return event
    maximum=min(actual,max(cfg["edge_group_sizes"]))
    if not maximum: return event
    rank=ema[layer].flatten().masked_fill(~alive,float("inf"))
    cutoff=torch.topk(rank,maximum,largest=False,sorted=False).values.max()
    lower=torch.nonzero(rank<cutoff,as_tuple=False).flatten()
    equal=torch.nonzero(rank==cutoff,as_tuple=False).flatten()
    ids=torch.cat((lower,equal[:maximum-len(lower)])).sort().values
    ids=ids[torch.argsort(rank[ids],stable=True)]
    values=model.layers[layer].weight.flatten()[ids].detach().clone()
    before=predict(model,hx,cfg["evaluation_batch_size"])
    ledger["proposal_function_samples"]+=len(hx)
    for count in sorted(set(min(maximum,n) for n in cfg["edge_group_sizes"])):
        with torch.no_grad():
            model.layers[layer].weight.flatten()[ids[:count]]=0
            model.masks[layer].flatten()[ids[:count]]=False
        after=predict(model,hx,cfg["evaluation_batch_size"])
        ledger["proposal_function_samples"]+=len(hx)
        measured=functional_test(before,after,hy,cfg)
        with torch.no_grad():
            model.layers[layer].weight.flatten()[ids]=values
            model.masks[layer].flatten()[ids]=True
        event["trials"].append({"count":count,**measured})
        if not measured["pass"]: break
        event["deleted"]=ids[:count].cpu().tolist()
        if time.perf_counter()-started>cfg["proposal_seconds_per_slot"]: raise TimeoutError("训练功能提议槽预算")
    event["proposal_seconds"]=time.perf_counter()-started
    if event["proposal_seconds"]>cfg["proposal_seconds_per_slot"]: raise TimeoutError("提议槽预算")
    return event


def close_candidate(model,ema,probe_x,cfg,ledger,path=None):
    started=time.perf_counter()
    device=next(model.parameters()).device
    before_struct=structure(model)
    before=predict(model,probe_x,cfg["evaluation_batch_size"]).detach().cpu().numpy()
    node_ids=[getattr(model,f"node_ids_{i}").cpu().tolist() for i in range(len(model.layers)-1)]
    original_widths=list(model.original_widths)
    compact,certificate=clean_graph(model.cpu(),node_ids)
    result=make_model([compact.layers[0].in_features]+[l.out_features for l in compact.layers],model.dropout,torch.device("cpu"),original_widths)
    result.load_state_dict(compact.state_dict(),strict=False)
    with torch.no_grad():
        for i,row in enumerate(certificate):
            getattr(result,f"node_ids_{i}").copy_(torch.tensor(row["surviving_original_ids"],dtype=torch.long))
    keep=[torch.ones(model.layers[0].in_features,dtype=torch.bool)]
    for ids,row in zip(node_ids,certificate):
        keep.append(torch.from_numpy(np.isin(np.asarray(ids),np.asarray(row["surviving_original_ids"]))))
    keep.append(torch.ones(model.layers[-1].out_features,dtype=torch.bool))
    new_ema=[e.detach().cpu()[keep[i+1]][:,keep[i]].clone().to(device) for i,e in enumerate(ema)]
    result.to(device)
    after=predict(result,probe_x,cfg["evaluation_batch_size"]).detach().cpu().numpy()
    comparison,errors=compare_logits(before,after,cfg)
    ledger["closure_evaluation_calls"]+=2
    ledger["closure_evaluation_samples"]+=2*len(probe_x)
    if path is not None:
        np.savez_compressed(path,before_logits=before,after_logits=after,max_abs_per_sample=errors)
    closed_struct=structure(result)
    closure={"before":before_struct,"after":closed_struct,"certificate":certificate,"comparison":comparison,
             "closed_deleted_per_layer":[a-b for a,b in zip(before_struct["nonzero_edges"],closed_struct["nonzero_edges"])],
             "seconds":time.perf_counter()-started}
    if path is not None:
        save_json(path.with_suffix(".json"),closure if comparison["finite"] else {"error":"闭合输出非有限","certificate":certificate})
    if not comparison["pass"]: raise ValueError("候选闭合前后输出不等价，立即停轮")
    del compact
    return result,new_ema,closure


def persist_identity(event,arm,slot):
    ids=np.asarray(event["deleted"],dtype=np.int32)
    if not len(ids): return
    path=arm/f"slot_{slot}_preclosure_deleted_edges.npz"
    np.savez_compressed(path,indices=ids,layer=np.int32(event["layer"]))
    event["deleted_indices_evidence"]={"file":path.name,"count":len(ids),"dtype":"int32","sha256":digest(path)}


def save_identity(event,arm,slot):
    if event["deleted"] and "deleted_indices_evidence" not in event: persist_identity(event,arm,slot)
    event["deleted_count"]=len(event.pop("deleted"))


def model_from_mother(source,meta,dropout,device):
    loaded=load_export(source)
    model=make_model(meta["widths"],dropout,device,meta["original_widths"])
    model.load_state_dict(loaded.state_dict(),strict=False)
    with torch.no_grad():
        for i,ids in enumerate(meta["node_ids"]):
            getattr(model,f"node_ids_{i}").copy_(torch.tensor(ids,device=device,dtype=torch.long))
    return model


def run_arm(name,cfg,base,data,device,folder,source,meta,fixed=None,main=None,session_started=None):
    arm=folder/name; arm.mkdir()
    model=model_from_mother(source,meta,base["dropout"],device)
    ema=[torch.zeros_like(l.weight) for l in model.layers]
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"])
    x,y=torch.from_numpy(data.fit_x).to(device),torch.from_numpy(data.fit_y).to(device)
    sx,sy=torch.from_numpy(data.select_x).to(device),torch.from_numpy(data.select_y).to(device)
    hi=[]
    for label in range(10):
        ids=np.flatnonzero(data.fit_y==label); hi.extend(ids[cfg["proposal_fit_per_class"]:])
    hx,hy=x[hi],y[hi]
    ledger={k:0 for k in ("attempted_updates","net_updates","optimization_samples","evaluation_calls","evaluation_samples",
                          "proposal_function_samples","closure_evaluation_calls","closure_evaluation_samples","rollbacks","candidate_adaptations")}
    rows,events,slot_reference=[],[],{}
    started=time.perf_counter(); stop_reason="finite_budget"; comparable=True; collapse_streak=0
    limit=cfg["epochs"] if main is None else len(main["rows"])
    if device.type=="cuda": torch.cuda.reset_peak_memory_stats()
    for additional in range(1,limit+1):
        epoch=cfg["parent_epoch"]+additional
        torch.manual_seed(cfg["model_seed"]+2000+epoch)
        stream=torch.Generator(device=device).manual_seed(cfg["model_seed"]+3000+epoch)
        order=torch.randperm(len(x),device=device,generator=stream)
        batches=[(augment(x[ids],stream,base),y[ids]) for ids in order.split(cfg["batch_size"])]
        event=None; edit=additional in cfg["edit_epochs"]
        if edit:
            del optimizer
            optimizer=torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"])
            optimizer.zero_grad(set_to_none=True)
            slot=cfg["edit_epochs"].index(additional)+1
            before_struct=structure(model)
            if name=="task":
                event=propose_edges(model,ema,slot,hx,hy,cfg,ledger)
                event.update({"epoch":epoch,"additional_epoch":additional,"accepted":False,"before":before_struct})
                if event["deleted"]:
                    persist_identity(event,arm,slot)
                    saved=snapshot(model,optimizer,ema)
                    original_loss=train_batches(model,optimizer,ema,batches[:cfg["adaptation_batches"]],cfg,ledger)
                    original,original_logits=measured_eval(model,sx,sy,cfg,ledger)
                    original_state=snapshot(model,optimizer,ema)
                    model,optimizer,ema=recover(model,optimizer,ema,saved,cfg)
                    apply_edges(model,event)
                    preclosed=structure(model)
                    try:
                        new_model,new_ema,closure=close_candidate(model,ema,hx,cfg,ledger,arm/f"slot_{slot}_closure_equivalence.npz")
                    except Exception:
                        torch.save({"epoch":epoch,"state":saved,"original_four_batch_state":original_state},arm/f"slot_{slot}_failure_states.pt")
                        raise
                    del saved
                    del model,optimizer,ema
                    model,ema=new_model,new_ema
                    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"])
                    del new_model,new_ema
                    candidate_loss=train_batches(model,optimizer,ema,batches[:cfg["adaptation_batches"]],cfg,ledger)
                    candidate,candidate_logits=measured_eval(model,sx,sy,cfg,ledger)
                    after_struct=structure(model)
                    accepted=acceptable(candidate,original,fixed["slot_reference"][str(epoch)],before_struct["nonzero_edges"],after_struct["nonzero_edges"],cfg)
                    event.update({"accepted":accepted,"preclosed":preclosed,"closure":closure,"after":after_struct,
                                  "preclosure_deleted_per_layer":[a-b for a,b in zip(before_struct["nonzero_edges"],preclosed["nonzero_edges"])],
                                  "net_deleted_per_layer":[a-b for a,b in zip(before_struct["nonzero_edges"],after_struct["nonzero_edges"])],
                                  "original_select":original,"candidate_select":candidate})
                    np.savez_compressed(arm/f"slot_{slot}_validation_trials.npz",original=original_logits,candidate=candidate_logits,labels=data.select_y)
                    ledger["candidate_adaptations"]+=1
                    if accepted: loss_sum=candidate_loss
                    else:
                        model,optimizer,ema=recover(model,optimizer,ema,original_state,cfg)
                        loss_sum=original_loss; ledger["rollbacks"]+=1
                    del original_state
                else:
                    loss_sum=train_batches(model,optimizer,ema,batches[:cfg["adaptation_batches"]],cfg,ledger)
                save_identity(event,arm,slot); events.append(event)
            elif name=="random":
                source_event=next(e for e in main["events"] if e["additional_epoch"]==additional)
                if source_event["accepted"]:
                    event=propose_edges(model,ema,slot,hx,hy,cfg,ledger,sum(source_event["preclosure_deleted_per_layer"]))
                    event.update({"epoch":epoch,"additional_epoch":additional,"accepted":True,"before":before_struct})
                    if not event["matching_available"]:
                        comparable=False; stop_reason="insufficient_preclosure_alive_edges"; event["accepted"]=False
                        save_identity(event,arm,slot); events.append(event); break
                    persist_identity(event,arm,slot)
                    apply_edges(model,event); preclosed=structure(model)
                    new_model,new_ema,closure=close_candidate(model,ema,hx,cfg,ledger,arm/f"slot_{slot}_closure_equivalence.npz")
                    del model,optimizer,ema
                    model,ema=new_model,new_ema
                    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"])
                    del new_model,new_ema
                    event.update({"preclosed":preclosed,"closure":closure,
                                  "preclosure_deleted_per_layer":[a-b for a,b in zip(before_struct["nonzero_edges"],preclosed["nonzero_edges"])],
                                  "net_deleted_per_layer":[a-b for a,b in zip(before_struct["nonzero_edges"],closure["after"]["nonzero_edges"])],
                                  "matched_preclosure_counts":True})
                    event["matched_preclosure_counts"]=event["preclosure_deleted_per_layer"]==source_event["preclosure_deleted_per_layer"]
                    comparable=comparable and event["matched_preclosure_counts"]
                    save_identity(event,arm,slot); events.append(event)
                loss_sum=train_batches(model,optimizer,ema,batches[:cfg["adaptation_batches"]],cfg,ledger)
            else:
                loss_sum=train_batches(model,optimizer,ema,batches[:cfg["adaptation_batches"]],cfg,ledger)
                metric,_=measured_eval(model,sx,sy,cfg,ledger); slot_reference[str(epoch)]=metric
            loss_sum+=train_batches(model,optimizer,ema,batches[cfg["adaptation_batches"]:],cfg,ledger)
        else:
            loss_sum=train_batches(model,optimizer,ema,batches,cfg,ledger)
        ledger["net_updates"]+=len(batches)
        selected,logits=measured_eval(model,sx,sy,cfg,ledger)
        fitted,fit_logits=measured_eval(model,x,y,cfg,ledger)
        current=structure(model)
        rows.append({"epoch":epoch,"additional_epoch":additional,"select":selected,"fit":fitted,"structure":current,"fit_augmented_cross_entropy":loss_sum/len(x)})
        np.savez_compressed(arm/f"epoch_{epoch}_predictions.npz",validation=logits,fit=fit_logits,validation_labels=data.select_y,fit_labels=data.fit_y)
        if edit or additional==limit:
            stage=arm/f"epoch_{epoch}"; stage.mkdir()
            exported=export_model(model,stage/"model.bgm")
            node_meta={"widths":current["widths"],"original_widths":model.original_widths,"node_ids":[getattr(model,f"node_ids_{i}").cpu().tolist() for i in range(3)]}
            save_json(stage/"structure.json",{**current,"export":exported,**node_meta})
            torch.save({"epoch":epoch,"state":snapshot(model,optimizer,ema),"events":events,"ledger":ledger},arm/"resume.pt")
        save_json(arm/"progress.json",{"rows":rows,"events":events,"ledger":ledger,"slot_reference":slot_reference})
        print(json.dumps({"arm":name,"epoch":epoch,"validation_accuracy":selected["accuracy"],"nonzero_edges":sum(current["nonzero_edges"]),
                          "widths":current["widths"],"accepted":event["accepted"] if event else None}),flush=True)
        collapsed=fitted["accuracy"]<cfg["collapse_fit_accuracy"] and selected["accuracy"]<cfg["collapse_validation_accuracy"]
        collapse_streak=collapse_streak+1 if collapsed else 0
        if name=="task" and collapse_streak>=cfg["collapse_consecutive_epochs"]:
            stop_reason="severe_task_collapse"; break
        if time.perf_counter()-session_started>cfg["session_wall_seconds"]: raise TimeoutError("会话预算")
        if device.type=="cuda" and torch.cuda.max_memory_allocated()>cfg["peak_memory_limit_bytes"]: raise MemoryError("CUDA峰预算")
        del batches
    current=structure(model); final_export=export_model(model,arm/"model.bgm")
    final_meta={"widths":current["widths"],"original_widths":model.original_widths,"node_ids":[getattr(model,f"node_ids_{i}").cpu().tolist() for i in range(3)]}
    save_json(arm/"node_ids.json",final_meta)
    torch.save({"epoch":rows[-1]["epoch"],"state":snapshot(model,optimizer,ema),"events":events,"ledger":ledger},arm/"resume.pt")
    result={"rows":rows,"events":events,"slot_reference":slot_reference,"select":rows[-1]["select"],"fit":rows[-1]["fit"],"structure":current,"export":final_export,
            "deployment_bytes_including_ids":final_export["bytes"]+(arm/"node_ids.json").stat().st_size,"ledger":ledger,"stop_reason":stop_reason,
            "wall_seconds":time.perf_counter()-started,"peak_cuda_memory_bytes":torch.cuda.max_memory_allocated() if device.type=="cuda" else None,
            "process_memory_at_arm_end_cumulative":memory_info(),"random_preclosure_comparable":comparable if name=="random" else None}
    save_json(arm/"result.json",result)
    del model,optimizer,ema; gc.collect()
    if device.type=="cuda": torch.cuda.empty_cache()
    return result


def main():
    cfg_path=ROOT/"configs/h1_capacity_closed_pruning_v1.json"; cfg=json.loads(cfg_path.read_text(encoding="utf-8"))
    base=json.loads((ROOT/"configs/capacity_baseline_v1.json").read_text(encoding="utf-8"))
    parent=ROOT/"experiments/runs"/cfg["parent_version"]/"mnist_static"/cfg["parent_arm"]
    source=parent/"model.bgm"; meta=json.loads((parent/"node_ids.json").read_text(encoding="utf-8"))
    expected=json.loads((parent/"result.json").read_text(encoding="utf-8"))
    if digest(source)!=expected["export"]["sha256"] or sum(expected["after"]["nonzero_edges"])!=17592108 or meta["widths"]!=[784,6272,2081,256,10] or cfg["model_seed"]!=11 or cfg["data_seed"]!=20261004:
        raise ValueError("冻结母图/固定根不符")
    folder=ROOT/"experiments/runs"/cfg["version"]/"mnist_static"; folder.mkdir(parents=True,exist_ok=True)
    if any(folder.iterdir()): raise FileExistsError("拒绝覆盖或重复训练")
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(11); np.random.seed(11); torch.manual_seed(11)
    registration={"config":cfg,"base_config":base,"config_sha256":digest(cfg_path),"source_sha256":digest(Path(__file__)),
                  "mother_model_sha256":digest(source),"mother_node_ids_sha256":digest(parent/"node_ids.json"),"mother_structure":expected["after"],
                  "helper_sha256":{p:digest(ROOT/p) for p in ("tools/train_capacity_natural.py","tools/clean_capacity_reachability.py","tools/train_capacity_pruning.py")},
                  "training_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"device":str(device),
                  "device_name":torch.cuda.get_device_name() if device.type=="cuda" else "CPU","torch":torch.__version__,"precision":"FP32","tf32":False,
                  "cpu_threads":2,"execution":"dense_zero_mask_with_physical_reachability_closure","same_final_edge_count_random":False}
    save_json(folder/"registration.json",registration)
    data=load_bundle("mnist",include_confirm=False)
    if data.fit_x.shape!=(3200,784) or data.select_x.shape!=(800,784) or len(data.confirm_x): raise ValueError("数据权限")
    save_json(folder/"data_manifest.json",data.manifest)
    started=time.perf_counter(); results={}
    try:
        results["fixed"]=run_arm("fixed",cfg,base,data,device,folder,source,meta,session_started=started)
        results["task"]=run_arm("task",cfg,base,data,device,folder,source,meta,results["fixed"],session_started=started)
        results["random"]=run_arm("random",cfg,base,data,device,folder,source,meta,main=results["task"],session_started=started)
        confirmed=load_bundle("mnist",include_confirm=True)
        for name,r in results.items():
            model=load_export(folder/name/"model.bgm").to(device)
            measured,logits=measured_eval(model,torch.from_numpy(confirmed.confirm_x).to(device),torch.from_numpy(confirmed.confirm_y).to(device),cfg,r["ledger"])
            r["control"]=measured; np.savez_compressed(folder/name/"control_final.npz",logits=logits,labels=confirmed.confirm_y)
            save_json(folder/name/"result.json",r); del model
        save_json(folder/"summary.json",{"registration":registration,"results":results,"session_wall_seconds":time.perf_counter()-started,
                                        "official_test_accessed":False,"sealed_remaining_ids_accessed":False,"same_final_edge_count_random":False})
    except Exception as exc:
        save_json(folder/"failure.json",{"error":repr(exc),"results":results,"session_wall_seconds":time.perf_counter()-started}); raise


if __name__=="__main__": main()
