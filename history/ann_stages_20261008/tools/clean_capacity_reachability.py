"""仅图的输出反向可达性清理；0训练，验证仅核验输出等价。"""
from __future__ import annotations
import gc
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from tools.train_capacity_baseline import ROOT, digest, evaluate, save_json
from tools.train_capacity_pruning import MaskedFC, load_export, export_model, sync
from tools.measure_capacity_deployment import memory_info


def structural_info(model):
    nonzero=[int(torch.count_nonzero(l.weight)) for l in model.layers]
    biases=sum(l.bias.numel() for l in model.layers)
    return {"widths":[model.layers[0].in_features]+[l.out_features for l in model.layers],
            "nonzero_edges":nonzero,"mask_slots":[int(m.sum()) for m in model.masks],"bias_count":biases,
            "independent_parameters":sum(nonzero)+biases,
            "tensor_numeric_bytes":sum(p.numel()*p.element_size() for p in model.parameters()),
            "tensor_mask_bytes":sum(m.numel()*m.element_size() for m in model.masks)}


def clean_graph(model,node_ids):
    if next(model.parameters()).device.type!="cpu": raise ValueError("图分析固定CPU")
    depth=len(model.layers)
    if len(node_ids)!=depth-1 or any(len(ids)!=model.layers[i].out_features for i,ids in enumerate(node_ids)):
        raise ValueError("原节点ID长度不符")
    if any(not torch.isfinite(l.weight).all() or not torch.isfinite(l.bias).all() for l in model.layers):
        raise FloatingPointError("源权重/bias非有限")
    reachable=[None]*(depth+1)
    reachable[depth]=torch.ones(model.layers[-1].out_features,dtype=torch.bool)
    certificate=[]
    for i in reversed(range(1,depth)):
        weight=model.layers[i].weight.detach()
        if not torch.isfinite(weight).all(): raise FloatingPointError("源权重非有限")
        reachable[i]=(weight[reachable[i+1]]!=0).any(0)
        direct=(weight!=0).any(0).logical_not()
        indirect=~reachable[i] & ~direct
        discarded=~reachable[i]
        if torch.count_nonzero(weight[reachable[i+1]][:,discarded]):
            raise ValueError("反向不可达证书违例")
        ids=np.asarray(node_ids[i-1],dtype=np.int32)
        row={"hidden_layer":i,"before_width":len(ids),"after_width":int(reachable[i].sum()),
             "direct_zero_outgoing_count":int(direct.sum()),"indirect_unreachable_count":int(indirect.sum()),
             "discarded_nonzero_bias_count":int(torch.count_nonzero(model.layers[i-1].bias[discarded])),
             "direct_zero_outgoing_original_ids":ids[direct.numpy()].tolist(),
             "indirect_unreachable_original_ids":ids[indirect.numpy()].tolist(),
             "surviving_original_ids":ids[reachable[i].numpy()].tolist(),"zero_to_reachable_certificate":True}
        certificate.append(row)
    reachable[0]=torch.ones(model.layers[0].in_features,dtype=torch.bool)
    new_widths=[int(r.sum()) for r in reachable]
    # 随机分配权重随即覆盖，不能改变既有模型/随机状态。
    with torch.random.fork_rng():
        cleaned=MaskedFC(new_widths,model.dropout)
    with torch.no_grad():
        for i,(old,new,mask,new_mask) in enumerate(zip(model.layers,cleaned.layers,model.masks,cleaned.masks)):
            expected_w=old.weight[reachable[i+1]][:,reachable[i]]
            expected_b=old.bias[reachable[i+1]]
            expected_m=mask[reachable[i+1]][:,reachable[i]]
            new.weight.copy_(expected_w); new.bias.copy_(expected_b); new_mask.copy_(expected_m)
            if not torch.equal(new.weight.view(torch.int32),expected_w.contiguous().view(torch.int32)) or not torch.equal(new.bias.view(torch.int32),expected_b.view(torch.int32)) or not torch.equal(new_mask,expected_m):
                raise ValueError("存活权重/bias/mask位模式改变")
    return cleaned,sorted(certificate,key=lambda row:row["hidden_layer"])


def compare_logits(reference,other,cfg):
    error=np.abs(reference.astype(np.float64)-other.astype(np.float64))
    bound=cfg["logit_atol"]+cfg["logit_rtol"]*np.abs(reference.astype(np.float64))
    finite=bool(np.isfinite(reference).all() and np.isfinite(other).all())
    within=(error<=bound).all(1)
    same=reference.argmax(1)==other.argmax(1)
    return {"pass":bool(finite and within.all() and same.all()),"finite":finite,
            "max_absolute_error":float(error.max()),"mean_absolute_error":float(error.mean()),
            "max_bound_ratio":float((error/bound).max()),"bitwise_equal":bool(np.array_equal(reference.view(np.uint32),other.view(np.uint32))),
            "allclose_sample_count":int(within.sum()),"argmax_equal_sample_count":int(same.sum()),"samples":len(reference)},error.max(1)


def run_arm(name,cfg,parent,expected,device,folder):
    started=time.perf_counter(); costs={}
    target=folder/name; target.mkdir()
    source=parent/name/"model.bgm"
    metadata_path=parent/name/f"epoch_{cfg['parent_epoch']}"/"structure.json"
    metadata=json.loads(metadata_path.read_text(encoding="utf-8"))
    if digest(source)!=expected["export"]["sha256"]: raise ValueError("原模型SHA不符")
    t=time.perf_counter(); model=load_export(source); costs["load_seconds"]=time.perf_counter()-t
    before=structural_info(model)
    if before["widths"]!=metadata["widths"] or before["nonzero_edges"]!=expected["structure"]["nonzero_edges"]:
        raise ValueError("精确重建结构不符")
    before_node_meta={"original_widths":[784,6272,3136,256,10],"widths":before["widths"],"node_ids":metadata["node_ids"],"sharing":"none"}
    save_json(target/"before_node_ids.json",before_node_meta)
    t=time.perf_counter(); cleaned,certificate=clean_graph(model,metadata["node_ids"]); costs["graph_and_slice_seconds"]=time.perf_counter()-t
    after=structural_info(cleaned)
    if any(a>b for a,b in zip(after["nonzero_edges"],before["nonzero_edges"])): raise ValueError("清理反而新增边")
    if device.type=="cuda": torch.cuda.reset_peak_memory_stats()
    with np.load(ROOT/"data/structure_mnist_v1/select.npz",allow_pickle=False) as data:
        x=torch.from_numpy(data["x"]).to(device); y=torch.from_numpy(data["y"]).to(device)
        labels=data["y"].copy()
    # 只反序列化原预测中validation字段，fit字段不读取。
    prediction_path=parent/name/f"epoch_{cfg['parent_epoch']}_predictions.npz"
    with np.load(prediction_path,allow_pickle=False) as saved:
        frozen=saved["validation"].copy()
        if not np.array_equal(saved["validation_labels"],labels): raise ValueError("验证标签对应异常")
    t=time.perf_counter(); model.to(device); before_metrics,before_logits=evaluate(model,x,y,cfg["evaluation_batch_size"]); sync(device)
    costs["before_evaluation_seconds"]=time.perf_counter()-t
    model.cpu(); gc.collect()
    t=time.perf_counter(); cleaned.to(device); after_metrics,after_logits=evaluate(cleaned,x,y,cfg["evaluation_batch_size"]); sync(device)
    costs["after_evaluation_seconds"]=time.perf_counter()-t
    comparisons={}; errors={}
    for key,a,b in (("before_after",before_logits,after_logits),("saved_before",frozen,before_logits),("saved_after",frozen,after_logits)):
        comparisons[key],errors[key]=compare_logits(a,b,cfg)
    # 即使超门槛，也先保存真实预测与失败证据。
    np.savez_compressed(target/"validation_equivalence.npz",before_logits=before_logits,after_logits=after_logits,frozen_logits=frozen,labels=labels,
                        **{f"{key}_max_abs_per_sample":v for key,v in errors.items()})
    save_json(target/"certificate.json",{"rule":"actual_nonzero_backward_reachability","layers":certificate,
                                          "original_widths":metadata["widths"],"original_node_ids":metadata["node_ids"]})
    if not all(v["pass"] for v in comparisons.values()):
        save_json(target/"failure.json",{"error":"输出一致性失败，后续立即停止","comparisons":comparisons,"before":before,"after":after,"costs":costs})
        raise ValueError("清理前后或冻结证据输出不一致")
    t=time.perf_counter(); cleaned.cpu(); exported=export_model(cleaned,target/"model.bgm")
    node_meta={"original_widths":before_node_meta["original_widths"],
               "widths":after["widths"],"node_ids":[r["surviving_original_ids"] for r in certificate],"sharing":"none"}
    save_json(target/"node_ids.json",node_meta)
    torch.save({"architecture":after["widths"],"state":cleaned.state_dict(),"node_ids":node_meta,"optimizer_state":None,"training_updates":0},target/"model_state.pt")
    costs["export_seconds"]=time.perf_counter()-t
    peak=torch.cuda.max_memory_allocated() if device.type=="cuda" else None
    output={"arm":name,"before":before,"after":after,"before_validation":before_metrics,"after_validation":after_metrics,
            "certificate":certificate,"comparisons":comparisons,"export":exported,"node_metadata_bytes":(target/"node_ids.json").stat().st_size,
            "before_model_bgm_bytes":source.stat().st_size,"before_node_metadata_bytes":(target/"before_node_ids.json").stat().st_size,
            "before_deployment_bytes_including_ids":source.stat().st_size+(target/"before_node_ids.json").stat().st_size,
            "deployment_bytes_including_ids":exported["bytes"]+(target/"node_ids.json").stat().st_size,
            "weight_recovery_pt_bytes":(target/"model_state.pt").stat().st_size,
            "source_sha256":digest(source),"source_node_metadata_sha256":digest(metadata_path),"source_validation_prediction_sha256":digest(prediction_path),
            "costs":costs,"wall_seconds":time.perf_counter()-started,"process_memory_at_arm_end_cumulative":memory_info(),
            "peak_cuda_allocated_bytes":peak,"training_updates":0,"evaluation_calls":2,"evaluation_samples":1600,"status":"pass"}
    save_json(target/"result.json",output)
    if peak is not None and peak>cfg["peak_cuda_memory_limit_bytes"]: raise MemoryError("CUDA峰工程预算")
    del model,cleaned,x,y
    if device.type=="cuda": torch.cuda.empty_cache()
    print(json.dumps({"arm":name,"nonzero_before":sum(before["nonzero_edges"]),"nonzero_after":sum(after["nonzero_edges"]),
                      "discarded_nodes":[r["before_width"]-r["after_width"] for r in certificate],"max_error":comparisons["before_after"]["max_absolute_error"]}),flush=True)
    return output


def main():
    cfg_path=ROOT/"configs/h1_capacity_reachability_v1.json"
    cfg=json.loads(cfg_path.read_text(encoding="utf-8"))
    parent=ROOT/"experiments/runs"/cfg["parent_version"]/"mnist_static"
    parent_report=ROOT/"reports/h1_capacity_natural_v1_mnist_static_20261008.json"
    expected=json.loads(parent_report.read_text(encoding="utf-8"))["results"]
    manifest=json.loads((ROOT/"data/structure_mnist_v1/manifest.json").read_text(encoding="utf-8"))
    if manifest["seed"]!=20261004 or digest(ROOT/"data/structure_mnist_v1/select.npz")!=manifest["cache_sha256"]["select"] or len(manifest["splits"]["select"])!=800:
        raise ValueError("验证集固定ID/SHA异常")
    folder=ROOT/"experiments/runs"/cfg["version"]/"mnist_static"; folder.mkdir(parents=True,exist_ok=True)
    if any(folder.iterdir()): raise FileExistsError("拒绝覆盖或重复执行")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    registration={"config":cfg,"config_sha256":digest(cfg_path),"source_sha256":digest(Path(__file__)),
                  "parent_report_sha256":digest(parent_report),"parent_models_sha256":{n:digest(parent/n/"model.bgm") for n in cfg["arms"]},
                  "validation_cache_sha256":manifest["cache_sha256"]["select"],"execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
                  "device":str(device),"device_name":torch.cuda.get_device_name() if device.type=="cuda" else platform.processor(),"torch":torch.__version__,
                  "precision":"FP32","cpu_threads":2,"execution":"dense_linear_physically_sliced","tf32":False,"no_label_used_for_node_selection":True}
    save_json(folder/"registration.json",registration)
    started=time.perf_counter(); results={}
    try:
        for name in cfg["arms"]:
            if time.perf_counter()-started>cfg["session_wall_seconds"]: raise TimeoutError("会话预算")
            results[name]=run_arm(name,cfg,parent,expected[name],device,folder)
        summary={"registration":registration,"results":results,"session_wall_seconds":time.perf_counter()-started,"training_updates":0,
                 "exploratory_test_evaluations":0,"official_test_accessed":False,"sealed_remaining_ids_accessed":False,"training_cache_accessed":False,
                 "evaluation_samples":sum(r["evaluation_samples"] for r in results.values()),"status":"pass"}
        save_json(folder/"summary.json",summary)
    except Exception as exc:
        save_json(folder/"failure.json",{"error":repr(exc),"results":results,"session_wall_seconds":time.perf_counter()-started})
        raise


if __name__=="__main__": main()
