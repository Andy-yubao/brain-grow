"""任务损伤决定群组及接受，边数仅为结果；五臂有限自然结构诊断。"""
from __future__ import annotations
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from tools.train_capacity_baseline import ROOT, augment, digest, evaluate, save_json
from tools.train_capacity_pruning import MaskedFC, capture, restore, export_model, load_export, sync
from braingrow.datasets import load_bundle


class NaturalFC(MaskedFC):
    def __init__(self, widths, dropout, original_widths=None):
        super().__init__(widths, dropout)
        self.original_widths = original_widths or list(widths)
        for i, width in enumerate(widths[1:-1]):
            self.register_buffer(f"node_ids_{i}", torch.arange(width))

    def forward(self, x):
        x = x.flatten(1)
        for i, layer in enumerate(self.layers[:-1]):
            x = F.relu(layer(x))
            if self.training and self.dropout:
                canonical = torch.rand((len(x), self.original_widths[i+1]), device=x.device)
                x = x * (canonical[:, getattr(self, f"node_ids_{i}")] >= self.dropout) / (1-self.dropout)
        return self.layers[-1](x)


def make_model(widths, dropout, device, original_widths=None):
    # 随机初始化只是分配；不能推进既有训练随机流。
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        return NaturalFC(widths, dropout, original_widths).to(device)


def widths(model):
    return [model.layers[0].in_features] + [l.out_features for l in model.layers]


def structure(model):
    nonzero = [int(torch.count_nonzero(l.weight)) for l in model.layers]
    mask_edges = [int(m.sum()) for m in model.masks]
    effective = []
    for i, layer in enumerate(model.layers[:-1]):
        incoming = (layer.weight != 0).any(1) | (layer.bias != 0)
        outgoing = (model.layers[i+1].weight != 0).any(0)
        effective.append(int((incoming & outgoing).sum()))
    return {"nonzero_edges": nonzero, "mask_edges": mask_edges, "widths": widths(model),
            "effective_hidden_nodes": effective, "bias_count": sum(l.bias.numel() for l in model.layers),
            "independent_parameters": sum(nonzero)+sum(l.bias.numel() for l in model.layers)}


def snapshot(model, optimizer, ema):
    saved = capture(model, optimizer, ema)
    saved.update({"widths": widths(model), "original_widths": model.original_widths,
                  "gradients": [None if p.grad is None else p.grad.detach().cpu().clone() for p in model.parameters()]})
    return saved


def recover(model, optimizer, ema, saved, cfg):
    device = next(model.parameters()).device
    if widths(model) != saved["widths"]:
        del optimizer, ema
        model = make_model(saved["widths"], model.dropout, device, saved["original_widths"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
        ema = [torch.zeros_like(l.weight) for l in model.layers]
    restore(model, optimizer, ema, saved)
    for p, g in zip(model.parameters(), saved["gradients"]):
        p.grad = None if g is None else g.to(device).clone()
    return model, optimizer, ema


def train_batches(model, optimizer, ema, batches, cfg, ledger):
    started=time.perf_counter()
    model.train()
    total = 0.
    for x, y in batches:
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y)
        if not torch.isfinite(loss):
            raise FloatingPointError("训练非有限")
        loss.backward()
        with torch.no_grad():
            for layer, mask, score in zip(model.layers, model.masks, ema):
                layer.weight.grad.mul_(mask)
                score.mul_(cfg["sensitivity_ema"]).add_((layer.weight * layer.weight.grad).abs(), alpha=1-cfg["sensitivity_ema"])
        optimizer.step()
        model.enforce(optimizer)
        ledger["attempted_updates"] += 1
        ledger["optimization_samples"] += len(y)
        total += float(loss.detach()) * len(y)
    sync(next(model.parameters()).device)
    ledger["training_seconds"] = ledger.get("training_seconds",0.)+time.perf_counter()-started
    return total


def predict(model, x, batch=512):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(z) for z in x.split(batch)])


def functional_test(before, after, y, cfg):
    kl = F.kl_div(F.log_softmax(after, 1), F.softmax(before, 1), reduction="batchmean")
    loss_delta = F.cross_entropy(after, y)-F.cross_entropy(before, y)
    accuracy_delta = (before.argmax(1)==y).float().mean()-(after.argmax(1)==y).float().mean()
    passed = bool(torch.isfinite(after).all() and loss_delta <= cfg["probe_loss_increase"]
                  and accuracy_delta <= cfg["probe_accuracy_drop"]+1e-7 and kl <= cfg["probe_kl"])
    return {"pass": passed, "cross_entropy_delta": float(loss_delta), "accuracy_drop": float(accuracy_delta), "kl": float(kl)}


def activations(model, x, layer, cfg, ledger):
    model.eval()
    required = len(x)*model.layers[layer].out_features*4*2
    if required > cfg["activation_cache_limit_bytes"]:
        raise MemoryError("单层激活与残差缓存超预算")
    rows = []
    with torch.no_grad():
        for batch in x.split(cfg["evaluation_batch_size"]):
            h = batch
            for l in model.layers[:layer+1]:
                h = F.relu(l(h))
            rows.append(h.cpu())
    ledger["proposal_activation_samples"] += len(x)
    return torch.cat(rows)


def library(h, cfg):
    residual = h.clone()
    norms = torch.linalg.vector_norm(h, dim=0)
    basis, q = [], []
    for _ in range(min(cfg["basis_size"], h.shape[1])):
        rank = residual.square().sum(0)
        if basis:
            rank[basis] = -1
        index = int(rank.argmax())
        if float(rank[index]) <= cfg["activation_epsilon"]**2:
            break
        v = h[:, index].clone()
        for _ in range(2):
            for qi in q:
                v -= qi * torch.dot(qi, v)
        length = torch.linalg.vector_norm(v)
        if float(length) <= cfg["activation_epsilon"]:
            break
        qi = v / length
        q.append(qi)
        basis.append(index)
        residual -= qi[:, None] * (qi @ residual)[None, :]
    if not basis:
        return [], [], torch.empty((0, 0)), []
    relative = torch.linalg.vector_norm(residual, dim=0) / norms.clamp_min(cfg["activation_epsilon"])
    relative[basis] = float("inf")
    pool = torch.argsort(relative, stable=True)[:min(cfg["node_pool"], h.shape[1]-len(basis))].tolist()
    coefficients = torch.linalg.lstsq(h[:, basis].double(), h[:, pool].double(), driver="gelsd").solution.float()
    eligible = [(i, j) for j, i in enumerate(pool) if float(norms[i]) <= cfg["activation_epsilon"] or
                int((coefficients[:, j].abs() > cfg["coefficient_epsilon"]).sum()) >= 2]
    return basis, [i for i, _ in eligible], coefficients[:, [j for _, j in eligible]], [float(norms[i]) for i, _ in eligible]


def apply_edges(model, event):
    layer = event["layer"]
    ids = torch.tensor(event["deleted"], device=model.masks[layer].device, dtype=torch.long)
    with torch.no_grad():
        model.masks[layer].flatten()[ids] = False
        model.layers[layer].weight.flatten()[ids] = 0


def node_compensation(model, event):
    layer = event["layer"]
    w = model.layers[layer+1].weight
    ids = torch.tensor(event["deleted"], device=w.device)
    basis = torch.tensor(event["basis"], device=w.device)
    coefficients = torch.tensor(event["coefficients"], dtype=w.dtype, device=w.device)
    with torch.no_grad():
        compensated = w.clone()
        compensated[:, basis] += w[:, ids] @ coefficients.T
        compensated[:, ids] = 0
    return compensated


def apply_nodes(model, event, ema=None):
    layer, ids = event["layer"], event["deleted"]
    device = next(model.parameters()).device
    keep = torch.tensor([i for i in range(model.layers[layer].out_features) if i not in set(ids)], device=device)
    new_widths = widths(model)
    new_widths[layer+1] -= len(ids)
    result = make_model(new_widths, model.dropout, device, model.original_widths)
    next_w = node_compensation(model, event)
    with torch.no_grad():
        for i, (old, new) in enumerate(zip(model.layers, result.layers)):
            new.weight.copy_(old.weight[keep] if i == layer else next_w[:, keep] if i == layer+1 else old.weight)
            new.bias.copy_(old.bias[keep] if i == layer else old.bias)
            # 节点支路没有删边，物理图仍以真实权重零值表示精确零。
            result.masks[i].copy_(model.masks[i][keep] if i == layer else model.masks[i][:, keep] if i == layer+1 else model.masks[i])
        for i in range(len(model.layers)-1):
            value = getattr(model, f"node_ids_{i}")
            getattr(result, f"node_ids_{i}").copy_(value[keep] if i == layer else value)
    scores = None if ema is None else [e[keep].clone() if i == layer else e[:, keep].clone() if i == layer+1 else e.clone() for i,e in enumerate(ema)]
    return result, scores


def edit_accounting(model,event):
    if not event["deleted"]:
        event.update({"nominal_removed_nonzero_edges":0,"compensation_new_nonzero_edges":0,"compensation_zeroed_survivor_edges":0})
        return
    if event["kind"]=="edge":
        event.update({"nominal_removed_nonzero_edges":int(torch.count_nonzero(model.layers[event["layer"]].weight.flatten()[event["deleted"]])),
                      "compensation_new_nonzero_edges":0,"compensation_zeroed_survivor_edges":0})
        return
    layer,ids=event["layer"],event["deleted"]
    old=model.layers[layer+1].weight
    new=node_compensation(model,event)
    keep=torch.ones(old.shape[1],dtype=torch.bool,device=old.device)
    keep[ids]=False
    event.update({"nominal_removed_nonzero_edges":int(torch.count_nonzero(model.layers[layer].weight[ids]))+int(torch.count_nonzero(old[:,ids])),
                  "compensation_new_nonzero_edges":int(((old[:,keep]==0)&(new[:,keep]!=0)).sum()),
                  "compensation_zeroed_survivor_edges":int(((old[:,keep]!=0)&(new[:,keep]==0)).sum()),
                  "deleted_original_node_ids":getattr(model,f"node_ids_{layer}")[ids].cpu().tolist()})


def propose(model, ema, mechanism, slot, fx, hx, hy, cfg, ledger, random_count=None):
    started = time.perf_counter()
    layer = (slot-1) % (len(model.layers) if mechanism == "edge" else len(model.layers)-1)
    event = {"kind": mechanism, "layer": layer, "trials": [], "deleted": [], "proposal_seconds": 0.}
    if mechanism == "edge":
        alive = model.masks[layer].flatten()
        maximum = min(int(alive.sum()), max(cfg["edge_group_sizes"]))
        rank = ema[layer].flatten().masked_fill(~alive, float("inf"))
        if random_count is not None:
            active = torch.nonzero(alive, as_tuple=False).flatten()
            stream = torch.Generator(device=alive.device).manual_seed(cfg["model_seed"]+6000+slot)
            event["deleted"] = active[torch.randperm(len(active), generator=stream, device=alive.device)[:random_count]].cpu().tolist()
            return event
        if maximum:
            cutoff = torch.topk(rank, maximum, largest=False, sorted=False).values.max()
            lower = torch.nonzero(rank < cutoff, as_tuple=False).flatten()
            equal = torch.nonzero(rank == cutoff, as_tuple=False).flatten()
            ids = torch.cat((lower, equal[:maximum-len(lower)]))
            ids = ids.sort().values
            ids = ids[torch.argsort(rank[ids], stable=True)]
        else:
            ids = torch.empty(0, device=alive.device, dtype=torch.long)
        trial_sizes = sorted(set(min(maximum,n) for n in cfg["edge_group_sizes"] if maximum))
        old_values = model.layers[layer].weight.flatten()[ids].detach().clone()
        before = predict(model, hx, cfg["evaluation_batch_size"])
        ledger["proposal_function_samples"] += len(hx)
        for count in trial_sizes:
            trial = {**event, "deleted": ids[:count].cpu().tolist()}
            apply_edges(model, trial)
            after = predict(model, hx, cfg["evaluation_batch_size"])
            ledger["proposal_function_samples"] += len(hx)
            measured = functional_test(before, after, hy, cfg)
            event["trials"].append({"count": count, **measured})
            with torch.no_grad():
                model.layers[layer].weight.flatten()[ids] = old_values
                model.masks[layer].flatten()[ids] = True
            if not measured["pass"]:
                break
            event["deleted"] = trial["deleted"]
            if time.perf_counter()-started > cfg["proposal_seconds_per_slot"]:
                raise TimeoutError("边提议槽预算")
    else:
        h = activations(model, fx, layer, cfg, ledger)
        basis, pool, coefficients, norms = library(h, cfg)
        del h
        event.update({"basis": basis, "pool_count": len(pool), "basis_count": len(basis)})
        if random_count is not None:
            stream = torch.Generator().manual_seed(cfg["model_seed"]+7000+slot)
            positions = torch.randperm(len(pool), generator=stream)[:random_count].tolist()
            event.update({"deleted": [pool[j] for j in positions], "coefficients": coefficients[:, positions].tolist(),
                          "matching_pool_sufficient": len(pool) >= random_count})
            if time.perf_counter()-started > cfg["proposal_seconds_per_slot"]:
                raise TimeoutError("随机节点提议槽预算")
            return event
        before = predict(model, hx, cfg["evaluation_batch_size"])
        ledger["proposal_function_samples"] += len(hx)
        sizes = sorted(set(min(len(pool),n) for n in cfg["node_group_sizes"] if pool))
        w = model.layers[layer+1].weight
        # 保存下一层矩阵，最多79MB；不克隆全部网络或构建逐边映射。
        original = w.detach().clone()
        for count in sizes:
            trial = {**event, "deleted": pool[:count], "coefficients": coefficients[:, :count].tolist()}
            compensated = node_compensation(model, trial)
            with torch.no_grad():
                w.copy_(compensated)
            after = predict(model, hx, cfg["evaluation_batch_size"])
            ledger["proposal_function_samples"] += len(hx)
            measured = functional_test(before, after, hy, cfg)
            with torch.no_grad():
                w.copy_(original)
            event["trials"].append({"count": count, **measured})
            if not measured["pass"]:
                break
            event.update({"deleted": trial["deleted"], "coefficients": trial["coefficients"],
                          "silent_deleted": sum(n <= cfg["activation_epsilon"] for n in norms[:count])})
            if time.perf_counter()-started > cfg["proposal_seconds_per_slot"]:
                raise TimeoutError("节点提议槽预算")
    event["proposal_seconds"] = time.perf_counter()-started
    if event["proposal_seconds"] > cfg["proposal_seconds_per_slot"]:
        raise TimeoutError("候选生成预算达到")
    return event


def measured_eval(model, x, y, cfg, ledger):
    started=time.perf_counter()
    measured, logits = evaluate(model, x, y, cfg["evaluation_batch_size"])
    ledger["evaluation_calls"] += 1
    ledger["evaluation_samples"] += len(x)
    sync(next(model.parameters()).device)
    ledger["evaluation_seconds"]=ledger.get("evaluation_seconds",0.)+time.perf_counter()-started
    return measured, logits


def acceptable(candidate, original, fixed, before_edges, after_edges, cfg):
    return (candidate["cross_entropy"] <= original["cross_entropy"]+cfg["local_loss_increase"] and
            candidate["accuracy"]+cfg["local_accuracy_drop"]+1e-7 >= original["accuracy"] and
            candidate["cross_entropy"] <= fixed["cross_entropy"]+cfg["cumulative_loss_increase"] and
            candidate["accuracy"]+cfg["cumulative_accuracy_drop"]+1e-7 >= fixed["accuracy"] and
            sum(after_edges) < sum(before_edges))


def run_arm(name, cfg, base, data, device, folder, parent_state, fixed=None, main=None, session_started=None):
    arm = folder/name
    arm.mkdir()
    model = make_model(base["fc_widths"], base["dropout"], device)
    model.load_state_dict(parent_state, strict=False)
    ema = [torch.zeros_like(l.weight) for l in model.layers]
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    x,y = torch.from_numpy(data.fit_x).to(device),torch.from_numpy(data.fit_y).to(device)
    sx,sy = torch.from_numpy(data.select_x).to(device),torch.from_numpy(data.select_y).to(device)
    fi,hi = [],[]
    for label in range(10):
        ids = np.flatnonzero(data.fit_y==label)
        fi.extend(ids[:cfg["proposal_fit_per_class"]]); hi.extend(ids[cfg["proposal_fit_per_class"]:])
    fx,hx,hy = x[fi],x[hi],y[hi]
    ledger = {k:0 for k in ("attempted_updates","net_updates","optimization_samples","evaluation_calls","evaluation_samples",
                            "proposal_function_samples","proposal_activation_samples","rollbacks","candidate_adaptations")}
    rows,events,slot_reference = [],[],{}
    started = time.perf_counter()
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats()
    mechanism = "edge" if name.startswith("edge") else "node"
    is_main, is_random = name.endswith("task"),name.endswith("random")
    epoch_limit = cfg["epochs"] if main is None else len(main["rows"])
    collapse_streak = 0
    stop_reason = "finite_budget"
    for epoch in range(1,epoch_limit+1):
        torch.manual_seed(cfg["model_seed"]+2000+epoch)
        stream = torch.Generator(device=device).manual_seed(cfg["model_seed"]+3000+epoch)
        order = torch.randperm(len(x), generator=stream, device=device)
        batches = [(augment(x[ids],stream,base),y[ids]) for ids in order.split(cfg["batch_size"])]
        event = None
        edit = epoch in cfg["edit_epochs"]
        if edit:
            del optimizer
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
            optimizer.zero_grad(set_to_none=True)
            slot = cfg["edit_epochs"].index(epoch)+1
            before_struct = structure(model)
            if is_main:
                proposal_start=time.perf_counter()
                event = propose(model,ema,mechanism,slot,fx,hx,hy,cfg,ledger)
                ledger["proposal_seconds"]=ledger.get("proposal_seconds",0.)+time.perf_counter()-proposal_start
                event.update({"epoch":epoch,"accepted":False,"before":before_struct})
                if event["deleted"]:
                    saved = snapshot(model,optimizer,ema)
                    original_loss = train_batches(model,optimizer,ema,batches[:cfg["adaptation_batches"]],cfg,ledger)
                    original, original_logits = measured_eval(model,sx,sy,cfg,ledger)
                    original_state = snapshot(model,optimizer,ema)
                    model,optimizer,ema = recover(model,optimizer,ema,saved,cfg)
                    del saved
                    edit_accounting(model,event)
                    if mechanism=="edge":
                        apply_edges(model,event)
                    else:
                        new_model,new_ema = apply_nodes(model,event,ema)
                        del model,optimizer,ema
                        model,ema = new_model,new_ema
                        optimizer = torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"])
                        del new_model,new_ema
                    candidate_loss = train_batches(model,optimizer,ema,batches[:cfg["adaptation_batches"]],cfg,ledger)
                    candidate,candidate_logits = measured_eval(model,sx,sy,cfg,ledger)
                    after_struct = structure(model)
                    accepted = acceptable(candidate,original,fixed["slot_reference"][str(epoch)],before_struct["nonzero_edges"],after_struct["nonzero_edges"],cfg)
                    event.update({"accepted":accepted,"original_select":original,"candidate_select":candidate,"after":after_struct,
                                  "net_deleted_per_layer":[a-b for a,b in zip(before_struct["nonzero_edges"],after_struct["nonzero_edges"])],
                                  "nominal_deleted_nodes":len(event["deleted"]) if mechanism=="node" else 0})
                    np.savez_compressed(arm/f"slot_{slot}_validation_trials.npz",original=original_logits,candidate=candidate_logits,labels=data.select_y)
                    ledger["candidate_adaptations"] += 1
                    if accepted:
                        loss_sum = candidate_loss
                    else:
                        model,optimizer,ema = recover(model,optimizer,ema,original_state,cfg)
                        loss_sum = original_loss
                        ledger["rollbacks"] += 1
                    del original_state
                else:
                    loss_sum = train_batches(model,optimizer,ema,batches[:cfg["adaptation_batches"]],cfg,ledger)
            elif is_random:
                source_event = next(e for e in main["events"] if e["epoch"]==epoch)
                if source_event["accepted"]:
                    proposal_start=time.perf_counter()
                    event = propose(model,ema,mechanism,slot,fx,hx,hy,cfg,ledger,len(source_event["deleted"]))
                    ledger["proposal_seconds"]=ledger.get("proposal_seconds",0.)+time.perf_counter()-proposal_start
                    event.update({"epoch":epoch,"accepted":True,"before":before_struct})
                    edit_accounting(model,event)
                    if mechanism=="edge": apply_edges(model,event)
                    elif event["deleted"]:
                        new_model,new_ema=apply_nodes(model,event,ema)
                        del model,optimizer,ema
                        model,ema=new_model,new_ema
                        optimizer=torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"])
                        del new_model,new_ema
                    after_struct=structure(model)
                    event.update({"after":after_struct,"net_deleted_per_layer":[a-b for a,b in zip(before_struct["nonzero_edges"],after_struct["nonzero_edges"])],
                                  "nominal_deleted_nodes":len(event["deleted"]) if mechanism=="node" else 0})
                    event["matched_net_deletion"] = event["net_deleted_per_layer"]==source_event["net_deleted_per_layer"] and len(event["deleted"])==len(source_event["deleted"])
                loss_sum=train_batches(model,optimizer,ema,batches[:cfg["adaptation_batches"]],cfg,ledger)
            else:
                loss_sum=train_batches(model,optimizer,ema,batches[:cfg["adaptation_batches"]],cfg,ledger)
                metric,_=measured_eval(model,sx,sy,cfg,ledger)
                slot_reference[str(epoch)]=metric
            loss_sum += train_batches(model,optimizer,ema,batches[cfg["adaptation_batches"]:],cfg,ledger)
            if is_main: events.append(event)
            elif event is not None: events.append(event)
        else:
            loss_sum=train_batches(model,optimizer,ema,batches,cfg,ledger)
        ledger["net_updates"] += len(batches)
        selected,logits=measured_eval(model,sx,sy,cfg,ledger)
        fitted,fit_logits=measured_eval(model,x,y,cfg,ledger)
        current=structure(model)
        row={"epoch":epoch,"select":selected,"fit":fitted,"structure":current,"fit_augmented_cross_entropy":loss_sum/len(x)}
        rows.append(row)
        np.savez_compressed(arm/f"epoch_{epoch}_predictions.npz",validation=logits,fit=fit_logits,validation_labels=data.select_y,fit_labels=data.fit_y)
        if edit or epoch==epoch_limit:
            slot_folder=arm/f"epoch_{epoch}"
            slot_folder.mkdir()
            exported=export_model(model,slot_folder/"model.bgm")
            torch.save({"epoch":epoch,"state":snapshot(model,optimizer,ema),"events":events,"ledger":ledger},arm/"resume.pt")
            save_json(slot_folder/"structure.json",{**current,"export":exported,"node_ids":[getattr(model,f"node_ids_{i}").cpu().tolist() for i in range(3)]})
        save_json(arm/"progress.json",{"rows":rows,"events":events,"ledger":ledger,"slot_reference":slot_reference})
        print(json.dumps({"arm":name,"epoch":epoch,"validation_accuracy":selected["accuracy"],"nonzero_edges":sum(current["nonzero_edges"]),
                          "widths":current["widths"],"accepted":event["accepted"] if event else None}),flush=True)
        collapsed=fitted["accuracy"]<cfg["collapse_fit_accuracy"] and selected["accuracy"]<cfg["collapse_validation_accuracy"]
        collapse_streak=collapse_streak+1 if collapsed else 0
        if is_main and collapse_streak>=cfg["collapse_consecutive_epochs"]:
            stop_reason="severe_task_collapse"; break
        if time.perf_counter()-session_started>cfg["session_wall_seconds"]: raise TimeoutError("会话工程预算")
        if device.type=="cuda" and torch.cuda.max_memory_allocated()>cfg["peak_memory_limit_bytes"]: raise MemoryError("CUDA峰工程预算")
        del batches
    final_export=export_model(model,arm/"model.bgm")
    torch.save({"epoch":epoch,"state":snapshot(model,optimizer,ema),"events":events,"ledger":ledger},arm/"resume.pt")
    result={"rows":rows,"events":events,"slot_reference":slot_reference,"select":selected,"fit":fitted,"structure":current,"export":final_export,"ledger":ledger,
            "stop_reason":stop_reason,"wall_seconds":time.perf_counter()-started,"peak_cuda_memory_bytes":torch.cuda.max_memory_allocated() if device.type=="cuda" else None,
            "random_net_deletion_comparable":all(e.get("matched_net_deletion",True) for e in events) if is_random else None}
    save_json(arm/"result.json",result)
    del model,optimizer,ema
    if device.type=="cuda": torch.cuda.empty_cache()
    return result


def main():
    cfg_path=ROOT/"configs/h1_capacity_natural_v1.json"
    cfg=json.loads(cfg_path.read_text(encoding="utf-8"))
    base=json.loads((ROOT/"configs/capacity_baseline_v1.json").read_text(encoding="utf-8"))
    if cfg["weight_decay"] != base["weight_decay"]:
        raise ValueError("基线权重衰减不一致")
    parent=ROOT/"experiments/runs/capacity_baseline_v1/mnist_static"
    parent_summary=json.loads((parent/"summary.json").read_text(encoding="utf-8"))
    if parent_summary["gate"]!="pass" or parent_summary["results"]["fc"]["best_select"]["epoch"]!=28 or cfg["model_seed"]!=11 or cfg["data_seed"]!=20261004:
        raise ValueError("起点/数据根不符合登记")
    folder=ROOT/"experiments/runs"/cfg["version"]/"mnist_static"
    folder.mkdir(parents=True,exist_ok=True)
    if any(folder.iterdir()): raise FileExistsError("拒绝覆盖或重复训练")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(11); np.random.seed(11); torch.manual_seed(11)
    registration={"config":cfg,"base_config":base,"config_sha256":digest(cfg_path),"source_sha256":digest(Path(__file__)),
                  "parent_checkpoint_sha256":digest(parent/"fc/best.pt"),"parent_helper_sha256":digest(ROOT/"tools/train_capacity_pruning.py"),
                  "training_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"device":str(device),
                  "device_name":torch.cuda.get_device_name() if device.type=="cuda" else "CPU","torch":torch.__version__,"precision":"FP32",
                  "cpu_threads":2,"execution":"dense_zero_mask_or_physically_sliced_dense","cnn_edges_is_target":False}
    save_json(folder/"registration.json",registration)
    data=load_bundle("mnist",include_confirm=False)
    if data.fit_x.shape!=(3200,784) or data.select_x.shape!=(800,784) or len(data.confirm_x): raise ValueError("数据权限")
    save_json(folder/"data_manifest.json",data.manifest)
    parent_state=torch.load(parent/"fc/best.pt",map_location="cpu",weights_only=False)["state"]
    results={}; started=time.perf_counter()
    try:
        for name in ("fixed","edge_task","edge_random","node_task","node_random"):
            reference=results.get("edge_task" if name=="edge_random" else "node_task") if name.endswith("random") else None
            results[name]=run_arm(name,cfg,base,data,device,folder,parent_state,results.get("fixed"),reference,started)
        confirmed=load_bundle("mnist",include_confirm=True)
        for name,result in results.items():
            model=load_export(folder/name/"model.bgm").to(device)
            measured,logits=measured_eval(model,torch.from_numpy(confirmed.confirm_x).to(device),torch.from_numpy(confirmed.confirm_y).to(device),cfg,result["ledger"])
            result["control"]=measured
            np.savez_compressed(folder/name/"control_final.npz",logits=logits,labels=confirmed.confirm_y)
            save_json(folder/name/"result.json",result)
            del model
        summary={"registration":registration,"results":results,"session_wall_seconds":time.perf_counter()-started,"official_test_accessed":False,
                 "sealed_remaining_ids_accessed":False,"cnn_edges_is_target":False,"model_selection":"fixed_final_epoch_no_test_feedback"}
        save_json(folder/"summary.json",summary)
        print(json.dumps({"results":{n:{"validation":r["select"]["accuracy"],"exploratory_test":r["control"]["accuracy"],"edges":sum(r["structure"]["nonzero_edges"])} for n,r in results.items()}}),flush=True)
    except Exception as exc:
        save_json(folder/"failure.json",{"error":repr(exc),"results":results,"session_wall_seconds":time.perf_counter()-started})
        raise


if __name__=="__main__": main()
