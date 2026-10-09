"""自然任务上的路径节点组正则、仅剪节点及匹配随机物理剪宽比较。"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .codec import decode_model
from .datasets import load_bundle
from .models import StructuredMLP
from .node_experiment import (checkpoint_state, fresh_adam, new_ledger, persist_session_cost,
                              reconcile_cached_evaluation, restore_model)
from .path_sparsity import delete_hidden_nodes, path_group_penalty, path_node_scores
from .pruning_experiment import (encoded, evaluate, initialize, metrics, pid_alive, restore_rng,
    rng_state, save_checkpoint, select_screen, sha, structure_stats, synchronize, within, write_json)

ROOT = Path(__file__).resolve().parents[2]
ARMS = ["fc", "plain_prune", "path_fixed", "path_prune", "path_random"]
PATH_ARMS = {"path_fixed", "path_prune", "path_random"}
SOURCES = ("tools/train_path_sparsity.py", "src/braingrow/path_sparsity_experiment.py",
    "src/braingrow/path_sparsity.py", "src/braingrow/node_experiment.py", "src/braingrow/node_reduction.py",
    "src/braingrow/pruning_experiment.py", "src/braingrow/models.py", "src/braingrow/codec.py",
    "src/braingrow/structural.py", "src/braingrow/datasets.py")
UPDATE_KEYS = ("attempted_updates", "candidate_updates", "rolled_back_updates", "fallback_updates")


def lambda_for(arm, epoch, config):
    if arm not in PATH_ARMS or epoch <= config["warmup_epochs"]:
        return 0.
    fraction = min(1., (epoch-config["warmup_epochs"]) /
                   (config["regularizer_full_epoch"]-config["warmup_epochs"]))
    return config["lambda_max"] * fraction


def batch_update(model, optimizer, x, y, ledger, coefficient, initial_widths, scale, candidate=False):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    task_loss = F.cross_entropy(model(x), y)
    regularizer = path_group_penalty(model, initial_widths, scale) if coefficient else task_loss.new_zeros(())
    total = task_loss + coefficient * regularizer
    if not torch.isfinite(total):
        raise FloatingPointError("任务/路径正则目标非有限")
    total.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise FloatingPointError("路径优化一阶梯度非有限")
    optimizer.step()
    ledger["optimization_forward_samples"] += len(x)
    ledger["optimization_backward_samples"] += len(x)
    ledger["attempted_updates"] += 1
    ledger["candidate_updates"] += int(candidate)
    ledger["regularizer_backprop_calls"] += int(coefficient > 0)
    return dict(task_loss=float(task_loss.detach()), normalized_regularizer=float(regularizer.detach()),
                total_objective=float(total.detach()))


def ordinary(model, optimizer, x, y, ledger, coefficient, config, scale):
    return batch_update(model, optimizer, x, y, ledger, coefficient, config["widths"], scale)


def candidate_slot(model, optimizer, x, y, sx, sy, screen, references, ledger,
                   coefficient, config, scale, epoch, proposal_index, layer_index):
    before_rng = rng_state()
    with torch.no_grad():
        scores = path_node_scores(model)[layer_index].detach().cpu().numpy()
    count = int(len(scores)*config["node_remove_fraction"])
    event = dict(epoch=epoch, proposal=proposal_index, layer_index=layer_index, accepted=False,
                 trial_adapted=False, removed_nodes=0, widths_before=list(model.widths),
                 coefficient=coefficient, reference_arms=list(references))
    if count < 1:
        event["reason"] = "no_node_within_fraction_budget"
        ordinary(model, optimizer, x, y, ledger, coefficient, config, scale)
        return model, optimizer, event
    dropped = np.argsort(scores, kind="stable")[:count]
    event.update(proposed_removed_nodes=count, dropped_indices=dropped.tolist(),
                 selected_path_scores=scores[dropped].tolist())
    before = encoded(model)
    compact = delete_hidden_nodes(model, layer_index, dropped)
    payload = encoded(compact)
    decoded = decode_model(payload)
    event.update(previous_export_bytes=len(before), proposed_export_bytes=len(payload),
                 proposed_widths=list(compact.widths), decoded_widths=list(decoded.widths))
    width_ok = decoded.widths == compact.widths and compact.widths[layer_index+1] == model.widths[layer_index+1]-count
    del decoded
    if not width_ok or len(payload) >= len(before):
        event["reason"] = "no_real_width_or_storage_reduction"
    else:
        trial_optimizer = fresh_adam(compact, optimizer.param_groups[0]["lr"], config)
        event["trial_adapted"] = True
        event["adaptation"] = batch_update(compact, trial_optimizer, x, y, ledger, coefficient,
                                           config["widths"], scale, candidate=True)
        screened, _ = evaluate(compact, sx[screen], sy[screen], config["evaluation_batch_size"], ledger, "candidate_screen")
        event["screen"] = screened
        if all(within(screened, ref["screen"], config) for ref in references.values()):
            selected, _ = evaluate(compact, sx, sy, config["evaluation_batch_size"], ledger, "candidate_full")
            event["select"] = selected
            adapted_payload = encoded(compact)
            event["adapted_export_bytes"] = len(adapted_payload)
            if (all(within(selected, ref["select"], config) for ref in references.values())
                    and len(adapted_payload) < len(before)):
                event.update(accepted=True, removed_nodes=count, reason="real_width_storage_and_task_preservation")
                return compact, trial_optimizer, event
            event["reason"] = "full_task_tolerance_or_adapted_storage_failed"
        else:
            event["reason"] = "screen_exceeds_reference_tolerance"
        ledger["rolled_back_updates"] += 1
    restore_rng(before_rng)
    ordinary(model, optimizer, x, y, ledger, coefficient, config, scale)
    ledger["fallback_updates"] += int(event["trial_adapted"])
    return model, optimizer, event


def random_slot(model, optimizer, x, y, ledger, coefficient, config, scale, source, epoch):
    event = dict(epoch=epoch, proposal=source["proposal"], matched_source_proposal=source["proposal"],
        layer_index=source["layer_index"], accepted=source["accepted"], trial_adapted=source["trial_adapted"],
        removed_nodes=source["removed_nodes"], widths_before=list(model.widths), coefficient=coefficient,
        acceptance_rule="同正则；主接受slot和数量匹配，均匀随机身份，不用选择反馈")
    if not source["accepted"]:
        if source["trial_adapted"]:
            before_rng = rng_state()
            trial = copy.deepcopy(model)  # 原图/参数/梯度/Adam保持不动。
            trial_optimizer = fresh_adam(trial, optimizer.param_groups[0]["lr"], config)
            batch_update(trial, trial_optimizer, x, y, ledger, coefficient, config["widths"], scale, candidate=True)
            ledger["rolled_back_updates"] += 1
            restore_rng(before_rng)
            del trial, trial_optimizer
            ledger["fallback_updates"] += 1
        ordinary(model, optimizer, x, y, ledger, coefficient, config, scale)
        event["reason"] = "matched_source_rejection_with_same_adaptation_cost"
        return model, optimizer, event
    layer_index, count = source["layer_index"], source["removed_nodes"]
    rng = np.random.default_rng(np.random.SeedSequence([11, epoch, source["proposal"], layer_index]))
    dropped = np.sort(rng.choice(model.widths[layer_index+1], size=count, replace=False))
    compact = delete_hidden_nodes(model, layer_index, dropped)
    trial_optimizer = fresh_adam(compact, optimizer.param_groups[0]["lr"], config)
    event["adaptation"] = batch_update(compact, trial_optimizer, x, y, ledger, coefficient,
                                       config["widths"], scale, candidate=True)
    event.update(dropped_indices=dropped.tolist(), widths_after=list(compact.widths),
                 reason="unconditional_same_regularizer_matched_random_nodes")
    return compact, trial_optimizer, event


def path_diagnostics(model, initial_widths, scale):
    with torch.no_grad():
        scores = [value.detach().cpu().numpy() for value in path_node_scores(model)]
        penalty = float(path_group_penalty(model, initial_widths, scale))
    return dict(normalized_regularizer=penalty, layers=[dict(width=len(value),
        quantiles=np.quantile(value, [0., .1, .25, .5, .9, 1.]).tolist(),
        quantile_levels=[0., .1, .25, .5, .9, 1.], exact_zero_groups=int(np.sum(value == 0)))
        for value in scores]), scores


def make_checkpoint(model, optimizer, rows, events, ledger, epoch, wall, arm, scale, coefficient):
    checkpoint = checkpoint_state(model, optimizer, rows, events, ledger, epoch, wall, arm)
    checkpoint.pop("node_experiment_version")
    checkpoint.update(path_experiment_version=1, initial_scale=scale, current_lambda=coefficient)
    return checkpoint


def run(args):
    began = time.perf_counter()
    config = json.loads((ROOT/args.config).read_text(encoding="utf-8"))
    if (config["model_seed"] != 11 or config["data_seed"] != 20261004 or config["arms"] != ARMS
            or not config["version"].startswith("h1_path_sparsity_")):
        raise ValueError("路径实验必须用独立版本、五臂及唯一模型/数据根")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    folder = ROOT/"experiments/runs"/config["version"]/"mnist_static"
    folder.mkdir(parents=True, exist_ok=True)
    lock = folder/"running.lock"
    if lock.exists():
        if pid_alive(int(lock.read_text())):
            raise RuntimeError("路径实验运行锁活动，拒绝重复启动")
        lock.unlink()
    fd = os.open(lock, os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    cost_path = folder/"session_cost.json"
    past_cost = json.loads(cost_path.read_text(encoding="utf-8")) if cost_path.exists() else dict(cumulative_wall_seconds=0, sessions=0)
    try:
        registration = dict(config=config, task="mnist_static", device=str(device), torch=torch.__version__,
            source_sha256={path: sha(ROOT/path) for path in SOURCES}, initialization="same_root11_dense_fc_all_arms")
        reg_path = folder/"registration.json"
        if reg_path.exists():
            if json.loads(reg_path.read_text(encoding="utf-8")) != registration or not args.resume:
                raise ValueError("运行已存在或源/config变化；仅允许相同实现--resume")
        else:
            write_json(reg_path, registration)
        if (folder/"summary.json").exists():
            print(json.dumps(dict(already_complete=True, run=str(folder))), flush=True)
            return
        if past_cost["cumulative_wall_seconds"] >= config["maximum_wall_seconds"]:
            raise RuntimeError("阶段累计墙钟预算用完，--resume不重获预算")
        write_json(folder/"running.json", dict(pid=os.getpid(), status="running"))
        previous_startup = ROOT/"experiments/runs/pruning_v1/mnist_static/startup.json"
        startup = json.loads(previous_startup.read_text(encoding="utf-8"))
        if not startup["ok"] or startup["widths"] != config["widths"]:
            raise ValueError("既有同尺寸GPU启动证据不匹配")
        write_json(folder/"startup.json", dict(reused_from=str(previous_startup), forward_samples=0,
            backward_samples=0, updates=0, additional_gpu_checks=0, new_cpu_check="test_path_sparsity一次通过"))
        bundle = load_bundle("mnist", include_confirm=False)
        if bundle.fit_x.shape != (3200, 784) or bundle.select_x.shape != (800, 784) or len(bundle.confirm_x):
            raise ValueError("只允许当前3200/800授权训练集，冻结前不加载control")
        write_json(folder/"data_manifest.json", bundle.manifest)
        fx, fy, sx, sy = [torch.from_numpy(value).to(device) for value in
                         (bundle.fit_x, bundle.fit_y, bundle.select_x, bundle.select_y)]
        screen = select_screen(sy, config["screen_samples"])
        candidate_layers = {item["epoch"]: item["layer_index"] for item in config["candidate_schedule"]}
        results = {}
        for arm in ARMS:
            af = folder/arm
            af.mkdir(exist_ok=True)
            resume_path = af/"resume.pt"
            if (af/"result.json").exists():
                result = json.loads((af/"result.json").read_text(encoding="utf-8"))
                saved = torch.load(resume_path, map_location=device, weights_only=False)
                result["ledger"] = saved["ledger"]
                if (af/"fit_final.npz").exists():
                    with np.load(af/"fit_final.npz", allow_pickle=False) as cache:
                        result["fit"] = metrics(cache["logits"], cache["labels"])
                    reconcile_cached_evaluation(result["ledger"], "frozen_fit", result["fit"]["samples"])
                else:
                    model = restore_model(saved, device, config["widths"])
                    result["fit"], scores = evaluate(model, fx, fy, config["evaluation_batch_size"], result["ledger"], "frozen_fit")
                    np.savez_compressed(af/"fit_final.npz", logits=scores, labels=bundle.fit_y)
                    del model
                saved["ledger"] = result["ledger"]
                save_checkpoint(resume_path, saved)
                write_json(af/"result.json", result)
                results[arm] = result
                continue
            resumed = resume_path.exists()
            if resumed:
                saved = torch.load(resume_path, map_location=device, weights_only=False)
                model = restore_model(saved, device, config["widths"])
                optimizer = fresh_adam(model, config["learning_rate"], config)
                optimizer.load_state_dict(copy.deepcopy(saved["optimizer"]))
                restore_rng(saved["rng"])
                rows, events, ledger = saved["rows"], saved["events"], saved["ledger"]
                next_epoch, wall, scale, checkpoint = saved["next_epoch"], saved["wall_seconds"], saved["initial_scale"], saved
            else:
                model = initialize("fc", dict(widths=config["widths"]), 11, device)
                initial_payload = encoded(model)
                initial_hash = hashlib.sha256(initial_payload).hexdigest()
                with torch.no_grad():
                    scale = float(sum(score.sum()/width for score, width in zip(path_node_scores(model), config["widths"][1:-1])))
                initialization = dict(initial_model_sha256=initial_hash, initial_scale=scale, widths=config["widths"], model_seed=11)
                if (folder/"initialization.json").exists():
                    if json.loads((folder/"initialization.json").read_text(encoding="utf-8")) != initialization:
                        raise ValueError("五臂初始化或固定路径归一化常数不一致")
                else:
                    write_json(folder/"initialization.json", initialization)
                (af/"initial_model.bgs").write_bytes(initial_payload)
                optimizer = fresh_adam(model, config["learning_rate"], config)
                rows, events, next_epoch, wall, checkpoint = [], [], 1, 0., None
                ledger = new_ledger()
                ledger["regularizer_backprop_calls"] = 0
            for epoch in range(next_epoch, config["epochs"]+1):
                synchronize(device)
                started = time.perf_counter()
                lr = config["learning_rate"]*.5**sum(epoch >= k for k in config["lr_milestones"])
                coefficient = lambda_for(arm, epoch, config)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                order = np.random.default_rng(np.random.SeedSequence([20261004, epoch])).permutation(len(fx))
                batches = [torch.as_tensor(order[start:start+64], device=device) for start in range(0, len(fx), 64)]
                objectives = [ordinary(model, optimizer, fx[ids], fy[ids], ledger, coefficient, config, scale) for ids in batches[:-1]]
                ids, event = batches[-1], None
                if epoch in candidate_layers:
                    optimizer = fresh_adam(model, lr, config)
                    ledger["adam_reset_epochs"].append(epoch)
                    if arm in {"plain_prune", "path_prune"}:
                        reference_arms = config["plain_reference"] if arm == "plain_prune" else config["path_reference"]
                        references = {name: results[name]["rows"][epoch-1] for name in reference_arms}
                        model, optimizer, event = candidate_slot(model, optimizer, fx[ids], fy[ids], sx, sy, screen,
                            references, ledger, coefficient, config, scale, epoch, len(events)+1, candidate_layers[epoch])
                    elif arm == "path_random":
                        source = [item for item in results["path_prune"]["events"] if item["epoch"] == epoch]
                        if len(source) != 1:
                            raise RuntimeError("匹配随机节点缺少唯一同轮主事件")
                        model, optimizer, event = random_slot(model, optimizer, fx[ids], fy[ids], ledger,
                            coefficient, config, scale, source[0], epoch)
                    else:
                        ordinary(model, optimizer, fx[ids], fy[ids], ledger, coefficient, config, scale)
                else:
                    ordinary(model, optimizer, fx[ids], fy[ids], ledger, coefficient, config, scale)
                if event is not None:
                    events.append(event)
                selected, predictions = evaluate(model, sx, sy, config["evaluation_batch_size"], ledger, "epoch_select")
                payload = encoded(model)
                structure = structure_stats(model, payload)
                diagnostics, path_scores = path_diagnostics(model, config["widths"], scale)
                net = ledger["attempted_updates"]-ledger["rolled_back_updates"]
                if net != epoch*50:
                    raise RuntimeError("候选未保留正常50净更新/轮")
                updates = {key: ledger[key] for key in UPDATE_KEYS}
                if arm == "path_random":
                    reference = results["path_prune"]["rows"][epoch-1]
                    if (list(model.widths) != reference["widths"] or updates != reference["updates"]
                            or ledger["adam_reset_epochs"] != reference["adam_reset_epochs"]):
                        raise RuntimeError("随机身份对照物理宽度/更新/Adam重置时序失配")
                synchronize(device)
                elapsed = time.perf_counter()-started
                wall += elapsed
                row = dict(epoch=epoch, select=selected, screen=metrics(predictions[screen.cpu().numpy()], bundle.select_y[screen.cpu().numpy()]),
                    structure=structure, widths=list(model.widths), lambda_value=coefficient, learning_rate=lr,
                    mean_ordinary_objective={key: float(np.mean([value[key] for value in objectives])) for key in objectives[0]},
                    ordinary_objective_batches=len(objectives), ordinary_objective_scope="所有臂同样前49个普通批次，候选/末批另记",
                    path_diagnostics=diagnostics, updates=updates,
                    net_updates=net, adam_reset_epochs=list(ledger["adam_reset_epochs"]), wall_seconds=elapsed)
                rows.append(row)
                np.savez_compressed(af/f"select_{epoch:03d}.npz", logits=predictions, labels=bundle.select_y)
                np.savez_compressed(af/f"path_scores_{epoch:03d}.npz", **{f"layer_{i}": value for i, value in enumerate(path_scores)})
                checkpoint = make_checkpoint(model, optimizer, rows, events, ledger, epoch+1, wall, arm, scale, coefficient)
                save_checkpoint(resume_path, checkpoint)
                persist_session_cost(cost_path, past_cost, began, config["maximum_wall_seconds"])
                write_json(af/"progress.json", dict(rows=rows, events=events, ledger=ledger, wall_seconds=wall))
                print(json.dumps(dict(arm=arm, epoch=epoch, accuracy=selected["accuracy"], loss=selected["loss"],
                    widths=list(model.widths), lambda_value=coefficient, export_bytes=len(payload), event=event and event["reason"])), flush=True)
                if (folder/"request_stop").exists() or past_cost["cumulative_wall_seconds"]+time.perf_counter()-began >= config["maximum_wall_seconds"]:
                    write_json(folder/"budget_stop.json", dict(arm=arm, epoch=epoch, reason="request_or_cumulative_stage_budget"))
                    return
            payload = encoded(model)
            (af/"final_model.bgs").write_bytes(payload)
            if (af/"fit_final.npz").exists():
                with np.load(af/"fit_final.npz", allow_pickle=False) as cache:
                    fitted = metrics(cache["logits"], cache["labels"])
                reconcile_cached_evaluation(ledger, "frozen_fit", fitted["samples"])
            else:
                fitted, predictions = evaluate(model, fx, fy, config["evaluation_batch_size"], ledger, "frozen_fit")
                np.savez_compressed(af/"fit_final.npz", logits=predictions, labels=bundle.fit_y)
            checkpoint.update(ledger=ledger, rng=rng_state())
            save_checkpoint(resume_path, checkpoint)
            result = dict(arm=arm, frozen=True, final_epoch=config["epochs"], fit=fitted, rows=rows, events=events,
                widths=list(model.widths), structure=structure_stats(model, payload), ledger=ledger, wall_seconds=wall)
            results[arm] = result
            write_json(af/"result.json", result)
            del model, optimizer
        write_json(folder/"freeze.json", dict(arms={name: result["final_epoch"] for name, result in results.items()}, source_sha256=registration["source_sha256"]))
        confirmed = load_bundle("mnist", include_confirm=True)
        cx, cy = torch.from_numpy(confirmed.confirm_x).to(device), torch.from_numpy(confirmed.confirm_y).to(device)
        for arm, result in results.items():
            af = folder/arm
            saved = torch.load(af/"resume.pt", map_location=device, weights_only=False)
            result["ledger"] = saved["ledger"]
            if (af/"confirm_final.npz").exists():
                with np.load(af/"confirm_final.npz", allow_pickle=False) as cache:
                    confirmation = metrics(cache["logits"], cache["labels"])
                reconcile_cached_evaluation(result["ledger"], "repeated_exploration", confirmation["samples"])
            else:
                model = restore_model(saved, device, config["widths"])
                confirmation, predictions = evaluate(model, cx, cy, config["evaluation_batch_size"], result["ledger"], "repeated_exploration")
                np.savez_compressed(af/"confirm_final.npz", logits=predictions, labels=confirmed.confirm_y)
                del model
            result["confirmation"] = confirmation
            saved["ledger"] = result["ledger"]
            save_checkpoint(af/"resume.pt", saved)
            write_json(af/"result.json", result)
        primary, fc = results["path_prune"], results["fc"]
        reduction = 1-primary["structure"]["export_bytes"]/fc["structure"]["export_bytes"]
        gate = dict(primary="path_prune", fc_learned=fc["rows"][-1]["select"]["accuracy"] >= config["learning_floor"],
            task_preserved=all(within(primary["rows"][-1]["select"], results[name]["rows"][-1]["select"], config) for name in config["path_reference"]),
            physical_width_reduced=sum(primary["widths"][1:-1]) < sum(config["widths"][1:-1]), storage_reduction=reduction,
            storage_passed=reduction >= config["minimum_storage_reduction"])
        gate["success"] = all(gate[key] for key in ("fc_learned", "task_preserved", "physical_width_reduced", "storage_passed"))
        comparison = {arm: dict(accepted=sum(item["accepted"] for item in result["events"]),
            select_accuracy_delta_vs_fc=result["rows"][-1]["select"]["accuracy"]-fc["rows"][-1]["select"]["accuracy"],
            export_bytes=result["structure"]["export_bytes"], widths=result["widths"]) for arm, result in results.items()}
        summary = dict(task="mnist_static", registration=registration, arms=results, gate=gate, comparison=comparison,
            confirmation_status=config["confirmation_role"], runtime_ratio_is_gate=False, random_widths_updates_matched=True,
            current_session_wall_seconds=time.perf_counter()-began,
            cumulative_wall_seconds=past_cost["cumulative_wall_seconds"]+time.perf_counter()-began,
            peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            recovery_accounting_scope="完整轮及已保存预测可协调；完全未落盘的中途成本不可完整追溯",
            execution_note="物理缩宽/参数槽减少，仍稠密gather；未测推断加速/能耗，正则值不是L0或成功证据")
        write_json(folder/"summary.json", summary)
        print(json.dumps(dict(complete=True, gate=gate, comparison=comparison)), flush=True)
    finally:
        persist_session_cost(cost_path, past_cost, began, config["maximum_wall_seconds"])
        lock.unlink(missing_ok=True)
        write_json(folder/"running.json", dict(pid=None, status="exited"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/h1_path_sparsity_v1.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
