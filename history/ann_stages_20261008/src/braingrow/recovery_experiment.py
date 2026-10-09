"""从封存剪枝末态延续，比较原图适应、任务恢复、匹配随机恢复。"""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from .datasets import load_bundle
from .models import StructuredMLP
from .edge_recovery import EdgeRecoveryController
from .structural import CandidateTransaction, Proposal
from .pruning_experiment import (write_json, save_checkpoint, sha, pid_alive, rng_state,
    restore_rng, synchronize, evaluate, train_batch, encoded, structure_stats, select_screen, within, metrics)

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ("tools/train_recovery.py", "src/braingrow/recovery_experiment.py",
           "src/braingrow/edge_recovery.py", "src/braingrow/pruning_experiment.py",
           "src/braingrow/models.py", "src/braingrow/codec.py", "src/braingrow/structural.py",
           "src/braingrow/datasets.py")


def ledger_new():
    return dict(optimization_forward_samples=0, optimization_backward_samples=0,
                attempted_updates=0, candidate_updates=0, rolled_back_updates=0,
                evaluation_forward_samples=0, evaluations=[], fallback_updates=0)


def active_counts(model):
    return [int((layer.group_index >= 0).sum()) for layer in model.shared_layers()]


def restore_model(saved, device, widths):
    model = StructuredMLP(widths).to(device)
    model.load_state_dict(saved["model"])
    for p, g in zip(model.parameters(), saved["parameter_grad"]):
        p.grad = None if g is None else g.clone()
    for layer, g in zip(model.layers, saved["edge_grad"]):
        layer.last_edge_grad = None if g is None else g.clone()
    return model


def fallback(model, optimizer, controller, x, y, ledger, config):
    if config["fallback_adaptation"]:
        train_batch(model, optimizer, x, y, ledger, controller)
        ledger["fallback_updates"] += 1


def candidate(model, optimizer, controller, x, y, sx, sy, screen, reference,
              reference_screen, ledger, config, epoch, start_counts):
    proposal = controller.propose(model, config["restore_fraction_original"])
    counts_now = active_counts(model)
    limited = []
    for li, indices in proposal.edits:
        remaining = int(start_counts[li] * config["maximum_growth_fraction_start"]) - (counts_now[li]-start_counts[li])
        indices = indices[:max(0, remaining)]
        if len(indices):
            limited.append((li, indices))
    proposal = Proposal("recover", limited, sum(len(ids) for _, ids in limited))
    counts = [0] * len(start_counts)
    for li, ids in limited:
        counts[li] = len(ids)
    event = dict(epoch=epoch, proposal=controller.proposals, accepted=False,
                 layer_edit_counts=counts, attempted_edges=proposal.count)
    if not proposal.count:
        train_batch(model, optimizer, x, y, ledger, controller)
        event["reason"] = "growth_budget_exhausted_or_no_inactive_edges"
        return event
    before, _ = evaluate(model, sx, sy, config["evaluation_batch_size"], ledger, "candidate_before")
    event["before"] = before
    started = time.perf_counter()
    with CandidateTransaction(model, optimizer, controller) as transaction:
        controller.apply(model, optimizer, proposal)
        train_batch(model, optimizer, x, y, ledger, candidate=True)
        screened, _ = evaluate(model, sx[screen], sy[screen], config["evaluation_batch_size"], ledger, "candidate_screen")
        event["screen"] = screened
        if within(screened, reference_screen, config):
            full, _ = evaluate(model, sx, sy, config["evaluation_batch_size"], ledger, "candidate_full")
            event["after"] = full
            improves = (full["loss"] <= before["loss"]-config["minimum_loss_improvement"] or
                        full["accuracy"] >= before["accuracy"]+config["minimum_accuracy_improvement"])
            preserves = full["accuracy"] >= before["accuracy"] and full["loss"] <= before["loss"]
            if within(full, reference, config) and improves and preserves:
                controller.accepted += 1
                transaction.commit()
                event.update(accepted=True, reason="improves_current_task_with_bounded_growth")
            else:
                event["reason"] = "insufficient_improvement_or_task_deterioration"
        else:
            event["reason"] = "screen_exceeds_fixed_tolerance"
        if not event["accepted"]:
            ledger["rolled_back_updates"] += 1
    event["transaction_wall_seconds"] = time.perf_counter()-started
    if not event["accepted"]:
        fallback(model, optimizer, controller, x, y, ledger, config)
    return event


def random_slot(model, optimizer, controller, x, y, ledger, config, source, epoch):
    event = copy.deepcopy({k: source[k] for k in ("epoch", "proposal", "accepted", "layer_edit_counts", "attempted_edges")})
    event["rule"] = "source_acceptance_and_counts_only_no_random_select_feedback"
    if source["reason"] == "growth_budget_exhausted_or_no_inactive_edges":
        controller.proposals += 1
        train_batch(model, optimizer, x, y, ledger, controller)
        event["reason"] = source["reason"]
        return event
    if source["accepted"]:
        proposal = controller.propose(model, config["restore_fraction_original"],
            random_match=source["layer_edit_counts"], seed_parts=[11, epoch, source["proposal"]])
        with CandidateTransaction(model, optimizer, controller) as transaction:
            controller.apply(model, optimizer, proposal)
            train_batch(model, optimizer, x, y, ledger, candidate=True)
            controller.accepted += 1
            transaction.commit()
        event["reason"] = "unconditional_matched_random_restoration"
    else:
        controller.proposals += 1
        with CandidateTransaction(model, optimizer, controller):
            train_batch(model, optimizer, x, y, ledger, candidate=True)
            ledger["rolled_back_updates"] += 1
        fallback(model, optimizer, controller, x, y, ledger, config)
        event["reason"] = "matched_rejection_then_original_graph_adaptation"
    return event


def run(args):
    began = time.perf_counter()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    if config["model_seed"] != 11 or config["data_seed"] != 20261004:
        raise ValueError("唯一根不允许改变")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    source_run = ROOT / config["source_run"]
    source_reg = json.loads((source_run / "registration.json").read_text(encoding="utf-8"))
    for file, expected in source_reg["source_sha256"].items():
        if sha(ROOT / file) != expected:
            raise ValueError(f"旧运行源不匹配：{file}")
    parent_summary = json.loads((source_run / "summary.json").read_text(encoding="utf-8"))
    if not all(v["frozen"] for v in parent_summary["arms"].values()):
        raise ValueError("父实验未全部冻结")
    source_checkpoint = source_run / config["source_arm"] / "resume.pt"
    widths = source_reg["config"]["tasks"]["mnist_static"]["widths"]
    folder = ROOT / "experiments/runs" / config["version"] / "mnist_static"
    folder.mkdir(parents=True, exist_ok=True)
    lock = folder / "running.lock"
    if lock.exists():
        if pid_alive(int(lock.read_text())):
            raise RuntimeError("运行锁活动，拒绝重复进程")
        lock.unlink()
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    costpath = folder / "session_cost.json"
    past_cost = json.loads(costpath.read_text(encoding="utf-8")) if costpath.exists() else {"cumulative_wall_seconds": 0, "sessions": 0}
    try:
        registration = dict(config=config, source_sha256={p: sha(ROOT/p) for p in SOURCES},
            source_checkpoint_sha256=sha(source_checkpoint), device=str(device), torch=torch.__version__,
            parent_run=config["source_run"], task="mnist_static", widths=widths)
        regpath = folder / "registration.json"
        if regpath.exists():
            if json.loads(regpath.read_text(encoding="utf-8")) != registration or not args.resume:
                raise ValueError("运行已存在或源配置不同；只能相同实现--resume")
        else:
            write_json(regpath, registration)
        if (folder / "summary.json").exists():
            print(json.dumps({"already_complete": True, "run": str(folder)}), flush=True)
            return
        if past_cost["cumulative_wall_seconds"] >= config["maximum_wall_seconds"]:
            raise RuntimeError("阶段累计墙钟预算已用完，需新阶段登记；--resume不重置预算")
        write_json(folder / "running.json", dict(pid=os.getpid(), status="running"))
        startup = json.loads((source_run / "startup.json").read_text(encoding="utf-8"))
        if not startup["ok"] or startup["widths"] != widths:
            raise ValueError("相同尺寸既有启动证据无效")
        write_json(folder / "startup.json", dict(reused_from=str(source_run / "startup.json"),
            forward_samples=0, backward_samples=0, updates=0, additional_gpu_checks=0,
            note="复用相同尺寸既有启动及新增恢复CPU检查；实际训练首批有非有限保护"))
        bundle = load_bundle("mnist", include_confirm=False)
        write_json(folder / "data_manifest.json", bundle.manifest)
        fx, fy, sx, sy = [torch.from_numpy(array).to(device) for array in
                         (bundle.fit_x, bundle.fit_y, bundle.select_x, bundle.select_y)]
        screen = select_screen(sy, config["screen_samples"])
        parent = torch.load(source_checkpoint, map_location=device, weights_only=False)
        results = {}
        for arm in config["arms"]:
            af = folder / arm
            af.mkdir(exist_ok=True)
            if (af / "result.json").exists():
                results[arm] = json.loads((af / "result.json").read_text(encoding="utf-8"))
                continue
            saved = torch.load(af / "resume.pt", map_location=device, weights_only=False) if (af / "resume.pt").exists() else parent
            model = restore_model(saved, device, widths)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=0)
            # 同设备load_state_dict可能沿用state张量，不能让先跑的fixed改写父状态。
            optimizer.load_state_dict(copy.deepcopy(saved["optimizer"]))
            controller = EdgeRecoveryController(model, config["gradient_decay"])
            resumed = "recovery_controller" in saved
            checkpoint = saved if resumed else None
            if resumed:
                controller.load_state_dict(saved["recovery_controller"])
            restore_rng(saved["rng"])
            rows, events = (saved["rows"], saved["events"]) if resumed else ([], [])
            ledger = saved["ledger"] if resumed else ledger_new()
            start_counts = saved["start_counts"] if resumed else active_counts(model)
            next_epoch = saved["next_epoch"] if resumed else 1
            streak = saved["capacity_streak"] if resumed else 0
            arm_wall = saved["wall_seconds"] if resumed else 0.0
            for epoch in range(next_epoch, config["epochs"]+1):
                synchronize(device)
                started = time.perf_counter()
                for group in optimizer.param_groups:
                    group["lr"] = config["learning_rate"] * .5**sum(epoch >= k for k in config["lr_milestones"])
                order = np.random.default_rng(np.random.SeedSequence([20261004, 25+epoch])).permutation(len(fx))
                batches = [torch.as_tensor(ids, device=device) for ids in np.array_split(order, len(fx)//config["batch_size"])]
                losses = [train_batch(model, optimizer, fx[ids], fy[ids], ledger,
                          controller if arm != "fixed" else None) for ids in batches[:-1]]
                mean_loss = float(np.mean(losses))
                streak = streak+1 if mean_loss > config["capacity_mean_fit_loss"] else 0
                ids = batches[-1]
                event = None
                if arm == "recover" and streak >= config["capacity_patience"] and controller.proposals < config["max_proposals"]:
                    ref = results["fixed"]["rows"][epoch-1]
                    event = candidate(model, optimizer, controller, fx[ids], fy[ids], sx, sy,
                        screen, ref["select"], ref["screen"], ledger, config, epoch, start_counts)
                elif arm == "random":
                    sources = [v for v in results["recover"]["events"] if v["epoch"] == epoch]
                    if sources:
                        event = random_slot(model, optimizer, controller, fx[ids], fy[ids], ledger, config, sources[0], epoch)
                    else:
                        train_batch(model, optimizer, fx[ids], fy[ids], ledger, controller)
                else:
                    train_batch(model, optimizer, fx[ids], fy[ids], ledger, controller if arm != "fixed" else None)
                if event:
                    events.append(event)
                selected, scores = evaluate(model, sx, sy, config["evaluation_batch_size"], ledger, "epoch_select")
                payload = encoded(model)
                counts = active_counts(model)
                if any(n > s+int(s*config["maximum_growth_fraction_start"]) for n, s in zip(counts, start_counts)):
                    raise RuntimeError("累计恢复超每层增长预算")
                if arm == "random":
                    ref = results["recover"]["rows"][epoch-1]
                    updates = {k: ledger[k] for k in ("attempted_updates", "candidate_updates", "rolled_back_updates", "fallback_updates")}
                    if counts != ref["active_counts"] or updates != ref["updates"]:
                        raise RuntimeError("随机恢复规模/适应更新失配")
                net = ledger["attempted_updates"]-ledger["rolled_back_updates"]
                if config["fallback_adaptation"] and net != epoch*len(batches):
                    raise RuntimeError("正常净训练更新被候选吞掉")
                synchronize(device)
                elapsed = time.perf_counter()-started
                arm_wall += elapsed
                row = dict(epoch=epoch, select=selected, screen=metrics(
                    scores[screen.cpu().numpy()], bundle.select_y[screen.cpu().numpy()]),
                    mean_ordinary_fit_loss=mean_loss, capacity_proxy_streak=streak, active_counts=counts,
                    structure=structure_stats(model, payload), wall_seconds=elapsed,
                    updates={k: ledger[k] for k in ("attempted_updates", "candidate_updates", "rolled_back_updates", "fallback_updates")}, net_updates=net)
                rows.append(row)
                np.savez_compressed(af / f"select_{epoch:03d}.npz", logits=scores, labels=bundle.select_y)
                checkpoint = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), recovery_controller=controller.state_dict(),
                    parameter_grad=[None if p.grad is None else p.grad.detach().clone() for p in model.parameters()],
                    edge_grad=[None if layer.last_edge_grad is None else layer.last_edge_grad.clone() for layer in model.layers],
                    rng=rng_state(), rows=rows, events=events, ledger=ledger, start_counts=start_counts,
                    next_epoch=epoch+1, capacity_streak=streak, wall_seconds=arm_wall)
                save_checkpoint(af / "resume.pt", checkpoint)
                write_json(af / "progress.json", dict(rows=rows, events=events, ledger=ledger))
                print(json.dumps(dict(arm=arm, epoch=epoch, accuracy=selected["accuracy"],
                    loss=selected["loss"], edges=sum(counts), mean_fit_loss=mean_loss, event=event and event["reason"])), flush=True)
                if (folder / "request_stop").exists() or past_cost["cumulative_wall_seconds"]+time.perf_counter()-began > config["maximum_wall_seconds"]:
                    write_json(folder / "budget_stop.json", dict(reason="request_or_wall_budget", completed_arm=arm, completed_epoch=epoch))
                    return
            payload = encoded(model)
            (af / "final_model.bgs").write_bytes(payload)
            fit, predictions = evaluate(model, fx, fy, config["evaluation_batch_size"], ledger, "frozen_fit")
            np.savez_compressed(af / "fit_final.npz", logits=predictions, labels=bundle.fit_y)
            checkpoint["ledger"] = ledger
            save_checkpoint(af / "resume.pt", checkpoint)
            result = dict(frozen=True, final_epoch=config["epochs"], fit=fit, rows=rows, events=events,
                          structure=structure_stats(model, payload), wall_seconds=arm_wall, ledger=ledger)
            results[arm] = result
            write_json(af / "result.json", result)
        write_json(folder / "freeze.json", dict(arms={k: v["final_epoch"] for k, v in results.items()},
                   before_repeated_exploratory_evaluation=True))
        confirmed = load_bundle("mnist", include_confirm=True)
        cx, cy = torch.from_numpy(confirmed.confirm_x).to(device), torch.from_numpy(confirmed.confirm_y).to(device)
        for arm, result in results.items():
            af = folder / arm
            if "confirmation" in result:
                continue
            saved = torch.load(af / "resume.pt", map_location=device, weights_only=False)
            model = restore_model(saved, device, widths)
            confirmation, scores = evaluate(model, cx, cy, config["evaluation_batch_size"], result["ledger"], "repeated_exploration")
            np.savez_compressed(af / "confirm_final.npz", logits=scores, labels=confirmed.confirm_y)
            result["confirmation"] = confirmation
            saved["ledger"] = result["ledger"]
            save_checkpoint(af / "resume.pt", saved)
            write_json(af / "result.json", result)
        comparison = {arm: dict(select_accuracy_delta_vs_fixed=result["rows"][-1]["select"]["accuracy"]-results["fixed"]["rows"][-1]["select"]["accuracy"],
            select_loss_delta_vs_fixed=result["rows"][-1]["select"]["loss"]-results["fixed"]["rows"][-1]["select"]["loss"],
            accepted=sum(e["accepted"] for e in result["events"])) for arm, result in results.items()}
        summary = dict(task="mnist_static", registration=registration, arms=results, comparison=comparison,
            runtime_ratio_is_gate=False, random_counts_and_updates_matched=True,
            confirmation_status=config["confirmation_role"], current_session_wall_seconds=time.perf_counter()-began,
            peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            execution_note="仍原大小稠密gather；边恢复和文件增长不是推断加速或能耗证据")
        write_json(folder / "summary.json", summary)
        print(json.dumps(dict(complete=True, comparison=comparison)), flush=True)
    finally:
        write_json(costpath, dict(cumulative_wall_seconds=past_cost["cumulative_wall_seconds"]+time.perf_counter()-began,
                                 sessions=past_cost["sessions"]+1, checked_at_complete_epoch_boundary=True))
        lock.unlink(missing_ok=True)
        write_json(folder / "running.json", dict(pid=None, status="exited"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/h1_recovery_v1.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
