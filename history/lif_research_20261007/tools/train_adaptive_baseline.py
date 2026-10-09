"""从暂停的三层基线接续：控制池交叉熵平台降学习率，保留原实验。"""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import ctypes
import hashlib
import json
import math
import shutil
import signal
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from tools import train_large_baseline as base


RUN_DIR = Path("experiments/runs/large_lif_baseline_adaptive_v1")
SOURCES = (*base.SOURCES, "tools/train_adaptive_baseline.py")
POLICY = dict(kind="ReduceLROnPlateau", metric="control.loss", mode="min",
              factor=0.5, patience=1, threshold=0.01, threshold_mode="rel",
              cooldown=1, min_lr=1e-5, eps=1e-8, control_interval_epochs=5,
              patience_note="连续2次完整控制评价未改善才下降；冷却1次评价")


class FileStopRequested(base.StopRequested):
    """后台运行也能请求在有效批次/评价批次边界安全暂停。"""
    def __init__(self, outdir):
        self.path = outdir / "stop_request.json"
        super().__init__()

    @property
    def requested(self):
        return self._requested or self.path.exists()

    @requested.setter
    def requested(self, value):
        self._requested = value


def process_alive(pid):
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            if ctypes.get_last_error() == 87:  # PID不存在；其他错误不能证明已结束。
                return False
            return True
        try:
            code = ctypes.c_ulong()
            return not kernel.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)  # Windows绝不能走此分支。
        return True
    except ProcessLookupError:
        return False


class RunLock:
    def __init__(self, outdir):
        self.path = outdir / "running.json"

    def __enter__(self):
        if self.path.exists():
            pid = int(json.loads(self.path.read_text(encoding="utf-8"))["pid"])
            if process_alive(pid):
                raise RuntimeError(f"该目录已有运行进程{pid}，禁止重复训练")
            self.path.replace(self.path.with_name(f"running_stale_{time.time_ns()}.json"))
        with self.path.open("x", encoding="utf-8") as handle:
            json.dump(dict(pid=os.getpid(), started=time.time()), handle)
        return self

    def __exit__(self, *_):
        self.path.unlink()


def source_hashes():
    return {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in SOURCES}


def make_scheduler(optimizer):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, **{key: POLICY[key] for key in
                      ("mode", "factor", "patience", "threshold", "threshold_mode",
                       "cooldown", "min_lr", "eps")})


def scalar_metrics(metrics):
    return {key: metrics[key] for key in
            ("loss", "digit_accuracy", "motion_accuracy", "joint_accuracy")}


def saved_controls(parent, state):
    """只继承安全检查点已经完成的评价，不读取之后的在途结果。"""
    rows = []
    for path in sorted((parent / "evaluations").glob("control_*.json")):
        epoch = int(path.stem.split("_")[-1])
        if epoch <= state["completed_epochs"]:
            rows.append(dict(epoch=epoch, **scalar_metrics(json.loads(path.read_text(encoding="utf-8")))))
    progress = state["progress"]
    if progress is not None and progress.get("control") is not None:
        rows.append(dict(epoch=progress["epoch"], **scalar_metrics(progress["control"])))
    return rows


def adopt_history(optimizer, state, rows):
    """继承已观察平台；如已满足条件，仅在接续时实际降一次，不虚构历史降率。"""
    scheduler = make_scheduler(optimizer)
    best, bad, previous_epoch = math.inf, 0, 0
    for row in rows:
        loss, epoch = float(row["loss"]), int(row["epoch"])
        if not math.isfinite(loss) or loss < 0 or epoch <= previous_epoch:
            raise ValueError("历史控制结果的损失或遍数不合法")
        if loss < best * (1 - POLICY["threshold"]):
            best, bad = loss, 0
        else:
            bad += 1
        previous_epoch = epoch
    before = [float(group["lr"]) for group in optimizer.param_groups]
    reduced = bad > POLICY["patience"] and any(lr > POLICY["min_lr"] + POLICY["eps"] for lr in before)
    if reduced:
        for group in optimizer.param_groups:
            group["lr"] = max(float(group["lr"]) * POLICY["factor"], POLICY["min_lr"])
        state["tracker"]["control_stale"] = state["tracker"]["fit_stale"] = 0
    progress = state["progress"]
    effective_epoch = state["completed_epochs"] + 1
    if progress is not None:
        effective_epoch = progress["epoch"] + (progress["stage"] != "train" or progress["next_sample"] >= 4480)
    after = [float(group["lr"]) for group in optimizer.param_groups]
    scheduler_state = scheduler.state_dict()
    scheduler_state.update(best=best, num_bad_epochs=0 if reduced else min(bad, POLICY["patience"]),
                           cooldown_counter=POLICY["cooldown"] if reduced else 0,
                           last_epoch=len(rows), _last_lr=after)
    scheduler.load_state_dict(scheduler_state)
    state["lr_policy"] = dict(scheduler=scheduler.state_dict(),
        last_control_epoch=previous_epoch, reductions=int(reduced),
        last_reduction_epoch=effective_epoch if reduced else None)
    # 若中途暂停，明确同一遍前半段与接续部分使用的不同学习率。
    if progress is not None:
        progress["lr_segments"] = [dict(start_sample=0, learning_rate=before[0])]
        if progress["stage"] == "train" and progress["next_sample"] < 4480 and reduced:
            progress["lr_segments"].append(dict(start_sample=progress["next_sample"], learning_rate=after[0]))
    return dict(kind="adopt_existing_control_feedback", historical_controls=rows,
                historical_bad_checks=bad, learning_rates_before=before, learning_rates_after=after,
                reduced=reduced, effective_epoch=effective_epoch,
                note="历史实际学习率保持原记录；本次最多降一次，未新增模型评价或优化更新")


def control_feedback(scheduler, state, metrics, epoch):
    policy = state["lr_policy"]
    if epoch == policy["last_control_epoch"]:
        return None  # 已完成评价阶段恢复不能再次累计平台次数。
    if epoch < policy["last_control_epoch"] or not math.isfinite(float(metrics["loss"])):
        raise ValueError("控制反馈顺序或损失不合法")
    before = [float(group["lr"]) for group in scheduler.optimizer.param_groups]
    scheduler.step(metrics["loss"])
    after = [float(group["lr"]) for group in scheduler.optimizer.param_groups]
    reduced = before != after
    if reduced:
        policy["reductions"] += 1
        policy["last_reduction_epoch"] = epoch + 1
        state["tracker"]["control_stale"] = state["tracker"]["fit_stale"] = 0
    policy["last_control_epoch"] = epoch
    policy["scheduler"] = scheduler.state_dict()
    return dict(kind="control_feedback", epoch=epoch, control=scalar_metrics(metrics),
                learning_rates_before=before, learning_rates_after=after, reduced=reduced,
                effective_epoch=epoch + 1, bad_checks=scheduler.num_bad_epochs,
                cooldown_checks=scheduler.cooldown_counter, best_control_loss=scheduler.best)


def append_policy_event(outdir, event):
    with (outdir / "learning_rate_events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(timestamp=time.time(), **event), ensure_ascii=False) + "\n")


def fork_paused_run(parent, outdir, args):
    parent = parent.resolve()
    if outdir.resolve() == parent or outdir.exists():
        raise FileExistsError("接续须使用不存在的新目录；原实验保留，已有接续目录使用--resume")
    config = json.loads((parent / "config.json").read_text(encoding="utf-8"))
    if config["source_sha256"] != base.source_hashes():
        raise ValueError("原实验实现与当前共享模型/训练代码不一致，拒绝接续")
    payload, checkpoint_path = base.load_resume(parent, expected_widths=base.WIDTHS)
    state = payload["state"]
    if state.get("stop_reason") not in ("user_interrupt", "user_interrupt_during_evaluation"):
        raise ValueError("原实验须先Ctrl+C安全暂停并保存，再接续学习率策略")
    if config["effective_batch"] != args.batch_size:
        raise ValueError("接续不能改变有效批量")
    if args.epochs > state["maximum_epochs"] and not args.extension_reason:
        raise ValueError("扩大原遍数预算须登记--extension-reason")
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))
                                  for _ in payload["optimizer"]["param_groups"][0]["params"]],
                                 lr=0.001, weight_decay=1e-4, foreach=False)
    optimizer.load_state_dict(payload["optimizer"])
    event = adopt_history(optimizer, state, saved_controls(parent, state))
    payload["optimizer"] = optimizer.state_dict()
    state["stop_reason"] = None
    state["maximum_epochs"] = max(state["maximum_epochs"], args.epochs)
    inherited = dict(parent_directory=str(parent), checkpoint=str(checkpoint_path),
        checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        config_sha256=hashlib.sha256((parent / "config.json").read_bytes()).hexdigest(),
        completed_epochs=state["completed_epochs"],
        progress=None if state["progress"] is None else
            {k: v for k, v in state["progress"].items() if k not in ("order", "control", "fit")},
        inherited_costs=state["costs"].copy(), policy_adoption=event,
        preserved="全部边权、Adam一/二阶矩与步数、NumPy/Torch/CUDA RNG、样本排列与位置、历史成本")
    outdir.mkdir(parents=True)
    (outdir / "evaluations").mkdir()
    (outdir / "checkpoints").mkdir()
    new_config = dict(config)
    new_config.update(source_sha256=source_hashes(), learning_rate_schedule=POLICY,
        parent_experiment=inherited, maximum_epochs=args.epochs,
        earliest_plateau_stop_epoch=None, minimum_epochs=50,
        plateau_minimum_reductions=2, plateau_grace_epochs_after_last_reduction=20,
        start_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        start_status=subprocess.check_output(["git", "status", "--short"], text=True),
        budget_note="总遍数包含原实验已完成部分；历史成本累积保留，不重复初始化或评价；额外重放另计")
    base.write_json(outdir / "config.json", new_config)
    base.write_json(outdir / "inheritance.json", inherited)
    shutil.copy2(parent / "config.json", outdir / "parent_config.json")
    snapshot = outdir / "source_snapshot"
    for name in (*SOURCES, "docs/adaptive_learning_rate_execution.md", "pyproject.toml"):
        destination = snapshot / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(name, destination)
    base.atomic_torch_save(outdir / "resume.pt", payload)
    append_policy_event(outdir, event)
    base.Ledger(outdir, state["costs"]).add("adaptive_policy_adoption", epoch=state["completed_epochs"])
    print(f"安全状态已接续到{outdir}；完成{state['completed_epochs']}遍；"
          f"学习率{event['learning_rates_before']}→{event['learning_rates_after']}。", flush=True)
    return inherited


def save_status(outdir, model, state, started, reason):
    result = base.status_result(model, state, started, reason)
    result.update(lr_policy=state["lr_policy"],
                  resume_command=".\\.venv\\Scripts\\python.exe -u -m tools.train_adaptive_baseline --resume")
    base.write_json(outdir / "summary.json", result)


def run(args):
    outdir = args.outdir
    payload, recovered_path = base.load_resume(outdir)
    state = payload["state"]
    config = json.loads((outdir / "config.json").read_text(encoding="utf-8"))
    if config["source_sha256"] != source_hashes() or config["learning_rate_schedule"] != POLICY:
        raise ValueError("接续源码或学习率策略与记录不一致")
    if "lr_policy" not in state or config["effective_batch"] != args.batch_size:
        raise ValueError("检查点缺少调度状态或有效批量不符")
    if args.prepare_only:
        print(f"恢复核对通过：完成{state['completed_epochs']}遍，lr={payload['optimizer']['param_groups'][0]['lr']:g}；"
              "未前向、未反传、未更新参数。", flush=True)
        return
    if state.get("stop_reason") in ("control_and_fit_plateau", "persistent_activity_fault") and not args.extension_reason:
        raise ValueError("平台/活动故障后继续须登记--extension-reason")
    if state["completed_epochs"] >= args.epochs and state["progress"] is None:
        print("已达到本次预算，不重复训练。", flush=True)
        return
    if args.epochs > state["maximum_epochs"] and not args.extension_reason:
        raise ValueError("扩大遍数预算须登记--extension-reason")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用，不退回CPU")
    if recovered_path.name == "resume_previous.pt" and (outdir / "resume.pt").exists():
        (outdir / "resume.pt").replace(outdir / f"resume_unreadable_{time.time_ns()}.pt")
    torch.manual_seed(11)
    torch.cuda.manual_seed_all(11)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    old_request = outdir / "stop_request.json"
    if old_request.exists():
        old_request.replace(outdir / f"stop_request_acknowledged_{time.time_ns()}.json")
    stop = FileStopRequested(outdir)
    original_sigint = signal.signal(signal.SIGINT, stop.receive)
    started = time.perf_counter()
    model = optimizer = rng = None
    try:
        fit, control, data = base.training_data()
        if config["data"] != data:
            raise ValueError("数据协议/哈希与原登记不符")
        device = torch.device("cuda")
        x, y = torch.as_tensor(fit[0], device=device), torch.as_tensor(fit[1], device=device)
        cx, cy = torch.as_tensor(control[0], device=device), torch.as_tensor(control[1], device=device)
        model = base.FixedLayeredSNN(base.WIDTHS, device=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4, foreach=False)
        rng = np.random.default_rng(11)
        base.restore_checkpoint(payload, model, optimizer, rng)
        del payload
        scheduler = make_scheduler(optimizer)
        scheduler.load_state_dict(state["lr_policy"]["scheduler"])
        ledger = base.Ledger(outdir, state["costs"])
        before = state["costs"].copy()
        ledger.recover_observed_costs()
        state["micro_batch"] = min(state["micro_batch"], args.micro_batch)
        state["maximum_epochs"] = max(state["maximum_epochs"], args.epochs)
        if args.extension_reason:
            state["tracker"]["control_stale"] = state["tracker"]["fit_stale"] = 0
            state["tracker"]["activity_fault_checks"] = state["zero_gradient_updates"] = 0
        state["stop_reason"] = None
        with (outdir / "resume_events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(timestamp=time.time(), checkpoint=str(recovered_path),
                completed_epochs=state["completed_epochs"], maximum_epochs=args.epochs,
                extension_reason=args.extension_reason,
                observed_costs_beyond_checkpoint={k: state["costs"][k] - before.get(k, 0) for k in state["costs"]},
                note="调度器和随机状态恢复；强制结束后已记录的回放成本保留"), ensure_ascii=False) + "\n")
        torch.cuda.reset_peak_memory_stats()
        print(f"GPU接续：隐藏层{model.widths}，已完成{state['completed_epochs']}遍，"
              f"lr={optimizer.param_groups[0]['lr']:g}，微批量{state['micro_batch']}。", flush=True)
        reason = "registered_epoch_budget"
        while state["completed_epochs"] < args.epochs:
            epoch = state["completed_epochs"] + 1
            if state["progress"] is None:
                state["progress"] = dict(epoch=epoch, order=rng.permutation(len(x)), next_sample=0,
                    loss_sum=0.0, stage="train", control=None, fit=None,
                    lr_segments=[dict(start_sample=0, learning_rate=float(optimizer.param_groups[0]["lr"]))])
            progress = state["progress"]
            epoch_started = time.perf_counter()
            if progress["stage"] == "train":
                model.train()
                while progress["next_sample"] < len(x):
                    positions = torch.as_tensor(progress["order"][progress["next_sample"]:
                        progress["next_sample"] + args.batch_size], device=device)
                    loss_sum, micro, norm = base.training_update(model, optimizer, x, y, positions,
                        state["micro_batch"], ledger, epoch)
                    state["micro_batch"] = micro
                    progress["loss_sum"] += loss_sum
                    progress["next_sample"] += len(positions)
                    state["zero_gradient_updates"] = state["zero_gradient_updates"] + 1 if norm == 0 else 0
                    if state["zero_gradient_updates"] >= 3:
                        base.save_resume(outdir, model, optimizer, rng, state)
                        raise RuntimeError("连续3批梯度全零，停止并保存")
                    batch_number = math.ceil(progress["next_sample"] / args.batch_size)
                    if batch_number % args.progress_every == 0 or progress["next_sample"] == len(x):
                        print(f"第{epoch}/{args.epochs}遍 {progress['next_sample']}/{len(x)}视频 | "
                              f"损失{loss_sum / len(positions):.4f} | lr={optimizer.param_groups[0]['lr']:g} | "
                              f"微批量{micro} | 本段{time.perf_counter() - epoch_started:.1f}秒", flush=True)
                    if stop.requested:
                        reason = "user_interrupt"
                        break
                if stop.requested:
                    break
                progress["stage"] = "control"
            if progress["stage"] == "control":
                if epoch % 5 == 0:
                    print(f"第{epoch}遍：评价1120个控制视频…", flush=True)
                    metrics, arrays = base.evaluate_pool(model, cx, cy, state["micro_batch"], ledger, "control", epoch, stop)
                    progress["control"] = metrics
                    base.write_json(outdir / "evaluations" / f"control_{epoch:03d}.json", metrics)
                    np.savez_compressed(outdir / "evaluations" / f"control_{epoch:03d}.npz", **arrays, raw_ids=control[2])
                    tracker = state["tracker"]
                    if base.accepted_control(metrics, tracker["accepted_control"]):
                        tracker["accepted_control"], tracker["control_stale"] = metrics, 0
                    else:
                        tracker["control_stale"] += 1
                    event = control_feedback(scheduler, state, metrics, epoch)
                    if event is not None:
                        append_policy_event(outdir, event)
                        if event["reduced"]:
                            print(f"控制损失连续未改善，学习率{event['learning_rates_before']}→"
                                  f"{event['learning_rates_after']}，第{epoch + 1}遍生效。", flush=True)
                    # best检查点带已处理调度状态与下一阶段，恢复不重复调度。
                    progress["stage"] = "fit"
                    score = sum(metrics[k] for k in ("digit_accuracy", "motion_accuracy", "joint_accuracy"))
                    if tracker["best_control"] is None or score > sum(tracker["best_control"][k]
                            for k in ("digit_accuracy", "motion_accuracy", "joint_accuracy")):
                        tracker["best_control"] = dict(epoch=epoch, **metrics)
                        base.atomic_torch_save(outdir / "best.pt", base.checkpoint_payload(model, optimizer, rng, state))
                    rates = metrics["spikes_per_lif_step_by_layer"]
                    faulty = any(r <= 0.00001 or r >= 0.95 for r in rates)
                    tracker["activity_fault_checks"] = tracker["activity_fault_checks"] + 1 if faulty else 0
                    base.atomic_torch_save(outdir / "checkpoints" / f"epoch_{epoch:03d}_model.pt",
                        dict(widths=base.WIDTHS, epoch=epoch, model=model.state_dict(), metrics=metrics))
                    print(f"控制：数字{metrics['digit_accuracy']:.2%} 运动{metrics['motion_accuracy']:.2%} "
                          f"联合{metrics['joint_accuracy']:.2%} | 损失{metrics['loss']:.4f} | Fire率{rates}", flush=True)
                progress["stage"] = "fit"
            if progress["stage"] == "fit":
                if epoch % 10 == 0:
                    print(f"第{epoch}遍：评价4480个拟合视频…", flush=True)
                    metrics, arrays = base.evaluate_pool(model, x, y, state["micro_batch"], ledger, "fit", epoch, stop)
                    progress["fit"] = metrics
                    base.write_json(outdir / "evaluations" / f"fit_{epoch:03d}.json", metrics)
                    np.savez_compressed(outdir / "evaluations" / f"fit_{epoch:03d}.npz", **arrays, raw_ids=fit[2])
                    tracker = state["tracker"]
                    score = sum(metrics[k] for k in ("digit_accuracy", "motion_accuracy", "joint_accuracy"))
                    loss_improved = tracker["best_fit_loss"] is None or metrics["loss"] < tracker["best_fit_loss"] - 0.001
                    score_improved = tracker["best_fit_score"] is None or score > tracker["best_fit_score"] + 0.002
                    tracker["fit_stale"] = 0 if loss_improved or score_improved else tracker["fit_stale"] + 1
                    tracker["best_fit_loss"] = metrics["loss"] if tracker["best_fit_loss"] is None else min(tracker["best_fit_loss"], metrics["loss"])
                    tracker["best_fit_score"] = score if tracker["best_fit_score"] is None else max(tracker["best_fit_score"], score)
                    print(f"拟合：数字{metrics['digit_accuracy']:.2%} 运动{metrics['motion_accuracy']:.2%} "
                          f"联合{metrics['joint_accuracy']:.2%}", flush=True)
                progress["stage"] = "finalize"
            row = dict(epoch=epoch, optimizer_mean_loss=progress["loss_sum"] / len(x),
                learning_rate=progress["lr_segments"][0]["learning_rate"],
                learning_rate_segments=progress["lr_segments"], next_learning_rate=float(optimizer.param_groups[0]["lr"]),
                lr_policy=state["lr_policy"], control=progress["control"], fit=progress["fit"],
                micro_batch=state["micro_batch"], costs=state["costs"].copy(),
                wall_seconds_this_session=time.perf_counter() - started,
                gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                process_peak_working_set_bytes=base.peak_working_set_bytes())
            base.write_json(outdir / f"epoch_{epoch:03d}.json", row)
            state["completed_epochs"], state["progress"] = epoch, None
            tracker, policy = state["tracker"], state["lr_policy"]
            if tracker["activity_fault_checks"] >= 2:
                reason = "persistent_activity_fault"
            elif (epoch >= 50 and epoch % 10 == 0 and policy["reductions"] >= 2
                  and epoch >= policy["last_reduction_epoch"] + 20
                  and tracker["control_stale"] >= 4 and tracker["fit_stale"] >= 2):
                reason = "control_and_fit_plateau"
            state["stop_reason"] = reason if reason != "registered_epoch_budget" else None
            base.save_resume(outdir, model, optimizer, rng, state)
            print(f"第{epoch}遍完成，优化平均损失{row['optimizer_mean_loss']:.4f}，"
                  f"下遍lr={row['next_learning_rate']:g}，检查点已保存。", flush=True)
            if reason != "registered_epoch_budget" or stop.requested:
                if stop.requested:
                    reason = "user_interrupt"
                break
        if state["progress"] is not None or stop.requested:
            state["stop_reason"] = reason
            base.save_resume(outdir, model, optimizer, rng, state)
        save_status(outdir, model, state, started, reason)
        print(f"本次运行结束：{reason}；完成{state['completed_epochs']}遍；{outdir.resolve()}。", flush=True)
    except base.EvaluationInterrupted:
        state["stop_reason"] = "user_interrupt_during_evaluation"
        base.save_resume(outdir, model, optimizer, rng, state)
        save_status(outdir, model, state, started, state["stop_reason"])
        print("评价中断已保存；恢复重跑未完成评价，保留已发生成本。", flush=True)
    except Exception as exc:
        base.write_json(outdir / "error.json", dict(error=repr(exc), timestamp=time.time(),
            note="上一个安全检查点保留；在途或强制终止的回放按日志计额外成本"))
        raise
    finally:
        signal.signal(signal.SIGINT, original_sigint)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--from-run", type=Path)
    selection.add_argument("--resume", action="store_true")
    selection.add_argument("--request-stop", action="store_true")
    parser.add_argument("--outdir", type=Path, default=RUN_DIR)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--micro-batch", type=int, default=32)
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--extension-reason", default="")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.micro_batch, args.progress_every) < 1 or args.micro_batch > args.batch_size:
        parser.error("遍数/批量/进度间隔须为正，微批量不得超过有效批量")
    if args.request_stop:
        if not (args.outdir / "config.json").exists():
            parser.error("目标接续目录不存在")
        base.write_json(args.outdir / "stop_request.json", dict(timestamp=time.time(), reason="user_requested_pause"))
        print("安全暂停请求已写入；等待当前批次结束、保存检查点并退出。", flush=True)
        return
    if args.from_run:
        fork_paused_run(args.from_run, args.outdir, args)
        args.from_run = None
    with RunLock(args.outdir):
        run(args)


if __name__ == "__main__":
    main()
