"""v6纯LIF三层基线：从头训练，反馈降率，累计最多25轮。"""
from __future__ import annotations
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
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
from braingrow.data.moving_mnist_v6 import training_data as load_v6, CONFIG as DATA_CONFIG
from braingrow.training_budget import TrainingBudget, load_budget
from tools import train_large_baseline as base
from tools import train_adaptive_baseline as adaptive

RUN_DIR = Path("experiments/runs/large_lif_baseline_v6")
POLICY = dict(adaptive.POLICY, control_interval_epochs=2)
SOURCES = (*adaptive.SOURCES, "tools/train_snn_baseline_v6.py",
    "src/braingrow/data/moving_mnist_v6.py", "src/braingrow/data/rendering.py",
    "src/braingrow/training_budget.py", "configs/next_snn_training.json")

def source_hashes():
    return {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in SOURCES}

def budget_for(epochs):
    configured = load_budget()
    if epochs > configured.maximum_epochs:
        raise ValueError("下一轮总预算最多25个epoch")
    return TrainingBudget(epochs, configured.control_interval_epochs, configured.fit_interval_epochs)

def training_data():
    fit, control = load_v6()
    path = Path("data/manifest_v6.json")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for key, source in (("renderer_source_sha256", "src/braingrow/data/moving_mnist_v6.py"),
                        ("shared_renderer_sha256", "src/braingrow/data/rendering.py")):
        if manifest[key] != hashlib.sha256(Path(source).read_bytes()).hexdigest():
            raise ValueError("v6渲染源码与已生成缓存不符")
    data = dict(version="v6", config=DATA_CONFIG, cache_sha256=manifest["cache_sha256"],
        manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        fit_original_ids=len(set(fit[2])), control_original_ids=len(set(control[2])))
    for name, pool, count in (("fit", fit, 4480), ("control", control, 1120)):
        x, y, ids = pool
        if (x.shape != (count, 8, 784) or y.shape != (count, 2)
                or not np.isfinite(x).all() or x.min() < 0 or x.max() > 1
                or not np.all((y[:, 0] >= 0) & (y[:, 0] < 10))
                or not np.all((y[:, 1] >= 0) & (y[:, 1] < 7))):
            raise ValueError("v6原像素或标签错误")
        data[name + "_ids_sha256"] = hashlib.sha256(ids.tobytes()).hexdigest()
        data[name + "_labels_sha256"] = hashlib.sha256(y.tobytes()).hexdigest()
    return fit, control, data

def configure_device():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用，不退回CPU")
    torch.manual_seed(11)
    torch.cuda.manual_seed_all(11)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(4)

def save_status(outdir, model, state, started, reason):
    result = base.status_result(model, state, started, reason)
    result.update(lr_policy=state["lr_policy"], data_version="v6",
        resume_command=".\\.venv\\Scripts\\python.exe -u -B -m tools.train_snn_baseline_v6 --resume --epochs 25")
    base.write_json(outdir / "summary.json", result)

def prepare(args):
    configure_device()
    fit, control, data = training_data()
    budget = budget_for(args.epochs)
    model = base.FixedLayeredSNN(base.WIDTHS, device="cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4, foreach=False)
    scheduler = adaptive.make_scheduler(optimizer)
    rng = np.random.default_rng(11)
    state = dict(completed_epochs=0, progress=None, maximum_epochs=args.epochs,
        micro_batch=args.micro_batch, costs=base.new_costs(), stop_reason=None,
        tracker=dict(best_control=None, accepted_control=None, control_stale=0,
            best_fit_loss=None, best_fit_score=None, fit_stale=0, activity_fault_checks=0),
        zero_gradient_updates=0, lr_policy=dict(scheduler=scheduler.state_dict(),
            last_control_epoch=0, reductions=0, last_reduction_epoch=None))
    outdir = args.outdir
    for directory in ("evaluations", "checkpoints"):
        (outdir / directory).mkdir()
    config = dict(schema=base.SCHEMA, experiment="snn_baseline_v6", widths=list(base.WIDTHS),
        root_model_seed=11, data_seed=20261004, data=data, fit_videos=len(fit[0]),
        control_videos=len(control[0]), maximum_epochs=args.epochs,
        effective_batch=args.batch_size, initial_micro_batch=args.micro_batch,
        control_interval=budget.control_interval_epochs, fit_interval=budget.fit_interval_epochs,
        control_evaluation_epochs=budget.evaluation_epochs("control"),
        fit_evaluation_epochs=budget.evaluation_epochs("fit"), learning_rate_schedule=POLICY,
        optimizer="AdamW(foreach=False)", initial_learning_rate=0.001, weight_decay=1e-4,
        gradient_clip_norm=1.0, objective="数字及运动硬标签交叉熵等权相加；无标签平滑",
        trainable="仅4组边权；无偏置；LIF泄露率和阈值固定", resources=model.resources(),
        neural_time_ms=160, step_ms=5, steps_per_frame=4, frames_per_video=8,
        video_time="归一化序列0至1；无物理帧率", layer_delay_steps=1,
        video_boundary="每视频清零电位、脉冲及输出", dtype="float32", matmul_precision="highest",
        surrogate="硬Fire前向；代理导数1/(1+5*abs(v-threshold))^2",
        initialization_scales=model.initial_scales,
        calibration="32拟合视频无标签校准，至多3次；四组边权梯度检查，无优化更新",
        plateau_early_stopping=False, activity_fault_stop="控制评价连续2次某层发放率<=1e-5或>=0.95",
        python=__import__("platform").python_version(), torch=torch.__version__, cuda_runtime=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0), gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
        start_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        start_status=subprocess.check_output(["git", "status", "--short"], text=True),
        source_sha256=source_hashes(), validation_arrays_unpacked=False, sealed_test_reads=0,
        maximum_train_forward_videos=args.epochs * len(fit[0]),
        maximum_train_backward_videos=args.epochs * len(fit[0]),
        maximum_control_forward_videos=len(budget.evaluation_epochs("control")) * len(control[0]),
        maximum_fit_evaluation_forward_videos=len(budget.evaluation_epochs("fit")) * len(fit[0]),
        maximum_calibration_forward_videos=96, diagnostic_forward_videos=32, diagnostic_backward_videos=32,
        budget_note="总轮数含恢复；OOM和中断回放额外计账；不读取原验证或封存测试")
    base.write_json(outdir / "config.json", config)
    for name in (*SOURCES, "docs/snn_baseline_v6_execution.md", "pyproject.toml"):
        destination = outdir / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(name, destination)
    ledger = base.Ledger(outdir, state["costs"])
    positions = rng.permutation(len(fit[0]))[:32]
    # 诊断抽样不推进正式训练的批次随机流。
    rng = np.random.default_rng(11)
    x = torch.as_tensor(fit[0][positions], device="cuda")
    y = torch.as_tensor(fit[1][positions], device="cuda")
    base.write_json(outdir / "diagnostic_positions.json", dict(fit_positions=positions.tolist(),
        raw_ids=fit[2][positions].tolist()))
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    calibration = base.calibrate(model, x, ledger)
    diagnostic = base.gradient_diagnostic(model, x, y, ledger)
    config["initialization_scales"] = list(model.initial_scales)
    base.write_json(outdir / "config.json", config)
    base.write_json(outdir / "prepared.json", dict(calibration=calibration, diagnostic=diagnostic,
        **base.status_result(model, state, started, "prepared_no_optimizer_updates")))
    base.save_resume(outdir, model, optimizer, rng, state)
    print("v6大图诊断通过，初始化状态已保存，优化更新0。", flush=True)

def run(args):
    outdir = args.outdir
    payload, recovered_path = base.load_resume(outdir)
    state = payload["state"]
    config = json.loads((outdir / "config.json").read_text(encoding="utf-8"))
    if config["source_sha256"] != source_hashes() or config["learning_rate_schedule"] != POLICY:
        raise ValueError("接续源码或学习率策略与记录不一致")
    if "lr_policy" not in state or config["effective_batch"] != args.batch_size:
        raise ValueError("检查点缺少调度状态或有效批量不符")
    budget = budget_for(args.epochs)
    if config["maximum_epochs"] != args.epochs or state["maximum_epochs"] != args.epochs:
        raise ValueError("恢复须沿用初始总预算，不能更改累计轮数")
    budget.remaining_epochs(state["completed_epochs"],
        None if state["progress"] is None else state["progress"]["epoch"])
    if args.prepare_only:
        print(f"恢复核对通过：完成{state['completed_epochs']}遍，lr={payload['optimizer']['param_groups'][0]['lr']:g}；"
              "未前向、未反传、未更新参数。", flush=True)
        return
    if state.get("stop_reason") == "persistent_activity_fault":
        raise ValueError("已因活动故障停止，须先诊断并另行登记新实验")
    if state["completed_epochs"] >= args.epochs and state["progress"] is None:
        print("已达到本次预算，不重复训练。", flush=True)
        return
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
    stop = adaptive.FileStopRequested(outdir)
    original_sigint = signal.signal(signal.SIGINT, stop.receive)
    started = time.perf_counter()
    model = optimizer = rng = None
    try:
        fit, control, data = training_data()
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
        scheduler = adaptive.make_scheduler(optimizer)
        scheduler.load_state_dict(state["lr_policy"]["scheduler"])
        ledger = base.Ledger(outdir, state["costs"])
        before = state["costs"].copy()
        ledger.recover_observed_costs()
        state["micro_batch"] = min(state["micro_batch"], args.micro_batch)
        state["stop_reason"] = None
        with (outdir / "resume_events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(timestamp=time.time(), checkpoint=str(recovered_path),
                completed_epochs=state["completed_epochs"], maximum_epochs=args.epochs,
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
                if epoch in budget.evaluation_epochs("control"):
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
                    event = adaptive.control_feedback(scheduler, state, metrics, epoch)
                    if event is not None:
                        adaptive.append_policy_event(outdir, event)
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
                if epoch in budget.evaluation_epochs("fit"):
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
    parser.add_argument("--outdir", type=Path, default=RUN_DIR)
    parser.add_argument("--epochs", type=int, default=load_budget().maximum_epochs)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--micro-batch", type=int, default=32)
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--resume", action="store_true", help="已有目录则恢复；不存在则从头准备")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--request-stop", action="store_true")
    args = parser.parse_args()
    try:
        budget_for(args.epochs)
        if min(args.batch_size, args.micro_batch, args.progress_every) < 1 or args.micro_batch > args.batch_size:
            raise ValueError("批量和间隔须为正，微批量不能超过有效批量")
    except ValueError as exc:
        parser.error(str(exc))
    if args.check_only:
        _, _, data = training_data()
        print(json.dumps(dict(data=data, maximum_epochs=args.epochs, training_started=False), ensure_ascii=False, indent=2))
        return
    if args.request_stop:
        if not (args.outdir / "resume.pt").exists():
            parser.error("该目录无可用训练检查点")
        base.write_json(args.outdir / "stop_request.json", dict(timestamp=time.time(), reason="user_requested_pause"))
        return
    if args.outdir.exists() and not args.resume:
        parser.error("目录已存在，请用--resume；拒绝覆盖证据")
    fresh = not args.outdir.exists()
    if fresh:
        args.outdir.mkdir(parents=True, exist_ok=False)
    configure_device()
    with adaptive.RunLock(args.outdir):
        if fresh:
            prepare(args)
        run(args)

if __name__ == "__main__":
    main()
