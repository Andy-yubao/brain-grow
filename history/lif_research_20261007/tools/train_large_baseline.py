"""5000/3000/2000固定LIF基线；CUDA、梯度累积、稀疏评价及安全恢复。"""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import hashlib
import json
import math
import pickle
import platform
import shutil
import signal
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from braingrow.evaluation.metrics import evaluate
from braingrow.network.fixed_layered import FixedLayeredSNN
from tools.run_layered_gradient import loss_for, peak_working_set_bytes, training_data, write_json


WIDTHS = (5000, 3000, 2000)
RUN_DIR = Path("experiments/runs/large_lif_baseline_v1")
SOURCES = ("src/braingrow/network/fixed_layered.py", "src/braingrow/network/layered_gradient.py",
           "tools/train_large_baseline.py", "tools/run_layered_gradient.py",
           "src/braingrow/evaluation/metrics.py", "src/braingrow/data/moving_mnist_v5.py")
SCHEMA = 1


def new_costs():
    return dict(train_forward_videos=0, train_backward_videos=0,
                calibration_forward_videos=0, diagnostic_forward_videos=0,
                diagnostic_backward_videos=0, control_forward_videos=0,
                fit_evaluation_forward_videos=0, control_metric_evaluations=0,
                fit_metric_evaluations=0, failed_forward_attempt_videos=0,
                failed_backward_attempt_videos=0, microbatch_retries=0)


class Ledger:
    def __init__(self, outdir, costs):
        self.outdir, self.costs = outdir, costs

    def add(self, kind, *, epoch=0, videos=0, **increments):
        for key, value in increments.items():
            self.costs[key] += int(value)
        with (self.outdir / "cost_journal.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(kind=kind, epoch=epoch, videos=videos,
                cumulative_costs=self.costs.copy(), timestamp=time.time()), ensure_ascii=False) + "\n")
            handle.flush()

    def recover_observed_costs(self):
        path = self.outdir / "cost_journal.jsonl"
        if not path.exists():
            return
        if path.stat().st_size and not path.read_bytes().endswith(b"\n"):
            with path.open("a", encoding="utf-8") as handle:
                handle.write("\n")
        # 强制结束可能留下最后一行残片；上一完整事件仍是已观察成本证据。
        for line in reversed(path.read_text(encoding="utf-8").splitlines()):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            for key, value in row["cumulative_costs"].items():
                self.costs[key] = max(self.costs.get(key, 0), value)
            break


class StopRequested:
    def __init__(self):
        self.requested = False

    def receive(self, *_):
        if not self.requested:
            print("\n收到停止请求；完成当前有效批次后保存检查点，请等待退出。", flush=True)
        self.requested = True


class EvaluationInterrupted(Exception):
    pass


def source_hashes():
    return {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in SOURCES}


def learning_rate(epoch):
    return 0.001 if epoch <= 50 else (0.0003 if epoch <= 80 else 0.0001)


def checkpoint_payload(model, optimizer, rng, state):
    return dict(schema=SCHEMA, widths=model.widths, model=model.state_dict(),
                optimizer=optimizer.state_dict(), numpy_rng=rng.bit_generator.state,
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if model.weights[0].is_cuda else None,
                initial_scales=model.initial_scales, state=state)


def atomic_torch_save(path, value, previous=None):
    temporary = path.with_name(path.name + ".pending")
    torch.save(value, temporary)
    if previous is not None and path.exists():
        path.replace(previous)
    temporary.replace(path)


def save_resume(outdir, model, optimizer, rng, state):
    if model.weights[0].is_cuda:
        torch.cuda.synchronize()
    atomic_torch_save(outdir / "resume.pt", checkpoint_payload(model, optimizer, rng, state),
                      outdir / "resume_previous.pt")


def load_resume(outdir, expected_widths=WIDTHS):
    paths = [outdir / "resume.pt", outdir / "resume_previous.pt"]
    errors = []
    for path in paths:
        if not path.exists():
            continue
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if payload["schema"] != SCHEMA or tuple(payload["widths"]) != tuple(expected_widths):
                raise ValueError("检查点版本或三层宽度不符")
            return payload, path
        except (OSError, RuntimeError, EOFError, ValueError, KeyError, TypeError, pickle.UnpicklingError) as exc:
            errors.append(f"{path}: {exc}")
    raise RuntimeError("没有可用的大容量基线检查点。" + "；".join(errors))


def restore_checkpoint(payload, model, optimizer, rng):
    if tuple(payload["widths"]) != model.widths:
        raise ValueError("恢复图宽度不符")
    model.load_state_dict(payload["model"])
    model.initial_scales = payload["initial_scales"]
    optimizer.load_state_dict(payload["optimizer"])
    rng.bit_generator.state = payload["numpy_rng"]
    torch.set_rng_state(payload["torch_rng"])
    if payload["cuda_rng"] is not None:
        torch.cuda.set_rng_state_all(payload["cuda_rng"])


@torch.no_grad()
def calibrate(model, x, ledger):
    trials = []
    for attempt in range(1, 4):
        _, activity = model(x, trace=True, voltage_diagnostics=True)
        ledger.add("initialization_calibration", videos=len(x), calibration_forward_videos=len(x))
        rates = [float(c.sum() / (len(x) * 32 * n))
                 for c, n in zip(activity["per_unit_fire"], model.widths)]
        rms = activity["positive_voltage_rms"].tolist()
        row = dict(attempt=attempt, rates=rates, positive_voltage_rms=rms,
                   positive_voltage_max=activity["positive_voltage_max"].tolist(),
                   scales=list(model.initial_scales))
        invalid = [i for i, rate in enumerate(rates) if rate < 0.005 or rate > 0.25]
        if not invalid:
            row["accepted"] = True
            trials.append(row)
            return trials
        if attempt == 3:
            row["accepted"] = False
            trials.append(row)
            write_json(ledger.outdir / "calibration_failed.json", dict(trials=trials))
            raise RuntimeError("三次无标签初始化校准后仍有失活或高发放层，停止准备；见calibration_failed.json")
        # 每次仅调整最靠前的异常层，避免用失活上游的统计调整下游。
        layer = invalid[0]
        if rates[layer] < 0.005:
            if rms[layer] <= 1e-12:
                raise RuntimeError(f"第{layer + 1}层完全无正电位，须检查上游脉冲或实现")
            factor = min(8.0, max(1.25, 0.30 / rms[layer]))
        else:
            factor = max(0.25, min(0.8, math.sqrt(0.05 / rates[layer])))
        model.weights[layer].mul_(factor)
        model.initial_scales[layer] *= factor
        row["adjustment"] = dict(layer=layer + 1, factor=factor)
        trials.append(row)
    raise AssertionError("不可达")


def gradient_diagnostic(model, x, y, ledger):
    model.zero_grad(set_to_none=True)
    potential, activity = model(x, trace=True)
    ledger.add("diagnostic_forward", videos=len(x), diagnostic_forward_videos=len(x))
    loss = loss_for(potential, y)
    loss.backward()
    ledger.add("diagnostic_backward", videos=len(x), diagnostic_backward_videos=len(x))
    gradients = [float(w.grad.abs().sum()) for w in model.weights]
    if not all(math.isfinite(g) and g > 0 for g in gradients):
        raise RuntimeError(f"四组边权梯度失效：{gradients}")
    rebuilt = activity["last_layer_counts"] @ model.weights[-1]
    torch.testing.assert_close(potential, rebuilt, rtol=1e-5, atol=1e-4)
    result = dict(videos=len(x), loss=float(loss), gradient_abs_sum=gradients,
                  rates=[float(c.sum() / (len(x) * 32 * n))
                         for c, n in zip(activity["per_unit_fire"], model.widths)],
                  output_reconstruction_max_error=float((potential - rebuilt).abs().max()),
                  optimizer_updates=0, resources=model.resources())
    model.zero_grad(set_to_none=True)
    return result


def training_update(model, optimizer, x, y, positions, micro_batch, ledger, epoch):
    """同一个有效批次累积梯度；仅前向/反传OOM可安全清梯度并原样重试。"""
    count = len(positions)
    while True:
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        potential = loss = None
        phase, active = "forward", 0
        try:
            for begin in range(0, count, micro_batch):
                selected = positions[begin:begin + micro_batch]
                active, phase = len(selected), "forward"
                potential = model(x[selected])
                ledger.add("train_forward", epoch=epoch, videos=active, train_forward_videos=active)
                loss = loss_for(potential, y[selected])
                if not torch.isfinite(loss):
                    raise RuntimeError("训练损失非有限值")
                phase = "backward"
                (loss * (active / count)).backward()
                ledger.add("train_backward", epoch=epoch, videos=active, train_backward_videos=active)
                loss_sum += float(loss.detach()) * active
                potential = loss = None
        except torch.OutOfMemoryError:
            ledger.add("microbatch_oom_retry", epoch=epoch, videos=active,
                       microbatch_retries=1,
                       **{f"failed_{phase}_attempt_videos": active})
            potential = loss = None
            optimizer.zero_grad(set_to_none=True)
            if micro_batch == 1:
                raise RuntimeError("微批量1仍显存不足，保留层宽和上个安全检查点；需检查显存占用")
            micro_batch = max(1, micro_batch // 2)
            torch.cuda.empty_cache()
            print(f"显存保护：微批量降为{micro_batch}，有效批量与三层宽度保持登记值。", flush=True)
            continue
        break
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    # optimizer.step不在OOM重试块内，避免重试已被部分更新的优化器。
    optimizer.step()
    return loss_sum, micro_batch, float(norm)


@torch.no_grad()
def evaluate_pool(model, x, y, batch_size, ledger, pool, epoch, stop):
    model.eval()
    potentials, frames, videos = [], [], []
    units = [torch.zeros(n, device=x.device) for n in model.widths]
    error = 0.0
    for start in range(0, len(x), batch_size):
        potential, activity = model(x[start:start + batch_size], trace=True)
        count = len(potential)
        key = "control_forward_videos" if pool == "control" else "fit_evaluation_forward_videos"
        ledger.add(f"{pool}_forward", epoch=epoch, videos=count, **{key: count})
        rebuilt = activity["last_layer_counts"] @ model.weights[-1]
        error = max(error, float((potential - rebuilt).abs().max()))
        if not torch.isfinite(potential).all():
            raise RuntimeError("评价输出出现非有限值")
        potentials.append(potential.cpu().numpy())
        frames.append(activity["frame_fire"].cpu().numpy())
        videos.append(activity["per_video_fire"].cpu().numpy())
        for i, counts in enumerate(activity["per_unit_fire"]):
            units[i] += counts
        if stop.requested:
            raise EvaluationInterrupted()
    arrays = dict(potential=np.concatenate(potentials), y=y.cpu().numpy(),
                  frame_fire_by_layer=np.concatenate(frames), video_fire_by_layer=np.concatenate(videos))
    arrays.update({f"per_unit_fire_layer{i + 1}": counts.cpu().numpy() for i, counts in enumerate(units)})
    metrics = evaluate(arrays["potential"], arrays["y"])
    total = int(arrays["video_fire_by_layer"].sum(dtype=np.float64))
    denominator = len(x) * 32
    per_lif_rates = [counts.cpu().numpy() / denominator for counts in units]
    metrics.update(videos=len(x), hidden_spikes=total, output_spikes=0,
                   mean_spikes_per_video=total / len(x), mean_spikes_per_frame=total / (len(x) * 8),
                   spikes_per_lif_step=total / (denominator * sum(model.widths)),
                   spikes_per_lif_step_by_layer=[float(r.sum() / n) for r, n in zip(per_lif_rates, model.widths)],
                   silent_lif_by_layer=[int((r == 0).sum()) for r in per_lif_rates],
                   high_rate_lif_by_layer=[int((r > 0.5).sum()) for r in per_lif_rates],
                   rate_quantiles_by_layer=[np.quantile(r, [0, 0.05, 0.5, 0.95, 1]).tolist() for r in per_lif_rates],
                   per_frame_mean_by_layer=arrays["frame_fire_by_layer"].mean(0, dtype=np.float64).tolist(),
                   output_reconstruction_max_error=error, resources=model.resources())
    key = "control_metric_evaluations" if pool == "control" else "fit_metric_evaluations"
    ledger.add(f"{pool}_metrics", epoch=epoch, **{key: 1})
    return metrics, arrays


def accepted_control(metrics, previous):
    return previous is None or (metrics["joint_accuracy"] >= previous["joint_accuracy"] + 0.005
        and metrics["digit_accuracy"] >= previous["digit_accuracy"] - 0.01
        and metrics["motion_accuracy"] >= previous["motion_accuracy"] - 0.01)


def record_start(outdir, args, data, model):
    outdir.mkdir(parents=True, exist_ok=False)
    (outdir / "evaluations").mkdir()
    (outdir / "checkpoints").mkdir()
    config = dict(schema=SCHEMA, widths=list(WIDTHS), root_model_seed=11, data_seed=20261004,
        data=data, fit_videos=4480, control_videos=1120, effective_batch=args.batch_size,
        initial_micro_batch=args.micro_batch, maximum_epochs=args.epochs,
        control_interval=5, fit_interval=10, minimum_epochs=50,
        plateau_control_checks=4, plateau_fit_checks=2, earliest_plateau_stop_epoch=70,
        learning_rate_schedule={"1-50": 0.001, "51-80": 0.0003, "81+": 0.0001},
        weight_decay=0.0001, gradient_clip_norm=1.0,
        optimizer="AdamW(foreach=False)", dtype="float32", matmul_precision="highest",
        trainable="仅4组相邻层边权；无偏置；泄露率/阈值固定", initialization="hidden sigma=0.15*sqrt(784/fan_in); output sigma=0.02*sqrt(512/last_width)",
        initialization_scales=model.initial_scales, calibration="32拟合视频、无标签、最多3次前向；依次修正最前异常层；健康率区间[0.005,0.25]仅用于初始化检查",
        surrogate="hard Fire forward; backward 1/(1+5*abs(v-threshold))^2",
        neural_time_ms=160, steps_per_frame=4, step_ms=5,
        video_time="8帧，实际播放时长未规定", video_boundary="电位/脉冲/输出每视频从0开始",
        layer_delay_steps=1, resources=model.resources(),
        start_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        start_status=subprocess.check_output(["git", "status", "--short"], text=True),
        python=platform.python_version(), torch=torch.__version__, cuda_runtime=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0), gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
        source_sha256=source_hashes(), validation_arrays_unpacked=False, validation_forward_evaluations=0,
        sealed_test_reads=0, maximum_train_forward_videos=args.epochs * 4480,
        maximum_train_backward_videos=args.epochs * 4480,
        maximum_control_forward_videos=(args.epochs // 5) * 1120,
        maximum_fit_evaluation_forward_videos=(args.epochs // 10) * 4480,
        maximum_calibration_forward_videos=96, diagnostic_forward_videos=32,
        diagnostic_backward_videos=32,
        budget_note="上限不含OOM重试、被强制终止后回放或中断评价重跑，额外成本单独累计")
    write_json(outdir / "config.json", config)
    snapshot = outdir / "source_snapshot"
    for name in (*SOURCES, "docs/large_lif_baseline_plan.md", "docs/large_lif_baseline_execution.md", "pyproject.toml"):
        destination = snapshot / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(name, destination)
    (outdir / "start_diff.patch").write_text(
        subprocess.check_output(["git", "diff"], text=True, encoding="utf-8"), encoding="utf-8")
    return config


def status_result(model, state, started, reason):
    return dict(status=reason, completed_epochs=state["completed_epochs"],
                progress=None if state["progress"] is None else
                    {k: v for k, v in state["progress"].items() if k != "order"},
                maximum_epochs=state["maximum_epochs"],
                costs=state["costs"], wall_seconds_this_session=time.perf_counter() - started,
                gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                process_peak_working_set_bytes=peak_working_set_bytes(), resources=model.resources(),
                best_control=state["tracker"]["best_control"], training_converged=False,
                validation_forward_evaluations=0, sealed_test_reads=0,
                resume_command=".\\.venv\\Scripts\\python.exe -u -m tools.train_large_baseline --resume")


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用；大容量基线不退回CPU")
    if args.epochs < 1 or args.batch_size < 1 or args.micro_batch < 1 or args.micro_batch > args.batch_size:
        raise ValueError("遍数和批量须为正，微批量不能超过有效批量")
    torch.manual_seed(11)
    torch.cuda.manual_seed_all(11)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    stop = StopRequested()
    original_sigint = signal.signal(signal.SIGINT, stop.receive)
    outdir, started = args.outdir, time.perf_counter()
    try:
        payload = None
        if args.resume:
            payload, recovered_path = load_resume(outdir)
            state = payload["state"]
            if args.prepare_only:
                print("准备检查点已存在；不重复诊断或更新参数。", flush=True)
                return
            if state.get("stop_reason") in ("control_and_fit_plateau", "persistent_activity_fault") and not args.extension_reason:
                print("该检查点已按登记平台信号停止。继续实验须登记转向/续训理由并提供--extension-reason。", flush=True)
                return
            if state["completed_epochs"] >= args.epochs and state["progress"] is None:
                print(f"已完成{state['completed_epochs']}遍；当前命令不重复训练。", flush=True)
                return
            config = json.loads((outdir / "config.json").read_text(encoding="utf-8"))
            if config["source_sha256"] != source_hashes():
                raise ValueError("实现源码与运行快照不同；拒绝静默修改恢复实验")
            if config["effective_batch"] != args.batch_size:
                raise ValueError("恢复的有效批量与原登记不符")
            if args.epochs > state["maximum_epochs"] and not args.extension_reason:
                raise ValueError("扩大遍数预算须提供--extension-reason登记原因")
            if recovered_path.name == "resume_previous.pt" and (outdir / "resume.pt").exists():
                # 保留坏文件证据；后续原子写入不能用坏的latest覆盖好的previous。
                (outdir / "resume.pt").replace(outdir / f"resume_unreadable_{time.time_ns()}.pt")
        elif outdir.exists():
            raise FileExistsError("运行目录已存在；使用--resume恢复，或显式指定新的--outdir，禁止覆盖证据")
        fit, control, data = training_data()
        if payload and config["data"] != data:
            raise ValueError("恢复的数据协议/哈希与初次登记不符")
        device = torch.device("cuda")
        x, y = torch.as_tensor(fit[0], device=device), torch.as_tensor(fit[1], device=device)
        cx, cy = torch.as_tensor(control[0], device=device), torch.as_tensor(control[1], device=device)
        model = FixedLayeredSNN(WIDTHS, device=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4, foreach=False)
        rng = np.random.default_rng(11)
        torch.cuda.reset_peak_memory_stats()
        if payload is None:
            config = record_start(outdir, args, data, model)
            state = dict(completed_epochs=0, progress=None, maximum_epochs=args.epochs,
                micro_batch=args.micro_batch, costs=new_costs(), stop_reason=None,
                tracker=dict(best_control=None, accepted_control=None, control_stale=0,
                             best_fit_loss=None, best_fit_score=None, fit_stale=0,
                             activity_fault_checks=0), zero_gradient_updates=0)
            ledger = Ledger(outdir, state["costs"])
            positions = np.random.default_rng(11).permutation(len(x))[:32]
            write_json(outdir / "diagnostic_positions.json", dict(fit_positions=positions.tolist(),
                raw_ids=fit[2][positions].tolist(), note="固定根11，无标签初始化校准"))
            sample = torch.as_tensor(positions, device=device)
            calibration = calibrate(model, x[sample], ledger)
            diagnostic = gradient_diagnostic(model, x[sample], y[sample], ledger)
            write_json(outdir / "prepared.json", dict(calibration=calibration, diagnostic=diagnostic,
                **status_result(model, state, started, "prepared_no_optimizer_updates")))
            save_resume(outdir, model, optimizer, rng, state)
            print(f"启动诊断通过：隐藏层{model.widths}，4组梯度非零，发放率{diagnostic['rates']}。", flush=True)
            if args.prepare_only or stop.requested:
                print("初始检查点已保存，未进行优化训练。训练使用同一入口加--resume。", flush=True)
                return
        else:
            restore_checkpoint(payload, model, optimizer, rng)
            ledger = Ledger(outdir, state["costs"])
            before = state["costs"].copy()
            ledger.recover_observed_costs()
            state["micro_batch"] = min(state["micro_batch"], args.micro_batch)
            if args.extension_reason:
                state["tracker"]["control_stale"] = state["tracker"]["fit_stale"] = 0
                state["tracker"]["activity_fault_checks"] = 0
                state["zero_gradient_updates"] = 0
                state["stop_reason"] = None
            state["maximum_epochs"] = max(state["maximum_epochs"], args.epochs)
            event = dict(timestamp=time.time(), checkpoint=str(recovered_path),
                completed_epochs=state["completed_epochs"], maximum_epochs=args.epochs,
                requested_micro_batch=args.micro_batch, actual_micro_batch=state["micro_batch"],
                extension_reason=args.extension_reason,
                observed_costs_beyond_checkpoint={k: state["costs"][k] - before.get(k, 0) for k in state["costs"]},
                abrupt_exit_note="强制结束后从安全检查点回放；完整日志记录的成本保留，未写入日志的在途工作无法精确计量")
            with (outdir / "resume_events.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
            del payload
            print(f"从第{state['completed_epochs']}遍后的安全状态恢复，微批量{state['micro_batch']}。", flush=True)
        reason = "registered_epoch_budget"
        while state["completed_epochs"] < args.epochs:
            epoch = state["completed_epochs"] + 1
            if state["progress"] is None:
                state["progress"] = dict(epoch=epoch, order=rng.permutation(len(x)), next_sample=0,
                    loss_sum=0.0, stage="train", control=None, fit=None)
                if epoch in (51, 81):
                    state["tracker"]["control_stale"] = state["tracker"]["fit_stale"] = 0
            progress = state["progress"]
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(epoch)
            epoch_started = time.perf_counter()
            if progress["stage"] == "train":
                model.train()
                while progress["next_sample"] < len(x):
                    start = progress["next_sample"]
                    positions = torch.as_tensor(progress["order"][start:start + args.batch_size], device=device)
                    loss_sum, micro, norm = training_update(model, optimizer, x, y, positions,
                        state["micro_batch"], ledger, epoch)
                    state["micro_batch"] = micro
                    progress["loss_sum"] += loss_sum
                    progress["next_sample"] += len(positions)
                    state["zero_gradient_updates"] = state["zero_gradient_updates"] + 1 if norm == 0 else 0
                    if state["zero_gradient_updates"] >= 3:
                        save_resume(outdir, model, optimizer, rng, state)
                        raise RuntimeError("连续3个有效批次梯度全零，停止并保留检查点；检查深层Fire")
                    batch_number = math.ceil(progress["next_sample"] / args.batch_size)
                    if batch_number % args.progress_every == 0 or progress["next_sample"] == len(x):
                        print(f"第{epoch}/{args.epochs}遍 {progress['next_sample']}/{len(x)}视频 | "
                              f"损失{loss_sum / len(positions):.4f} | lr={learning_rate(epoch):g} | "
                              f"微批量{micro} | 本段{time.perf_counter() - epoch_started:.1f}秒", flush=True)
                    if stop.requested:
                        reason = "user_interrupt"
                        break
                if stop.requested:
                    break
                progress["stage"] = "control"
            if progress["stage"] == "control":
                if epoch % 5 == 0:
                    print(f"第{epoch}遍：评价1120个结构控制视频…", flush=True)
                    metrics, arrays = evaluate_pool(model, cx, cy, state["micro_batch"], ledger, "control", epoch, stop)
                    progress["control"] = metrics
                    write_json(outdir / "evaluations" / f"control_{epoch:03d}.json", metrics)
                    np.savez_compressed(outdir / "evaluations" / f"control_{epoch:03d}.npz", **arrays, raw_ids=control[2])
                    tracker = state["tracker"]
                    if accepted_control(metrics, tracker["accepted_control"]):
                        tracker["accepted_control"], tracker["control_stale"] = metrics, 0
                    else:
                        tracker["control_stale"] += 1
                    score = sum(metrics[k] for k in ("digit_accuracy", "motion_accuracy", "joint_accuracy"))
                    if tracker["best_control"] is None or score > sum(tracker["best_control"][k]
                            for k in ("digit_accuracy", "motion_accuracy", "joint_accuracy")):
                        tracker["best_control"] = dict(epoch=epoch, **metrics)
                        atomic_torch_save(outdir / "best.pt", checkpoint_payload(model, optimizer, rng, state))
                    rates = metrics["spikes_per_lif_step_by_layer"]
                    faulty = any(r <= 0.00001 or r >= 0.95 for r in rates)
                    tracker["activity_fault_checks"] = tracker["activity_fault_checks"] + 1 if faulty else 0
                    atomic_torch_save(outdir / "checkpoints" / f"epoch_{epoch:03d}_model.pt",
                        dict(widths=WIDTHS, epoch=epoch, model=model.state_dict(), metrics=metrics))
                    print(f"控制：数字{metrics['digit_accuracy']:.2%} 运动{metrics['motion_accuracy']:.2%} "
                          f"联合{metrics['joint_accuracy']:.2%} | 逐层Fire率{rates}", flush=True)
                progress["stage"] = "fit"
            if progress["stage"] == "fit":
                if epoch % 10 == 0:
                    print(f"第{epoch}遍：评价4480个拟合视频…", flush=True)
                    metrics, arrays = evaluate_pool(model, x, y, state["micro_batch"], ledger, "fit", epoch, stop)
                    progress["fit"] = metrics
                    write_json(outdir / "evaluations" / f"fit_{epoch:03d}.json", metrics)
                    np.savez_compressed(outdir / "evaluations" / f"fit_{epoch:03d}.npz", **arrays, raw_ids=fit[2])
                    tracker = state["tracker"]
                    score = sum(metrics[k] for k in ("digit_accuracy", "motion_accuracy", "joint_accuracy"))
                    loss_improved = tracker["best_fit_loss"] is None or metrics["loss"] < tracker["best_fit_loss"] - 0.001
                    score_improved = tracker["best_fit_score"] is None or score > tracker["best_fit_score"] + 0.002
                    if loss_improved or score_improved:
                        tracker["fit_stale"] = 0
                    else:
                        tracker["fit_stale"] += 1
                    tracker["best_fit_loss"] = metrics["loss"] if tracker["best_fit_loss"] is None else min(tracker["best_fit_loss"], metrics["loss"])
                    tracker["best_fit_score"] = score if tracker["best_fit_score"] is None else max(tracker["best_fit_score"], score)
                    print(f"拟合：数字{metrics['digit_accuracy']:.2%} 运动{metrics['motion_accuracy']:.2%} "
                          f"联合{metrics['joint_accuracy']:.2%}", flush=True)
                progress["stage"] = "finalize"
            row = dict(epoch=epoch, optimizer_mean_loss=progress["loss_sum"] / len(x),
                learning_rate=learning_rate(epoch), control=progress["control"], fit=progress["fit"],
                micro_batch=state["micro_batch"], costs=state["costs"].copy(),
                wall_seconds_this_session=time.perf_counter() - started,
                gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                process_peak_working_set_bytes=peak_working_set_bytes())
            write_json(outdir / f"epoch_{epoch:03d}.json", row)
            state["completed_epochs"], state["progress"] = epoch, None
            tracker = state["tracker"]
            if tracker["activity_fault_checks"] >= 2:
                reason = "persistent_activity_fault"
            # 给第51遍开始的第一次学习率下降至少20遍，不在第50遍直接平台停止。
            elif epoch >= 70 and epoch % 10 == 0 and tracker["control_stale"] >= 4 and tracker["fit_stale"] >= 2:
                reason = "control_and_fit_plateau"
            state["stop_reason"] = reason if reason != "registered_epoch_budget" else None
            save_resume(outdir, model, optimizer, rng, state)
            print(f"第{epoch}遍完成，优化平均损失{row['optimizer_mean_loss']:.4f}，检查点已保存。", flush=True)
            if reason != "registered_epoch_budget" or stop.requested:
                if stop.requested:
                    reason = "user_interrupt"
                break
        if state["progress"] is not None or stop.requested:
            state["stop_reason"] = reason
            save_resume(outdir, model, optimizer, rng, state)
        write_json(outdir / "summary.json", status_result(model, state, started, reason))
        print(f"本次运行结束：{reason}。完整训练遍数{state['completed_epochs']}，日志/检查点在{outdir.resolve()}。", flush=True)
    except EvaluationInterrupted:
        state["stop_reason"] = "user_interrupt_during_evaluation"
        save_resume(outdir, model, optimizer, rng, state)
        write_json(outdir / "summary.json", status_result(model, state, started, state["stop_reason"]))
        print("评价中断，检查点已保存；恢复会重跑该评价，已发生的前向成本保留。", flush=True)
    except Exception as exc:
        if outdir.exists():
            write_json(outdir / "error.json", dict(error=repr(exc), timestamp=time.time(),
                note="未覆盖上个安全检查点；失败或强制结束后的回放成本须从独立日志保留"))
        raise
    finally:
        signal.signal(signal.SIGINT, original_sigint)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=RUN_DIR)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--micro-batch", type=int, default=32)
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--extension-reason", default="")
    args = parser.parse_args()
    if args.progress_every < 1:
        parser.error("--progress-every须为正")
    run(args)


if __name__ == "__main__":
    main()
