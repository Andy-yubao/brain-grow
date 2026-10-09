"""纯LIF配对泛化实验：刷新轨迹、标签平滑，按登记门槛自动选择/收束。"""
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
from torch.nn import functional as F

from braingrow.data.online_v5 import selected_original_images, epoch_cache
from tools import train_large_baseline as base
from tools import train_adaptive_baseline as adaptive


ROOT = Path("experiments/runs/lif_generalization_v1")
WARM = adaptive.RUN_DIR / "checkpoints/epoch_020_model.pt"
ARMS = {"fixed": dict(refresh=False, smoothing=0.0),
        "refresh": dict(refresh=True, smoothing=0.0),
        "refresh_smooth": dict(refresh=True, smoothing=0.1)}
SOURCES = (*adaptive.SOURCES, "src/braingrow/data/rendering.py",
           "src/braingrow/data/online_v5.py", "tools/train_generalization_tuning.py")
TASKS = ("digit_accuracy", "motion_accuracy", "joint_accuracy")


class Interrupted(Exception):
    pass


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def hashes():
    return {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in SOURCES}


def costs():
    return dict(**base.new_costs(), optimizer_updates=0, data_render_videos=0,
                data_render_epochs=0, validation_forward_videos=0, validation_metric_evaluations=0)


def regularized_loss(potential, target, smoothing):
    return (F.cross_entropy(potential[:, :10], target[:, 0], label_smoothing=smoothing)
            + F.cross_entropy(potential[:, 10:], target[:, 1], label_smoothing=smoothing))


def training_update(model, optimizer, x, y, positions, micro, ledger, epoch, smoothing):
    count = len(positions)
    while True:
        optimizer.zero_grad(set_to_none=True)
        total, active, phase = 0.0, 0, "forward"
        potential = loss = None
        try:
            for begin in range(0, count, micro):
                chosen = positions[begin:begin + micro]
                active, phase = len(chosen), "forward"
                potential = model(x[chosen])
                ledger.add("train_forward", epoch=epoch, videos=active, train_forward_videos=active)
                loss = regularized_loss(potential, y[chosen], smoothing)
                if not torch.isfinite(loss):
                    raise RuntimeError("训练目标非有限")
                phase = "backward"
                (loss * (active / count)).backward()
                ledger.add("train_backward", epoch=epoch, videos=active, train_backward_videos=active)
                total += float(loss.detach()) * active
                potential = loss = None
        except torch.OutOfMemoryError:
            ledger.add("microbatch_oom_retry", epoch=epoch, videos=active, microbatch_retries=1,
                       **{f"failed_{phase}_attempt_videos": active})
            potential = loss = None
            optimizer.zero_grad(set_to_none=True)
            if micro == 1:
                raise RuntimeError("微批量1仍OOM；保留层宽及上个安全状态")
            micro = max(1, micro // 2)
            torch.cuda.empty_cache()
            continue
        break
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()  # 不在OOM重试范围内。
    ledger.add("optimizer_update", epoch=epoch, optimizer_updates=1)
    return total, micro, float(norm)


def add_joint_confusion(metrics, arrays):
    p, y = arrays["potential"], arrays["y"]
    pred = p[:, :10].argmax(1) * 7 + p[:, 10:].argmax(1)
    joint = np.zeros((70, 70), dtype=np.int64)
    np.add.at(joint, (y[:, 0] * 7 + y[:, 1], pred), 1)
    metrics["joint_confusion"] = joint.tolist()
    return metrics


def paired_comparison(summaries):
    averaged = {}
    for name, summary in summaries.items():
        rows = [row["metrics"] for row in summary["history"] if row["epoch"] in (4, 6)]
        if len(rows) != 2:
            raise ValueError("比较须有第4/6遍完整控制评价")
        averaged[name] = {key: float(np.mean([row[key] for row in rows])) for key in (*TASKS, "loss")}
    reference = averaged["fixed"]
    passed = []
    effects = {}
    for name in ("refresh", "refresh_smooth"):
        effects[name] = {key: averaged[name][key] - reference[key] for key in (*TASKS, "loss")}
        effect = effects[name]
        if (effect["joint_accuracy"] >= 0.02 - 1e-12 and effect["digit_accuracy"] >= -0.01 - 1e-12
                and effect["motion_accuracy"] >= -0.01 - 1e-12 and effect["loss"] <= 1e-12):
            passed.append(name)
    passed.sort(key=lambda name: (averaged[name]["joint_accuracy"],
                sum(averaged[name][key] for key in TASKS), -averaged[name]["loss"]), reverse=True)
    return dict(averaged=averaged, effects=effects, passed=passed,
                winner=passed[0] if passed else None,
                rule="第4/6遍均值：联合+0.02、单项各≥-0.01、硬标签控制损失不升")


def prepare(root, fit, control, data):
    if root.exists():
        raise FileExistsError("目录存在；使用--resume，不覆盖原实验")
    parent_config = read(adaptive.RUN_DIR / "config.json")
    if parent_config["source_sha256"] != adaptive.source_hashes():
        raise ValueError("起点实现哈希与原实验不符")
    warm = torch.load(WARM, map_location="cpu", weights_only=False)
    if tuple(warm["widths"]) != base.WIDTHS or warm["epoch"] != 20:
        raise ValueError("起点必须是固定三层第20遍")
    root.mkdir(parents=True)
    torch.manual_seed(11)
    torch.cuda.manual_seed_all(11)
    model = base.FixedLayeredSNN(device="cuda")
    model.load_state_dict(warm["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-4, foreach=False)
    shared = costs()
    ledger = base.Ledger(root, shared)
    torch.cuda.reset_peak_memory_stats()
    diagnostic = base.gradient_diagnostic(model, torch.as_tensor(fit[0][:16], device="cuda"),
                                          torch.as_tensor(fit[1][:16], device="cuda"), ledger)
    config = dict(source_sha256=hashes(), warm_checkpoint=str(WARM.resolve()),
        warm_sha256=hashlib.sha256(WARM.read_bytes()).hexdigest(), warm_epoch=20,
        warm_control=warm["metrics"], data=data, arms=ARMS, widths=list(base.WIDTHS),
        model_seed=11, data_seed=20261004, optimizer="new AdamW(foreach=False)",
        initial_lr=0.0003, weight_decay=1e-4, effective_batch=128, initial_micro_batch=32,
        pilot_epochs=6, maximum_extension_epochs=6, control_interval=2,
        label_smoothing=0.1, selection_rule="paired_comparison：第4/6遍均值；预登记门槛",
        learning_rate_policy=dict(adaptive.POLICY, control_interval_epochs=2), fit_pool_videos=4480, control_pool_videos=1120,
        input="v5原灰度；同160拟合ID，刷新variant；控制保持固定",
        sealed_test_reads=0, validation_policy="冻结后最多两模型，各一次1400视频；不用作调优反馈",
        maximum_train_forward_videos=107520, maximum_train_backward_videos=107520,
        maximum_control_forward_videos=13440, maximum_fit_forward_videos=17920,
        maximum_validation_forward_videos=2800, diagnostic=diagnostic,
        start_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        start_status=subprocess.check_output(["git", "status", "--short"], text=True),
        device=torch.cuda.get_device_name(0), torch=torch.__version__,
        gpu_diagnostic_peak_bytes=torch.cuda.max_memory_allocated())
    base.write_json(root / "config.json", config)
    initial = base.checkpoint_payload(model, optimizer, np.random.default_rng(11), dict(costs=shared))
    base.atomic_torch_save(root / "common_initial.pt", initial)
    for name in (*SOURCES, "docs/generalization_tuning_execution.md"):
        destination = root / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(name, destination)
    suite = dict(phase="pilot", arm_index=0, winner=None, status="prepared", common_costs=shared,
                 summaries={}, validation_done=[], active_arm=None, started_timestamp=time.time())
    base.write_json(root / "suite.json", suite)
    print(f"共同第20遍模型诊断通过；原优化器不继承，三臂新Adam。梯度{diagnostic['gradient_abs_sum']}。", flush=True)


def new_state():
    return dict(completed_epochs=0, progress=None, maximum_epochs=6, micro_batch=32,
        costs=costs(), stop_reason=None, history=[], best_control=None,
        extension_reference=None, extension_stale=0, zero_gradient_updates=0,
        tracker=dict(control_stale=0, fit_stale=0, activity_fault_checks=0))


def arm_summary(state, started, reason):
    return dict(status=reason, completed_epochs=state["completed_epochs"],
                progress=None if state["progress"] is None else
                    {k: v for k, v in state["progress"].items() if k != "order"},
                history=state["history"], best_control=state["best_control"], costs=state["costs"],
                current_learning_rate=state["lr_policy"]["scheduler"]["_last_lr"][0],
                wall_seconds_this_session=time.perf_counter() - started,
                gpu_peak_bytes=torch.cuda.max_memory_allocated(), process_peak_bytes=base.peak_working_set_bytes())


def run_arm(root, name, maximum, fit, control, images, ids, digits, suite, stop):
    directory = root / name
    model = base.FixedLayeredSNN(device="cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-4, foreach=False)
    rng = np.random.default_rng(11)
    if directory.exists():
        payload, recovered = base.load_resume(directory)
        if recovered.name == "resume_previous.pt" and (directory / "resume.pt").exists():
            (directory / "resume.pt").replace(directory / f"resume_unreadable_{time.time_ns()}.pt")
        state = payload["state"]
        base.restore_checkpoint(payload, model, optimizer, rng)
    else:
        directory.mkdir()
        (directory / "evaluations").mkdir()
        payload = torch.load(root / "common_initial.pt", map_location="cpu", weights_only=False)
        base.restore_checkpoint(payload, model, optimizer, rng)
        state = new_state()
        adaptive.adopt_history(optimizer, state, [])
    del payload
    if maximum > 6 and state["stop_reason"] == "extension_plateau":
        return read(directory / "summary.json")
    scheduler = adaptive.make_scheduler(optimizer)
    scheduler.load_state_dict(state["lr_policy"]["scheduler"])
    if state["completed_epochs"] == 0 and state["progress"] is None:
        saved = scheduler.state_dict()
        saved["best"] = read(root / "config.json")["warm_control"]["loss"]
        scheduler.load_state_dict(saved)
        state["lr_policy"]["scheduler"] = scheduler.state_dict()
        base.save_resume(directory, model, optimizer, rng, state)
    if maximum > 6 and state["maximum_epochs"] == 6:
        state["extension_reference"] = state["history"][-1]["metrics"]
        state["extension_stale"] = 0
    state["maximum_epochs"] = maximum
    ledger = base.Ledger(directory, state["costs"])
    ledger.recover_observed_costs()
    common_ledger = base.Ledger(root, suite["common_costs"])
    cx, cy = torch.as_tensor(control[0], device="cuda"), torch.as_tensor(control[1], device="cuda")
    oldx, y = torch.as_tensor(fit[0], device="cuda"), torch.as_tensor(fit[1], device="cuda")
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    reason = "pilot_budget" if maximum == 6 else "extension_budget"
    try:
        while state["completed_epochs"] < maximum:
            epoch = state["completed_epochs"] + 1
            print(f"[{name}] 调优第{epoch}/{maximum}遍，lr={optimizer.param_groups[0]['lr']:g}。", flush=True)
            if ARMS[name]["refresh"]:
                arrays, metadata = epoch_cache(root / "augmentation", images, ids, digits, epoch)
                if not np.array_equal(arrays["raw_ids"], fit[2]) or not np.array_equal(arrays["y"], fit[1]):
                    raise ValueError("刷新轨迹改变了ID/标签顺序")
                if metadata["generated"]:
                    common_ledger.add("online_data_render", epoch=epoch, videos=len(arrays["x"]),
                        data_render_videos=len(arrays["x"]), data_render_epochs=1)
                    base.write_json(root / "suite.json", suite)
                x = torch.as_tensor(arrays["x"], device="cuda")
                del arrays, metadata
            else:
                x = oldx
            if state["progress"] is None:
                state["progress"] = dict(epoch=epoch, order=rng.permutation(len(x)), next_sample=0,
                                         loss_sum=0.0, stage="train", control=None, fit=None)
            progress = state["progress"]
            if stop.requested:
                raise Interrupted()
            if progress["stage"] == "train":
                model.train()
                while progress["next_sample"] < len(x):
                    positions = torch.as_tensor(progress["order"][progress["next_sample"]:progress["next_sample"] + 128], device="cuda")
                    loss, micro, norm = training_update(model, optimizer, x, y, positions, state["micro_batch"],
                                                       ledger, epoch, ARMS[name]["smoothing"])
                    state["micro_batch"] = micro
                    progress["loss_sum"] += loss
                    progress["next_sample"] += len(positions)
                    state["zero_gradient_updates"] = state["zero_gradient_updates"] + 1 if norm == 0 else 0
                    if state["zero_gradient_updates"] >= 3:
                        raise RuntimeError("连续3批梯度为零")
                    if (progress["next_sample"] // 128) % 5 == 0 or progress["next_sample"] == len(x):
                        print(f"[{name}] 第{epoch}遍 {progress['next_sample']}/{len(x)} | "
                              f"训练目标{loss / len(positions):.4f} | lr={optimizer.param_groups[0]['lr']:g}", flush=True)
                    if stop.requested:
                        raise Interrupted()
                progress["stage"] = "control"
            if progress["stage"] == "control":
                if epoch % 2 == 0:
                    metrics, saved = base.evaluate_pool(model, cx, cy, state["micro_batch"], ledger, "control", epoch, stop)
                    add_joint_confusion(metrics, saved)
                    progress["control"] = metrics
                    base.write_json(directory / "evaluations" / f"control_{epoch:03d}.json", metrics)
                    np.savez_compressed(directory / "evaluations" / f"control_{epoch:03d}.npz", **saved, raw_ids=control[2])
                    state["history"].append(dict(epoch=epoch, metrics=metrics))
                    event = adaptive.control_feedback(scheduler, state, metrics, epoch)
                    if event:
                        adaptive.append_policy_event(directory, event)
                    state["tracker"]["activity_fault_checks"] = state["tracker"]["activity_fault_checks"] + 1 if any(
                        r <= 1e-5 or r >= 0.95 for r in metrics["spikes_per_lif_step_by_layer"]) else 0
                    if maximum > 6:
                        if base.accepted_control(metrics, state["extension_reference"]):
                            state["extension_reference"], state["extension_stale"] = metrics, 0
                        else:
                            state["extension_stale"] += 1
                    progress["stage"] = "fit"
                    rank = lambda m: (m["joint_accuracy"], sum(m[key] for key in TASKS), -m["loss"])
                    if state["best_control"] is None or rank(metrics) > rank(state["best_control"]):
                        state["best_control"] = dict(epoch=epoch, **metrics)
                        base.atomic_torch_save(directory / "best.pt", base.checkpoint_payload(model, optimizer, rng, state))
                    print(f"[{name}] 控制：数字{metrics['digit_accuracy']:.2%} 运动{metrics['motion_accuracy']:.2%} "
                          f"联合{metrics['joint_accuracy']:.2%} | 硬标签损失{metrics['loss']:.4f}", flush=True)
                progress["stage"] = "fit"
            stop_extension = maximum > 6 and state["extension_stale"] >= 2
            if progress["stage"] == "fit":
                if epoch == 6 or epoch == maximum or stop_extension:
                    metrics, saved = base.evaluate_pool(model, oldx, y, state["micro_batch"], ledger, "fit", epoch, stop)
                    add_joint_confusion(metrics, saved)
                    progress["fit"] = metrics
                    base.write_json(directory / "evaluations" / f"fit_{epoch:03d}.json", metrics)
                    np.savez_compressed(directory / "evaluations" / f"fit_{epoch:03d}.npz", **saved, raw_ids=fit[2])
                progress["stage"] = "finalize"
            base.write_json(directory / f"epoch_{epoch:03d}.json", dict(epoch=epoch,
                training_objective_mean=progress["loss_sum"] / len(x), smoothing=ARMS[name]["smoothing"],
                control=progress["control"], fit=progress["fit"], next_learning_rate=optimizer.param_groups[0]["lr"],
                costs=state["costs"].copy(), resources=model.resources(),
                gpu_peak_bytes=torch.cuda.max_memory_allocated(), wall_seconds_this_session=time.perf_counter() - started))
            state["completed_epochs"], state["progress"] = epoch, None
            if state["tracker"]["activity_fault_checks"] >= 2:
                reason = "persistent_activity_fault"
            elif stop_extension:
                reason = "extension_plateau"
            state["stop_reason"] = reason if (epoch == maximum or reason in ("persistent_activity_fault", "extension_plateau")) else None
            base.save_resume(directory, model, optimizer, rng, state)
            result = arm_summary(state, started, reason)
            base.write_json(directory / "summary.json", result)
            suite["summaries"][name] = result
            base.write_json(root / "suite.json", suite)
            if reason in ("persistent_activity_fault", "extension_plateau"):
                break
        result = arm_summary(state, started, reason)
        base.write_json(directory / "summary.json", result)
        return result
    except (Interrupted, base.EvaluationInterrupted):
        state["stop_reason"] = "user_interrupt"
        base.save_resume(directory, model, optimizer, rng, state)
        result = arm_summary(state, started, "user_interrupt")
        base.write_json(directory / "summary.json", result)
        suite["summaries"][name] = result
        raise Interrupted()
    except Exception as exc:
        # 参数更新异常不覆盖上个安全检查点；保留错误与成本。
        base.write_json(directory / "error.json", dict(error=repr(exc), timestamp=time.time()))
        raise


class ValidationLedger:
    def __init__(self, ledger):
        self.ledger = ledger

    def add(self, kind, **kwargs):
        for old, new in (("fit_evaluation_forward_videos", "validation_forward_videos"),
                         ("fit_metric_evaluations", "validation_metric_evaluations")):
            if old in kwargs:
                kwargs[new] = kwargs.pop(old)
        self.ledger.add(kind.replace("fit", "validation"), **kwargs)


def frozen_validation(root, suite, fit, control, stop):
    if not suite.get("frozen"):
        suite["frozen"] = {name: dict(path=str(root / name / "best.pt"),
            sha256=hashlib.sha256((root / name / "best.pt").read_bytes()).hexdigest())
            for name in ("fixed", suite["winner"])}
        base.write_json(root / "suite.json", suite)
    with np.load("data/dynamic_v5.npz", allow_pickle=False) as cache:
        x, y, raw_ids = (cache[key] for key in ("validation_x", "validation_y", "validation_ids"))
    if len(x) != 1400 or len(set(raw_ids)) != 100 or set(raw_ids) & (set(fit[2]) | set(control[2])):
        raise ValueError("冻结确认ID隔离失败")
    vx, vy = torch.as_tensor(x, device="cuda"), torch.as_tensor(y, device="cuda")
    ledger = ValidationLedger(base.Ledger(root, suite["common_costs"]))
    for name, frozen in suite["frozen"].items():
        if name in suite["validation_done"]:
            continue
        path = Path(frozen["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != frozen["sha256"]:
            raise ValueError("冻结模型哈希已改变")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model = base.FixedLayeredSNN(device="cuda")
        model.load_state_dict(payload["model"])
        epoch = payload["state"]["best_control"]["epoch"]
        metrics, saved = base.evaluate_pool(model, vx, vy, 32, ledger, "fit", epoch, stop)
        add_joint_confusion(metrics, saved)
        base.write_json(root / f"validation_{name}.json", dict(metrics=metrics, frozen=frozen,
            selection="先冻结后确认；结果不参与调优反馈；旧验证池曾被探索", sealed_test_reads=0))
        np.savez_compressed(root / f"validation_{name}.npz", **saved, raw_ids=raw_ids)
        suite["validation_done"].append(name)
        base.write_json(root / "suite.json", suite)
        print(f"冻结确认[{name}]：数字{metrics['digit_accuracy']:.2%} 运动{metrics['motion_accuracy']:.2%} "
              f"联合{metrics['joint_accuracy']:.2%}", flush=True)
        del model, payload


def export_report(root, suite):
    comparison = suite.get("comparison")
    report = dict(status=suite["status"], phase=suite["phase"], comparison=comparison,
                  summaries=suite["summaries"], common_costs=suite["common_costs"],
                  validation={name: read(root / f"validation_{name}.json") for name in suite["validation_done"]},
                  validation_role="冻结后探索性确认，非封存测试", sealed_test_reads=0,
                  root_directory=str(root.resolve()))
    base.write_json(root / "comparison.json", report)
    text = ["# LIF泛化调优实际结果", "", f"状态：{suite['status']}；阶段：{suite['phase']}。", "",
            "三臂共同第20遍模型；新Adam、同预算、同根11。刷新不改变拟合ID或输入定义，平滑仅作用于训练目标。", "",
            "| 臂 | 完成调优遍数 | 最佳控制数字 | 最佳控制运动 | 最佳控制联合 |", "|---|---:|---:|---:|---:|"]
    for name, summary in suite["summaries"].items():
        metric = summary["best_control"]
        if metric:
            text.append(f"| {name} | {summary['completed_epochs']} | {metric['digit_accuracy']:.3%} | "
                        f"{metric['motion_accuracy']:.3%} | {metric['joint_accuracy']:.3%} |")
    if comparison:
        text += ["", "第4/6遍均值的配对比较：", "", "| 臂 | 数字 | 运动 | 联合 | 控制损失 |",
                 "|---|---:|---:|---:|---:|"]
        for name, metrics in comparison["averaged"].items():
            text.append(f"| {name} | {metrics['digit_accuracy']:.3%} | {metrics['motion_accuracy']:.3%} | "
                        f"{metrics['joint_accuracy']:.3%} | {metrics['loss']:.4f} |")
        text += ["", f"通过门槛：{comparison['passed']}；继续候选：{comparison['winner']}。未通过不扩大。"]
    text += ["", "控制用于选择；原验证只在候选冻结后确认，仍为探索结果。最终测试未读取。",
             "所有检查点/恢复状态、混淆矩阵与逐层Fire保留在各臂目录，comparison.json包含实际成本。预算到期不表示泛化问题已解决。"]
    (root / "comparison.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用，不退回CPU")
    torch.manual_seed(11)
    torch.cuda.manual_seed_all(11)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    fit, control, data = base.training_data()
    if not args.resume:
        prepare(args.outdir, fit, control, data)
    root = args.outdir
    config, suite = read(root / "config.json"), read(root / "suite.json")
    if config["source_sha256"] != hashes() or config["data"] != data:
        raise ValueError("源码/数据哈希改变，拒绝静默恢复")
    if args.prepare_only:
        print("共同图准备/哈希核对完成，尚未执行三臂训练。", flush=True)
        return
    if suite["phase"] == "complete":
        print("本轮已完成，不重复训练或确认。", flush=True)
        return
    stop_path = root / "stop_request.json"
    if stop_path.exists():
        stop_path.replace(root / f"stop_request_acknowledged_{time.time_ns()}.json")
    stop = adaptive.FileStopRequested(root)
    old_handler = signal.signal(signal.SIGINT, stop.receive)
    ids = list(dict.fromkeys(fit[2].tolist()))
    digits = [int(fit[1][np.flatnonzero(fit[2] == raw_id)[0], 0]) for raw_id in ids]
    images = selected_original_images(ids)
    ledger = base.Ledger(root, suite["common_costs"])
    ledger.recover_observed_costs()
    try:
        suite["status"] = "running"
        if suite["phase"] == "pilot":
            names = list(ARMS)
            while suite["arm_index"] < len(names):
                name = names[suite["arm_index"]]
                suite["active_arm"] = name
                base.write_json(root / "suite.json", suite)
                summary = run_arm(root, name, 6, fit, control, images, ids, digits, suite, stop)
                suite["summaries"][name] = summary
                if summary["status"] == "persistent_activity_fault":
                    raise RuntimeError("候选出现持续活动故障，停止扩大")
                suite["arm_index"] += 1
                base.write_json(root / "suite.json", suite)
            suite["comparison"] = paired_comparison(suite["summaries"])
            suite["winner"] = suite["comparison"]["winner"]
            suite["phase"] = "extension" if suite["winner"] else "complete"
            base.write_json(root / "suite.json", suite)
            export_report(root, suite)
        if suite["phase"] == "extension":
            name = suite["winner"]
            suite["active_arm"] = name
            suite["summaries"][name] = run_arm(root, name, 12, fit, control, images, ids, digits, suite, stop)
            if suite["summaries"][name]["status"] == "persistent_activity_fault":
                raise RuntimeError("扩展出现持续活动故障")
            suite["phase"] = "validation"
            base.write_json(root / "suite.json", suite)
        if suite["phase"] == "validation":
            suite["active_arm"] = "frozen_validation"
            frozen_validation(root, suite, fit, control, stop)
            suite["phase"] = "complete"
        suite["status"], suite["active_arm"] = "complete_registered_round", None
        base.write_json(root / "suite.json", suite)
        export_report(root, suite)
        print(f"本轮完成，比较报告：{(root / 'comparison.md').resolve()}。", flush=True)
    except (Interrupted, base.EvaluationInterrupted):
        suite["status"] = "user_interrupt"
        base.write_json(root / "suite.json", suite)
        export_report(root, suite)
        print("本轮已安全暂停并保存阶段/检查点，--resume继续。", flush=True)
    except Exception as exc:
        suite["status"] = "error"
        base.write_json(root / "suite.json", suite)
        base.write_json(root / "error.json", dict(error=repr(exc), timestamp=time.time()))
        export_report(root, suite)
        raise
    finally:
        signal.signal(signal.SIGINT, old_handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=ROOT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--request-stop", action="store_true")
    args = parser.parse_args()
    if args.request_stop:
        if not (args.outdir / "config.json").exists():
            parser.error("调优运行不存在")
        base.write_json(args.outdir / "stop_request.json", dict(timestamp=time.time()))
        print("已请求安全暂停，等待批次保存退出。", flush=True)
        return
    if not args.resume:
        preparation_only = args.prepare_only
        args.prepare_only = True
        run(args)
        if preparation_only:
            return
        args.resume, args.prepare_only = True, False
    with adaptive.RunLock(args.outdir):
        run(args)


if __name__ == "__main__":
    main()
