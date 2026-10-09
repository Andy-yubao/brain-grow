"""v6原像素的三个ANN基线；复用历史通用工具，独立数据与恢复证据。"""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import copy
import gc
import hashlib
import json
import msvcrt
import shutil
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from braingrow.ann_feedback import POLICY, initial_feedback, make_scheduler, step_feedback
from braingrow.data.moving_mnist_v6 import CONFIG, training_data as v6_training_data
from braingrow.network.ann_controls import ARMS, make_model, resources
from tools import train_ann_comparison as legacy
from tools.train_ann_comparison import (
    Ledger, atomic_save, costs_zero, diagnostic, evaluate_pool, load_resume,
    loss_for, optimizer_for, restore, save_resume, score, seed, sha, write_json,
)

OUT = Path("experiments/runs/ann_baselines_v6")
SOURCES = (*legacy.SOURCES, "tools/train_ann_baselines_v6.py",
           "src/braingrow/data/moving_mnist_v6.py", "src/braingrow/data/rendering.py",
           "src/braingrow/ann_feedback.py")


def source_hashes():
    return {name: sha(name) for name in SOURCES}


def training_data():
    fit, control = v6_training_data()
    manifest_path = Path("data/manifest_v6.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest["renderer_source_sha256"] != sha("src/braingrow/data/moving_mnist_v6.py")
            or manifest["shared_renderer_sha256"] != sha("src/braingrow/data/rendering.py")):
        raise ValueError("v6渲染源码与已审计缓存不符，保留数据并停止")
    data = dict(version="v6", config=CONFIG, cache_sha256=manifest["cache_sha256"],
                manifest_sha256=sha(manifest_path),
                fit_original_ids=len(set(fit[2])), control_original_ids=len(set(control[2])))
    for name, pool in (("fit", fit), ("control", control)):
        x, y, ids = pool
        if (x.shape != (len(x), 8, 784) or y.shape != (len(x), 2)
                or not np.isfinite(x).all() or x.min() < 0 or x.max() > 1
                or not np.all((y[:, 0] >= 0) & (y[:, 0] < 10))
                or not np.all((y[:, 1] >= 0) & (y[:, 1] < 7))):
            raise ValueError(f"{name}原像素/标签维度或取值错误")
        data[f"{name}_id_sequence_sha256"] = hashlib.sha256(ids.tobytes()).hexdigest()
        data[f"{name}_label_sequence_sha256"] = hashlib.sha256(y.tobytes()).hexdigest()
    return fit, control, data


def make_config(arm, model, args, data):
    config = legacy.make_config(arm, model, args, data)
    config.update(schema=2, experiment="ann_baselines_v6", source_sha256=source_hashes(),
        preprocessing="v6原灰度[0,1]；仅重排8×784维度，无额外增强、特征或预训练",
        snn_reference=None,
        shared_gpu_note="三臂顺序执行；不修改其他GPU任务，墙钟不等于实测能耗")
    config.pop("lr_schedule")
    config.update(initial_learning_rate=0.001, lr_policy=POLICY,
        early_stop="控制损失连续5次未相对改善1%，且已完成至少10轮；降率不重置早停计数",
        control_interval=2, fit_interval=5, evaluate_final_epoch=True,
        maximum_control_forward_videos=((args.epochs + 1) // 2) * 1120,
        maximum_fit_forward_videos=(((args.epochs + 4) // 5) + 1) * 4480,
        fit_budget_note="含可能在非5倍数早停轮增加的一次末态拟合评价；最佳冻结后的拟合另计")
    return config


def result_for(model, state, started, session_base, reason):
    result = legacy.result_for(model, state, started, session_base, reason)
    result.update(feedback=state["feedback"], finish_reason=state.get("finish_reason"),
                  data_version="v6", interpretation="控制参与检查点选择，为探索成绩；墙钟不解释为实测能耗。")
    return result


def validate_existing(args, data):
    # 先核对全部待恢复臂，再写入任何运行证据或启动模型。
    for arm in args.arms:
        directory = args.outdir / arm
        if not directory.exists():
            continue
        if not args.resume:
            raise ValueError(f"{arm}目录已存在，需要--resume")
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        if (config.get("experiment") != "ann_baselines_v6" or config["data"] != data
                or config["source_sha256"] != source_hashes()
                or config["maximum_epochs"] != args.epochs
                or config["effective_batch"] != args.batch_size
                or config.get("lr_policy") != POLICY):
            raise ValueError(f"{arm}恢复源码/数据/预算/有效批量不符，禁止覆盖")
        saved = load_resume(directory)
        if not saved["state"].get("diagnostic_ok"):
            raise RuntimeError(f"{arm}准备诊断未完成，保留现场；不得自动重试或正式训练")
        del saved


@contextmanager
def run_lock(outdir):
    # Windows字节锁随进程退出释放；文件本身保留，避免并发启动同一实验。
    with (outdir / "run.lock").open("a+b") as handle:
        if handle.seek(0, 2) == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError("同一运行目录已有任务，请勿重复启动") from exc
        try:
            yield
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def prepare_arm(arm, args, data, x, y):
    directory = args.outdir / arm
    if directory.exists():
        raise ValueError(f"目录已存在，使用--resume保留证据：{directory}")
    directory.mkdir()
    (directory / "evaluations").mkdir()
    seed()
    rng = np.random.default_rng(11)
    model = make_model(arm).cuda()
    optimizer = optimizer_for(model)
    config = make_config(arm, model, args, data)
    write_json(directory / "config.json", config)
    for name in (*SOURCES, "docs/ann_baselines_v6_execution.md", "pyproject.toml"):
        destination = directory / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(name, destination)
    (directory / "start_diff.patch").write_text(
        subprocess.check_output(["git", "diff"], text=True, encoding="utf-8"), encoding="utf-8")
    scheduler = make_scheduler(optimizer)
    state = dict(arm=arm, completed_epochs=0, maximum_epochs=args.epochs,
        feedback=initial_feedback(scheduler),
        best=None, costs=costs_zero(), wall_seconds=0.0, optimizer_training_wall_seconds=0.0)
    save_resume(directory, model, optimizer, rng, state)
    diagnostic_costs = costs_zero()
    ledger = Ledger(directory, diagnostic_costs)
    positions = torch.as_tensor(np.random.default_rng(11).permutation(len(x))[:32], device=x.device)
    try:
        diag = diagnostic(directory, model, optimizer, rng, x[positions], y[positions], ledger)
    except Exception as exc:
        write_json(directory / "failure.json", dict(phase="diagnostic", error=repr(exc), costs=ledger.costs))
        raise
    saved = load_resume(directory)
    state = restore(saved, model, optimizer, rng)
    state["costs"] = diagnostic_costs
    state["wall_seconds"] = diag["wall_seconds"]
    state["diagnostic_ok"] = True
    save_resume(directory, model, optimizer, rng, state)
    print(f"{arm}诊断通过：参数{config['resources']['trainable_parameters']:,}，恢复重放完全一致。", flush=True)
    del saved, model, optimizer
    gc.collect()
    torch.cuda.empty_cache()


def train_arm(arm, args, data, x_fit, y_fit, x_control, y_control):
    directory = args.outdir / arm
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    if (config["data"] != data or config["source_sha256"] != source_hashes()
            or config["maximum_epochs"] != args.epochs or config["effective_batch"] != args.batch_size
            or config.get("lr_policy") != POLICY):
        raise ValueError(f"{arm}恢复源码/数据/预算/有效批量不符")
    saved = load_resume(directory)
    if not saved["state"].get("diagnostic_ok"):
        raise RuntimeError(f"{arm}尚未完成准备诊断，不能启动训练")
    if (directory / "result.json").exists():
        previous = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        if previous["status"] in ("completed", "early_stopped"):
            print(f"{arm}已完成，不重新前向或训练。", flush=True)
            return True
    seed()
    model = make_model(arm).cuda()
    optimizer = optimizer_for(model)
    rng = np.random.default_rng(11)
    state = restore(saved, model, optimizer, rng)
    if not 0 <= state["completed_epochs"] <= args.epochs:
        raise ValueError("断点累计轮数超出预算")
    scheduler = make_scheduler(optimizer)
    scheduler.load_state_dict(state["feedback"]["scheduler"])
    del saved
    ledger = Ledger(directory, state["costs"])
    ledger.recover()
    started = time.perf_counter()
    session_base = 0.0
    torch.cuda.reset_peak_memory_stats()
    try:
        remaining = () if state.get("finish_reason") else range(
            state["completed_epochs"] + 1, state["maximum_epochs"] + 1)
        for epoch in remaining:
            if (args.outdir / "STOP").exists() or (directory / "STOP").exists():
                write_json(directory / "result.json", result_for(model, state, started, session_base, "stopped"))
                return False
            epoch_lr = float(optimizer.param_groups[0]["lr"])
            order = rng.permutation(len(x_fit))
            loss_sum = 0.0
            norm_max = 0.0
            train_started = time.perf_counter()
            for start in range(0, len(order), args.batch_size):
                positions = torch.as_tensor(order[start:start + args.batch_size], device=x_fit.device)
                # 前向与反传分别记账；失败在最后安全遍恢复，不重试部分optimizer更新。
                model.train()
                optimizer.zero_grad(set_to_none=True)
                output = model(x_fit[positions])
                ledger.add("train_forward", epoch, train_forward_videos=len(positions))
                loss = loss_for(output, y_fit[positions])
                if not torch.isfinite(loss):
                    raise RuntimeError("训练损失非有限")
                loss.backward()
                ledger.add("train_backward", epoch, train_backward_videos=len(positions))
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                ledger.add("optimizer_update", epoch, optimizer_updates=1)
                loss_sum += float(loss.detach()) * len(positions)
                norm_max = max(norm_max, float(norm))
            torch.cuda.synchronize()
            train_wall = time.perf_counter() - train_started
            state["optimizer_training_wall_seconds"] += train_wall
            control = fit = feedback_event = None
            early_stop = False
            if epoch % 2 == 0 or epoch == args.epochs:
                control = evaluate_pool(model, x_control, y_control, directory, "control", epoch, ledger, args.batch_size)
                feedback_event = step_feedback(scheduler, state["feedback"], control["loss"], epoch)
                early_stop = feedback_event["early_stop"]
                if state["best"] is None or score(control) > score(state["best"]["metrics"]):
                    state["best"] = dict(epoch=epoch, metrics=control)
                    atomic_save(directory / "best.pt", dict(arm=arm, epoch=epoch, model=model.state_dict()))
                    write_json(directory / "best_meta.json", state["best"])
            if epoch % 5 == 0 or epoch == args.epochs or early_stop:
                fit = evaluate_pool(model, x_fit, y_fit, directory, "fit", epoch, ledger, args.batch_size)
            row = dict(epoch=epoch, optimizer_mean_loss=loss_sum / len(x_fit), learning_rate=epoch_lr, feedback_event=feedback_event,
                gradient_norm_max=norm_max, optimizer_training_wall_seconds=train_wall,
                control=control, fit=fit, costs=state["costs"].copy())
            write_json(directory / f"epoch_{epoch:03d}.json", row)
            if epoch == 10:
                atomic_save(directory / "epoch_010_model.pt", dict(arm=arm, epoch=10, model=model.state_dict()))
            state["completed_epochs"] = epoch
            if epoch == args.epochs or early_stop:
                state["finish_reason"] = "completed" if epoch == args.epochs else "early_stopped"
            elapsed = time.perf_counter() - started
            state["wall_seconds"] += elapsed - session_base
            session_base = elapsed
            save_resume(directory, model, optimizer, rng, state)
            print(f"{arm}第{epoch}轮已保存，平均损失{row['optimizer_mean_loss']:.4f}，lr={epoch_lr:g}。", flush=True)
            if feedback_event and feedback_event["reduced"]:
                print(f"控制损失平台，下一轮lr降至{feedback_event['learning_rates_after'][0]:g}。", flush=True)
            if control is not None:
                print(f"{arm}第{epoch}遍 控制：数字{control['digit_accuracy']:.2%} 运动{control['motion_accuracy']:.2%} 联合{control['joint_accuracy']:.2%}；平均损失{row['optimizer_mean_loss']:.4f}", flush=True)
            if fit is not None:
                print(f"{arm}第{epoch}遍 拟合：数字{fit['digit_accuracy']:.2%} 运动{fit['motion_accuracy']:.2%} 联合{fit['joint_accuracy']:.2%}", flush=True)
            if early_stop:
                print(f"{arm}控制损失连续5次未改善，第{epoch}轮早停。", flush=True)
                break
        # 冻结最高控制联合检查点后，只评价其拟合集，不再更新参数或选择。
        if not state.get("selected_fit_done"):
            best = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
            model.load_state_dict(best["model"])
            selected_fit = evaluate_pool(model, x_fit, y_fit, directory, "selected_fit",
                best["epoch"], ledger, args.batch_size)
            state["selected_fit_done"] = True
            state["selected_fit_metrics"] = selected_fit
            # 保留最后遍的恢复模型/优化器一致性；额外评价不改变正式模型随机流。
            saved = load_resume(directory)
            observed = copy.deepcopy(state)
            restore(saved, model, optimizer, rng)
            state = observed
            save_resume(directory, model, optimizer, rng, state)
        result = result_for(model, state, started, session_base, state["finish_reason"])
        result["best_control_fit"] = state["selected_fit_metrics"]
        result["resources"] = resources(model, arm)
        write_json(directory / "result.json", result)
        print(f"{arm}结束训练：{state['finish_reason']}，累计{state['completed_epochs']}/{state['maximum_epochs']}轮，最佳控制联合{state['best']['metrics']['joint_accuracy']:.2%}（第{state['best']['epoch']}遍）。", flush=True)
        return True
    except (Exception, KeyboardInterrupt) as exc:
        write_json(directory / "failure.json", dict(phase="training", error=repr(exc),
            last_completed_epoch=state["completed_epochs"], costs=ledger.costs,
            note="保留按遍恢复检查点与实际成本日志；在途工作恢复后重放并额外计数。"))
        raise
    finally:
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()


def run(args):
    if not 10 <= args.epochs <= 25 or args.batch_size < 1:
        raise ValueError("累计预算须为10–25轮；有效批量须为正")
    if len(set(args.arms)) != len(args.arms):
        raise ValueError("不得重复指定同一个模型")
    fit, control, data = training_data()
    validate_existing(args, data)
    if args.check_only:
        print(json.dumps(dict(ok=True, data=data, arms=args.arms, epochs=args.epochs,
            fit_videos=len(fit[0]), control_videos=len(control[0]),
            model_forward_videos=0, optimizer_updates=0,
            note="仅CPU数据/配置预检；未诊断GPU或训练"), ensure_ascii=False, indent=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用，不能启动已登记GPU基线")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    args.outdir.mkdir(parents=True, exist_ok=True)
    with run_lock(args.outdir):
        validate_existing(args, data)
        evidence = args.outdir / "split_evidence.npz"
        if evidence.exists():
            with np.load(evidence) as saved:
                for key, expected in dict(fit_ids=fit[2], fit_y=fit[1],
                        control_ids=control[2], control_y=control[1]).items():
                    if not np.array_equal(saved[key], expected):
                        raise ValueError("已保存划分证据不一致，禁止覆盖")
        else:
            np.savez_compressed(evidence, fit_ids=fit[2], fit_y=fit[1],
                                control_ids=control[2], control_y=control[1])
        print(f"v6：拟合{len(fit[0])}、控制{len(control[0])}视频；"
              f"每模型最多{args.epochs}遍，顺序执行{args.arms}。", flush=True)
        x_fit, y_fit = (torch.as_tensor(a, device="cuda") for a in fit[:2])
        x_control, y_control = (torch.as_tensor(a, device="cuda") for a in control[:2])
        for arm in args.arms:
            if not (args.outdir / arm).exists():
                prepare_arm(arm, args, data, x_fit, y_fit)
        if args.prepare_only:
            print("指定模型准备完成；新建模型正式训练完成遍数为0。", flush=True)
            return
        for arm in args.arms:
            if not train_arm(arm, args, data, x_fit, y_fit, x_control, y_control):
                return
        summary = dict(completed=True, arms=args.arms, results={arm: json.loads(
            (args.outdir / arm / "result.json").read_text(encoding="utf-8")) for arm in args.arms},
            data=data, validation_forward_evaluations=0, sealed_test_reads=0,
            resume_entry="tools.train_ann_baselines_v6 --resume（保留原arms/epochs/batch-size/outdir）")
        write_json(args.outdir / "summary.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=OUT)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--prepare-only", action="store_true", help="仅实际GPU诊断，正式训练0遍")
    parser.add_argument("--resume", action="store_true", help="恢复已有臂；缺失臂从头诊断并创建")
    parser.add_argument("--check-only", action="store_true", help="仅CPU数据/配置预检，不写运行目录")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
