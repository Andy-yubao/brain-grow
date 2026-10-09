"""同v5数据的三个非SNN对照，固定预算、成本记录及按遍恢复。"""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import copy
import gc
import hashlib
import json
import platform
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from braingrow.evaluation.metrics import evaluate
from braingrow.network.ann_controls import ARMS, make_model, resources
from tools.run_layered_gradient import loss_for, peak_working_set_bytes, training_data, write_json


OUT = Path("experiments/runs/ann_comparison_v1")
REFERENCE = Path("experiments/runs/large_lif_baseline_v1")
SOURCES = ("src/braingrow/network/ann_controls.py", "tools/train_ann_comparison.py",
           "tools/run_layered_gradient.py", "src/braingrow/evaluation/metrics.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    return {name: sha(name) for name in SOURCES}


def seed():
    torch.manual_seed(11)
    torch.cuda.manual_seed_all(11)


def lr_for(epoch):
    return 0.001 if epoch <= 50 else 0.0003 if epoch <= 80 else 0.0001


def optimizer_for(model):
    return torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4, foreach=False)


def costs_zero():
    return dict(train_forward_videos=0, train_backward_videos=0, optimizer_updates=0,
        control_forward_videos=0, fit_forward_videos=0,
        control_metric_evaluations=0, fit_metric_evaluations=0,
        selected_fit_forward_videos=0, selected_fit_metric_evaluations=0,
        diagnostic_forward_videos=0, diagnostic_backward_videos=0,
        diagnostic_optimizer_updates=0)


class Ledger:
    def __init__(self, directory, costs):
        self.directory, self.costs = directory, costs

    def add(self, kind, epoch=0, **increments):
        for key, value in increments.items():
            self.costs[key] += int(value)
        row = dict(kind=kind, epoch=epoch, cumulative_costs=self.costs.copy(), timestamp=time.time())
        with (self.directory / "cost_journal.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()

    def recover(self):
        path = self.directory / "cost_journal.jsonl"
        if not path.exists():
            return
        if path.stat().st_size and not path.read_bytes().endswith(b"\n"):
            with path.open("a", encoding="utf-8") as handle:
                handle.write("\n")
        for line in reversed(path.read_text(encoding="utf-8").splitlines()):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            for key, value in row["cumulative_costs"].items():
                self.costs[key] = max(self.costs[key], value)
            break


def payload(model, optimizer, rng, state):
    return dict(schema=1, model=model.state_dict(), optimizer=optimizer.state_dict(),
        numpy_rng=copy.deepcopy(rng.bit_generator.state), torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all(), state=copy.deepcopy(state))


def restore(saved, model, optimizer, rng):
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    rng.bit_generator.state = saved["numpy_rng"]
    torch.set_rng_state(saved["torch_rng"])
    torch.cuda.set_rng_state_all(saved["cuda_rng"])
    return copy.deepcopy(saved["state"])


def atomic_save(path, value, previous=None):
    torch.cuda.synchronize()
    pending = path.with_name(path.name + ".pending")
    torch.save(value, pending)
    if previous is not None and path.exists():
        path.replace(previous)
    pending.replace(path)


def save_resume(directory, model, optimizer, rng, state):
    atomic_save(directory / "resume.pt", payload(model, optimizer, rng, state),
                directory / "resume_previous.pt")


def load_resume(directory):
    errors = []
    for name in ("resume.pt", "resume_previous.pt"):
        path = directory / name
        if path.exists():
            try:
                saved = torch.load(path, map_location="cpu", weights_only=False)
                if saved["schema"] != 1:
                    raise ValueError("检查点版本不符")
                return saved
            except (OSError, RuntimeError, ValueError, EOFError) as exc:
                errors.append(str(exc))
    raise RuntimeError(f"无可恢复检查点：{directory} {errors}")


def nested_equal(first, second):
    if isinstance(first, torch.Tensor):
        return torch.equal(first.cpu(), second.cpu())
    if isinstance(first, dict):
        return first.keys() == second.keys() and all(nested_equal(first[k], second[k]) for k in first)
    if isinstance(first, (list, tuple)):
        return len(first) == len(second) and all(nested_equal(a, b) for a, b in zip(first, second))
    return first == second


def update(model, optimizer, x, y):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(x)
    loss = loss_for(output, y)
    if not torch.isfinite(loss):
        raise RuntimeError("损失非有限，保留上一完整遍检查点")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach()), float(norm)


def diagnostic(directory, model, optimizer, rng, x, y, ledger):
    started = time.perf_counter()
    model.train()
    output = model(x)
    loss = loss_for(output, y)
    ledger.add("diagnostic_initial_forward", diagnostic_forward_videos=len(x))
    loss.backward()
    ledger.add("diagnostic_initial_backward", diagnostic_backward_videos=len(x))
    gradients = {name: float(p.grad.abs().sum()) for name, p in model.named_parameters()
                 if "weight" in name}
    if output.shape != (len(x), 17) or not all(np.isfinite(g) and g > 0 for g in gradients.values()):
        raise RuntimeError(f"输出/梯度诊断失败：{gradients}")
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    ledger.add("diagnostic_update", diagnostic_optimizer_updates=1)
    checkpoint = directory / "diagnostic_restore.pt"
    atomic_save(checkpoint, payload(model, optimizer, rng, {}))
    # 使用保存的随机流生成批次，连同优化器和模型进行真实更新重放。
    positions = torch.as_tensor(rng.permutation(len(x)), device=x.device)
    next_loss, _ = update(model, optimizer, x[positions], y[positions])
    ledger.add("diagnostic_next_update", diagnostic_forward_videos=len(x),
               diagnostic_backward_videos=len(x), diagnostic_optimizer_updates=1)
    reference = copy.deepcopy(payload(model, optimizer, rng, {}))
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    restore(saved, model, optimizer, rng)
    replay_positions = torch.as_tensor(rng.permutation(len(x)), device=x.device)
    replay_loss, _ = update(model, optimizer, x[replay_positions], y[replay_positions])
    ledger.add("diagnostic_replay_update", diagnostic_forward_videos=len(x),
               diagnostic_backward_videos=len(x), diagnostic_optimizer_updates=1)
    after = payload(model, optimizer, rng, {})
    replay_ok = next_loss == replay_loss and nested_equal(reference, after)
    if not replay_ok:
        raise RuntimeError("模型/优化器/随机流恢复重放不一致")
    model.eval()
    with torch.no_grad():
        probe = x[:16]
        whole = model(probe)
        split = torch.cat((model(probe[:8]), model(probe[8:])))
        shuffled = model(probe.flip(0)).flip(0)
        reverse_time = model(probe.flip(1))
        ledger.add("diagnostic_video_isolation_and_time", diagnostic_forward_videos=64)
        split_error = float((whole - split).abs().max())
        shuffle_error = float((whole - shuffled).abs().max())
        order_difference = float((whole - reverse_time).abs().max())
        torch.testing.assert_close(whole, split, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(whole, shuffled, rtol=2e-5, atol=2e-5)
        if order_difference <= 1e-7:
            raise RuntimeError("本次拟合诊断无法观察到帧序影响")
    result = dict(ok=True, videos=len(x), initial_loss=float(loss.detach()),
        next_loss=next_loss, gradient_abs_sums=gradients,
        full_optimizer_and_rng_replay_exact=replay_ok,
        split_video_batch_max_error=split_error, shuffle_video_batch_max_error=shuffle_error,
        reverse_frame_order_max_difference=order_difference,
        wall_seconds=time.perf_counter() - started, costs=ledger.costs.copy(),
        gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        process_peak_working_set_bytes=peak_working_set_bytes(),
        note="诊断含3次更新（其中一次重放）；正式训练恢复独立第0遍，诊断不是准确率成绩。")
    write_json(directory / "diagnostic.json", result)
    # 保留诊断检查点作为恢复证据；释放大模型的临时复制。
    del reference, after, saved, output, loss
    return result


@torch.no_grad()
def evaluate_pool(model, x, y, directory, pool, epoch, ledger, batch_size):
    model.eval()
    outputs = []
    for start in range(0, len(x), batch_size):
        output = model(x[start:start + batch_size])
        if not torch.isfinite(output).all():
            raise RuntimeError("评价输出非有限")
        outputs.append(output.cpu().numpy())
        ledger.add(f"{pool}_forward", epoch, **{f"{pool}_forward_videos": len(output)})
    logits = np.concatenate(outputs)
    targets = y.cpu().numpy()
    metrics = evaluate(logits, targets)
    metrics["videos"] = len(x)
    ledger.add(f"{pool}_metrics", epoch, **{f"{pool}_metric_evaluations": 1})
    stem = f"{pool}_{epoch:03d}"
    write_json(directory / "evaluations" / f"{stem}.json", metrics)
    np.savez_compressed(directory / "evaluations" / f"{stem}.npz", logits=logits, y=targets)
    return metrics


def snapshot_reference(outdir, data):
    destination = outdir / "snn_reference"
    config = json.loads((REFERENCE / "config.json").read_text(encoding="utf-8"))
    if config["data"] != {key: data[key] for key in config["data"]}:
        raise ValueError("对照数据哈希/ID数量与SNN不一致")
    paths = [REFERENCE / "config.json", REFERENCE / "epoch_010.json"]
    paths += [REFERENCE / "evaluations" / f"{pool}_010.{ext}"
              for pool in ("control", "fit") for ext in ("json", "npz")]
    info = dict(epoch=10, files={str(path): sha(path) for path in paths},
        model_checkpoint=str(REFERENCE / "checkpoints/epoch_010_model.pt"),
        model_checkpoint_sha256=sha(REFERENCE / "checkpoints/epoch_010_model.pt"),
        note="只复制用户指定第10遍的已有评价证据；未运行新的SNN前向或更新。")
    destination.mkdir(parents=True, exist_ok=True)
    for path in paths:
        target = destination / path.relative_to(REFERENCE)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and sha(target) != info["files"][str(path)]:
            raise ValueError("冻结SNN参照发生变化")
        if not target.exists():
            shutil.copy2(path, target)
    write_json(destination / "provenance.json", info)
    return info


def make_config(arm, model, args, data):
    return dict(schema=1, arm=arm, model_seed=11, data_seed=20261004, data=data,
        maximum_epochs=args.epochs, effective_batch=args.batch_size, eval_batch=args.batch_size,
        optimizer="AdamW(foreach=False)", weight_decay=1e-4, gradient_clip_norm=1.0,
        lr_schedule={"1-50": 0.001, "51-80": 0.0003, "81-100": 0.0001},
        training_objective="数字10类交叉熵 + 运动7类交叉熵，全部参数反传",
        preprocessing="同SNN原灰度[0,1]；仅重排维度，无额外增强、归一化、特征或预训练",
        initialization="PyTorch模块默认初始化；各臂根11",
        fit_videos=4480, control_videos=1120, control_interval=5, fit_interval=10,
        dtype="float32", matmul_precision="highest", tf32=False, deterministic_algorithms=True,
        gpu=torch.cuda.get_device_name(0), gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
        python=platform.python_version(), torch=torch.__version__, cuda_runtime=torch.version.cuda,
        cpu_threads=4, resources=resources(model, arm),
        steps_per_video=8 if arm == "cnn_gru" else 1,
        neural_simulation_time=None, video_time="8帧；实际播放时长未规定",
        video_boundary="每视频独立，GRU初始状态零；无跨视频状态",
        control_selection="最高联合准确率；平分依次比较数字准确率、运动准确率、较早遍数",
        validation_arrays_unpacked=False, validation_forward_evaluations=0, sealed_test_reads=0,
        maximum_train_forward_videos=args.epochs * 4480,
        maximum_train_backward_videos=args.epochs * 4480,
        maximum_control_forward_videos=(args.epochs // 5) * 1120,
        maximum_fit_forward_videos=(args.epochs // 10) * 4480,
        maximum_selected_fit_forward_videos=4480,
        diagnostic_forward_videos=160, diagnostic_backward_videos=96,
        diagnostic_optimizer_updates=3,
        source_sha256=source_hashes(),
        start_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        start_status=subprocess.check_output(["git", "status", "--short"], text=True),
        shared_gpu_note="用户SNN可能同时运行；不将墙钟解释为独占速度或能耗",
        resume_policy="按遍恢复；完整成本账保留未保存遍的额外工作，回放额外成本可由超出完成遍预算量核算")


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
    for name in (*SOURCES, "docs/ann_comparison_execution.md", "pyproject.toml"):
        destination = directory / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(name, destination)
    (directory / "start_diff.patch").write_text(
        subprocess.check_output(["git", "diff"], text=True, encoding="utf-8"), encoding="utf-8")
    state = dict(arm=arm, completed_epochs=0, maximum_epochs=args.epochs,
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


def score(metrics):
    return tuple(metrics[k] for k in ("joint_accuracy", "digit_accuracy", "motion_accuracy"))


def result_for(model, state, started, session_base, reason):
    return dict(status=reason, completed_epochs=state["completed_epochs"],
        maximum_epochs=state["maximum_epochs"], best_control=state["best"], costs=state["costs"],
        wall_seconds_including_diagnostic=state["wall_seconds"] + time.perf_counter() - started - session_base,
        optimizer_training_wall_seconds=state["optimizer_training_wall_seconds"],
        gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        process_peak_working_set_bytes=peak_working_set_bytes(),
        validation_forward_evaluations=0, sealed_test_reads=0,
        extra_replayed_train_forward_videos=max(0, state["costs"]["train_forward_videos"] - state["completed_epochs"] * 4480),
        interpretation="控制参与检查点选择，为探索成绩；GPU共享，墙钟不代表独占速度或能耗。")


def train_arm(arm, args, data, x_fit, y_fit, x_control, y_control):
    directory = args.outdir / arm
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    if (config["data"] != data or config["source_sha256"] != source_hashes()
            or config["maximum_epochs"] != args.epochs or config["effective_batch"] != args.batch_size):
        raise ValueError(f"{arm}恢复源码/数据/预算/有效批量不符")
    saved = load_resume(directory)
    if not saved["state"].get("diagnostic_ok"):
        raise RuntimeError(f"{arm}尚未完成准备诊断，不能启动训练")
    if (directory / "result.json").exists():
        previous = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        if previous["status"] == "completed":
            print(f"{arm}已完成，不重新前向或训练。", flush=True)
            return True
    seed()
    model = make_model(arm).cuda()
    optimizer = optimizer_for(model)
    rng = np.random.default_rng(11)
    state = restore(saved, model, optimizer, rng)
    del saved
    ledger = Ledger(directory, state["costs"])
    ledger.recover()
    started = time.perf_counter()
    session_base = 0.0
    torch.cuda.reset_peak_memory_stats()
    try:
        for epoch in range(state["completed_epochs"] + 1, state["maximum_epochs"] + 1):
            if (args.outdir / "STOP").exists() or (directory / "STOP").exists():
                write_json(directory / "result.json", result_for(model, state, started, session_base, "stopped"))
                return False
            for group in optimizer.param_groups:
                group["lr"] = lr_for(epoch)
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
            control = fit = None
            if epoch % 5 == 0:
                control = evaluate_pool(model, x_control, y_control, directory, "control", epoch, ledger, args.batch_size)
                if state["best"] is None or score(control) > score(state["best"]["metrics"]):
                    state["best"] = dict(epoch=epoch, metrics=control)
                    atomic_save(directory / "best.pt", dict(arm=arm, epoch=epoch, model=model.state_dict()))
                    write_json(directory / "best_meta.json", state["best"])
            if epoch % 10 == 0:
                fit = evaluate_pool(model, x_fit, y_fit, directory, "fit", epoch, ledger, args.batch_size)
            row = dict(epoch=epoch, optimizer_mean_loss=loss_sum / len(x_fit), learning_rate=lr_for(epoch),
                gradient_norm_max=norm_max, optimizer_training_wall_seconds=train_wall,
                control=control, fit=fit, costs=state["costs"].copy())
            write_json(directory / f"epoch_{epoch:03d}.json", row)
            if epoch == 10:
                atomic_save(directory / "epoch_010_model.pt", dict(arm=arm, epoch=10, model=model.state_dict()))
            state["completed_epochs"] = epoch
            elapsed = time.perf_counter() - started
            state["wall_seconds"] += elapsed - session_base
            session_base = elapsed
            save_resume(directory, model, optimizer, rng, state)
            if control is not None:
                print(f"{arm}第{epoch}遍 控制：数字{control['digit_accuracy']:.2%} 运动{control['motion_accuracy']:.2%} 联合{control['joint_accuracy']:.2%}；平均损失{row['optimizer_mean_loss']:.4f}", flush=True)
            elif epoch == 1:
                print(f"{arm}第1遍完成，平均损失{row['optimizer_mean_loss']:.4f}，优化耗时{train_wall:.1f}秒。", flush=True)
            if fit is not None:
                print(f"{arm}第{epoch}遍 拟合：数字{fit['digit_accuracy']:.2%} 运动{fit['motion_accuracy']:.2%} 联合{fit['joint_accuracy']:.2%}", flush=True)
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
        result = result_for(model, state, started, session_base, "completed")
        result["best_control_fit"] = state["selected_fit_metrics"]
        result["resources"] = resources(model, arm)
        write_json(directory / "result.json", result)
        print(f"{arm}完成固定{state['completed_epochs']}遍预算，最佳控制联合{state['best']['metrics']['joint_accuracy']:.2%}（第{state['best']['epoch']}遍）。", flush=True)
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
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用，不能启动已登记GPU对照")
    if args.epochs < 10 or args.epochs % 10 or args.batch_size < 1:
        raise ValueError("预算至少10遍且为10的倍数；有效批量须为正")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    fit, control, data = training_data()
    for name, pool in (("fit", fit), ("control", control)):
        data[f"{name}_id_sequence_sha256"] = hashlib.sha256(pool[2].tobytes()).hexdigest()
        data[f"{name}_label_sequence_sha256"] = hashlib.sha256(pool[1].tobytes()).hexdigest()
    args.outdir.mkdir(parents=True, exist_ok=True)
    snapshot_reference(args.outdir, data)
    x_fit, y_fit = (torch.as_tensor(a, device="cuda") for a in fit[:2])
    x_control, y_control = (torch.as_tensor(a, device="cuda") for a in control[:2])
    np.savez_compressed(args.outdir / "split_evidence.npz", fit_ids=fit[2], fit_y=fit[1],
                        control_ids=control[2], control_y=control[1])
    for arm in args.arms:
        if not (args.outdir / arm).exists():
            prepare_arm(arm, args, data, x_fit, y_fit)
        elif not args.resume:
            raise ValueError(f"{arm}目录已存在，需要--resume")
    if args.prepare_only:
        print("三个模型准备完成；正式训练完成遍数均为0。", flush=True)
        return
    for arm in args.arms:
        if not train_arm(arm, args, data, x_fit, y_fit, x_control, y_control):
            return
    summary = dict(completed=True, arms=args.arms, results={arm: json.loads(
        (args.outdir / arm / "result.json").read_text(encoding="utf-8")) for arm in args.arms},
        data=data, validation_forward_evaluations=0, sealed_test_reads=0,
        resume_command=".\\.venv\\Scripts\\python.exe -u -m tools.train_ann_comparison --resume")
    write_json(args.outdir / "summary.json", summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=OUT)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
