"""同一已训练LIF图上的固定继续训练与两层全连接加点诊断。"""
from __future__ import annotations

import copy
import json
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import hashlib
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from braingrow.network.layered_gradient import LayeredGradientSNN
from tools.run_layered_gradient import (evaluate_control, loss_for, peak_working_set_bytes,
                                        record_cost, training_data, write_json)


BASE = Path("experiments/runs/layered_gradient_v1/baseline_resume.pt")
OUT = Path("experiments/runs/layered_growth_d1")


def optimizer_after_growth(old: torch.optim.AdamW, model: LayeredGradientSNN,
                           pairs: list[tuple[torch.nn.Parameter, torch.nn.Parameter]]) -> torch.optim.AdamW:
    """保留所有旧边的AdamW步数/动量，新边动量为零。"""
    new = torch.optim.AdamW(model.parameters(), lr=old.param_groups[0]["lr"],
                            weight_decay=old.param_groups[0]["weight_decay"])
    for before, after in pairs:
        if before not in old.state:
            continue
        for key, value in old.state[before].items():
            if not torch.is_tensor(value):
                new.state[after][key] = copy.deepcopy(value)
            elif value.ndim == 0:
                new.state[after][key] = value.clone()
            else:
                expanded = torch.zeros_like(after)
                old_region = tuple(slice(0, size) for size in before.shape)
                expanded[old_region] = value
                new.state[after][key] = expanded
    return new


def make_branch(checkpoint: dict, arm: str, count: int, device: torch.device,
                sample: torch.Tensor, birth_mode: str, parents_by_layer: dict[int, torch.Tensor]):
    model = LayeredGradientSNN(tuple(checkpoint["widths"]), device=device)
    model.load_state_dict(checkpoint["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    optimizer.load_state_dict(checkpoint["optimizer"])
    with torch.no_grad():
        before = model(sample)
    birth_error = 0.0
    if arm != "fixed":
        layer = 0 if arm == "grow_layer1" else 1
        if birth_mode == "split_active":
            pairs = model.split_active(layer, parents_by_layer[layer], proposal_id=layer + 1)
        else:
            pairs = model.grow(layer, count, proposal_id=layer + 1)
        optimizer = optimizer_after_growth(optimizer, model, pairs)
        with torch.no_grad():
            birth_error = float((model(sample) - before).abs().max().item())
        if birth_error > 1e-4:
            raise RuntimeError(f"新节点出生时改变了旧函数: {arm} {birth_error}")
    rng = np.random.default_rng()
    rng.bit_generator.state = copy.deepcopy(checkpoint["numpy_rng"])
    return model, optimizer, rng, birth_error


@torch.no_grad()
def active_parents(checkpoint: dict, x_fit: torch.Tensor, count: int,
                   batch_size: int) -> dict[int, torch.Tensor]:
    model = LayeredGradientSNN(tuple(checkpoint["widths"]), device=x_fit.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    totals = [torch.zeros(n, device=x_fit.device) for n in model.widths]
    for start in range(0, len(x_fit), batch_size):
        _, activity = model(x_fit[start:start + batch_size], trace=True)
        for layer in (0, 1):
            totals[layer] += activity["per_unit_fire"][layer]
    return {layer: torch.argsort(totals[layer], descending=True, stable=True)[:count]
            for layer in (0, 1)}


def class_rates(confusion: list[list[int]]) -> np.ndarray:
    table = np.asarray(confusion)
    return np.diag(table) / table.sum(axis=1)


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用；本实验不退回CPU")
    if (args.outdir / "summary.json").exists():
        print("D1已完成；恢复入口不重复训练", flush=True)
        return
    if args.outdir.exists() and not args.resume:
        raise FileExistsError("生长诊断目录已存在；请显式使用--resume")
    if args.resume and not args.outdir.exists():
        raise FileNotFoundError("没有可恢复的D1目录")
    checkpoint = torch.load(args.base, map_location="cpu", weights_only=False)
    if checkpoint["epoch"] != 21 or tuple(checkpoint["widths"]) != (512, 512):
        raise ValueError("本次D1只接受预登记的第21遍共同图")
    torch.manual_seed(11)
    torch.cuda.manual_seed_all(11)
    torch.use_deterministic_algorithms(True)
    fit, control, data_info = training_data()
    device = torch.device("cuda")
    x_fit, y_fit = (torch.as_tensor(a, device=device) for a in fit[:2])
    x_control, y_control = (torch.as_tensor(a, device=device) for a in control[:2])
    sha = hashlib.sha256(args.base.read_bytes()).hexdigest()
    if args.resume:
        config = json.loads((args.outdir / "config.json").read_text(encoding="utf-8"))
        if (config["base_checkpoint_sha256"] != sha or config["new_nodes_per_growth_arm"] != args.count
                or config["birth_mode"] != args.birth_mode
                or config["training_epochs_per_arm"] != args.epochs):
            raise ValueError("恢复配置/共同起点不符")
    else:
        args.outdir.mkdir(parents=True)
        write_json(args.outdir / "config.json", {"base_checkpoint": str(args.base),
        "base_checkpoint_sha256": sha, "base_epoch": 21,
        "start_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "start_diff": subprocess.check_output(["git", "diff"], text=True, encoding="utf-8"),
        "data": data_info, "root_seed": 11, "device": torch.cuda.get_device_name(0),
        "arms": ["fixed", "grow_layer1", "grow_layer2"], "new_nodes_per_growth_arm": args.count,
        "birth_mode": args.birth_mode,
        "new_edges": "new LIF fully connected to immediately preceding and following layers; random mode has zero outgoing weights; split mode duplicates active incoming edges and conserves outgoing sum",
        "training_epochs_per_arm": args.epochs, "batch_size": args.batch_size,
        "candidate_selection_pool": "train_ID_holdout_40; validation_unopened",
        "maximum_train_forward_videos": 3 * args.epochs * len(x_fit),
        "maximum_train_backward_videos": 3 * args.epochs * len(x_fit),
        "maximum_control_forward_videos": 3 * args.epochs * len(x_control),
        "maximum_parent_rank_forward_videos": len(x_fit) if args.birth_mode == "split_active" else 0,
        "validation_forward_evaluations": 0, "sealed_test_reads": 0,
        "accept_signal": "growth joint >= fixed +0.02; digit/motion >= fixed -0.01; every class <=0.05 decline",
        "neural_time_ms": 160, "video_time": "8帧，实际播放时长未规定"})
        snapshot = args.outdir / "source_snapshot"
        snapshot.mkdir()
        for name in ("src/braingrow/network/layered_gradient.py", "tools/run_layered_gradient.py",
                     "tools/run_layered_growth.py", "docs/layered_gradient_execution.md"):
            target = snapshot / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(name, target)
    costs = {"train_forward_videos": 0, "train_backward_videos": 0,
             "control_forward_videos": 0, "metric_evaluations": 0,
             "birth_probes_forward_videos": 0, "parent_rank_forward_videos": 0}
    if args.resume:
        journal = args.outdir / "cost_journal.jsonl"
        if journal.exists() and journal.stat().st_size:
            costs = json.loads(journal.read_text(encoding="utf-8").splitlines()[-1])["cumulative_costs"]
    results = {arm: json.loads((args.outdir / f"{arm}_result.json").read_text(encoding="utf-8"))
               for arm in ("fixed", "grow_layer1", "grow_layer2")
               if (args.outdir / f"{arm}_result.json").exists()}
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    parents_by_layer = {}
    if args.birth_mode == "split_active":
        parent_path = args.outdir / "parent_indices.json"
        if parent_path.exists():
            stored = json.loads(parent_path.read_text(encoding="utf-8"))
            parents_by_layer = {int(layer): torch.tensor(indices, device=device)
                                for layer, indices in stored.items()}
        else:
            parents_by_layer = active_parents(checkpoint, x_fit, args.count, args.batch_size)
            write_json(parent_path, {str(layer): indices.cpu().tolist()
                                     for layer, indices in parents_by_layer.items()})
            costs["parent_rank_forward_videos"] += len(x_fit)
            record_cost(args.outdir, costs, "parent_rank_fit_only", len(x_fit), 21)
    for arm in ("fixed", "grow_layer1", "grow_layer2"):
        if arm in results:
            continue
        arm_resume = args.outdir / f"{arm}_resume.pt"
        if arm_resume.exists():
            saved = torch.load(arm_resume, map_location="cpu", weights_only=False)
            model = LayeredGradientSNN(tuple(saved["widths"]), device=device)
            model.load_state_dict(saved["model"])
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
            optimizer.load_state_dict(saved["optimizer"])
            rng = np.random.default_rng()
            rng.bit_generator.state = saved["rng"]
            birth_error = saved["birth_error"]
            start_offset = int(saved["offset"]) + 1
            best = saved["best"]
        else:
            model, optimizer, rng, birth_error = make_branch(
                checkpoint, arm, args.count, device, x_fit[:16], args.birth_mode, parents_by_layer)
            costs["birth_probes_forward_videos"] += 16 * (1 if arm == "fixed" else 2)
            record_cost(args.outdir, costs, f"birth_probe_{arm}", 16 * (1 if arm == "fixed" else 2), 21)
            start_offset, best = 1, None
        for offset in range(start_offset, args.epochs + 1):
            model.train()
            order = rng.permutation(len(x_fit))
            fit_loss_sum = 0.0
            for begin in range(0, len(order), args.batch_size):
                positions = torch.as_tensor(order[begin:begin + args.batch_size], device=device)
                optimizer.zero_grad(set_to_none=True)
                potential = model(x_fit[positions])
                loss = loss_for(potential, y_fit[positions])
                if not torch.isfinite(loss):
                    raise RuntimeError(f"{arm} 损失非有限值")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                fit_loss_sum += float(loss.item()) * len(positions)
                costs["train_forward_videos"] += len(positions)
                costs["train_backward_videos"] += len(positions)
                record_cost(args.outdir, costs, f"train_{arm}", len(positions), 21 + offset)
            metrics, arrays = evaluate_control(model, x_control, y_control, args.batch_size)
            costs["control_forward_videos"] += len(x_control)
            costs["metric_evaluations"] += 1
            record_cost(args.outdir, costs, f"control_{arm}", len(x_control), 21 + offset)
            score = sum(metrics[key] for key in ("digit_accuracy", "motion_accuracy", "joint_accuracy"))
            row = {"arm": arm, "epoch_from_base": offset, "fit_loss": fit_loss_sum / len(x_fit),
                   "control": metrics, "score": score, "costs": costs.copy(),
                   "wall_seconds": time.perf_counter() - started}
            with (args.outdir / "epochs.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
            if best is None or score > best["score"]:
                best = {"score": score, "epoch_from_base": offset, "metrics": metrics}
                write_json(args.outdir / f"{arm}_best_meta.json", best)
                np.savez_compressed(args.outdir / f"{arm}_best_outputs.npz", **arrays)
                torch.save({"model": model.state_dict(), "widths": model.widths,
                            "optimizer": optimizer.state_dict(), "rng": rng.bit_generator.state,
                            "base_checkpoint_sha256": sha, "arm": arm, "epoch_from_base": offset},
                           args.outdir / f"{arm}_best.pt")
            torch.save({"model": model.state_dict(), "widths": model.widths,
                        "optimizer": optimizer.state_dict(), "rng": rng.bit_generator.state,
                        "birth_error": birth_error, "offset": offset, "best": best},
                       arm_resume.with_suffix(".pending"))
            arm_resume.with_suffix(".pending").replace(arm_resume)
            print(json.dumps({"arm": arm, "epoch": offset, "fit_loss": row["fit_loss"],
                  "digit": metrics["digit_accuracy"], "motion": metrics["motion_accuracy"],
                  "joint": metrics["joint_accuracy"], "score": score}, ensure_ascii=False), flush=True)
        results[arm] = {"birth_output_max_error": birth_error, "best": best,
                        "final_resources": model.resources()}
        write_json(args.outdir / f"{arm}_result.json", results[arm])
    fixed = results["fixed"]["best"]["metrics"]
    accepted = []
    for arm in ("grow_layer1", "grow_layer2"):
        proposed = results[arm]["best"]["metrics"]
        per_class_ok = all(np.all(class_rates(proposed[key]) >= class_rates(fixed[key]) - 0.05)
                           for key in ("digit_confusion", "motion_confusion"))
        passes = (proposed["joint_accuracy"] >= fixed["joint_accuracy"] + 0.02
                  and proposed["digit_accuracy"] >= fixed["digit_accuracy"] - 0.01
                  and proposed["motion_accuracy"] >= fixed["motion_accuracy"] - 0.01
                  and per_class_ok)
        if passes:
            accepted.append(arm)
    write_json(args.outdir / "summary.json", {"completed": True, "results": results,
        "growth_arms_passing_predeclared_signal": accepted,
        "costs": costs, "wall_seconds": time.perf_counter() - started,
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "process_peak_working_set_bytes": peak_working_set_bytes(),
        "validation_forward_evaluations": 0, "sealed_test_reads": 0,
        "interpretation_limit": "同一结构控制池已参与训练决策；这里仅是D1机制信号，不是独立验证成绩"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--outdir", type=Path, default=OUT)
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--birth-mode", choices=("random", "split_active"), default="random")
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
