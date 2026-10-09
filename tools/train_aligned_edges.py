"""节点对齐FC的独立、可恢复全局剪边实验；从项目根以module运行。"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import random
import subprocess
import tempfile
import time

import numpy as np
import torch
from torch.nn import functional as F

from tools.train_capacity_baseline import ROOT, augment, digest, evaluate, save_json
from tools.train_capacity_pruning import MaskedFC, capture, restore, export_model, load_export, sync
from braingrow.datasets import load_bundle


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def edge_counts(model):
    return {"mask_edges": [int(m.sum()) for m in model.masks],
            "nonzero_edges": [int(torch.count_nonzero(l.weight)) for l in model.layers],
            "nodes": [l.out_features for l in model.layers], "sharing": "none"}


def scores_and_protection(model):
    scores, protect = [], []
    for i, (layer, mask) in enumerate(zip(model.layers, model.masks)):
        values = layer.weight.detach().abs()
        rms = values[mask].square().mean().sqrt().clamp_min(1e-12)
        score = (values / rms).masked_fill(~mask, float("inf"))
        reserved = torch.zeros_like(mask)
        if i == len(model.layers) - 1:
            if not mask.any(1).all():
                raise ValueError("输出行已无输入边")
            index = values.masked_fill(~mask, -1).argmax(1)
            reserved[torch.arange(mask.shape[0], device=mask.device), index] = True
        scores.append(score)
        protect.append(reserved)
    return scores, protect


def propose(model, fraction, random_counts=None, random_seed=11):
    """仅接收权重/mask和通用比例；不读取任务坐标或CNN信息。"""
    scores, protect = scores_and_protection(model)
    selected = []
    with torch.no_grad():
        if random_counts is None:
            flat = torch.cat([s.masked_fill(p, float("inf")).flatten() for s, p in zip(scores, protect)])
            count = min(math.floor(sum(int(m.sum()) for m in model.masks) * fraction),
                        sum(int((m & ~p).sum()) for m, p in zip(model.masks, protect)))
            indices = torch.topk(flat, count, largest=False, sorted=False).indices
            start = 0
            for mask in model.masks:
                selected.append(indices[(indices >= start) & (indices < start + mask.numel())] - start)
                start += mask.numel()
        else:
            stream = torch.Generator(device=model.masks[0].device).manual_seed(random_seed)
            for mask, reserved, count in zip(model.masks, protect, random_counts):
                available = torch.nonzero((mask & ~reserved).flatten()).flatten()
                if count > len(available):
                    raise ValueError("随机对照可删边不足")
                selected.append(available[torch.randperm(len(available), device=available.device, generator=stream)[:count]])
        for mask, indices in zip(model.masks, selected):
            mask.flatten()[indices] = False
        model.enforce()
    return [int(len(i)) for i in selected]


def training_epoch(model, optimizer, teacher, x, y, cfg, epoch, period, ledger):
    torch.manual_seed(cfg["model_seed"] + 2000 + epoch)
    stream = torch.Generator(device=x.device).manual_seed(cfg["model_seed"] + 3000 + epoch)
    order = torch.randperm(len(x), device=x.device, generator=stream)
    optimizer.param_groups[0]["lr"] = cfg["minimum_learning_rate"] + .5 * (cfg["learning_rate"] - cfg["minimum_learning_rate"]) * (1 + math.cos(math.pi * ((epoch - 1) % period) / period))
    model.train()
    loss_sum, ce_sum, teacher_seconds = 0., 0., 0.
    for ids in order.split(cfg["batch_size"]):
        inputs = augment(x[ids], stream, cfg)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        ce = F.cross_entropy(logits, y[ids])
        loss = ce
        if teacher is not None:
            t = time.perf_counter()
            with torch.no_grad():
                target = teacher(inputs)
            sync(x.device)
            teacher_seconds += time.perf_counter() - t
            temperature = cfg["distillation_temperature"]
            kd = F.kl_div(F.log_softmax(logits / temperature, dim=1), F.softmax(target / temperature, dim=1), reduction="batchmean") * temperature ** 2
            loss = (1 - cfg["distillation_weight"]) * ce + cfg["distillation_weight"] * kd
            ledger["teacher_samples"] += len(ids)
        if not torch.isfinite(loss):
            raise FloatingPointError("训练损失非有限")
        loss.backward()
        with torch.no_grad():
            for layer, mask in zip(model.layers, model.masks):
                layer.weight.grad.mul_(mask)
        optimizer.step()
        model.enforce(optimizer)
        ledger["attempted_updates"] += 1
        ledger["optimization_samples"] += len(ids)
        loss_sum += float(loss.detach()) * len(ids)
        ce_sum += float(ce.detach()) * len(ids)
    optimizer.zero_grad(set_to_none=True)
    return {"training_objective": loss_sum / len(x), "hard_label_ce": ce_sum / len(x), "teacher_seconds": teacher_seconds}


def new_model(cfg, device, state=None):
    seed(cfg["model_seed"])
    model = MaskedFC(cfg["fc_widths"], cfg["dropout"]).to(device)
    if state is not None:
        model.load_state_dict(state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    return model, optimizer


def checkpoint(path, model, optimizer, info):
    torch.save({"state": capture(model, optimizer, []), "info": info}, path.with_suffix(".tmp"))
    path.with_suffix(".tmp").replace(path)


def empty_ledger():
    return {"attempted_updates": 0, "net_updates": 0, "optimization_samples": 0, "teacher_samples": 0,
            "evaluation_calls": 0, "evaluation_samples": 0, "rollbacks": 0}


def measure(model, sx, sy, cfg, ledger):
    result, logits = evaluate(model, sx, sy, cfg["evaluation_batch_size"])
    ledger["evaluation_calls"] += 1
    ledger["evaluation_samples"] += len(sx)
    return result, logits


def function_ok(selected, reference, cfg):
    return (selected["accuracy"] + cfg["accuracy_drop"] + 1e-7 >= reference["accuracy"]
            and selected["cross_entropy"] <= reference["cross_entropy"] + cfg["cross_entropy_increase"])


class Budget:
    def __init__(self, limit, prior=0.):
        self.started = time.perf_counter()
        self.limit, self.prior = limit, prior

    def elapsed(self):
        return self.prior + time.perf_counter() - self.started

    def check(self, device):
        if self.elapsed() >= self.limit:
            raise TimeoutError("本轮1800秒预算达到；保留checkpoint，未自动追加预算")
        if device.type == "cuda" and torch.cuda.max_memory_allocated() > 8589934592:
            raise MemoryError("本轮8GiB峰值预算达到")


def baseline(cfg, data, device, folder, budget):
    arm = folder / "baseline"
    arm.mkdir(exist_ok=True)
    if (arm / "result.json").exists():
        return json.loads((arm / "result.json").read_text())
    model, optimizer = new_model(cfg, device)
    ledger, rows, best = empty_ledger(), [], None
    epoch_start = 1
    if (arm / "resume.pt").exists():
        saved = torch.load(arm / "resume.pt", map_location="cpu", weights_only=False)
        restore(model, optimizer, [], saved["state"])
        ledger, rows, best = saved["info"]["ledger"], saved["info"]["rows"], saved["info"]["best"]
        epoch_start = saved["info"]["epoch"] + 1
    x, y = torch.from_numpy(data.fit_x).to(device), torch.from_numpy(data.fit_y).to(device)
    sx, sy = torch.from_numpy(data.select_x).to(device), torch.from_numpy(data.select_y).to(device)
    started = time.perf_counter()
    for epoch in range(epoch_start, cfg["baseline_epochs"] + 1):
        t = time.perf_counter()
        trained = training_epoch(model, optimizer, None, x, y, cfg, epoch, cfg["baseline_epochs"], ledger)
        sync(device)
        trained["training_seconds"] = time.perf_counter() - t
        t = time.perf_counter()
        selected, logits = measure(model, sx, sy, cfg, ledger)
        sync(device)
        row = {"epoch": epoch, "select": selected, "evaluation_seconds": time.perf_counter() - t, **trained}
        rows.append(row)
        ledger["net_updates"] += math.ceil(len(x) / cfg["batch_size"])
        if best is None or (selected["accuracy"], -selected["cross_entropy"]) > (best["select"]["accuracy"], -best["select"]["cross_entropy"]):
            best = {"epoch": epoch, "select": selected}
            checkpoint(arm / "best.pt", model, optimizer, {"epoch": epoch, "ledger": ledger, "rows": rows})
            np.savez_compressed(arm / "select_best.npz", logits=logits, labels=data.select_y)
        checkpoint(arm / "resume.pt", model, optimizer, {"epoch": epoch, "ledger": ledger, "rows": rows, "best": best})
        save_json(arm / "progress.json", {"rows": rows, "best": best, "ledger": ledger, "session_wall_seconds": budget.elapsed()})
        print(json.dumps({"arm": "baseline", "epoch": epoch, "accuracy": selected["accuracy"], "best": best["select"]["accuracy"]}), flush=True)
        budget.check(device)
        if epoch >= cfg["baseline_minimum_epochs"] and best["select"]["accuracy"] >= cfg["baseline_select_target"]:
            break
    saved = torch.load(arm / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(saved["state"]["model"])
    fit, fit_logits = measure(model, x, y, cfg, ledger)
    np.savez_compressed(arm / "fit_final.npz", logits=fit_logits, labels=data.fit_y)
    result = {"best": best, "fit": fit, "rows": rows, "ledger": ledger, "structure": {**export_model(model, arm / "model.bgm"), **edge_counts(model)},
              "wall_seconds": time.perf_counter() - started, "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
              "gate": "pass" if best["select"]["accuracy"] >= cfg["baseline_function_floor"] else "below_function_floor"}
    save_json(arm / "result.json", result)
    return result


def run_arm(name, cfg, data, device, folder, parent, reference, events, budget):
    arm = folder / name
    arm.mkdir(exist_ok=True)
    if (arm / "result.json").exists():
        result = json.loads((arm / "result.json").read_text())
        if name == "distilled":
            events.update({int(k): v for k, v in result["events"].items()})
        return result
    model, optimizer = new_model(cfg, device, parent)
    teacher = None
    if name in ("distilled", "random"):
        teacher, _ = new_model(cfg, device, parent)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
    x, y = torch.from_numpy(data.fit_x).to(device), torch.from_numpy(data.fit_y).to(device)
    sx, sy = torch.from_numpy(data.select_x).to(device), torch.from_numpy(data.select_y).to(device)
    ledger, rows, stages = empty_ledger(), [], []
    costs = {k: 0. for k in ("training_seconds", "teacher_seconds", "selection_seconds", "proposal_seconds", "snapshot_seconds", "export_seconds")}
    state = {"stage": 1, "epoch_in_block": 0, "mode": "new", "saved_before": None, "event": None}
    t = time.perf_counter()
    initial = {**export_model(model, arm / "stage_0.bgm"), **edge_counts(model)}
    costs["export_seconds"] += time.perf_counter() - t
    stages.append({"stage": 0, "select": reference, "structure": initial, "accepted": True})
    if (arm / "resume.pt").exists():
        saved = torch.load(arm / "resume.pt", map_location="cpu", weights_only=False)
        restore(model, optimizer, [], saved["state"])
        info = saved["info"]
        ledger, rows, stages, costs, state = [info[k] for k in ("ledger", "rows", "stages", "costs", "controller")]
        if name == "distilled":
            events.update(info["events"])
    started = time.perf_counter()

    def save():
        checkpoint(arm / "resume.pt", model, optimizer, {"ledger": ledger, "rows": rows, "stages": stages, "costs": costs,
                   "controller": state, "events": events, "session_wall_seconds": budget.elapsed()})
        save_json(arm / "progress.json", {"rows": rows, "stages": stages, "ledger": ledger, "costs": costs, "controller": {k: v for k, v in state.items() if k != "saved_before"}, "session_wall_seconds": budget.elapsed()})

    while state["stage"] <= cfg["stages"]:
        stage = state["stage"]
        if state["mode"] == "new":
            state["epoch_in_block"] = 0
            state["event"] = None
            if name in ("magnitude", "distilled"):
                t = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                state["saved_before"] = capture(model, optimizer, [])
                costs["snapshot_seconds"] += time.perf_counter() - t
                t = time.perf_counter()
                counts = propose(model, cfg["delete_fraction"])
                model.enforce(optimizer)
                sync(device)
                costs["proposal_seconds"] += time.perf_counter() - t
                state["event"] = {"stage": stage, "deleted_per_layer": counts, "before": stages[-1]["structure"]}
                state["mode"] = "candidate"
            elif name == "random" and events[stage]["accepted"]:
                t = time.perf_counter()
                counts = propose(model, cfg["delete_fraction"], events[stage]["deleted_per_layer"], cfg["model_seed"] + 4000 + stage)
                model.enforce(optimizer)
                sync(device)
                costs["proposal_seconds"] += time.perf_counter() - t
                state["event"] = {"stage": stage, "deleted_per_layer": counts, "forced_match": True}
                state["mode"] = "forced"
            else:
                state["mode"] = "ordinary"
            save()
        for step in range(state["epoch_in_block"] + 1, cfg["adaptation_epochs"] + 1):
            epoch = (stage - 1) * cfg["adaptation_epochs"] + step
            t = time.perf_counter()
            trained = training_epoch(model, optimizer, teacher, x, y, cfg, epoch, cfg["adaptation_epochs"], ledger)
            sync(device)
            costs["training_seconds"] += time.perf_counter() - t
            costs["teacher_seconds"] += trained["teacher_seconds"]
            rows.append({"stage": stage, "epoch": epoch, "mode": state["mode"], **trained})
            state["epoch_in_block"] = step
            save()
            print(json.dumps({"arm": name, "stage": stage, "epoch": epoch, "mode": state["mode"], "hard_ce": trained["hard_label_ce"]}), flush=True)
            budget.check(device)
        t = time.perf_counter()
        selected, logits = measure(model, sx, sy, cfg, ledger)
        sync(device)
        costs["selection_seconds"] += time.perf_counter() - t
        t = time.perf_counter()
        exported = {**export_model(model, arm / f"stage_{stage}_{state['mode']}.bgm"), **edge_counts(model)}
        costs["export_seconds"] += time.perf_counter() - t
        if state["mode"] == "candidate":
            accepted = function_ok(selected, reference, cfg) and exported["bytes"] < state["event"]["before"]["bytes"]
            state["event"].update({"accepted": accepted, "candidate_select": selected, "candidate_structure": exported})
            if name == "distilled":
                events[stage] = copy.deepcopy(state["event"])
            if not accepted:
                t = time.perf_counter()
                restore(model, optimizer, [], state["saved_before"])
                costs["snapshot_seconds"] += time.perf_counter() - t
                ledger["rollbacks"] += 1
                state["saved_before"] = None
                state["mode"], state["epoch_in_block"] = "fallback", 0
                save()
                continue
        else:
            accepted = name == "fixed" or state["mode"] == "forced"
        # Complete stage snapshot also preserves every rejected candidate metric.
        export_model(model, arm / f"stage_{stage}.bgm")
        np.savez_compressed(arm / f"select_stage_{stage}.npz", logits=logits, labels=data.select_y)
        stages.append({"stage": stage, "select": selected, "structure": exported, "accepted": accepted, "event": state["event"]})
        ledger["net_updates"] += cfg["adaptation_epochs"] * math.ceil(len(x) / cfg["batch_size"])
        checkpoint(arm / f"stage_{stage}.pt", model, optimizer, {"stage": stage, "ledger": ledger, "costs": costs, "event": state["event"]})
        print(json.dumps({"arm": name, "stage_complete": stage, "accuracy": selected["accuracy"], "nonzero_edges": sum(exported["nonzero_edges"]), "accepted": accepted}), flush=True)
        state = {"stage": stage + 1, "epoch_in_block": 0, "mode": "new", "saved_before": None, "event": None}
        save()
        budget.check(device)
    qualified = [r for r in stages if function_ok(r["select"], reference, cfg)]
    chosen = min(qualified, key=lambda r: (sum(r["structure"]["nonzero_edges"]), -r["select"]["accuracy"], r["select"]["cross_entropy"]))
    model = load_export(arm / f"stage_{chosen['stage']}.bgm").to(device)
    fit, fit_logits = measure(model, x, y, cfg, ledger)
    np.savez_compressed(arm / "fit_final.npz", logits=fit_logits, labels=data.fit_y)
    exported = {**export_model(model, arm / "model.bgm"), **edge_counts(model)}
    result = {"arm": name, "stages": stages, "rows": rows, "chosen_stage": chosen["stage"], "select": chosen["select"], "fit": fit,
              "structure": exported, "ledger": ledger, "costs": costs, "events": events if name == "distilled" else {}, "wall_seconds": time.perf_counter() - started,
              "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None}
    save_json(arm / "result.json", result)
    return result


def check():
    torch.set_num_threads(2)
    seed(11)
    cfg = {"fc_widths": [8, 7, 5, 3, 10], "dropout": 0., "model_seed": 11, "learning_rate": .001, "weight_decay": .0001}
    model, optimizer = new_model(cfg, torch.device("cpu"))
    x, y = torch.randn(6, 8), torch.arange(6)
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    saved = capture(model, optimizer, [])
    expected = model(x).detach().clone()
    perm = torch.tensor([3, 0, 6, 1, 7, 2, 5, 4])
    shuffled, other_opt = new_model(cfg, torch.device("cpu"), saved["model"])
    with torch.no_grad():
        shuffled.layers[0].weight.copy_(model.layers[0].weight[:, perm])
    counts = propose(model, .5)
    other_counts = propose(shuffled, .5)
    assert counts == other_counts
    assert torch.equal(shuffled.masks[0], model.masks[0][:, perm])
    assert all(torch.equal(a, b) for a, b in zip(shuffled.masks[1:], model.masks[1:]))
    assert model.masks[-1].any(1).all()
    torch.testing.assert_close(shuffled(x[:, perm]), model(x))
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    for layer, mask in zip(model.layers, model.masks):
        layer.weight.grad.mul_(mask)
    optimizer.step()
    model.enforce(optimizer)
    for layer, mask in zip(model.layers, model.masks):
        assert not torch.count_nonzero(layer.weight[~mask])
        for key in ("exp_avg", "exp_avg_sq"):
            assert not torch.count_nonzero(optimizer.state[layer.weight][key][~mask])
    restore(model, optimizer, [], saved)
    torch.testing.assert_close(model(x), expected, rtol=0, atol=0)
    again = capture(model, optimizer, [])
    def equal(a, b):
        if isinstance(a, torch.Tensor):
            return torch.equal(a, b)
        if isinstance(a, np.ndarray):
            return np.array_equal(a, b)
        if isinstance(a, dict):
            return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
        if isinstance(a, (list, tuple)):
            return len(a) == len(b) and all(equal(i, j) for i, j in zip(a, b))
        return a == b
    assert equal(saved, again)
    with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as temporary:
        path = Path(temporary) / "model.bgm"
        export_model(model, path)
        loaded = load_export(path)
        torch.testing.assert_close(loaded(x), expected, rtol=0, atol=0)
    print("新增CPU检查通过：全局剪边、输出保护、置乱对应、mask/Adam零约束、完整回退、导出一致性", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.check:
        (ROOT / "tmp").mkdir(exist_ok=True)
        check()
        return
    cfg_path = ROOT / "configs/h1_aligned_edges_v1.json"
    cfg = json.loads(cfg_path.read_text())
    assert cfg["model_seed"] == 11 and cfg["data_seed"] == 20261004
    assert cfg["fc_widths"] == [784, 3136, 1568, 128, 10]
    folder = ROOT / "experiments/runs" / cfg["version"] / "mnist_static"
    folder.mkdir(parents=True, exist_ok=True)
    if any(folder.iterdir()) and not args.resume:
        raise FileExistsError("已有证据：仅--resume接续，拒绝覆盖")
    if (folder / "summary.json").exists():
        print((folder / "summary.json").read_text(), flush=True)
        return
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    registration = {"config": cfg, "config_sha256": digest(cfg_path), "source_sha256": digest(Path(__file__)),
                    "reused_source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in (ROOT / "tools/train_capacity_baseline.py", ROOT / "tools/train_capacity_pruning.py")},
                    "training_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    "device": str(device), "device_name": torch.cuda.get_device_name() if device.type == "cuda" else "CPU", "precision": "FP32", "torch": torch.__version__,
                    "execution": "dense_linear_zero_mask", "cpu_threads": 2, "initial_edges": sum(a * b for a, b in zip(cfg["fc_widths"], cfg["fc_widths"][1:])),
                    "non_input_nodes": sum(cfg["fc_widths"][1:]), "official_test_accessed": False, "sealed_remaining_ids_accessed": False}
    prior = 0.
    if (folder / "registration.json").exists():
        old = json.loads((folder / "registration.json").read_text())
        if any(old[k] != registration[k] for k in ("config_sha256", "source_sha256", "reused_source_sha256")):
            raise ValueError("源/配置哈希变化，拒绝恢复")
        registration = old
        elapsed = [json.loads(p.read_text()).get("session_wall_seconds", 0.) for p in folder.glob("*/progress.json")]
        prior = max(elapsed, default=0.)
    else:
        save_json(folder / "registration.json", registration)
    budget = Budget(cfg["wall_seconds_total"], prior)
    results, events = {}, {}
    try:
        budget.check(device)
        data = load_bundle("mnist", include_confirm=False)
        assert data.fit_x.shape == (3200, 784) and data.select_x.shape == (800, 784) and not len(data.confirm_x)
        save_json(folder / "data_manifest.json", data.manifest)
        base = baseline(cfg, data, device, folder, budget)
        results["baseline"] = base
        if base["gate"] != "pass":
            save_json(folder / "summary.json", {"registration": registration, "results": results, "gate": "baseline_below_function_floor", "session_wall_seconds": budget.elapsed()})
            return
        parent_file = folder / "baseline/best.pt"
        parent = torch.load(parent_file, map_location="cpu", weights_only=False)["state"]["model"]
        for name in ("fixed", "magnitude", "distilled", "random"):
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            results[name] = run_arm(name, cfg, data, device, folder, parent, base["best"]["select"], events, budget)
        # Every candidate and selected stage is frozen before this historical exploratory access.
        confirmed = load_bundle("mnist", include_confirm=True)
        cx, cy = torch.from_numpy(confirmed.confirm_x).to(device), torch.from_numpy(confirmed.confirm_y).to(device)
        for name, result in results.items():
            if "control" in result:
                continue
            model = load_export(folder / name / "model.bgm").to(device)
            control, logits = evaluate(model, cx, cy, cfg["evaluation_batch_size"])
            result["control"] = control
            np.savez_compressed(folder / name / "control_final.npz", logits=logits, labels=confirmed.confirm_y)
            save_json(folder / name / "result.json", result)
            del model
            budget.check(device)
        summary = {"registration": registration, "parent_sha256": digest(parent_file), "results": results, "events": events,
                   "session_wall_seconds": budget.elapsed(), "gate": "completed", "official_test_accessed": False, "sealed_remaining_ids_accessed": False}
        save_json(folder / "summary.json", summary)
        print(json.dumps({"gate": "completed", "wall_seconds": budget.elapsed(), "results": {n: {"select": r.get("select", r.get("best", {}).get("select")), "control": r["control"], "edges": r["structure"]["nonzero_edges"]} for n, r in results.items()}}, ensure_ascii=False), flush=True)
    except Exception as exc:
        save_json(folder / "failure.json", {"error": repr(exc), "session_wall_seconds": budget.elapsed(), "completed_arms": list(results), "resume": "python -m tools.train_aligned_edges --resume"})
        raise


if __name__ == "__main__":
    main()
