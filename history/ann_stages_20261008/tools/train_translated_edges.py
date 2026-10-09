"""位置均衡MNIST与节点贡献剪边；独立入口，不修改既有结果。"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from tools.train_capacity_baseline import ROOT, digest, evaluate, save_json
from tools.train_capacity_pruning import MaskedFC, capture, restore, export_model, load_export, sync
from tools.train_aligned_edges import seed, edge_counts, propose, function_ok, Budget, empty_ledger, checkpoint
from braingrow.datasets import load_bundle


class CircularCNN(nn.Module):
    def __init__(self, dropout=.2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, stride=2, padding=1, padding_mode="circular")
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1, padding_mode="circular")
        self.fc1 = nn.Linear(1568, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = dropout

    def forward(self, x):
        x = F.relu(self.conv1(x.reshape(-1, 1, 28, 28)))
        x = F.relu(self.conv2(x)).flatten(1)
        x = F.dropout(F.relu(self.fc1(x)), self.dropout, self.training)
        return self.fc2(x)


def shifts(x, offsets):
    """每样本独立整数循环平移；仅索引复制，不插值/裁剪。"""
    offsets = offsets.to(x.device)
    rows = (torch.arange(28, device=x.device)[None, :, None] - offsets[:, 0, None, None]) % 28
    cols = (torch.arange(28, device=x.device)[None, None, :] - offsets[:, 1, None, None]) % 28
    indices = (rows * 28 + cols).reshape(len(x), 784)
    return x.flatten(1).gather(1, indices)


def fixed_offsets(count, repeats, root):
    """每ID位置互异；全体位移频数至多相差1，不用标签分配位置。"""
    rng = np.random.default_rng(root)
    sequence = np.resize(rng.permutation(784), count * repeats).reshape(count, repeats)
    sequence = sequence[rng.permutation(count)]
    return np.stack((sequence // 28, sequence % 28), axis=-1).astype(np.int64)


def evaluation_data(x, y, repeats, root):
    offsets = fixed_offsets(len(x), repeats, root)
    tx = shifts(x.repeat_interleave(repeats, 0), torch.from_numpy(offsets.reshape(-1, 2)))
    return tx, y.repeat_interleave(repeats), offsets


def measure(model, x, y, cfg, ledger):
    result, logits = evaluate(model, x, y, cfg["evaluation_batch_size"])
    ledger["evaluation_calls"] += 1
    ledger["evaluation_samples"] += len(x)
    repeats = cfg["validation_positions_per_id"]
    correct = logits.argmax(1) == y.cpu().numpy()
    grouped = correct.reshape(-1, repeats)
    prediction = logits.argmax(1).reshape(-1, repeats)
    result.update({"original_ids": len(grouped), "positions_per_id": repeats,
                   "all_positions_correct_fraction": float(grouped.all(1).mean()),
                   "prediction_consistency_fraction": float((prediction == prediction[:, :1]).all(1).mean()),
                   "sample_independence": "同ID多个位置有关联，非独立样本"})
    return result, logits


def new_model(cfg, device, architecture="fc", state=None):
    seed(cfg["model_seed"])
    model = (CircularCNN(cfg["dropout"]) if architecture == "cnn" else MaskedFC(cfg["fc_widths"], cfg["dropout"])).to(device)
    if state is not None:
        model.load_state_dict(state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    return model, optimizer


def train_epoch(model, optimizer, x, y, cfg, epoch, period, ledger):
    torch.manual_seed(cfg["model_seed"] + 2000 + epoch)
    stream = torch.Generator(device=x.device).manual_seed(cfg["data_seed"] + 3000 + epoch)
    order = torch.randperm(len(x), device=x.device, generator=stream)
    offsets = torch.randint(28, (len(x), 2), device=x.device, generator=stream)
    histogram = np.bincount((offsets[:, 0]*28+offsets[:, 1]).cpu().numpy(), minlength=784)
    ledger["training_position_histogram"] = (np.asarray(ledger.get("training_position_histogram", [0]*784)) + histogram).tolist()
    optimizer.param_groups[0]["lr"] = cfg["minimum_learning_rate"] + .5 * (cfg["learning_rate"] - cfg["minimum_learning_rate"]) * (1 + math.cos(math.pi * ((epoch - 1) % period) / period))
    model.train()
    loss_sum = 0.
    for ids in order.split(cfg["batch_size"]):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(shifts(x[ids], offsets[ids])), y[ids])
        if not torch.isfinite(loss):
            raise FloatingPointError("训练损失非有限")
        loss.backward()
        if hasattr(model, "masks"):
            with torch.no_grad():
                for layer, mask in zip(model.layers, model.masks):
                    layer.weight.grad.mul_(mask)
        optimizer.step()
        if hasattr(model, "enforce"):
            model.enforce(optimizer)
        ledger["attempted_updates"] += 1
        ledger["optimization_samples"] += len(ids)
        loss_sum += float(loss.detach()) * len(ids)
    optimizer.zero_grad(set_to_none=True)
    return {"hard_label_ce": loss_sum / len(x), "learning_rate": optimizer.param_groups[0]["lr"]}


def exported(model, path):
    if isinstance(model, MaskedFC):
        return {**export_model(model, path), **edge_counts(model)}
    torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()}}, path)
    return {"bytes": path.stat().st_size, "sha256": digest(path), "nonzero_edges": [28224, 225792, 200704, 1280],
            "nodes": [3136, 1568, 128, 10], "sharing": "卷积同通道位置共享核参数",
            "independent_parameters": sum(p.numel() for p in model.parameters()), "format": "torch_checkpoint"}


def baseline(name, cfg, tensors, device, folder, budget):
    arm = folder / name
    arm.mkdir(exist_ok=True)
    if (arm / "result.json").exists():
        return json.loads((arm / "result.json").read_text())
    model, optimizer = new_model(cfg, device, "cnn" if name == "cnn" else "fc")
    ledger, rows, best, first = empty_ledger(), [], None, 1
    if (arm / "resume.pt").exists():
        saved = torch.load(arm / "resume.pt", map_location="cpu", weights_only=False)
        restore(model, optimizer, [], saved["state"])
        ledger, rows, best = [saved["info"][k] for k in ("ledger", "rows", "best")]
        first = saved["info"]["epoch"] + 1
    x, y, sx, sy = tensors
    started = time.perf_counter()
    for epoch in range(first, cfg["baseline_epochs"] + 1):
        t = time.perf_counter()
        trained = train_epoch(model, optimizer, x, y, cfg, epoch, cfg["baseline_epochs"], ledger)
        sync(device)
        trained["training_seconds"] = time.perf_counter() - t
        t = time.perf_counter()
        selected, logits = measure(model, sx, sy, cfg, ledger)
        sync(device)
        row = {"epoch": epoch, "validation": selected, "evaluation_seconds": time.perf_counter() - t, **trained}
        rows.append(row)
        ledger["net_updates"] += math.ceil(len(x) / cfg["batch_size"])
        if best is None or (selected["accuracy"], -selected["cross_entropy"]) > (best["validation"]["accuracy"], -best["validation"]["cross_entropy"]):
            best = {"epoch": epoch, "validation": selected}
            checkpoint(arm / "best.pt", model, optimizer, {"epoch": epoch})
            np.savez_compressed(arm / "validation_best.npz", logits=logits, labels=sy.cpu().numpy())
        checkpoint(arm / "resume.pt", model, optimizer, {"epoch": epoch, "ledger": ledger, "rows": rows, "best": best})
        save_json(arm / "progress.json", {"rows": rows, "best": best, "ledger": ledger, "session_wall_seconds": budget.elapsed()})
        print(json.dumps({"arm": name, "epoch": epoch, "accuracy": selected["accuracy"], "best": best["validation"]["accuracy"]}), flush=True)
        budget.check(device)
        if epoch >= cfg["baseline_minimum_epochs"] and best["validation"]["accuracy"] >= cfg["baseline_select_target"]:
            break
    saved = torch.load(arm / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(saved["state"]["model"])
    result = {"best": best, "validation": best["validation"], "rows": rows, "ledger": ledger,
              "structure": exported(model, arm / ("model.pt" if name == "cnn" else "model.bgm")),
              "wall_seconds": time.perf_counter() - started,
              "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
              "gate": "pass" if best["validation"]["accuracy"] >= cfg["baseline_function_floor"] else "below_function_floor"}
    save_json(arm / "result.json", result)
    return result


def calibration(model, x, batch):
    """只读取训练输入，收集各线性层输入二阶矩和节点输入/输出扰动用证据。"""
    model.eval()
    squares = [torch.zeros(l.in_features, device=x.device) for l in model.layers]
    count = 0
    with torch.no_grad():
        for h in x.split(batch):
            for i, layer in enumerate(model.layers):
                squares[i] += h.square().sum(0)
                z = layer(h)
                h = F.relu(z) if i < len(model.layers) - 1 else z
            count += len(h)
    return [(s / count).sqrt() for s in squares]


def node_proposal(model, input_rms, fraction):
    row_counts, deleted = [], []
    with torch.no_grad():
        for layer, mask, rms in zip(model.layers, model.masks, input_rms):
            energy = layer.weight.square() * rms.square()[None, :]
            ranked, order = energy.masked_fill(~mask, float("inf")).sort(dim=1, stable=True)
            alive = mask.sum(1)
            cumulative = ranked.masked_fill(torch.isinf(ranked), 0).cumsum(1)
            total = energy.masked_fill(~mask, 0).sum(1)
            positions = torch.arange(mask.shape[1], device=mask.device)[None, :]
            eligible = (cumulative <= fraction * total[:, None]) & (positions < (alive - 1).clamp_min(0)[:, None])
            chosen = torch.zeros_like(mask).scatter_(1, order, eligible)
            # 评分全零时也保留一条现存最强权重边；不制造节点删除。
            strongest = layer.weight.abs().masked_fill(~mask, -1).argmax(1)
            chosen[torch.arange(len(mask), device=mask.device), strongest] = False
            selected = torch.nonzero(chosen.flatten()).flatten()
            mask.flatten()[selected] = False
            row_counts.append(chosen.sum(1).cpu().numpy().astype(np.int32))
            deleted.append(selected.cpu().numpy().astype(np.int32))
        model.enforce()
    return row_counts, deleted


def random_rows(model, counts, root):
    stream = torch.Generator(device=model.masks[0].device).manual_seed(root)
    deleted = []
    with torch.no_grad():
        for layer, mask, per_row in zip(model.layers, model.masks, counts):
            strongest = layer.weight.abs().masked_fill(~mask, -1).argmax(1)
            scores = torch.rand(mask.shape, device=mask.device, generator=stream).masked_fill(~mask, 2.)
            scores[torch.arange(len(mask), device=mask.device), strongest] = 2.
            order = scores.argsort(1)
            eligible = torch.arange(mask.shape[1], device=mask.device)[None, :] < torch.as_tensor(per_row, device=mask.device)[:, None]
            chosen = torch.zeros_like(mask).scatter_(1, order, eligible)
            assert not (chosen & ~mask).any()
            indices = torch.nonzero(chosen.flatten()).flatten()
            mask.flatten()[indices] = False
            deleted.append(indices.cpu().numpy().astype(np.int32))
        model.enforce()
    return deleted


def disturbance(model, before_state, x, batch):
    """固定剪前层输入，测联删造成的线性输入/激活相对RMS变化；不作接受阈值。"""
    sums = [{"input_energy": 0., "input_delta_energy": 0., "activation_energy": 0., "activation_delta_energy": 0.} for _ in model.layers]
    model.eval()
    with torch.no_grad():
        for h in x.split(batch):
            for i, layer in enumerate(model.layers):
                w = before_state[f"layers.{i}.weight"].to(x.device)
                b = before_state[f"layers.{i}.bias"].to(x.device)
                old, new = F.linear(h, w, b), layer(h)
                a, c = (F.relu(old), F.relu(new)) if i < len(model.layers) - 1 else (old, new)
                for key, value in (("input_energy", old), ("input_delta_energy", new-old), ("activation_energy", a), ("activation_delta_energy", c-a)):
                    sums[i][key] += float(value.square().sum())
                h = a
    return [{**s, "relative_input_rms": math.sqrt(s["input_delta_energy"] / max(s["input_energy"], 1e-30)),
             "relative_activation_rms": math.sqrt(s["activation_delta_energy"] / max(s["activation_energy"], 1e-30))} for s in sums]


def run_arm(name, cfg, tensors, calibration_x, device, folder, parent, reference, budget):
    arm = folder / name
    arm.mkdir(exist_ok=True)
    if (arm / "result.json").exists():
        return json.loads((arm / "result.json").read_text())
    model, optimizer = new_model(cfg, device, state=parent)
    ledger, rows, stages, events = empty_ledger(), [], [], []
    costs = {k: 0. for k in ("training_seconds", "validation_seconds", "proposal_seconds", "random_pair_seconds", "snapshot_seconds", "export_seconds")}
    controller = {"stage": 1, "step": 0, "mode": "new", "before": None, "event": None}
    initial = exported(model, arm / "stage_0.bgm")
    stages.append({"stage": 0, "validation": reference, "structure": initial, "accepted": True})
    if (arm / "resume.pt").exists():
        saved = torch.load(arm / "resume.pt", map_location="cpu", weights_only=False)
        restore(model, optimizer, [], saved["state"])
        ledger, rows, stages, events, costs, controller = [saved["info"][k] for k in ("ledger", "rows", "stages", "events", "costs", "controller")]
    x, y, sx, sy = tensors
    started = time.perf_counter()

    def save():
        save_started = time.perf_counter()
        checkpoint(arm / "resume.pt", model, optimizer, {"ledger": ledger, "rows": rows, "stages": stages, "events": events, "costs": costs, "controller": controller})
        costs["snapshot_seconds"] += time.perf_counter() - save_started
        save_json(arm / "progress.json", {"rows": rows, "stages": stages, "ledger": ledger, "costs": costs,
                   "controller": {k: v for k, v in controller.items() if k not in ("before", "event")}, "session_wall_seconds": budget.elapsed()})

    while controller["stage"] <= cfg["stages"]:
        stage = controller["stage"]
        if controller["mode"] == "new":
            controller["step"] = 0
            controller["event"] = None
            if name == "fixed":
                controller["mode"] = "ordinary"
            else:
                t = time.perf_counter()
                controller["before"] = capture(model, optimizer, [])
                costs["snapshot_seconds"] += time.perf_counter()-t
                t = time.perf_counter()
                before_masks = [m.clone() for m in model.masks]
                if name == "magnitude":
                    counts = propose(model, cfg["delete_fraction"])
                    deleted = [torch.nonzero((a & ~b).flatten()).flatten().cpu().numpy().astype(np.int32) for a, b in zip(before_masks, model.masks)]
                    row_counts = [(a & ~b).sum(1).cpu().numpy().astype(np.int32) for a, b in zip(before_masks, model.masks)]
                    perturb = None
                else:
                    rms = calibration(model, calibration_x, cfg["evaluation_batch_size"])
                    row_counts, deleted = node_proposal(model, rms, cfg["node_energy_fraction"])
                    counts = [int(c.sum()) for c in row_counts]
                    perturb = disturbance(model, controller["before"]["model"], calibration_x, cfg["evaluation_batch_size"])
                    np.savez_compressed(arm / f"calibration_stage_{stage}.npz", **{f"input_rms_{i}": a.cpu().numpy() for i, a in enumerate(rms)})
                model.enforce(optimizer)
                sync(device)
                costs["proposal_seconds"] += time.perf_counter() - t
                np.savez_compressed(arm / f"deletion_stage_{stage}.npz", **{f"deleted_{i}": a for i, a in enumerate(deleted)}, **{f"row_counts_{i}": a for i, a in enumerate(row_counts)})
                controller["event"] = {"stage": stage, "deleted_per_layer": counts, "before": stages[-1]["structure"], "disturbance": perturb}
                controller["mode"] = "candidate"
            save()
        if controller["mode"] in ("candidate", "ordinary", "fallback"):
            for step in range(controller["step"] + 1, cfg["adaptation_epochs"] + 1):
                epoch = (stage - 1) * cfg["adaptation_epochs"] + step
                t = time.perf_counter()
                trained = train_epoch(model, optimizer, x, y, cfg, epoch, cfg["adaptation_epochs"], ledger)
                sync(device)
                costs["training_seconds"] += time.perf_counter() - t
                rows.append({"stage": stage, "epoch": epoch, "mode": controller["mode"], **trained})
                controller["step"] = step
                save()
                print(json.dumps({"arm": name, "stage": stage, "epoch": epoch, "mode": controller["mode"]}), flush=True)
                budget.check(device)
            t = time.perf_counter()
            selected, logits = measure(model, sx, sy, cfg, ledger)
            sync(device)
            costs["validation_seconds"] += time.perf_counter() - t
            t = time.perf_counter()
            structure = exported(model, arm / f"stage_{stage}_{controller['mode']}.bgm")
            costs["export_seconds"] += time.perf_counter()-t
            np.savez_compressed(arm / f"validation_stage_{stage}_{controller['mode']}.npz", logits=logits, labels=sy.cpu().numpy())
            if controller["mode"] == "candidate":
                accepted = function_ok(selected, reference, cfg) and structure["bytes"] < controller["event"]["before"]["bytes"]
                controller["event"].update({"accepted": accepted, "candidate_validation": selected, "candidate_structure": structure})
                if name == "node":
                    # 独立候选比较：每次随机对照从本槽同一剪前图/Adam/RNG开始。
                    candidate_saved = capture(model, optimizer, [])
                    save()
                    t = time.perf_counter()
                    restore(model, optimizer, [], controller["before"])
                    with np.load(arm / f"deletion_stage_{stage}.npz", allow_pickle=False) as archive:
                        per_row = [archive[f"row_counts_{i}"] for i in range(len(model.layers))]
                    random_deleted = random_rows(model, per_row, cfg["model_seed"] + 9000 + stage)
                    model.enforce(optimizer)
                    random_ledger, first_random_step, random_seconds = empty_ledger(), 1, 0.
                    branch = arm / f"random_pair_stage_{stage}"
                    branch.mkdir(exist_ok=True)
                    if (branch / "resume.pt").exists():
                        random_saved = torch.load(branch/"resume.pt", map_location="cpu", weights_only=False)
                        restore(model, optimizer, [], random_saved["state"])
                        random_ledger = random_saved["info"]["ledger"]
                        first_random_step = random_saved["info"]["step"]+1
                        random_seconds = random_saved["info"]["seconds"]
                    for step in range(first_random_step, cfg["adaptation_epochs"] + 1):
                        epoch_started = time.perf_counter()
                        train_epoch(model, optimizer, x, y, cfg, (stage-1) * cfg["adaptation_epochs"] + step, cfg["adaptation_epochs"], random_ledger)
                        sync(device)
                        random_seconds += time.perf_counter()-epoch_started
                        checkpoint(branch/"resume.pt", model, optimizer, {"ledger": random_ledger, "step": step, "seconds": random_seconds})
                        save_json(branch/"progress.json", {"ledger": random_ledger, "step": step, "seconds": random_seconds, "session_wall_seconds": budget.elapsed()})
                        budget.check(device)
                    if (branch/"result.json").exists():
                        pair_result = json.loads((branch/"result.json").read_text())
                    else:
                        random_validation, random_logits = measure(model, sx, sy, cfg, random_ledger)
                        random_structure = exported(model, arm / f"random_pair_stage_{stage}.bgm")
                        np.savez_compressed(arm / f"random_pair_stage_{stage}.npz", logits=random_logits, labels=sy.cpu().numpy(), **{f"deleted_{i}": a for i, a in enumerate(random_deleted)})
                        pair_result = {"validation": random_validation, "structure": random_structure, "ledger": random_ledger,
                            "same_parent": True, "forced_retained_timing": accepted, "function_ok": function_ok(random_validation, reference, cfg), "training_seconds": random_seconds}
                        save_json(branch/"result.json", pair_result)
                    controller["event"]["random_pair"] = pair_result
                    restore(model, optimizer, [], candidate_saved)
                    costs["random_pair_seconds"] += time.perf_counter() - t
                if not accepted:
                    events.append(copy.deepcopy(controller["event"]))
                    restore(model, optimizer, [], controller["before"])
                    ledger["rollbacks"] += 1
                    controller["mode"], controller["step"] = "fallback", 0
                    save()
                    continue
            else:
                accepted = name == "fixed"
            if controller["mode"] == "candidate":
                events.append(copy.deepcopy(controller["event"]))
            t = time.perf_counter()
            exported(model, arm / f"stage_{stage}.bgm")
            costs["export_seconds"] += time.perf_counter()-t
            stages.append({"stage": stage, "validation": selected, "structure": structure, "accepted": accepted, "event": controller["event"]})
            ledger["net_updates"] += cfg["adaptation_epochs"] * math.ceil(len(x)/cfg["batch_size"])
            checkpoint(arm / f"stage_{stage}.pt", model, optimizer, {"stage": stage})
            print(json.dumps({"arm": name, "stage_complete": stage, "accuracy": selected["accuracy"], "edges": sum(structure["nonzero_edges"]), "accepted": accepted}), flush=True)
            controller = {"stage": stage + 1, "step": 0, "mode": "new", "before": None, "event": None}
            save()
            budget.check(device)
    qualified = [r for r in stages if function_ok(r["validation"], reference, cfg)]
    chosen = min(qualified, key=lambda r: (sum(r["structure"]["nonzero_edges"]), -r["validation"]["accuracy"], r["validation"]["cross_entropy"]))
    model = load_export(arm / f"stage_{chosen['stage']}.bgm").to(device)
    result = {"arm": name, "stages": stages, "rows": rows, "events": events, "chosen_stage": chosen["stage"], "validation": chosen["validation"],
              "structure": exported(model, arm / "model.bgm"), "ledger": ledger, "costs": costs, "wall_seconds": time.perf_counter()-started,
              "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None}
    save_json(arm / "result.json", result)
    return result


def check():
    torch.set_num_threads(2)
    seed(11)
    x = torch.arange(784).float()[None, :].repeat(784, 1)
    offsets = torch.tensor([(a, b) for a in range(28) for b in range(28)])
    shifted = shifts(x, offsets)
    assert torch.equal(shifted.sort(1).values, x)
    assert torch.equal(shifted[:, 0].sort().values, x[0])
    positions = fixed_offsets(800, 8, 20261004)
    codes = positions[..., 0]*28 + positions[..., 1]
    freq = np.bincount(codes.flatten(), minlength=784)
    assert int(freq.max()-freq.min()) == 1
    assert all(len(set(row)) == 8 for row in codes)
    cfg = {"model_seed": 11, "fc_widths": [8, 7, 5, 3, 10], "dropout": 0., "learning_rate": .001, "weight_decay": .0001}
    model, optimizer = new_model(cfg, torch.device("cpu"))
    inputs = torch.randn(16, 8)
    F.cross_entropy(model(inputs), torch.arange(16)%10).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    before = capture(model, optimizer, [])
    rms = calibration(model, inputs, 8)
    counts, deleted = node_proposal(model, rms, .05)
    changes = disturbance(model, before["model"], inputs, 8)
    assert sum(map(len, deleted)) > 0 and all(math.isfinite(a["relative_input_rms"]) for a in changes)
    for i, (l, mask, c) in enumerate(zip(model.layers, model.masks, counts)):
        old_mask = before["model"][f"mask_{i}"]
        e = before["model"][f"layers.{i}.weight"].square()*rms[i].square()[None, :]
        assert ((e*(old_mask & ~mask)).sum(1) <= .05*(e*old_mask).sum(1)+1e-8).all()
        assert mask.any(1).all()
    model.enforce(optimizer)
    assert all(not torch.count_nonzero(l.weight[~m]) for l, m in zip(model.layers, model.masks))
    restore(model, optimizer, [], before)
    rand = random_rows(model, counts, 11)
    assert [len(a) for a in rand] == [len(a) for a in deleted]
    for i, mask in enumerate(model.masks):
        assert np.array_equal((before["model"][f"mask_{i}"] & ~mask).sum(1).numpy(), counts[i])
    restore(model, optimizer, [], before)
    again = capture(model, optimizer, [])
    assert all(torch.equal(again["model"][k], v) for k, v in before["model"].items())
    assert torch.equal(again["torch_rng"], before["torch_rng"])
    print(json.dumps({"ok": True, "device": "CPU", "pixel_preservation": True, "balanced_validation_positions": True,
                      "node_energy_bound": True, "same_parent_random_row_counts": True, "new_gpu_check": False}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.check:
        check()
        return
    cfg_path = ROOT / "configs/h1_translated_edges_v1.json"
    cfg = json.loads(cfg_path.read_text())
    assert cfg["model_seed"] == 11 and cfg["data_seed"] == 20261004
    assert cfg["fc_widths"] == [784,3136,1568,128,10]
    assert cfg["cnn_expanded_edges"] == 3136*9+1568*16*9+1568*128+128*10
    folder = ROOT / "experiments/runs" / cfg["version"] / "mnist_circular"
    folder.mkdir(parents=True, exist_ok=True)
    if any(folder.iterdir()) and not args.resume:
        raise FileExistsError("已有证据，仅--resume接续，拒绝覆盖")
    if (folder / "summary.json").exists():
        print((folder / "summary.json").read_text(), flush=True)
        return
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reused = (ROOT / "tools/train_capacity_baseline.py", ROOT / "tools/train_capacity_pruning.py", ROOT / "tools/train_aligned_edges.py", ROOT / "src/braingrow/datasets.py")
    registration = {"config": cfg, "config_sha256": digest(cfg_path), "source_sha256": digest(Path(__file__)),
                    "reused_source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in reused},
                    "training_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    "device": str(device), "device_name": torch.cuda.get_device_name() if device.type == "cuda" else "CPU", "torch": torch.__version__,
                    "precision": "FP32", "execution": "dense_linear_zero_mask_or_dense_CNN", "cpu_threads": 2,
                    "official_test_accessed": False, "sealed_remaining_ids_accessed": False}
    prior = 0.
    if (folder / "registration.json").exists():
        old = json.loads((folder / "registration.json").read_text())
        if any(old[k] != registration[k] for k in ("config_sha256", "source_sha256", "reused_source_sha256")):
            raise ValueError("源/配置哈希变化，拒绝恢复")
        registration = old
        elapsed_paths = list(folder.rglob("progress.json"))
        if (folder/"failure.json").exists():
            elapsed_paths.append(folder/"failure.json")
        prior = max([json.loads(p.read_text()).get("session_wall_seconds", 0.) for p in elapsed_paths], default=0.)
    else:
        save_json(folder / "registration.json", registration)
    budget = Budget(cfg["wall_seconds_total"], prior)
    results = {}
    try:
        budget.check(device)
        data = load_bundle("mnist", include_confirm=False)
        assert data.fit_x.shape == (3200,784) and data.select_x.shape == (800,784) and not len(data.confirm_x)
        x, y = torch.from_numpy(data.fit_x).to(device), torch.from_numpy(data.fit_y).to(device)
        sx, sy, offsets = evaluation_data(torch.from_numpy(data.select_x).to(device), torch.from_numpy(data.select_y).to(device), cfg["validation_positions_per_id"], cfg["data_seed"]+7000)
        calibration_offsets = fixed_offsets(len(x), 1, cfg["data_seed"]+8000)
        calibration_x = shifts(x, torch.from_numpy(calibration_offsets[:,0]))
        np.savez_compressed(folder / "position_manifest.npz", validation_offsets=offsets, calibration_offsets=calibration_offsets,
                            train_original_ids=np.asarray(data.manifest["splits"]["fit"]), validation_original_ids=np.asarray(data.manifest["splits"]["select"]))
        save_json(folder / "data_manifest.json", {"original": data.manifest, "position_sha256": digest(folder / "position_manifest.npz"), "validation_samples": len(sx),
                   "positions_rule": "固定排列784位移循环拼接，ID行随机排列；每ID8位置互异，全体频数8或9", "calibration_training_only": True})
        tensors = (x,y,sx,sy)
        for name in ("baseline", "cnn"):
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            results[name] = baseline(name,cfg,tensors,device,folder,budget)
        if any(results[name]["gate"] != "pass" for name in ("baseline", "cnn")):
            save_json(folder / "summary.json", {"registration": registration, "results": results, "gate": "baseline_below_function_floor", "session_wall_seconds": budget.elapsed(), "historical_test_accessed": False})
            print(json.dumps({"gate":"baseline_below_function_floor", "accuracy": {name: results[name]["validation"]["accuracy"] for name in ("baseline", "cnn")}}),flush=True)
            return
        parent_file = folder / "baseline/best.pt"
        parent = torch.load(parent_file,map_location="cpu",weights_only=False)["state"]["model"]
        for name in ("fixed","magnitude","node"):
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            results[name] = run_arm(name,cfg,tensors,calibration_x,device,folder,parent,results["baseline"]["validation"],budget)
        for name,result in results.items():
            if "training_fixed_one_position" in result:
                continue
            if name == "cnn":
                model,_ = new_model(cfg,device,"cnn")
                model.load_state_dict(torch.load(folder/name/"model.pt",map_location="cpu",weights_only=False)["model"])
            else:
                model = load_export(folder/name/"model.bgm").to(device)
            train_metrics,train_logits = evaluate(model,calibration_x,y,cfg["evaluation_batch_size"])
            result["training_fixed_one_position"] = {**train_metrics,"positions_per_id":1,"note":"训练ID固定1位移，非全部训练位移表现"}
            result["ledger"]["evaluation_calls"] += 1
            result["ledger"]["evaluation_samples"] += len(y)
            np.savez_compressed(folder/name/"training_fixed_one_position.npz",logits=train_logits,labels=y.cpu().numpy(),original_ids=np.asarray(data.manifest["splits"]["fit"]))
            save_json(folder/name/"result.json",result)
            del model
        confirmed = load_bundle("mnist",include_confirm=True)
        cx,cy,test_offsets = evaluation_data(torch.from_numpy(confirmed.confirm_x).to(device),torch.from_numpy(confirmed.confirm_y).to(device),cfg["test_positions_per_id"],cfg["data_seed"]+10000)
        np.savez_compressed(folder / "test_positions.npz",offsets=test_offsets,original_ids=np.asarray(confirmed.manifest["splits"]["confirm"]))
        for name,result in results.items():
            if "exploratory_test" in result:
                continue
            if name == "cnn":
                model,_ = new_model(cfg,device,"cnn")
                model.load_state_dict(torch.load(folder / name / "model.pt",map_location="cpu",weights_only=False)["model"])
            else:
                model = load_export(folder / name / "model.bgm").to(device)
            test,logits = measure(model,cx,cy,cfg,result["ledger"])
            result["exploratory_test"] = test
            np.savez_compressed(folder/name/"exploratory_test.npz",logits=logits,labels=cy.cpu().numpy())
            save_json(folder/name/"result.json",result)
            del model
            budget.check(device)
        selected_node = results["node"]["chosen_stage"]
        if selected_node:
            branch_path = folder / "node" / f"random_pair_stage_{selected_node}.bgm"
            event = next(e for e in results["node"]["events"] if e["stage"] == selected_node)
            random_result = copy.deepcopy(event["random_pair"])
        else:
            branch_path = folder / "node/stage_0.bgm"
            random_result = {"validation": results["baseline"]["validation"], "structure": results["baseline"]["structure"], "ledger": empty_ledger(), "same_parent": True}
        if (folder/"random_pair_result.json").exists():
            random_result = json.loads((folder/"random_pair_result.json").read_text())
        else:
            model = load_export(branch_path).to(device)
            train_metrics,train_logits = evaluate(model,calibration_x,y,cfg["evaluation_batch_size"])
            random_result["training_fixed_one_position"] = {**train_metrics,"positions_per_id":1}
            random_result["exploratory_test"], logits = measure(model,cx,cy,cfg,random_result["ledger"])
            random_result["selected_stage"] = selected_node
            random_result["selection_rule"] = "完全跟随节点主臂验证冻结阶段，不据随机成绩选阶段；非连续随机臂"
            np.savez_compressed(folder / "node/random_pair_exploratory_test.npz",logits=logits,labels=cy.cpu().numpy(),original_ids=np.asarray(confirmed.manifest["splits"]["confirm"]))
            save_json(folder/"random_pair_result.json",random_result)
        results["random_pair"] = random_result
        budget.check(device)
        save_json(folder/"summary.json",{"registration":registration,"results":results,"parent_sha256":digest(parent_file),"gate":"completed","session_wall_seconds":budget.elapsed(),
                  "official_test_accessed":False,"sealed_remaining_ids_accessed":False})
        print(json.dumps({"gate":"completed","wall_seconds":budget.elapsed(),"results":{n:{"validation":r["validation"]["accuracy"],"edges":sum(r["structure"]["nonzero_edges"])} for n,r in results.items()}}),flush=True)
    except Exception as exc:
        save_json(folder/"failure.json",{"error":repr(exc),"session_wall_seconds":budget.elapsed(),"completed_arms":list(results),"resume":"python -m tools.train_translated_edges --resume"})
        raise


if __name__ == "__main__":
    main()
