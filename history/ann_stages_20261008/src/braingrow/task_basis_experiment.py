"""共享冻结FC快照、两层独立群组分叉；固定控制只实际训练一次。"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from .codec import decode_model
from .datasets import load_bundle
from .node_reduction import apply_node_reduction
from .node_experiment import (checkpoint_state, fit_partition, fresh_adam, new_ledger,
    persist_session_cost, reconcile_cached_evaluation, restore_model)
from .pruning_experiment import (encoded, evaluate, metrics, pid_alive, restore_rng,
    rng_state, save_checkpoint, sha, structure_stats, synchronize, train_batch, within, write_json)
from .task_basis import build_library, extract_snapshot, reduction_proposal, score_record, select_group

ROOT = Path(__file__).resolve().parents[2]
BRANCHES = ["fixed", "layer0_residual", "layer0_task", "layer0_random",
            "layer1_residual", "layer1_task", "layer1_random"]
SOURCES = ("tools/train_task_basis.py", "src/braingrow/task_basis_experiment.py", "src/braingrow/task_basis.py",
    "src/braingrow/node_reduction.py", "src/braingrow/node_experiment.py", "src/braingrow/pruning_experiment.py",
    "src/braingrow/models.py", "src/braingrow/codec.py", "src/braingrow/structural.py", "src/braingrow/datasets.py")


def atomic_npz(path, **values):
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **values)
    os.replace(temporary, path)


def digest_tensor(value):
    return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()


def prepare(folder, parent, fx, fy, partition, config, device):
    """快照提取缓存与账本共同原子落盘；提议只从这个共享缓存产生。"""
    path = folder/"preparation.pt"
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=False)
    model = restore_model(parent, device, config["widths"])
    restore_rng(parent["rng"])
    synchronize(device)
    began = time.perf_counter()
    snapshots = extract_snapshot(model, fx, fy, config["proposal_batch_size"],
        config["maximum_extra_cpu_array_bytes"], config["maximum_cuda_allocated_bytes"])
    synchronize(device)
    extraction_seconds = time.perf_counter()-began
    led = dict(proposal_forward_samples=len(fx), proposal_backward_samples=len(fx),
        proposal_derivative_backward_calls=int(np.ceil(len(fx)/config["proposal_batch_size"])),
        extraction_seconds=extraction_seconds, cpu_search_seconds=0., encoding_seconds=0.,
        gradients_source="母快照逐样本拟合CE对下一层Z的sum-loss导数；包括固定640内部留出，不更新参数")
    # 提取缓存先原子保存，搜索中断后不会再做提取前向/反传。
    extraction_path = folder/"snapshot_cache.pt"
    save_checkpoint(extraction_path, dict(records=snapshots, ledger=led))
    return finish_preparation(folder, parent, partition, config, snapshots, led)


def finish_preparation(folder, parent, partition, config, snapshots, ledger):
    started = time.perf_counter()
    proposals, metadata, libraries = {}, {}, {}
    for layer in config["layers"]:
        record = snapshots[layer]
        library = build_library(record, layer, config["widths"], partition["fit_indices"],
            partition["holdout_indices"], config["basis_budget"][layer], config["minimum_exact_c32_support"])
        arrays_bytes = sum(value.numel()*value.element_size() for item in snapshots for value in item.values())
        if arrays_bytes*2 > config["maximum_extra_cpu_array_bytes"]:
            raise MemoryError("共享库与群组工作空间CPU数组预算不足")
        library_hash = digest_tensor(library.coefficients)
        pool_hash = hashlib.sha256(np.asarray(library.pool, dtype=np.int64).tobytes()).hexdigest()
        libraries[layer] = dict(basis=library.basis, pool=library.pool, coefficients=library.coefficients,
            support_counts=library.support_counts, relative_residuals=library.relative_residuals,
            sigma=library.sigma, library_sha256=library_hash, pool_sha256=pool_hash)
        snapshot_hashes = {key: digest_tensor(value) for key, value in record.items() if key != "errors"}
        for method in ("residual", "task", "random"):
            branch = f"layer{layer}_{method}"
            dropped, steps = select_group(record, library, method, config["remove_count"], config["j_limit"], config["q_limit"])
            info = dict(layer_index=layer, method=method, proposal_possible=dropped is not None,
                pool_count=len(library.pool), basis_indices=list(library.basis), pool_sha256=pool_hash,
                library_sha256=library_hash, snapshot_sha256=snapshot_hashes, greedy_steps=steps,
                source_widths=config["widths"], chosen_before_adaptation=True,
                random_is_acceptance_matched=False, fixed_reference="../fixed",
                failure_rule="池不足8：诊断未成立；保留原图并完成相同250更新，不补静默节点/降数/调阈值")
            if dropped is None:
                proposals[branch] = None
                info.update(dropped_indices=[], reason="shared_pool_less_than_8", removed_nodes=0)
            else:
                proposals[branch] = reduction_proposal(library, dropped)
                info.update(dropped_indices=list(dropped), removed_nodes=len(dropped),
                    diagnostics=score_record(record, library, dropped, config["j_limit"], config["q_limit"]))
            metadata[branch] = info
    ledger["cpu_search_seconds"] += time.perf_counter()-started
    result = dict(proposals=proposals, metadata=metadata, libraries=libraries, ledger=ledger,
                  parent_rng=parent["rng"], fit_partition=partition)
    save_checkpoint(folder/"preparation.pt", result)
    export_preparation(folder, result)
    return result


def export_preparation(folder, preparation):
    """从完整提议缓存幂等补齐可审阅文件，文件中断不重新模型提取。"""
    write_json(folder/"proposal_metadata.json", dict(branches=preparation["metadata"], ledger=preparation["ledger"],
                                                   fit_partition=preparation["fit_partition"]))
    for layer, library in preparation["libraries"].items():
        atomic_npz(folder/f"layer{layer}_library.npz", **{key: np.asarray(value) for key, value in library.items()
            if key not in {"library_sha256", "pool_sha256"}})


def checkpoint(model, optimizer, rows, ledger, next_epoch, wall, branch):
    value = checkpoint_state(model, optimizer, rows, [], ledger, next_epoch, wall, branch)
    value.pop("node_experiment_version")
    value["task_basis_version"] = 1
    return value


def cached_evaluation(path, model, x, y, labels, batch_size, ledger, purpose):
    if path.exists():
        with np.load(path, allow_pickle=False) as cache:
            result = metrics(cache["logits"], cache["labels"])
        reconcile_cached_evaluation(ledger, purpose, result["samples"])
        return result
    result, logits = evaluate(model, x, y, batch_size, ledger, purpose)
    atomic_npz(path, logits=logits, labels=labels)
    return result


def timed_encoded(model, ledger):
    began = time.perf_counter()
    payload = encoded(model)
    ledger["encoding_seconds"] = ledger.get("encoding_seconds", 0.)+time.perf_counter()-began
    return payload


def run(args):
    began = time.perf_counter()
    config = json.loads((ROOT/args.config).read_text(encoding="utf-8"))
    if (config["model_seed"] != 11 or config["data_seed"] != 20261004 or config["actual_branches"] != BRANCHES
            or config["source_arm"] != "fc" or config["source_epoch"] != 25 or config["epochs"] != 5):
        raise ValueError("只允许登记的根、FC25母快照及7个实际分支")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    folder = ROOT/"experiments/runs"/config["version"]/"mnist_static"
    folder.mkdir(parents=True, exist_ok=True)
    lock, cost_path = folder/"running.lock", folder/"session_cost.json"
    if lock.exists():
        if pid_alive(int(lock.read_text())):
            raise RuntimeError("分叉实验运行锁活动，拒绝重复启动")
        lock.unlink()
    fd = os.open(lock, os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    past = json.loads(cost_path.read_text(encoding="utf-8")) if cost_path.exists() else dict(cumulative_wall_seconds=0., sessions=0)
    try:
        source = ROOT/config["source_run"]
        parent_reg = json.loads((source/"registration.json").read_text(encoding="utf-8"))
        for path, expected in parent_reg["source_sha256"].items():
            if sha(ROOT/path) != expected:
                raise ValueError(f"父登记旧源已变化：{path}")
        startup_path = source/"startup.json"
        startup = json.loads(startup_path.read_text(encoding="utf-8"))
        if not startup["ok"] or startup["widths"] != config["widths"]:
            raise ValueError("既有同尺寸GPU启动证据不符")
        bundle = load_bundle("mnist", include_confirm=False)
        if bundle.fit_x.shape != (3200, 784) or bundle.select_x.shape != (800, 784) or len(bundle.confirm_x):
            raise ValueError("冻结前只加载授权3200/800，不加载旧control")
        partition = fit_partition(bundle.fit_y)
        registration = dict(config=config, task="mnist_static", device=str(device), torch=torch.__version__,
            source_sha256={path: sha(ROOT/path) for path in SOURCES},
            parent_checkpoint_sha256=sha(source/"fc/resume.pt"), parent_registration_sha256=sha(source/"registration.json"),
            parent_config_sha256=sha(ROOT/"configs/pruning_v1.json"),
            data_manifest_sha256=sha(ROOT/"data/structure_mnist_v1/manifest.json"), fit_partition=partition,
            logical_branches=8, actual_branches=7, fixed_reuse="同一个fixed末态哈希供两个独立层诊断引用，实际成本只计一次")
        rp = folder/"registration.json"
        if rp.exists():
            if json.loads(rp.read_text(encoding="utf-8")) != registration or not args.resume:
                raise ValueError("已有运行仅允许相同源/配置/母快照--resume")
        else:
            write_json(rp, registration)
        if (folder/"summary.json").exists():
            print(json.dumps(dict(already_complete=True, run=str(folder))), flush=True)
            return
        if past["cumulative_wall_seconds"] >= config["maximum_wall_seconds"]:
            raise RuntimeError("累计阶段预算用完，resume不重获预算")
        if not (folder/"training_commit.json").exists():
            raise ValueError("训练前必须保存独立实现真实提交metadata")
        parent = torch.load(source/"fc/resume.pt", map_location=device, weights_only=False)
        # 旧pruning_v1 checkpoint没有arm字段；FC身份由已核验父登记及固定fc路径确定。
        if parent["next_epoch"] != 26 or parent.get("arm", config["source_arm"]) != "fc":
            raise ValueError("母checkpoint不是未剪枝FC25末态")
        write_json(folder/"data_manifest.json", bundle.manifest)
        fx, fy, sx, sy = [torch.from_numpy(value).to(device) for value in
                         (bundle.fit_x, bundle.fit_y, bundle.select_x, bundle.select_y)]
        prep_path, extraction_path = folder/"preparation.pt", folder/"snapshot_cache.pt"
        if prep_path.exists():
            preparation = torch.load(prep_path, map_location="cpu", weights_only=False)
        elif extraction_path.exists():
            saved = torch.load(extraction_path, map_location="cpu", weights_only=False)
            preparation = finish_preparation(folder, parent, partition, config, saved["records"], saved["ledger"])
        else:
            preparation = prepare(folder, parent, fx, fy, partition, config, device)
        export_preparation(folder, preparation)
        persist_session_cost(cost_path, past, began, config["maximum_wall_seconds"])
        write_json(folder/"startup.json", dict(reused_from="experiments/runs/pruning_v1/mnist_static/startup.json",
            forward_samples=0, backward_samples=0, updates=0, additional_gpu_checks=0,
            new_cpu_check="仅test_task_basis一次真实CPU机制检查"))
        results = {}
        for branch in BRANCHES:
            if (folder/"request_stop").exists() or past["cumulative_wall_seconds"]+time.perf_counter()-began >= config["maximum_wall_seconds"]:
                write_json(folder/"budget_stop.json", dict(branch=branch, reason="request_or_stage_budget"))
                return
            af = folder/branch
            af.mkdir(exist_ok=True)
            cp = af/"resume.pt"
            if cp.exists():
                saved = torch.load(cp, map_location=device, weights_only=False)
                model = restore_model(saved, device, config["widths"])
                optimizer = fresh_adam(model, config["learning_rate"], config)
                optimizer.load_state_dict(copy.deepcopy(saved["optimizer"]))
                restore_rng(saved["rng"])
                rows, ledger, wall, next_epoch = saved["rows"], saved["ledger"], saved["wall_seconds"], saved["next_epoch"]
            else:
                model = restore_model(parent, device, config["widths"])
                if tuple(model.widths) != tuple(config["widths"]):
                    raise ValueError("来源FC动态宽度不符")
                proposal = preparation["proposals"].get(branch)
                if proposal is not None:
                    model = apply_node_reduction(model, proposal)
                model.zero_grad(set_to_none=True)
                for layer in model.layers:
                    layer.last_edge_grad = None
                restore_rng(parent["rng"])
                optimizer = fresh_adam(model, config["learning_rate"], config)
                rows, ledger, wall, next_epoch = [], new_ledger(), 0., 1
                ledger["adam_reset_epochs"] = [0]
                initial = timed_encoded(model, ledger)
                decoded = decode_model(initial)
                if tuple(decoded.widths) != tuple(model.widths):
                    raise RuntimeError("物理缩宽编码/解码维度不符")
                del decoded
                (af/"initial_model.bgs").write_bytes(initial)
                info = dict(branch=branch, initial_structure=structure_stats(model, initial),
                    mother_checkpoint_sha256=registration["parent_checkpoint_sha256"],
                    fresh_adam_all_branches=True, proposal=preparation["metadata"].get(branch))
                write_json(af/"initialization.json", info)
                saved = checkpoint(model, optimizer, rows, ledger, 1, wall, branch)
                save_checkpoint(cp, saved)
            for epoch in range(next_epoch, config["epochs"]+1):
                synchronize(device)
                started = time.perf_counter()
                global_epoch = config["order_global_epoch_start"]+epoch-1
                order = np.random.default_rng(np.random.SeedSequence([20261004, global_epoch])).permutation(len(fx))
                losses = []
                for start in range(0, len(order), config["batch_size"]):
                    ids = torch.as_tensor(order[start:start+config["batch_size"]], device=device)
                    losses.append(train_batch(model, optimizer, fx[ids], fy[ids], ledger))
                    if device.type == "cuda" and torch.cuda.memory_allocated(device) > config["maximum_cuda_allocated_bytes"]:
                        raise MemoryError("CUDA allocated工程上限用完")
                # 当前轮若中断后从上一完整checkpoint重训，旧孤立cache不能替代新末态。
                selected, logits = evaluate(model, sx, sy, config["evaluation_batch_size"], ledger, f"epoch_select_{epoch}")
                atomic_npz(af/f"select_{epoch:03d}.npz", logits=logits, labels=bundle.select_y)
                payload = timed_encoded(model, ledger)
                synchronize(device)
                elapsed = time.perf_counter()-started
                wall += elapsed
                if ledger["attempted_updates"] != epoch*50:
                    raise RuntimeError("分支未保持每轮50净更新")
                rows.append(dict(epoch=epoch, global_epoch=global_epoch, learning_rate=config["learning_rate"],
                    select=selected, mean_classification_loss=float(np.mean(losses)),
                    widths=list(model.widths), structure=structure_stats(model, payload),
                    attempted_updates=ledger["attempted_updates"], net_updates=ledger["attempted_updates"], wall_seconds=elapsed))
                saved = checkpoint(model, optimizer, rows, ledger, epoch+1, wall, branch)
                save_checkpoint(cp, saved)
                persist_session_cost(cost_path, past, began, config["maximum_wall_seconds"])
                write_json(af/"progress.json", dict(rows=rows, ledger=ledger))
                print(json.dumps(dict(branch=branch, epoch=epoch, accuracy=selected["accuracy"],
                    loss=selected["loss"], widths=list(model.widths), export_bytes=len(payload))), flush=True)
                if (folder/"request_stop").exists() or past["cumulative_wall_seconds"]+time.perf_counter()-began >= config["maximum_wall_seconds"]:
                    write_json(folder/"budget_stop.json", dict(branch=branch, epoch=epoch, reason="request_or_stage_budget"))
                    return
            payload = timed_encoded(model, ledger)
            (af/"final_model.bgs").write_bytes(payload)
            fitted = cached_evaluation(af/"fit_final.npz", model, fx, fy, bundle.fit_y,
                config["evaluation_batch_size"], ledger, "frozen_fit")
            saved.update(ledger=ledger, rng=rng_state())
            save_checkpoint(cp, saved)
            results[branch] = dict(branch=branch, frozen=True, final_epoch=config["epochs"], rows=rows, fit=fitted,
                widths=list(model.widths), structure=structure_stats(model, payload), ledger=ledger,
                wall_seconds=wall, final_model_sha256=sha(af/"final_model.bgs"), proposal=preparation["metadata"].get(branch))
            write_json(af/"result.json", results[branch])
            del model, optimizer
        write_json(folder/"freeze.json", dict(branches={branch: result["final_model_sha256"] for branch, result in results.items()},
            all_actual_branches_frozen=True, fixed_reference_reused_by_layers=[0, 1]))
        confirmed = load_bundle("mnist", include_confirm=True)
        cx, cy = torch.from_numpy(confirmed.confirm_x).to(device), torch.from_numpy(confirmed.confirm_y).to(device)
        for branch, result in results.items():
            af = folder/branch
            saved = torch.load(af/"resume.pt", map_location=device, weights_only=False)
            result["ledger"] = saved["ledger"]
            model = restore_model(saved, device, config["widths"])
            result["confirmation"] = cached_evaluation(af/"confirm_final.npz", model, cx, cy, confirmed.confirm_y,
                config["evaluation_batch_size"], result["ledger"], "repeated_exploration")
            saved["ledger"] = result["ledger"]
            save_checkpoint(af/"resume.pt", saved)
            write_json(af/"result.json", result)
            del model
        fixed = results["fixed"]
        diagnostics = {}
        for layer in config["layers"]:
            arms = {}
            for method in ("residual", "task", "random"):
                result = results[f"layer{layer}_{method}"]
                possible = result["proposal"]["proposal_possible"]
                guard = possible and result["proposal"]["diagnostics"]["protection_passed"]
                task_ok = within(result["rows"][-1]["select"], fixed["rows"][-1]["select"], config)
                smaller = result["structure"]["export_bytes"] < fixed["structure"]["export_bytes"]
                width_ok = result["widths"][layer+1] == config["widths"][layer+1]-config["remove_count"]
                arms[method] = dict(proposal_possible=possible, functional_guard=guard, task_preserved=task_ok,
                    physical_width_reduced=width_ok, real_storage_reduced=smaller,
                    accepted=bool(possible and guard and task_ok and smaller and width_ok
                        and fixed["rows"][-1]["select"]["accuracy"] >= config["learning_floor"]),
                    random_acceptance_is_diagnostic_only=method == "random",
                    select_accuracy_delta_vs_fixed=result["rows"][-1]["select"]["accuracy"]-fixed["rows"][-1]["select"]["accuracy"])
            diagnostics[str(layer)] = dict(fixed_branch="fixed", fixed_model_sha256=fixed["final_model_sha256"], arms=arms,
                mechanism_diagnostic_established=all(value["proposal_possible"] for value in arms.values()))
        cost = {key: sum(result["ledger"][key] for result in results.values()) for key in
            ("optimization_forward_samples", "optimization_backward_samples", "attempted_updates", "evaluation_forward_samples")}
        cost.update(preparation["ledger"])
        cost["encoding_seconds"] += sum(result["ledger"].get("encoding_seconds", 0.) for result in results.values())
        cost["net_updates"] = cost["attempted_updates"]
        cost["total_model_forward_samples"] = cost["optimization_forward_samples"]+cost["proposal_forward_samples"]+cost["evaluation_forward_samples"]
        cost["total_model_backward_samples"] = cost["optimization_backward_samples"]+cost["proposal_backward_samples"]
        summary = dict(task="mnist_static", registration=registration, branches=results, diagnostics=diagnostics,
            logical_branches=8, actual_branches=7, cost=cost, fixed_reused=True,
            peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            cumulative_wall_seconds=past["cumulative_wall_seconds"]+time.perf_counter()-began,
            confirmation_status=config["confirmation_role"], budget_includes_encoding=True,
            attribution="仅同一FC25快照的群组选择诊断，非累计简化/多根稳健结论；随机不匹配接受轨迹",
            recovery_accounting_scope="完整轮及已保存预测/提取缓存幂等协调；完全未落盘的中断成本不能完整追溯")
        write_json(folder/"summary.json", summary)
        print(json.dumps(dict(complete=True, diagnostics=diagnostics, cost=cost)), flush=True)
    finally:
        persist_session_cost(cost_path, past, began, config["maximum_wall_seconds"])
        lock.unlink(missing_ok=True)
        write_json(folder/"running.json", dict(pid=None, status="exited"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/h1_task_basis_v1.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
