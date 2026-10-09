"""修正版诊断公共入口：只解包训练数组、逐前向成本与可靠完成恢复。"""
from __future__ import annotations
import hashlib, json, platform, shutil, subprocess, time
from pathlib import Path
import numpy as np
from braingrow.data.moving_mnist_v5 import CONFIG
from braingrow.evaluation.metrics import evaluate
from tools.run_discrete_growth import subset, save, write_json, peak_memory


def training_data():
    path = Path("data/dynamic_v5.npz")
    manifest = json.loads(Path("data/manifest_v5.json").read_text(encoding="utf-8"))
    if manifest["config"] != CONFIG or hashlib.sha256(path.read_bytes()).hexdigest() != manifest["cache_sha256"]:
        raise ValueError("v5缓存协议/哈希不符，拒绝运行")
    # 共享压缩文件的哈希读字节，不解包validation成员、不作验证前向或评价。
    with np.load(path) as z:
        return {k: z[k] for k in ("train_x", "train_y", "train_ids", "train_variants")}


def record_cost(state, videos, kind, outdir):
    state["simulation_videos"] += int(videos)
    row = {"simulation_videos_cumulative": state["simulation_videos"],
        "metric_evaluations_cumulative": state["metric_evaluations"], "videos": int(videos), "kind": kind, "time": time.time()}
    with (outdir / "cost_journal.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n"); handle.flush()


class BudgetExhausted(RuntimeError):
    pass


def ensure_budget(state, videos, outdir):
    if state["simulation_videos"] + videos > state["simulation_budget"]:
        state["stopped_budget"] = True
        save(outdir / "resume.pkl", state)
        write_json(outdir / "budget_stop.json", {"completed": False, "reason": "实际累计预算不足以执行下一批，保留部分轮次状态，不重放或扩预算",
            "actual_simulation_videos": state["simulation_videos"], "budget": state["simulation_budget"], "next_batch_videos": videos})
        raise BudgetExhausted("实际预算已耗尽，保存并停止")


def restore_cost(state, outdir):
    path = outdir / "cost_journal.jsonl"
    if path.exists():
        rows = path.read_text(encoding="utf-8").splitlines()
        if rows:
            row = json.loads(rows[-1])
            state["simulation_videos"] = max(state["simulation_videos"], row["simulation_videos_cumulative"])
            state["metric_evaluations"] = max(state["metric_evaluations"], row["metric_evaluations_cumulative"])


def evaluate_net(net, pool, state, name, outdir):
    x, y, ids = pool; outputs = []; frames = []
    for begin in range(0, len(x), 84):
        batch = x[begin:begin + 84]
        ensure_budget(state, len(batch), outdir)
        p, _, trace = net.forward(batch, trace=True)
        record_cost(state, len(batch), "evaluation_" + name, outdir)
        outputs.append(p); frames.append(trace)
    potential = np.concatenate(outputs); trace = np.concatenate(frames); counts = trace.sum(axis=1)
    rebuilt = counts.astype(np.float64) @ net.out.astype(np.float64)
    np.testing.assert_allclose(potential, rebuilt, atol=1e-4, rtol=1e-6)
    metrics = evaluate(potential, y)
    state["metric_evaluations"] += 1; record_cost(state, 0, "metric_complete_" + name, outdir)
    total = float(trace.sum())
    result = {"name": name, "metrics": metrics, "resources": net.resources(), "unique_ids": len(set(ids.tolist())),
        "videos": len(x), "potential_reconstruction_max_absolute_error": float(np.abs(potential - rebuilt).max()),
        "activity": {"hidden_spikes": int(total), "output_spikes": 0, "mean_spikes_per_frame": total / (len(x) * 8),
            "mean_spikes_per_video": total / len(x), "spikes_per_lif_step": total / (len(x) * 32 * net.n),
            "per_frame_mean": (trace.sum(axis=(0, 2)) / len(x)).tolist(), "silent_units": int((counts.sum(axis=0) == 0).sum())}}
    np.savez_compressed(outdir / (name + "_outputs.npz"), potential=potential, counts=counts, frame_counts=trace, y=y, ids=ids)
    write_json(outdir / (name + ".json"), result)
    return result


def provenance(outdir, configuration):
    configuration.update(start_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        data_hashes={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ("data/dynamic_v5.npz", "data/manifest_v5.json")},
        root_model_seed=11, data_seed=20261004, validation_arrays_unpacked=False, validation_forward_evaluations=0,
        sealed_test_reads=0, device="CPU", environment="local_venv", numpy=np.__version__, platform=platform.platform(),
        neural_time_ms=160, video_time="8帧无固定实际播放时长", dependencies_added=[])
    write_json(outdir / "config.json", configuration)
    (outdir / "start_diff.patch").write_text(subprocess.check_output(["git", "diff"], text=True, encoding="utf-8"), encoding="utf-8")
    for directory in ("src", "tools"):
        shutil.copytree(directory, outdir / "source_snapshot" / directory, ignore=shutil.ignore_patterns("__pycache__"))


def prediction(outdir, name):
    with np.load(outdir / (name + "_outputs.npz")) as z: p = z["potential"]
    return np.column_stack((p[:, :10].argmax(axis=1), p[:, 10:].argmax(axis=1)))
