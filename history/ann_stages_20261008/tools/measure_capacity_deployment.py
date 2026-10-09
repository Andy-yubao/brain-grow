"""冻结模型真实CSR部署与串行独立进程端到端测量。"""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import gc
import json
from pathlib import Path
import platform
import struct
import subprocess
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from tools.train_capacity_baseline import ROOT, FourLayerFC, TwoConvTwoFC, digest, evaluate, save_json


class CSRFC(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.depth = len(layers)
        for i, row in enumerate(layers):
            weight = torch.sparse_csr_tensor(row["crow"], row["col"], row["values"], size=row["shape"], check_invariants=True)
            self.register_buffer(f"weight_{i}", weight)
            self.register_buffer(f"bias_{i}", row["bias"])

    def forward(self, x):
        x = x.flatten(1)
        for i in range(self.depth):
            x = torch.sparse.mm(getattr(self, f"weight_{i}"), x.t().contiguous()).t() + getattr(self, f"bias_{i}")
            if i + 1 < self.depth:
                x = F.relu(x)
        return x


def read_bgm(path):
    layers = []
    with path.open("rb") as source:
        if source.read(4) != b"BGM1":
            raise ValueError("BGM格式错误")
        header = json.loads(source.read(struct.unpack("<I", source.read(4))[0]))
        for row in header["layers"]:
            alive = np.unpackbits(np.frombuffer(source.read(row["bitmap_bytes"]), dtype=np.uint8), bitorder="little")[:int(np.prod(row["shape"]))].astype(bool).reshape(row["shape"])
            values = np.frombuffer(source.read(4 * row["edges"]), dtype="<f4").copy()
            bias = np.frombuffer(source.read(4 * row["bias_count"]), dtype="<f4").copy()
            if int(alive.sum()) != row["edges"] or not np.isfinite(values).all() or not np.isfinite(bias).all():
                raise ValueError("导出位图/数值异常")
            layers.append({"alive": alive, "values": values, "bias": bias, "shape": row["shape"]})
        if source.read(1):
            raise ValueError("BGM尾部异常")
    return header, layers


def csr_layers(layers):
    result = []
    for row in layers:
        counts = row["alive"].sum(1, dtype=np.int64)
        crow = np.concatenate(([0], counts.cumsum())).astype(np.int32)
        if crow[-1] != len(row["values"]) or int(crow[-1]) >= 2**31 - 1:
            raise ValueError("CSR int32范围或边数异常")
        col = np.nonzero(row["alive"])[1].astype(np.int32)
        result.append({"crow": torch.from_numpy(crow), "col": torch.from_numpy(col), "values": torch.from_numpy(row["values"]),
                       "bias": torch.from_numpy(row["bias"]), "shape": row["shape"]})
    return result


def dense_bgm(path):
    header, layers = read_bgm(path)
    model = FourLayerFC(header["widths"], header["dropout"])
    with torch.no_grad():
        for layer, row in zip(model.layers, layers):
            weights = np.zeros(row["shape"], dtype=np.float32)
            weights[row["alive"]] = row["values"]
            layer.weight.copy_(torch.from_numpy(weights))
            layer.bias.copy_(torch.from_numpy(row["bias"]))
    return model


def memory_info():
    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [(k, ctypes.c_size_t) for k in
            ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage", "PrivateUsage")]
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return {"working_set_bytes": counters.WorkingSetSize, "peak_working_set_bytes": counters.PeakWorkingSetSize, "private_usage_bytes": counters.PrivateUsage}


def model_storage(model):
    numeric, indices = 0, 0
    for tensor in list(model.parameters()) + list(model.buffers()):
        if tensor.layout == torch.sparse_csr:
            numeric += tensor.values().numel() * tensor.values().element_size()
            indices += tensor.crow_indices().numel() * tensor.crow_indices().element_size() + tensor.col_indices().numel() * tensor.col_indices().element_size()
        else:
            numeric += tensor.numel() * tensor.element_size()
    return {"numeric_bytes": numeric, "index_bytes": indices, "model_tensor_bytes": numeric + indices}


def worker(folder, variant, device_name):
    cfg = json.loads((folder / "registration.json").read_text(encoding="utf-8"))["config"]
    torch.set_num_threads(cfg["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(device_name)
    result = {"variant": variant, "device": device_name, "memory_before_load": memory_info()}
    started = time.perf_counter()
    try:
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA不可用")
            torch.cuda.reset_peak_memory_stats()
        parent = ROOT / "experiments/runs/h1_capacity_adaptation_v1/mnist_static"
        baseline = ROOT / "experiments/runs/capacity_baseline_v1/mnist_static"
        if variant == "task_csr":
            saved = torch.load(folder / "task_csr.pt", map_location="cpu", weights_only=True)
            model = CSRFC(saved["layers"])
            del saved
            reference_file = parent / "task/select_epoch_16.npz"
        elif variant in ("task_dense", "fixed_dense"):
            name = variant.split("_")[0]
            model = dense_bgm(parent / name / "model.bgm")
            reference_file = parent / name / "select_epoch_16.npz"
        else:
            cfg_base = json.loads((ROOT / "configs/capacity_baseline_v1.json").read_text(encoding="utf-8"))
            model = TwoConvTwoFC(cfg_base["cnn_channels"], cfg_base["cnn_hidden"], cfg_base["dropout"])
            state = torch.load(baseline / "cnn/best.pt", map_location="cpu", weights_only=False)
            model.load_state_dict(state["state"])
            del state
            reference_file = baseline / "cnn/select_best.npz"
        model.eval().requires_grad_(False)
        result["storage"] = model_storage(model)
        model.to(device)
        gc.collect()
        result["memory_after_load"] = memory_info()
        result["cuda_model_allocated_bytes"] = torch.cuda.memory_allocated() if device.type == "cuda" else None
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        with np.load(ROOT / "data/structure_mnist_v1/select.npz", allow_pickle=False) as data:
            host_x = torch.from_numpy(data["x"])
            labels = data["y"]
        with np.load(reference_file, allow_pickle=False) as saved:
            reference = saved["logits"]
            if not np.array_equal(labels, saved["labels"]):
                raise ValueError("验证标签与保存证据不一致")
        with torch.inference_mode():
            measured, logits = evaluate(model, host_x.to(device), torch.from_numpy(labels).to(device), cfg["evaluation_batch_size"])
            difference = np.abs(logits.astype(np.float64) - reference.astype(np.float64))
            result["validation"] = measured
            result["consistency"] = {"maximum_absolute_error": float(difference.max()), "maximum_relative_error": float((difference / np.maximum(np.abs(reference), 1e-8)).max()),
                                     "argmax_identical": bool(np.array_equal(logits.argmax(1), reference.argmax(1))), "reference_sha256": digest(reference_file)}
            np.savez_compressed(folder / f"{variant}_{device_name}_validation.npz", logits=logits, labels=labels)
            passed = difference.max() <= cfg["absolute_error_limit"] and result["consistency"]["argmax_identical"]
            if not passed:
                raise ValueError("与对应保存预测误差超标，停止该格式计时")
            timings = {}
            for batch in cfg["batch_sizes"]:
                inputs = host_x[:batch].contiguous()
                def request():
                    output = model(inputs.to(device)).cpu()
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    return output
                for _ in range(cfg["warmup_iterations"]):
                    request()
                samples = []
                for _ in range(cfg["timing_iterations"]):
                    t = time.perf_counter_ns()
                    request()
                    samples.append((time.perf_counter_ns() - t) / 1e6)
                timings[str(batch)] = {"milliseconds": samples, "mean_ms": float(np.mean(samples)), "p50_ms": float(np.median(samples)), "p95_ms": float(np.percentile(samples, 95)),
                                       "p50_ms_per_sample": float(np.median(samples) / batch)}
            result["timings"] = timings
        result["memory_final"] = memory_info()
        result["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated() if device.type == "cuda" else None
        result["cuda_reserved_bytes"] = torch.cuda.memory_reserved() if device.type == "cuda" else None
        if device.type == "cuda" and result["cuda_peak_allocated_bytes"] > cfg["peak_cuda_memory_limit_bytes"]:
            raise MemoryError("峰CUDA超过登记预算")
        result["status"] = "pass"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = repr(exc)
    result["worker_wall_seconds"] = time.perf_counter() - started
    save_json(folder / f"{variant}_{device_name}.json", result)
    print(json.dumps({"variant": variant, "device": device_name, "status": result["status"], "timings": {k: v["p50_ms"] for k, v in result.get("timings", {}).items()}, "error": result.get("error")}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=("task_csr", "task_dense", "fixed_dense", "cnn"))
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    folder = ROOT / "experiments/runs/h1_capacity_deployment_v1/mnist_static"
    if args.worker:
        worker(folder, args.worker, args.device)
        return
    cfg_path = ROOT / "configs/h1_capacity_deployment_v1.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    folder.mkdir(parents=True, exist_ok=True)
    if any(folder.iterdir()):
        raise FileExistsError("拒绝覆盖/重复测量")
    started = time.perf_counter()
    parent = ROOT / "experiments/runs/h1_capacity_adaptation_v1/mnist_static"
    summary = json.loads((parent / "summary.json").read_text(encoding="utf-8"))
    models = {n: parent / n / "model.bgm" for n in ("task", "fixed")}
    for name, path in models.items():
        if digest(path) != summary["results"][name]["structure"]["sha256"]:
            raise ValueError("父模型SHA不匹配")
    manifest = json.loads((ROOT / "data/structure_mnist_v1/manifest.json").read_text(encoding="utf-8"))
    if manifest["seed"] != 20261004 or digest(ROOT / "data/structure_mnist_v1/select.npz") != manifest["cache_sha256"]["select"] or len(manifest["splits"]["select"]) != 800:
        raise ValueError("验证数据SHA/固定ID不一致")
    registration = {"config": cfg, "config_sha256": digest(cfg_path), "source_sha256": digest(Path(__file__)),
                    "training_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    "parent_model_sha256": {n: digest(p) for n, p in models.items()}, "cnn_checkpoint_sha256": digest(ROOT / "experiments/runs/capacity_baseline_v1/mnist_static/cnn/best.pt"),
                    "validation_cache_sha256": manifest["cache_sha256"]["select"], "torch": torch.__version__,
                    "cpu": platform.processor(), "platform": platform.platform(), "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                    "cuda_runtime": torch.version.cuda, "tf32": False}
    save_json(folder / "registration.json", registration)
    header, layers = read_bgm(models["task"])
    csr = csr_layers(layers)
    torch.save({"layers": csr, "architecture": header["widths"]}, folder / "task_csr.pt")
    save_json(folder / "csr_artifact.json", {"bytes": (folder / "task_csr.pt").stat().st_size, "sha256": digest(folder / "task_csr.pt"),
                                           "active_edges": [len(row["values"]) for row in csr], "numerically_nonzero_edges": [int(torch.count_nonzero(row["values"])) for row in csr]})
    del layers, csr
    results = []
    for device in ("cpu", "cuda"):
        for variant in ("task_dense", "task_csr", "fixed_dense", "cnn"):
            if time.perf_counter() - started > cfg["session_seconds"]:
                save_json(folder / "failure.json", {"error": "整次工程时间达到", "results": results})
                return
            command = [str(ROOT / ".venv/Scripts/python.exe"), "-m", "tools.measure_capacity_deployment", "--worker", variant, "--device", device]
            try:
                run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=cfg["worker_seconds"])
                (folder / f"{variant}_{device}_process.txt").write_text(run.stdout + run.stderr, encoding="utf-8")
                print(run.stdout.strip(), flush=True)
                result_file = folder / f"{variant}_{device}.json"
                result = json.loads(result_file.read_text(encoding="utf-8")) if result_file.exists() else {"variant": variant, "device": device, "status": "failed", "error": "工作进程无结果", "exit_code": run.returncode}
            except subprocess.TimeoutExpired:
                result = {"variant": variant, "device": device, "status": "failed", "error": "120秒worker上限达到"}
            results.append(result)
    save_json(folder / "summary.json", {"registration": registration, "results": results, "session_wall_seconds": time.perf_counter() - started,
                                         "training_updates": 0, "test_set_evaluations": 0, "official_test_accessed": False, "sealed_remaining_ids_accessed": False})


if __name__ == "__main__":
    main()
