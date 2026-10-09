"""部署诊断只读报告；读取保存预测/资源时序，无模型前向。"""
from pathlib import Path
import json
import shutil
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    started = time.perf_counter()
    folder = ROOT / "experiments/runs/h1_capacity_deployment_v1/mnist_static"
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    artifact = json.loads((folder / "csr_artifact.json").read_text(encoding="utf-8"))
    title = "h1_capacity_deployment_v1_mnist_static_20261008"
    target = ROOT / "reports" / title
    evidence = target.with_name(title + "_evidence")
    evidence.mkdir(exist_ok=True)
    labels = {"task_dense": "主图同权重dense", "task_csr": "主图真实CSR", "fixed_dense": "完整FC dense", "cnn": "CNN参照"}
    for result in summary["results"]:
        if "validation" not in result:
            continue
        name = f"{result['variant']}_{result['device']}"
        file = folder / f"{name}_validation.npz"
        with np.load(file) as saved:
            logits, y = saved["logits"].astype(np.float64), saved["labels"]
        prediction = logits.argmax(1)
        loss = np.log(np.exp(logits-logits.max(1,keepdims=True)).sum(1)) + logits.max(1) - logits[np.arange(len(y)), y]
        matrix = np.bincount(y*10+prediction, minlength=100).reshape(10, 10)
        if abs((prediction==y).mean()-result["validation"]["accuracy"])>1e-7 or abs(loss.mean()-result["validation"]["cross_entropy"])>1e-6 or matrix.tolist()!=result["validation"]["confusion_matrix"]:
            raise ValueError("保存预测与登记评价不一致")
        shutil.copy2(file, evidence / file.name)
    cnn_state = torch.load(ROOT / "experiments/runs/capacity_baseline_v1/mnist_static/cnn/best.pt", map_location="cpu", weights_only=False)["state"]
    cnn_edges = {}
    for name, side in (("conv1", 28), ("conv2", 14)):
        output = side // 2
        occurrence = np.array([sum(0 <= position*2-1+offset < side for position in range(output)) for offset in range(3)])
        nonzero = (cnn_state[name+".weight"].numpy()!=0).sum((0,1))
        cnn_edges[name] = int((nonzero * np.outer(occurrence, occurrence)).sum())
    cnn_edges["fc1"] = int(torch.count_nonzero(cnn_state["fc1.weight"]))
    cnn_edges["fc2"] = int(torch.count_nonzero(cnn_state["fc2.weight"]))
    cnn_edges["total"] = sum(cnn_edges.values())
    summary["csr_artifact"] = artifact
    summary["cnn_effective_nonzero_edges"] = cnn_edges
    lines = ["# 扩容FC真实CSR部署实际报告", "", "2026-10-08；测量前实现/登记提交`"+summary["registration"]["training_commit"]+"`。只读task16/fixed16及CNN25，未训练/删边/量化，不读取训练集、探索性测试集、官方MNIST测试集或剩余封存ID。仅已有800验证集；每设备/模型一个独立子进程串行，CPU线程2、FP32、TF32关闭。", "", "| 设备 / 执行 | 状态 | 验证准确率 / CE | 最大绝对输出误差 | 800 argmax一致 |", "| --- | --- | --- | --- | --- |"]
    for r in summary["results"]:
        v, c = r.get("validation",{}), r.get("consistency",{})
        lines.append(f"| {r['device']} / {labels[r['variant']]} | {r['status']} | {v.get('accuracy',float('nan')):.3%} / {v.get('cross_entropy',float('nan')):.6f} | {c.get('maximum_absolute_error',float('nan')):.8f} | {c.get('argmax_identical')} |")
        if r["status"] != "pass":
            lines.append(f"\n失败保留：{r.get('error')}\n")
    lines += ["", "一致性基准是各模型此前保存的完整验证logits，CSR对应task16。相对误差（近零值可能放大）另在JSON，不作调阈值依据。CSR采用原位图每个活边的FP32值/int32 crow与col；没有减小非零边，采用[官方 sparse.mm 接口](https://docs.pytorch.org/docs/2.14/generated/torch.sparse.mm.html)，实际是否支持/加速以本机2.13.0+cu130测量为准。", "", "| 设备 / 执行 | batch1 p50 / p95 ms | batch64 p50 / p95 ms |", "| --- | --- | --- |"]
    for r in summary["results"]:
        if r["status"] == "pass":
            a, b = r["timings"]["1"], r["timings"]["64"]
            lines.append(f"| {r['device']} / {labels[r['variant']]} | {a['p50_ms']:.6f} / {a['p95_ms']:.6f} | {b['p50_ms']:.6f} / {b['p95_ms']:.6f} |")
    for device in ("cpu", "cuda"):
        matched = {r["variant"]: r for r in summary["results"] if r["device"]==device and r["status"]=="pass"}
        if "task_dense" in matched and "task_csr" in matched:
            ratios = {b: matched["task_csr"]["timings"][b]["p50_ms"]/matched["task_dense"]["timings"][b]["p50_ms"] for b in ("1","64")}
            lines += ["", f"{device}同权重CSR/dense的p50时延比：batch1 {ratios['1']:.3f}倍、batch64 {ratios['64']:.3f}倍（小于1才是该条件加速）。结果依赖设备/批量/实现，不能概括所有部署更快。"]
    lines += ["", "端到端：固定验证前1/64张host原灰度FP32→目标设备→前向→10类输出回CPU→同步；包括GPU pageable输入/输出传输，无pinned、CUDA Graph或并发。每尺寸10次预热+100次独立计时，原始时序、均值及每样本p50在JSON。模型载入/解码/预处理不在请求计时内；没有外部请求服务/网络，不能称整套线上服务延迟。", "", "| 设备 / 执行 | 数值 / 索引 / 总张量字节 | 载入后工作集 / 最终峰工作集字节 | CUDA模型分配 / 阶段峰分配字节 |", "| --- | --- | --- | --- |"]
    for r in summary["results"]:
        if "storage" in r:
            s, m = r["storage"], r.get("memory_final",r.get("memory_after_load",{}))
            lines.append(f"| {r['device']} / {labels[r['variant']]} | {s['numeric_bytes']} / {s['index_bytes']} / {s['model_tensor_bytes']} | {r.get('memory_after_load',{}).get('working_set_bytes')} / {m.get('peak_working_set_bytes')} | {r.get('cuda_model_allocated_bytes')} / {r.get('cuda_peak_allocated_bytes')} |")
    lines += ["", f"实际CSR部署文件{artifact['bytes']}字节；原位图+活值BGM文件44814701字节。CSR索引增大文件，CSR运行张量与BGM存储格式不同；dense同权重数值张量101605160字节，训练mask/Adam未部署。所有{sum(artifact['numerically_nonzero_edges'])}活边均实际非零。进程工作集/峰工作集来自Windows GetProcessMemoryInfo，包含PyTorch、CPU分配/转换/载入，不是模型张量内存；GPUtorch分配/保留缓存不包含全部CUDA上下文与驱动内存，初始工作集及缓存另列JSON。以上均不等于能耗。", "", f"CNN真实有效非零连接独立核对：conv1={cnn_edges['conv1']}、conv2={cnn_edges['conv2']}、FC1={cnn_edges['fc1']}、FC2={cnn_edges['fc2']}，排除padding合计{cnn_edges['total']}。按实际保存权重非零及每kernel位置合法输入次数计，偏置不算输入边；CNN独立参数206922仅作附带信息。当前FC有效非零边{sum(artifact['numerically_nonzero_edges'])}，约CNN {sum(artifact['numerically_nonzero_edges'])/cnn_edges['total']:.3f}倍。用户最新主目标是有效非零连接接近CNN，独立参数/文件不是压缩目标或门槛，后续不强制各剪枝点97%，完整报告性能下降。", "", f"完整部署诊断会话{summary['session_wall_seconds']:.3f}秒（含8个独立进程启动与转换）；设备{summary['registration']['cpu']} / {summary['registration']['gpu_name']}，平台{summary['registration']['platform']}。一次小CPU真实CSR/dense前向与int32索引检查通过0.071秒，无训练更新/反传，800验证×8=6400评价样本，预热/计时1760调用57200推断样本，合计63600。旧测试/GPU启动诊断未重跑；本轮一致性与计时是正式测量。", "", "下一轮优先有效边10×/5×/2×/1× CNN（4336800/2168400/867360/433680）压缩曲线，保持相同边数身份随机对照，观察任务效果而不硬守97%。当前任意CSR的执行结果仅提示部署约束，不能用它作为新删边阈值搜索或预置卷积结构依据。", "", f"报告仅读取保存预测/时序及CNN权重计数，模型前向0；生成{time.perf_counter()-started:.3f}秒。完整CSR部署文件本地`experiments/runs/h1_capacity_deployment_v1/mnist_static/task_csr.pt`，SHA见JSON。报告附8份验证预测与完整原始时序/资源JSON，可复核输出/时延，不自动重复测量。", ""]
    target.with_suffix(".md").write_text("\n".join(lines),encoding="utf-8")
    target.with_suffix(".json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"report":str(target.with_suffix('.md')),"model_forwards":0,"report_seconds":time.perf_counter()-started},ensure_ascii=False))


if __name__ == "__main__":
    main()
