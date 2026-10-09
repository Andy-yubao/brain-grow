"""读取已保存预测和编码头生成中文报告；不运行模型。"""
from pathlib import Path
import json
import shutil
import struct
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    started = time.perf_counter()
    folder = ROOT / "experiments/runs/h1_capacity_pruning_v1/mnist_static"
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    title = "h1_capacity_pruning_v1_mnist_static_20261008"
    target = ROOT / "reports" / title
    evidence = target.with_name(title + "_evidence")
    evidence.mkdir(exist_ok=True)
    for name, result in summary["results"].items():
        arm = folder / name
        for split, prediction_file in (("select", "select_epoch_8.npz"), ("fit", "fit_final.npz"), ("control", "control_final.npz")):
            with np.load(arm / prediction_file) as saved:
                logits, labels = saved["logits"].astype(np.float64), saved["labels"]
            prediction = logits.argmax(1)
            loss = np.log(np.exp(logits - logits.max(1, keepdims=True)).sum(1)) + logits.max(1) - logits[np.arange(len(labels)), labels]
            accuracy = (prediction == labels).mean()
            confusion = np.bincount(labels * 10 + prediction, minlength=100).reshape(10, 10)
            expected = result[split]
            if abs(accuracy - expected["accuracy"]) > 1e-7 or abs(loss.mean() - expected["cross_entropy"]) > 1e-6 or confusion.tolist() != expected["confusion_matrix"]:
                raise ValueError(f"{name}/{split}保存预测不一致")
            shutil.copy2(arm / prediction_file, evidence / f"{name}_{split}.npz")
        masks = {}
        with (arm / "model.bgm").open("rb") as source:
            assert source.read(4) == b"BGM1"
            header = json.loads(source.read(struct.unpack("<I", source.read(4))[0]))
            for i, layer in enumerate(header["layers"]):
                bitmap = np.frombuffer(source.read(layer["bitmap_bytes"]), dtype=np.uint8).copy()
                actual = int(np.unpackbits(bitmap, bitorder="little")[:int(np.prod(layer["shape"]))].sum())
                if actual != layer["edges"]:
                    raise ValueError("结构位图计数不符")
                masks[f"layer_{i}"] = bitmap
                source.seek(4 * (layer["edges"] + layer["bias_count"]), 1)
            masks["header_utf8"] = np.frombuffer(json.dumps(header, ensure_ascii=False).encode(), dtype=np.uint8)
        np.savez_compressed(evidence / f"{name}_structure.npz", **masks)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    labels = {"fixed": "普通训练", "task": "任务敏感剪枝", "random": "匹配随机剪枝"}
    for name, result in summary["results"].items():
        rows = result["rows"]
        epochs = [row["epoch"] for row in rows]
        axes[0].plot(epochs, [r["select"]["accuracy"] * 100 for r in rows], marker="o", label=labels[name])
        axes[1].plot(epochs, [r["select"]["cross_entropy"] for r in rows], marker="o")
        axes[2].plot(epochs, [sum(r["structure"]["active_edges"]) / 1e6 for r in rows], marker="o")
    for axis, ylabel in zip(axes, ("选择准确率（%）", "选择交叉熵", "活边（百万）")):
        axis.set(xlabel="新增训练轮", ylabel=ylabel)
        axis.grid(alpha=.2)
    axes[0].axhline(96.5, linestyle="--", color="gray")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(target.with_name(title + "_curves.png"), dpi=160)
    plt.close(fig)
    lines = ["# 扩容FC任务敏感剪枝实际报告", "", "2026-10-08；模型根11、数据20261004，同第28轮冻结FC权重。旧CNN/FC均97%基线已通过后新增本轮预算。训练前提交`" + summary["registration"]["training_commit"] + "`；固定8轮/臂，无结果驱动追调。", "", "| 臂 | 选择准确率 / CE | 旧control重复探索 / CE | 活边 | 独立参数含偏置 | 实际部署字节 |", "| --- | --- | --- | --- | --- | --- |"]
    for name, r in summary["results"].items():
        lines.append(f"| {labels[name]} | {r['select']['accuracy']:.3%} / {r['select']['cross_entropy']:.6f} | {r['control']['accuracy']:.3%} / {r['control']['cross_entropy']:.6f} | {sum(r['structure']['active_edges'])} | {r['structure']['independent_parameters']} | {r['structure']['bytes']} |")
    task, fixed, rand = (summary["results"][n] for n in ("task", "fixed", "random"))
    accepted = sum(e["accepted"] for e in summary["events"].values())
    lines += ["", f"主接受{accepted}/4次，实际文件减少{summary['storage_reduction']:.3%}；效果门槛{summary['performance_pass']}、20%实际字节门槛{summary['storage_pass']}，本轮判定`{summary['gate']}`。完整选择反馈参与逐候选接受；不是最终独立测试。主相对随机选择差{100*(task['select']['accuracy']-rand['select']['accuracy']):+.3f}pp、重复探索差{100*(task['control']['accuracy']-rand['control']['accuracy']):+.3f}pp；同时对照交叉熵，单根结果不称统计显著。", "", "位图+活FP32+偏置是无损导出，实际文件删除了被剪权重，不使用标量量化或zip权重压缩。完整位图与架构头随报告保存，共享映射为none，独立参数=活边+偏置。隐藏节点6272/3136/256及训练参数槽25401290不变；训练及推断仍dense Linear，权重/梯度/Adam矩在删边位置归零。尚无稀疏计算执行、推断加速或能耗证据；文件缩小不能推断GPU内存减少。", "", "| 臂 | 尝试 / 净更新 | 优化样本 | 评价调用 / 样本 | 回退 / fallback轮 | 墙钟秒 | 峰CUDA字节 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, r in summary["results"].items():
        a = r["ledger"]
        lines.append(f"| {labels[name]} | {a['attempted_updates']} / {a['net_updates']} | {a['optimization_samples']} | {a['evaluation_calls']} / {a['evaluation_samples']} | {a['rollbacks']} / {a['fallback_epochs']} | {r['wall_seconds']:.3f} | {r['peak_cuda_memory_bytes']} |")
    total_updates = sum(r["ledger"]["attempted_updates"] for r in summary["results"].values())
    total_samples = sum(r["ledger"]["optimization_samples"] for r in summary["results"].values())
    lines += ["", f"新增优化前向/反传各{total_samples}样本、{total_updates}尝试更新。净更新三臂各400，同增强/批次/dropout流；主拒绝后的额外适应独立计账，random只继承主接受的逐层数量/时序，未按随机效果重选。会话完整墙钟{summary['session_wall_seconds']:.3f}秒含最终旧control；表中臂墙钟含训练/选择/候选/CPU事务/编码/checkpoint与fit，最终control在会话层另计。分项costs保存于JSON，checkpoint/最终fit的耗时未独立计入分项，不把分项之和称全会话。", "", "三个臂统一计算EMA和mask维护作共同记账；fixed的EMA不驱动结构，仅作测量开销，因此这里的fixed耗时不是去掉测量的纯普通训练。fixed是本进程首臂，包含首次CUDA算子初始化；主墙钟比fixed短不构成算法训练加速证据。random的大层randperm导致更高峰内存，各峰仅是对应臂本次执行的分配统计。", "", f"设备{summary['registration']['device_name']}，FP32、训练batch64、评价512、CPU线程2，禁止TF32，dense执行。文件写入、排序/候选与全状态CPU复制均计入墙钟；不是参数或FLOPs换算的资源。", "", "唯一新增CPU行为检查通过（导出输出精确相等、固定零权重/Adam矩、拒绝全模型/Adam/EMA/RNG恢复），首跑因项目tmp目录缺失失败，创建后仅重跑该项；无GPU诊断/旧检查/恢复重放。workspace轻量链接/语法检查另记。", "", "三臂冻结后才对已探索的1000 control各评价一次；官方测试和剩余封存ID访问False。新用户基本97%左右门槛操作化为≥96.5%，当前判定保留在完整JSON；不把旧control称独立未见测试。", "", f"保存预测三臂三个划分与完整结构位图均已从本地真实产物核对。报告生成耗时{time.perf_counter()-started:.3f}秒，模型前向/训练0。部署模型与完整轮恢复在`experiments/runs/h1_capacity_pruning_v1/mnist_static/`（本地大型生成物），源码/配置/父权重SHA见JSON；拒绝覆盖的训练入口不会自动重复运行。", "", f"![实际选择与结构曲线]({title}_curves.png)", ""]
    target.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    target.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(target.with_suffix('.md')), "model_forwards": 0, "report_seconds": time.perf_counter()-started}, ensure_ascii=False))


if __name__ == "__main__":
    main()
