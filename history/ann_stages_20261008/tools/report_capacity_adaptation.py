"""只读冻结结构适应的真实预测/位图，模型前向为零。"""
from pathlib import Path
import json
import shutil
import struct
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    started = time.perf_counter()
    folder = ROOT / "experiments/runs/h1_capacity_adaptation_v1/mnist_static"
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    parent = json.loads((ROOT / "experiments/runs/h1_capacity_pruning_v1/mnist_static/summary.json").read_text(encoding="utf-8"))
    title = "h1_capacity_adaptation_v1_mnist_static_20261008"
    target = ROOT / "reports" / title
    evidence = target.with_name(title + "_evidence")
    evidence.mkdir(exist_ok=True)
    labels = {"fixed": "固定完整图", "task": "任务敏感剪枝图", "random": "匹配随机剪枝图"}
    for name, result in summary["results"].items():
        arm = folder / name
        for split, file in (("select", "select_epoch_16.npz"), ("fit", "fit_final.npz"), ("control", "control_final.npz")):
            with np.load(arm / file) as saved:
                logits, y = saved["logits"].astype(np.float64), saved["labels"]
            prediction = logits.argmax(1)
            loss = np.log(np.exp(logits - logits.max(1, keepdims=True)).sum(1)) + logits.max(1) - logits[np.arange(len(y)), y]
            expected = result[split]
            matrix = np.bincount(y * 10 + prediction, minlength=100).reshape(10, 10)
            if abs((prediction == y).mean() - expected["accuracy"]) > 1e-7 or abs(loss.mean() - expected["cross_entropy"]) > 1e-6 or matrix.tolist() != expected["confusion_matrix"]:
                raise ValueError("保存预测核对失败")
            shutil.copy2(arm / file, evidence / f"{name}_{split}.npz")
        masks = {}
        with (arm / "model.bgm").open("rb") as source:
            assert source.read(4) == b"BGM1"
            header = json.loads(source.read(struct.unpack("<I", source.read(4))[0]))
            for i, row in enumerate(header["layers"]):
                bitmap = np.frombuffer(source.read(row["bitmap_bytes"]), dtype=np.uint8).copy()
                if int(np.unpackbits(bitmap, bitorder="little")[:int(np.prod(row["shape"]))].sum()) != row["edges"]:
                    raise ValueError("位图活边计数不符")
                masks[f"layer_{i}"] = bitmap
                source.seek(4 * (row["edges"] + row["bias_count"]), 1)
        previous = ROOT / f"reports/h1_capacity_pruning_v1_mnist_static_20261008_evidence/{name}_structure.npz"
        with np.load(previous) as saved:
            if any(not np.array_equal(a, saved[k]) for k, a in masks.items()):
                raise ValueError("前轮与末态位图不一致")
        masks["header_utf8"] = np.frombuffer(json.dumps(header, ensure_ascii=False).encode(), dtype=np.uint8)
        np.savez_compressed(evidence / f"{name}_structure.npz", **masks)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for name, result in summary["results"].items():
        rows = [parent["results"][name]["rows"][-1]] + result["rows"]
        x = [r["epoch"] for r in rows]
        axes[0].plot(x, [r["select"]["accuracy"] * 100 for r in rows], marker="o", label=labels[name])
        axes[1].plot(x, [r["select"]["cross_entropy"] for r in rows], marker="o")
        axes[2].plot([8] + [r["epoch"] for r in result["rows"]], [parent["results"][name]["rows"][-1]["fit_augmented_cross_entropy"]] + [r["fit_augmented_cross_entropy"] for r in result["rows"]], marker="o")
    for axis, ylabel in zip(axes, ("验证准确率（%）", "验证交叉熵", "增强训练交叉熵")):
        axis.set(xlabel="全局轮数（8为父末态）", ylabel=ylabel)
        axis.grid(alpha=.2)
    axes[0].axhline(96.5, linestyle="--", color="gray")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(target.with_name(title + "_curves.png"), dpi=160)
    plt.close(fig)
    lines = ["# 扩容FC冻结结构等预算适应实际报告", "", "2026-10-08；训练前登记/实现提交`" + summary["registration"]["training_commit"] + "`。从前轮三臂第8轮完整模型、Adam、EMA与RNG延续各8轮，固定父末轮学习率1.72314444e-5，未重训前轮、未重置Adam、未新增删边；模型根11、数据20261004。第16轮统一冻结，未按验证集/探索性测试集选最佳轮。", "", "| 臂 | 前轮→末态验证准确率 | 前轮→末态验证CE | 末态探索性测试准确率 / CE | 末态训练准确率 / CE |", "| --- | --- | --- | --- | --- |"]
    for name, r in summary["results"].items():
        old = parent["results"][name]
        lines.append(f"| {labels[name]} | {old['select']['accuracy']:.3%}→{r['select']['accuracy']:.3%} | {old['select']['cross_entropy']:.6f}→{r['select']['cross_entropy']:.6f} | {r['control']['accuracy']:.3%} / {r['control']['cross_entropy']:.6f} | {r['fit']['accuracy']:.3%} / {r['fit']['cross_entropy']:.6f} |")
    task, fixed, rand = (summary["results"][n] for n in ("task", "fixed", "random"))
    lines += ["", f"随机追上预登记判据（验证准确率距task≤0.5pp且CE≤task+.025）={summary['random_caught_up']}；主对fixed效果保持={summary['task_retention_pass']}。随机相对主验证准确率差{100*(rand['select']['accuracy']-task['select']['accuracy']):+.3f}pp、CE差{rand['select']['cross_entropy']-task['select']['cross_entropy']:+.6f}；训练CE差{rand['fit']['cross_entropy']-task['fit']['cross_entropy']:+.6f}，帮助判断有限适应后的拟合差距。", "", "本结果只回答固定父末轮小学习率的额外8轮。未追上不能推出长期容量劣势，追上也不证明随机结构在所有预算都等价。探索性测试集已历史使用，1000样本不是全新独立最终测试；3200训练/800验证原ID隔离，官方MNIST测试集及其余封存ID均未访问。", "", "| 臂 | 是否≥96.5%验证及探索性测试 | 起末mask不变 | 活边 / 含偏置独立参数 | 起末统一导出字节 | 新增更新 / 优化样本 | 峰CUDA字节 | 臂墙钟秒 |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    old_gap = parent["results"]["random"]["select"]["cross_entropy"] - parent["results"]["task"]["select"]["cross_entropy"]
    new_gap = rand["select"]["cross_entropy"] - task["select"]["cross_entropy"]
    lines.insert(12, f"随机验证CE与主的差距由{old_gap:.6f}缩为{new_gap:.6f}，其末轮CE{rand['rows'][-2]['select']['cross_entropy']:.6f}→{rand['rows'][-1]['select']['cross_entropy']:.6f}仍下降，尚未显示收敛平台。原CE落后包含有限适应因素，本轮不把剩余差距全归于结构容量。")
    lines.insert(13, "")
    for name, r in summary["results"].items():
        lines.append(f"| {labels[name]} | {r['branch_accuracy_pass']} | {r['masks_unchanged']} | {sum(r['structure']['active_edges'])} / {r['structure']['independent_parameters']} | {r['initial_structure']['bytes']}→{r['structure']['bytes']} | {r['ledger']['attempted_updates']} / {r['ledger']['optimization_samples']} | {r['peak_cuda_memory_bytes']} | {r['wall_seconds']:.3f} |")
    failed = [labels[n] for n, r in summary["results"].items() if not r["branch_accuracy_pass"]]
    lines += ["", f"分支准确率失败：{'、'.join(failed) if failed else '无'}。失败分支停止对应扩展，不改已经通过的CNN/FC基本基线，不据探索性测试成绩追调。三个图的完整位图已与前轮保存证据精确核对一致；权重正常更新，部署结构/真实字节没有新增简化。", "", f"新增优化前向/反传各76800样本、1200尝试/净更新，三臂各400；无候选/回退，评价30次/31800样本。完整会话{summary['session_wall_seconds']:.3f}秒，冻结后三臂探索性测试阶段{summary['control_phase_seconds']:.3f}秒（含导出载入/模型构建/保存）；分项训练/验证/导出/checkpoint/训练集评价成本在JSON。表中臂墙钟从加载父完整状态/起点导出后起算，不含装载准备，准备计入会话。", "", f"设备{summary['registration']['device_name']}，FP32、训练batch64/评价512、CPU线程2、dense Linear+零mask、禁止TF32。训练EMA/mask维护三臂相同，固定完整图首臂含CUDA算子初始化；不能以本轮臂墙钟差称算法加速。宽度/参数槽仍25401290，字节/边数不是实测能耗，无稀疏计算加速证据。", "", "复用前轮已通过真实小CPU导出/完整恢复行为检查，没有重跑测试/GPU诊断/旧训练。新轮只检查父epoch/源码SHA/固定种子、同更新预算及位图不变；保存末态预测的准确率、CE和混淆矩阵逐项核对一致。报告0模型前向，代码不加载模型做评价。", "", f"报告生成{time.perf_counter()-started:.3f}秒。完整三臂预测与完整位图/头随报告保存；大型部署模型与各轮checkpoint/预测在`experiments/runs/h1_capacity_adaptation_v1/mnist_static/`本地保留，训练入口拒绝覆盖。父checkpoint SHA、源/配置SHA与提交号保存在JSON，不改父证据。", "", f"![实际验证与训练曲线]({title}_curves.png)", ""]
    target.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    target.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(target.with_suffix('.md')), "model_forwards": 0, "report_seconds": time.perf_counter()-started}, ensure_ascii=False))


if __name__ == "__main__":
    main()
