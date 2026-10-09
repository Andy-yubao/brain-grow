"""只读已保存预测的功能群组报告；不运行模型，图表中文。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from braingrow.pruning_experiment import metrics  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="experiments/runs/h1_task_basis_v1/mnist_static")
    parser.add_argument("--reuse-prediction-check", action="store_true")
    args = parser.parse_args()
    folder = ROOT/args.run
    summary = json.loads((folder/"summary.json").read_text(encoding="utf-8"))
    cost = json.loads((folder/"session_cost.json").read_text(encoding="utf-8"))
    if not args.reuse_prediction_check:
        for branch, result in summary["branches"].items():
            paths = [("fit_final.npz", result["fit"]), ("confirm_final.npz", result["confirmation"])]
            paths += [(f"select_{row['epoch']:03d}.npz", row["select"]) for row in result["rows"]]
            for name, expected in paths:
                with np.load(folder/branch/name, allow_pickle=False) as cache:
                    observed = metrics(cache["logits"], cache["labels"])
                assert observed["confusion_matrix"] == expected["confusion_matrix"]
                assert abs(observed["accuracy"]-expected["accuracy"]) < 1e-12
                assert abs(observed["loss"]-expected["loss"]) < 1e-12
    stem = "h1_task_basis_v1_mnist_static_20261008"
    reports = ROOT/"reports"
    evidence = reports/f"{stem}_evidence"
    evidence.mkdir(exist_ok=True)
    for name in ("registration.json", "training_commit.json", "data_manifest.json", "proposal_metadata.json",
                 "freeze.json", "session_cost.json", "layer0_library.npz", "layer1_library.npz"):
        shutil.copy2(folder/name, evidence/name)
    for branch in summary["branches"]:
        target = evidence/branch
        target.mkdir(exist_ok=True)
        for name in ("final_model.bgs", "fit_final.npz", "confirm_final.npz", "select_005.npz", "initialization.json"):
            shutil.copy2(folder/branch/name, target/name)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for layer in (0, 1):
        for branch in ("fixed", f"layer{layer}_residual", f"layer{layer}_task", f"layer{layer}_random"):
            rows = summary["branches"][branch]["rows"]
            epochs = [row["epoch"] for row in rows]
            for column, values in enumerate(([row["select"]["accuracy"]*100 for row in rows],
                    [row["select"]["loss"] for row in rows], [row["structure"]["export_bytes"]/1024 for row in rows])):
                axes[layer, column].plot(epochs, values, marker="o", label=branch)
        for column, title in enumerate(("结构选择准确率（%）", "结构选择交叉熵", "完整导出（KiB）")):
            axes[layer, column].set_title(f"隐藏层{layer}：{title}")
            axes[layer, column].set_xlabel("等预算适应轮")
            axes[layer, column].grid(alpha=.25)
        axes[layer, 0].legend(fontsize=8)
    fig.suptitle("同一冻结FC快照：独立群组选择与等预算适应")
    fig.savefig(reports/f"{stem}_curves.png", dpi=150)
    plt.close(fig)
    lines = ["# 固定快照任务敏感群组简化：实际报告", "",
        "逻辑8分支、实际7分支；fixed只训练/评价一次，两个层诊断引用同一个模型哈希。"
        "同一根11未剪枝FC25快照，两隐藏层独立删8节点，不累积剪宽。数据根20261004；1000control仅末态重复探索。", "",
        "| 实际分支 | 拟合准确率 | 选择准确率 | 选择CE | 探索准确率 | 探索CE | 隐藏宽度 | 文件字节 |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for branch, result in summary["branches"].items():
        selected, confirmation = result["rows"][-1]["select"], result["confirmation"]
        lines.append(f"| {branch} | {result['fit']['accuracy']:.3%} | {selected['accuracy']:.3%} | {selected['loss']:.6f} | "
            f"{confirmation['accuracy']:.3%} | {confirmation['loss']:.6f} | {result['widths'][1:-1]} | {result['structure']['export_bytes']} |")
    lines += ["", "## 接受与机制证据", "", "母态只作来源/恢复入口，不作等预算最终控制；所有7分支fresh Adam、同5轮250更新。"
        "J/Q保护失败候选仍完成同预算适应并保留结果，接受判断仅末态一次；random为共享池均匀身份，不是主接受轨迹匹配对照。", ""]
    for layer, diagnostic in summary["diagnostics"].items():
        lines += [f"### 隐藏层{layer}", "", f"共享fixed模型SHA：{diagnostic['fixed_model_sha256']}。诊断成立={diagnostic['mechanism_diagnostic_established']}。", ""]
        for method, gate in diagnostic["arms"].items():
            info = summary["branches"][f"layer{layer}_{method}"]["proposal"]
            lines.append(f"- {method}：末态诊断通过={gate['accepted']}，功能保护={gate['functional_guard']}，"
                f"效果保持={gate['task_preserved']}，真实文件缩小={gate['real_storage_reduced']}；池大小{info['pool_count']}。")
            if info["proposal_possible"]:
                scores = info["diagnostics"]
                lines.append(f"  拟合J/Q={scores['fit_j']}/{scores['fit_q']}，内部留出J/Q={scores['holdout_j']}/{scores['holdout_q']}，"
                    f"拟合主导项{scores['fit_dominant']}；C32支持数{scores['support_counts']}。")
                if method == "task":
                    lines.append(f"  逐次贪心主导项：{[step['dominant'] for step in info['greedy_steps']]}。")
        lines.append("")
    lines += ["J是拟合任务的一阶敏感度代理，Q是归一化下游扰动。若Q主导或任务梯度近零，低J不能证明梯度选组有效。"
        "两层是同一模型的不同局部诊断，不是独立种子；小差异不称显著，不能据单次结果宣称累计压缩或通用架构发现。", "",
        "## 实际成本与证据", "", "```json", json.dumps(summary["cost"], ensure_ascii=False, indent=2), "```", "",
        f"持久session_cost墙钟{cost['cumulative_wall_seconds']:.6f}秒，设备{summary['registration']['device']}、FP32、batch64、评价512；"
        f"CUDA整会话峰值{summary['peak_cuda_memory_bytes']}字节。编码、CPU求解/贪心和梯度提取均在成本内，未测推断延迟或能耗。", "",
        "完整轮状态、已保存预测和提取缓存可幂等恢复；完全未落盘的中断成本无法完全追溯。"
        "保存拒绝/失败分支，不回写母FC；完成resume不再训练/确认。官方测试与剩余原ID仍封存。", "",
        f"[中文曲线]({stem}_curves.png)、[源与数据登记]({stem}_evidence/registration.json)及末态模型/预测/共同系数库已保存。"
        "报告仅复核保存logits，新增模型前向0；额外GPU检查0。", ""]
    (reports/f"{stem}.md").write_text("\n".join(lines), encoding="utf-8")
    (reports/f"{stem}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(dict(report=str(reports/f"{stem}.md"), model_forward_samples=0, predictions_verified=True), ensure_ascii=False))


if __name__ == "__main__":
    main()
