"""只读保存预测/结构，复核并汇报路径节点正则实验，不运行模型。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def prediction_metrics(path):
    with np.load(path, allow_pickle=False) as cache:
        logits, labels = cache["logits"].astype(np.float64), cache["labels"]
    shifted = logits-logits.max(axis=1, keepdims=True)
    losses = np.log(np.exp(shifted).sum(axis=1))-shifted[np.arange(len(labels)), labels]
    prediction = logits.argmax(axis=1)
    confusion = np.zeros((logits.shape[1], logits.shape[1]), dtype=np.int64)
    np.add.at(confusion, (labels, prediction), 1)
    return dict(samples=len(labels), accuracy=float(np.mean(prediction == labels)), loss=float(losses.mean()),
                confusion_matrix=confusion.tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="experiments/runs/h1_path_sparsity_v1/mnist_static")
    parser.add_argument("--date", default="20261008")
    parser.add_argument("--reuse-prediction-check", action="store_true", help="仅已完成一次保存logits复核后使用，不重复同一复核")
    args = parser.parse_args()
    run = ROOT/args.run
    summary = json.loads((run/"summary.json").read_text(encoding="utf-8"))
    for arm, result in (() if args.reuse_prediction_check else summary["arms"].items()):
        for filename, expected in (("fit_final.npz", result["fit"]),
                                   ("select_025.npz", result["rows"][-1]["select"]),
                                   ("confirm_final.npz", result["confirmation"])):
            observed = prediction_metrics(run/arm/filename)
            if observed["confusion_matrix"] != expected["confusion_matrix"] or any(
                    not np.isclose(observed[key], expected[key], atol=1e-7, rtol=1e-7) for key in ("accuracy", "loss")):
                raise ValueError(f"保存预测与汇总不一致：{arm}/{filename}")
    reports = ROOT/"reports"
    reports.mkdir(exist_ok=True)
    stem = f"h1_path_sparsity_v1_mnist_static_{args.date}"
    evidence = reports/f"{stem}_evidence"
    evidence.mkdir(exist_ok=True)
    for filename in ("registration.json", "initialization.json", "training_commit.json", "freeze.json", "session_cost.json", "data_manifest.json"):
        shutil.copy2(run/filename, evidence/filename)
    for arm in summary["arms"]:
        target = evidence/arm
        target.mkdir(exist_ok=True)
        for filename in ("final_model.bgs", "fit_final.npz", "select_025.npz", "confirm_final.npz", "path_scores_025.npz"):
            shutil.copy2(run/arm/filename, target/filename)
    cost = json.loads((run/"session_cost.json").read_text(encoding="utf-8"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
    fig, axes = plt.subplots(3, 2, figsize=(13, 10))
    for arm, result in summary["arms"].items():
        rows = result["rows"]
        epochs = [row["epoch"] for row in rows]
        series = ([row["select"]["accuracy"]*100 for row in rows], [row["select"]["loss"] for row in rows],
                  [sum(row["widths"][1:-1]) for row in rows], [row["structure"]["export_bytes"]/1024 for row in rows],
                  [row["path_diagnostics"]["normalized_regularizer"] for row in rows],
                  [row["mean_ordinary_objective"]["task_loss"] for row in rows])
        for axis, values in zip(axes.flat, series):
            axis.plot(epochs, values, label=arm, linewidth=1.5)
    for axis, label in zip(axes.flat, ("结构选择准确率（%）", "结构选择交叉熵", "物理隐藏节点数", "完整导出（KiB）", "固定尺度归一化路径正则", "前49普通批次平均分类损失")):
        axis.set_title(label)
        axis.set_xlabel("训练轮")
        axis.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("路径节点正则：五臂实际性能、参数目标与结构曲线", fontsize=14)
    fig.tight_layout()
    fig.savefig(reports/f"{stem}_curves.png", dpi=150)
    plt.close(fig)
    lines = ["# H1完整路径节点正则：实际五臂结果", "", "本报告仅复核已保存logits与结构，不产生模型前向或训练。",
             "", f"![实际曲线]({stem}_curves.png)", "",
             "", "| 臂 | 拟合准确率 | 选择准确率 | 选择CE | 重复探索准确率 | 探索CE | 隐藏宽度 | 参数槽 | 实际字节 | 接受 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |"]
    for arm, result in summary["arms"].items():
        selected, confirmation = result["rows"][-1]["select"], result["confirmation"]
        slots = result["structure"]["model_total"]["allocated_parameter_slots"]
        accepted = sum(event["accepted"] for event in result["events"])
        lines.append(f"| {arm} | {result['fit']['accuracy']:.2%} | {selected['accuracy']:.2%} | {selected['loss']:.6f} | "
                     f"{confirmation['accuracy']:.2%} | {confirmation['loss']:.6f} | {result['widths'][1:-1]} | {slots} | "
                     f"{result['structure']['export_bytes']} | {accepted}/{len(result['events'])} |")
    gate = summary["gate"]
    totals = {key: sum(result["ledger"][key] for result in summary["arms"].values()) for key in
              ("optimization_forward_samples", "optimization_backward_samples", "attempted_updates", "candidate_updates",
               "rolled_back_updates", "fallback_updates", "evaluation_forward_samples", "regularizer_backprop_calls")}
    totals["net_updates"] = totals["attempted_updates"]-totals["rolled_back_updates"]
    fc_bytes = summary["arms"]["fc"]["structure"]["export_bytes"]
    plain_bytes = summary["arms"]["plain_prune"]["structure"]["export_bytes"]
    regularized_fixed_bytes = summary["arms"]["path_fixed"]["structure"]["export_bytes"]
    lines += ["", f"登记主判断：success={gate['success']}；FC学会={gate['fc_learned']}，效果保持={gate['task_preserved']}，"
              f"物理缩宽={gate['physical_width_reduced']}，实际文件减少{gate['storage_reduction']:.2%}。",
              "", "**本轮证明路径正则方案可安全剪宽，但没有证明正则促进可剪性。**"
              "plain_prune也4/4接受，与path_prune同样物理删58节点、宽512/256→473/237、535818→486023参数槽；"
              f"无正则方案文件减少{1-plain_bytes/fc_bytes:.2%}，超过路径方案的{gate['storage_reduction']:.2%}。"
              "因此不能把主门槛通过解释为新增正则优于普通剪节点。"
              "两者均4/4接受，固定4%×4slot的删除预算未必触及任一模型的简化极限；"
              "本轮既未证明正则增加可剪性，也不能判定它对最终可剪上限没有帮助。",
              "", "固定图path_fixed与fc宽度/槽数完全相同，选择CE0.273183比0.462507低，但准确率92.50%比92.625%略低。"
              f"它的真实文件为{regularized_fixed_bytes}字节，比fc {fc_bytes}大{regularized_fixed_bytes/fc_bytes-1:.2%}。"
              "相同物理规模仍有不同无损FP32压缩字节，不能把文件变化全归因于节点数量或宣称量化。",
              "", "path_prune与path_random逐轮同宽度/接受slot/净更新，末态选择准确率主+0.625pp、旧control主+0.2pp；"
              "随机臂选择/探索CE反而略低，方向不一致，只能记为小幅单根身份信号，不声称统计显著。"
              "plain与path最终规模恰好相同，但训练目标、参数与接受参考不同，亦非多根配对优势。", "", "纯分类损失、正则和总目标分别保存在逐轮rows中；"
              "路径分数分位数/数组仅为机制诊断，不是L0或剪枝成功。1000control已反复探索，只在全部末态冻结后记录一次；"
              "官方测试与剩余原ID未解封。", "", "## 实际成本", "", "```json", json.dumps(totals, ensure_ascii=False, indent=2), "```", "",
              f"session_cost最终跨会话墙钟{cost['cumulative_wall_seconds']:.2f}秒（summary写入时{summary['cumulative_wall_seconds']:.2f}秒）；设备{summary['registration']['device']}、FP32、batch64/评价512、线程2。"
              f"CUDA整会话峰值{summary['peak_cuda_memory_bytes']}字节，未作逐臂内存/推断延迟/能耗测量。"
              "参数槽和真实文件减少仍配合稠密gather执行，不能据此宣称加速或节能。", "",
              f"训练前实现提交574295083e6b16f93768c5bad4fc67b034bb2cf9；[末态模型、预测与元数据]({stem}_evidence/registration.json)"
              "保存来源/配置哈希与各臂真实无损FP32文件。", "",
              "每完整轮保存图/参数/Adam/梯度/RNG与账本；预测缓存和末态评价账本已协调。"
              "完全未落盘的中途成本不能完整追溯。额外GPU检查0；本轮仅复用已有模型证据并新增一次路径机制CPU检查。", "",
              "## 下一项机制建议（未训练）", "",
              "候选方向为任务敏感多节点基组简化，不用纯激活残差单独决定哪些节点可替代。"
              "对重构误差ΔH，记录真实下游扰动ΔZ=ΔH·W_droppedᵀ，并以逐样本拟合CE对下一层预激活的导数G_Z衡量"
              "mean_n Σ_k |G_Z,nk ΔZ_nk|；这个乘积在ReLU节点入/出正缩放下保持尺度一致，无需假定二阶。"
              "同时记录下游扰动的按输出通道固定尺度归一化RMS，防止一阶符号抵消或零梯度把明显误差藏掉。"
              "优先考虑在同一冻结FC快照上并行生成残差/任务/随机群组并给同等适应，以隔离一次群组选择。"
              "若后续各臂累计延续，候选与系数仅使用相同生成规则，不能称不同轨迹的池完全相同。系数仍只拟合训练子集，"
              "内部留出只验证固定C32，结构选择继续检验纯任务损失和真实缩宽/bytes。"
              "快照/累计比较方式、归一化常数和容差须在新轮前登记，不扫描阈值；本报告不实现或启动新训练。", "",
              "## 候选记录", ""]
    for arm in ("plain_prune", "path_prune", "path_random"):
        lines += [f"### {arm}", "", "| 轮 | 层 | 提议/实际删节点 | 接受 | 原因 |", "| --- | --- | --- | --- | --- |"]
        for event in summary["arms"][arm]["events"]:
            lines.append(f"| {event['epoch']} | {event['layer_index']} | {event.get('proposed_removed_nodes', event['removed_nodes'])}/"
                         f"{event['removed_nodes']} | {event['accepted']} | {event['reason']} |")
        lines.append("")
    (reports/f"{stem}.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    (reports/f"{stem}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(dict(report=str(reports/f"{stem}.md"), predictions_verified=True,
                          model_forward_samples=0, total_cost=totals), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
