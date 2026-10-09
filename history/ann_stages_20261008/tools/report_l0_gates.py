"""逐边门控报告：保存预测/文件/门值只读，0模型前向，不加载checkpoint。"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import time

import numpy as np

from .report_pruning import check_metrics, confusion_table, digest, read_json, saved_metrics
from .report_nodes_nonzero import file_widths

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("fixed", "learned", "gate_no_l0", "random")


def make_plot(arms, path):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    fixed_bytes = {row["epoch"]: row["structure"]["export_bytes"] for row in arms["fixed"]["rows"]}
    for arm in ARMS:
        rows = arms[arm]["rows"]
        epochs = [row["epoch"] for row in rows]
        values = ([r["select"]["accuracy"] * 100 for r in rows], [r["select"]["loss"] for r in rows],
                  [r["structure"]["model_total"]["active_edges"] for r in rows],
                  [r["structure"]["export_bytes"] / fixed_bytes[r["epoch"]] for r in rows],
                  [r["gates"]["expected_active_edges"] if r["gates"] else np.nan for r in rows],
                  [r["mean_fit_batch_ce"] for r in rows])
        for ax, values_one in zip(axes.flat, values):
            ax.plot(epochs, values_one, label=arm, linewidth=1.6)
    titles = (("结构选择准确率", "准确率（%）"), ("结构选择交叉熵", "交叉熵"),
              ("确定性部署活跃边", "边数"), ("相对同轮fixed实际文件", "字节比"),
              ("随机门期望活跃边", "期望边数"), ("真实训练批平均分类损失", "交叉熵"))
    for ax, (title, ylabel) in zip(axes.flat, titles):
        ax.set_title(title)
        ax.set_xlabel("从父FC25继续的轮数")
        ax.set_ylabel(ylabel)
        ax.axvline(20.5, color="black", linestyle=":", linewidth=1)
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.suptitle("H1-L0逐边门控：第20轮后固定图；random共享learned前缀", fontsize=13)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    began = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="h1_l0_gates_v1")
    parser.add_argument("--date", default="20261008")
    args = parser.parse_args()
    run = ROOT / "experiments/runs" / args.version / "mnist_static"
    summary, registration = read_json(run / "summary.json"), read_json(run / "registration.json")
    if summary["registration"] != registration or registration["config"]["version"] != "h1_l0_gates_v1" or set(summary["arms"]) != set(ARMS):
        raise ValueError("要求已登记完成的四臂门控实验")
    arms, config = summary["arms"], registration["config"]
    if not all(result["frozen"] and result["final_epoch"] == 25 and len(result["rows"]) == 25 for result in arms.values()):
        raise ValueError("四臂必须完成25逻辑轮并冻结")
    commit = read_json(run / "training_commit.json")
    reports = ROOT / "reports"
    stem = f"{args.version}_mnist_static_{args.date}"
    evidence = reports / f"{stem}_evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    predictions, hashes = {}, {}
    def copy_file(source, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        hashes[str(destination.relative_to(reports))] = digest(destination)
    for arm in ARMS:
        af, result = run / arm, arms[arm]
        predictions[arm] = {}
        recorded = dict(fit=result["fit"], select=result["rows"][-1]["select"], confirm=result["confirmation"])
        for split, filename in (("fit", "fit_final.npz"), ("select", "select_025.npz"), ("confirm", "confirm_final.npz")):
            actual = saved_metrics(af / filename)
            check_metrics(recorded[split], actual, f"{arm}:{split}")
            predictions[arm][split] = actual
            copy_file(af / filename, evidence / arm / filename)
        if file_widths(af / "final_model.bgs") != registration["initial_widths"]:
            raise ValueError("门控轮不应改变节点维度")
        if (af / "final_model.bgs").stat().st_size != result["structure"]["export_bytes"]:
            raise ValueError("实际部署文件长度不符")
        for filename in ("final_model.bgs", "initial_frozen_model.bgs", "freeze_boundary.json", "freeze_cost.json"):
            copy_file(af / filename, evidence / arm / filename)
        if arm != "fixed":
            copy_file(af / "freeze_gate_values.npz", evidence / arm / "freeze_gate_values.npz")
    # 不构造模型，直接用保存FP32门值/置乱核对分叉身份和活跃位置。
    permutation_evidence = []
    with np.load(run / "learned/freeze_gate_values.npz", allow_pickle=False) as learned, np.load(run / "random/freeze_gate_values.npz", allow_pickle=False) as random:
        for layer in range(len(registration["initial_widths"])-1):
            left, right, perm = learned[f"gate_{layer}"], random[f"gate_{layer}"], random[f"permutation_{layer}"]
            if left.dtype != np.float32 or right.dtype != np.float32 or not np.array_equal(np.sort(perm), np.arange(len(left))) or not np.array_equal(left[perm], right):
                raise ValueError("不是同层完整门值置乱")
            count = int(np.count_nonzero(left > 0))
            if count != int(np.count_nonzero(right > 0)) or count != arms["learned"]["freeze"]["layers"][layer]["active_gate_positions"]:
                raise ValueError("主随机活跃位置匹配失败")
            permutation_evidence.append(dict(layer_index=layer, active_gate_positions=count, zero_gates=int(np.count_nonzero(left == 0)),
                fractional_gates=int(np.count_nonzero((left > 0) & (left < 1))), full_gate_multiset_verified=True))
    copy_file(run / "random/select_freeze_boundary.npz", evidence / "random/select_freeze_boundary.npz")
    for arm in config["prefix_arms"]:
        copy_file(run / "prefixes" / arm / "select_020.npz", evidence / "prefixes" / arm / "select_020.npz")
        copy_file(run / "prefixes" / arm / "deterministic_parent_model.bgs", evidence / "prefixes" / arm / "deterministic_parent_model.bgs")
    for filename in ("training_commit.json", "registration.json", "startup.json", "data_manifest.json", "acceptance.json", "freeze.json", "session_cost.json"):
        copy_file(run / filename, evidence / filename)
    costs = summary["actual_cost_total"]
    actual_updates = sum(item["ledger"]["attempted_updates"] for item in list(summary["prefixes"].values()) + list(arms.values()))
    if actual_updates != 4000 or costs["optimization_forward_samples"] != 256000 or costs["evaluation_forward_samples"] > 81600:
        raise ValueError("实际账本未符合固定预算")
    plot_path = reports / f"{stem}_curves.png"
    make_plot(arms, plot_path)
    hashes[plot_path.name] = digest(plot_path)
    difference = {split: dict(accuracy_delta=predictions["learned"][split]["accuracy"] - predictions["random"][split]["accuracy"],
                               loss_delta=predictions["learned"][split]["loss"] - predictions["random"][split]["loss"])
                  for split in ("fit", "select", "confirm")}
    report = copy.deepcopy(summary)
    report.update(training_commit=commit, prediction_metrics=predictions, verified_permutation=permutation_evidence,
                  learned_minus_random=difference, evidence_sha256=hashes,
                  report_execution="保存预测/模型架构头/门值/文件/账本只读；0模型前向反传，无checkpoint加载",
                  report_wall_seconds=time.perf_counter()-began)
    (reports / f"{stem}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    lines = ["# H1-L0逐边可训练门控：实际报告", "", f"{args.date}；启动前实现提交`{commit['gitHead']}`。",
        "", f"[完整JSON]({stem}.json)、[证据索引]({stem}_evidence/README.md)保存曲线、门统计、真实预测、成本和来源。",
        "", "## 固定机制与公平边界", "",
        "从旧未剪枝FC25延续，模型根11/数据20261004，784→512→256→10，3200fit/800select；"
        "四臂冻结并完成接受决定后各重复探索旧1000control一次。官方测试及剩余封存ID未访问。",
        "", "Louizos等既有hard-concrete L0机制的逐边应用，不称首创；作者Dense按输入单元门，本轮每边门、bias独立。"
        "β2/3、γ-.1、ζ1.1、a初值2.5；训练随机门，评价使用不除β的z_det。"
        "CE+λ·sum(p_open)/3200，λ前5轮线性至.001并保持至20轮；门lr.02、θlr.00025第16轮减半、AdamW eps1e-8、wd0，无扫描。",
        "", "p_open是理想未截断Uniform(0,1)解析期望；实际Uniform端点保护后仅作为近似统计/惩罚。累计1800秒在新轮开始前及轮末检查，单个已开始轮可跨过阈值，实际成本照记。",
        "", "fixed/learned/gate_no_l0三个实际20轮前缀；random复用learned20母态。四最终分支各5轮，同批次/Adamreset，固定掩码CE训练。"
        "random逐层置乱完整z_det（含零），冻结同theta×z_perm；主为theta×z_det。"
        "位置及衰减幅度对应联合身份是归因对象，不能称纯二值身份或独立随机训练全过程。"
        "gate_no_l0的门仍接受CE梯度，控制显式L0项，不能称固定dropout。",
        "", "冻结仅折入确定性门并删除精确零门，无额外阈值、排序强剪或最佳轮选择。训练a/Adam和门采样成本单报；"
        "部署文件删除门元数据、保留无损FP32有效权重、偏置和掩码。原槽/节点维度保持，稠密gather执行不意味着推断加速或能耗收益。",
        "", "## 完整预算末态成绩", "", "| 臂 | 集合 | 样本 | 准确率 | 交叉熵 |", "| --- | --- | --- | --- | --- |"]
    labels = dict(fit="拟合", select="结构选择", confirm="重复探索")
    for arm in ARMS:
        for split in labels:
            m = predictions[arm][split]
            lines.append(f"| {arm} | {labels[split]} | {m['samples']} | {m['accuracy']:.3%} | {m['loss']:.6f} |")
    lines += ["", "## 确定性图与真实大小", "", "| 臂 | 活跃边 | FP32非零边 | 独立参数含偏置 | 原分配槽 | 真实字节 | 相对fixed减少 |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    fixed_bytes = arms["fixed"]["structure"]["export_bytes"]
    for arm in ARMS:
        s, t = arms[arm]["structure"], arms[arm]["structure"]["model_total"]
        lines.append(f"| {arm} | {t['active_edges']} | {t['nonzero_edges']} | {t['independent_parameters']} | {t['allocated_parameter_slots']} | {s['export_bytes']} | {1-s['export_bytes']/fixed_bytes:.3%} |")
    lines += ["", "每层门值（含零）多重集、保存置乱索引、活跃位置已只读核对；活跃代表可为零，与非零边数分开。"
              "实际文件长度含权重、bias、宽度、掩码/索引及压缩元数据，不用期望L0代替实际大小。",
              "", "| 层 | 期望边（learned20） | 主随机共同活跃位置 | 零门 | 分数门 |", "| --- | --- | --- | --- | --- |"]
    last_gates = summary["prefixes"]["learned"]["rows"][-1]["gates"]
    for entry in permutation_evidence:
        index = entry["layer_index"]
        lines.append(f"| {index} | {last_gates['layers'][index]['expected_active_edges']:.2f} | {entry['active_gate_positions']} | {entry['zero_gates']} | {entry['fractional_gates']} |")
    lines += ["", f"![真实中文曲线]({plot_path.name})", "", "random前20轮曲线是共享learned母态，不是重复训练。虚线处冻结，之后没有期望门指标；训练批CE与确定性评价CE不同。",
              "", "## 实际成本", "", "| 实际执行段 | 优化样本前向/反传 | 更新 | 评价调用/样本 | Adam重置轮 | 段计时秒 | 编码构图秒 |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for phase, items in (("前缀", summary["prefixes"]), ("分支", arms)):
        for arm, item in items.items():
            c = item["ledger"]
            lines.append(f"| {phase}:{arm} | {c['optimization_forward_samples']}/{c['optimization_backward_samples']} | {c['attempted_updates']} | {len(c['evaluations'])}/{c['evaluation_forward_samples']} | {c['adam_reset_epochs']} | {item['wall_seconds']:.3f} | {item['encoding_wall_seconds']:.3f} |")
    lines += ["", f"新增实际{costs['attempted_updates']}更新，优化前向/反传各{costs['optimization_forward_samples']}样本，"
        f"评价{costs['evaluation_calls']}次/{costs['evaluation_forward_samples']}样本。random逻辑前20轮64000样本共享，不重复算实际成本。"
        "主/λ0/random后5轮均250适应更新；失败适应不退成本、不补fallback。没有新增激活拟合、模型检查或GPU诊断。",
        "", f"会话{summary['current_session_wall_seconds']:.3f}秒，累计{summary['cumulative_wall_seconds']:.3f}秒，"
        f"设备{registration['hardware']}（{registration['device']}）、FP32、训练batch64/评价512、CPU线程2；"
        f"整次CUDA峰分配{summary['peak_cuda_memory_bytes']}字节。段计时排除checkpoint/预测写入及末态fit/探索，编码构图是段计时子集不能重复相加。"
        "前缀checkpoint字节完整JSON列明；训练额外535040门参数及Adam矩不能省略。没有实测推断速度或能耗。",
        "", "## 一次接受判定与机制归因", "", f"最终accepted={summary['acceptance']['accepted']}；任务保持={summary['acceptance']['task_preserved']}；"
        f"真实文件减少{summary['acceptance']['storage_reduction']:.3%}，20%阶段信号={summary['acceptance']['storage_signal_at_least_20_percent']}。"
        "判据为相对fixed25选择1pp/.05、fit≥90%且真实文件更小，旧control不参与。",
        "", "| 集合 | learned−random准确率百分点 | learned−random交叉熵 |", "| --- | --- | --- |"]
    for split, d in difference.items():
        lines.append(f"| {labels[split]} | {100*d['accuracy_delta']:+.3f} | {d['loss_delta']:+.6f} |")
    lines += ["", "单根、反复使用的select与旧control不足以称显著性/稳健性/跨任务发现。"
        "主随机差只能解释同theta下完整门值分配；λ0比较显式L0增量，普通FC比较整套门训练代价和效果。"
        "期望L0下降、确定性零门、真实导出、任务保持及身份优势分别判定。",
        "", "若接受失败，25轮失败候选预测、大小和全部预算仍保留；仅恢复20轮母态完整状态作后续入口，不把恢复态冒充公平25轮模型。"
        "原母态参数/Adam/日程/RNG/门流checkpoint未受分支改写，未追加训练、改λ/门阈值或扫描种子。",
        "", "## 证据与恢复", "", "完整checkpoint与逐轮日志留原运行目录；证据包括四候选真实模型/预测、"
        "初始冻结模型、完整门值与置乱、20轮确定性母模型及预测、训练提交、源配置哈希和接受账本。报告不加载checkpoint/模型，0前向。",
        "", "```powershell", ".venv/Scripts/python.exe -u -B -m tools.train_l0_gates --resume", "```", "",
        "完成summary的恢复入口直接返回，不复跑训练或评价；失败母态继续需另登记新轮。", "", "## 混淆矩阵", ""]
    for arm in ARMS:
        for split in labels:
            lines += [f"### {arm}：{labels[split]}", "", *confusion_table(predictions[arm][split]["confusion_matrix"]), ""]
    (reports / f"{stem}.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    (evidence / "README.md").write_text("# H1-L0实际证据\n\n四臂25轮候选的FP32模型与保存预测、20轮确定性母模型/预测、完整门值与随机置乱、"
        "冻结及接受登记按实际字节复制，哈希在上级JSON。无checkpoint加载或模型前向。\n\n"
        "门训练完整20轮母态checkpoint留experiments原目录，仅作为恢复入口；失败候选25轮结果仍完整报告，"
        "恢复20轮不是公平25轮模型。random共享learned前缀，实际执行账本不重复计成本。"
        "官方测试封存，旧control只重复探索。\n", encoding="utf-8")
    print(json.dumps(dict(report=str(reports / f"{stem}.md"), actual_cost=costs,
        acceptance=summary["acceptance"], learned_minus_random=difference), ensure_ascii=False))


if __name__ == "__main__":
    main()
