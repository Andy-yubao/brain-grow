"""只读已冻结四臂结果，生成本轮中文报告及标准静态科学图。"""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/"experiments/runs/h1_adaptive_edges_v1/mnist_circular"
OUT = ROOT/"docs/reports/2026-10-08/h1_adaptive_edges_v1"
NAMES = {"magnitude_old": "幅度旧控制＋闭合", "magnitude_adaptive": "幅度自适应＋闭合",
         "node_old": "节点旧控制＋闭合", "node_adaptive": "节点自适应＋闭合"}


def main():
    summary = json.loads((RUN/"summary.json").read_text(encoding="utf-8"))
    assert summary["gate"] == "completed"
    results = summary["results"]
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(RUN/"summary.json", OUT/"result.json")
    registration = summary["registration"]
    rows = []
    events_text = []
    for name, r in results.items():
        accepted = [e for e in r["events"] if e["accepted"]]
        direct = [sum(e["direct_deleted_per_layer"][i] for e in accepted) for i in range(4)]
        closure = [sum(e["closure"]["removed_nonzero_per_layer"][i] for e in accepted) for i in range(4)]
        initial = sum(r["initial_closure"]["after"]["nonzero_edges"])
        final = sum(r["structure"]["nonzero_edges"])
        original = sum(r["initial_closure"]["before"]["nonzero_edges"])
        rows.append(f"| {NAMES[name]} | {original:,} | {initial:,} | {final:,} | {r['validation']['accuracy']*100:.4f}% | {r['validation']['cross_entropy']:.6f} | {len(accepted)} | {r['wall_seconds']:.2f} |")
        events_text.append(f"### {NAMES[name]}\n\n起始闭合额外删除{sum(r['initial_closure']['removed_nonzero_per_layer']):,}边；被接受操作直接删边分层为`{direct}`，闭合新增删边为`{closure}`。最终各层真实边`{r['structure']['nonzero_edges']}`；前两层分别{r['structure']['nonzero_edges'][0]:,}/{r['structure']['nonzero_edges'][1]:,}。终止原因`{r['stop']}`，选中完整状态`{r['selected_full_checkpoint']}`。\n\n| 父槽/尝试 | 强度 | 适应轮数 | 结论 | 验证准确率 | 验证CE |\n| --- | --- | --- | --- | --- | --- |")
        for e in r["events"]:
            metric = e.get("validation", {})
            accuracy = f"{metric['accuracy']*100:.4f}%" if metric else "未到验证点"
            ce = f"{metric['cross_entropy']:.6f}" if metric else "—"
            events_text.append(f"| {e['slot']}/{e['attempt']} | {e['strength']:.6g} | {e.get('adaptation_epochs',0)} | `{e['status']}` | {accuracy} | {ce} |")
        events_text.append("\n固定验证点明细（旧候选与回退原图适应分开；同一历史点只列一次）：\n\n| 槽/尝试 | 轮 | 决策 | 准确率 | CE | 训练状态 | 验证仍改善 |\n| --- | --- | --- | --- | --- | --- | --- |")
        seen = set()
        for event in r["events"]:
            for point in event["validation_points"]:
                key = (event["slot"], event["attempt"], point["step"], point["decision"])
                if key in seen:
                    continue
                seen.add(key)
                metric = point["validation"]
                events_text.append(f"| {event['slot']}/{event['attempt']} | {point['step']} | `{point['decision']}` | {metric['accuracy']*100:.4f}% | {metric['cross_entropy']:.6f} | `{point['training_status']}` | {point.get('validation_improving',False)} |")
        selected_slot = 0 if r["selected_full_checkpoint"] == "initial.pt" else int(r["selected_full_checkpoint"].split("_")[1].split(".")[0])
        selected_updates = sum(s["event"]["adaptation_epochs"]*50 for s in r["stages"] if s["slot"] and s["slot"] <= selected_slot)
        events_text.append(f"\n最终固定单位置训练校准准确率{r['training_fixed_one_position']['accuracy']*100:.4f}%、CE{r['training_fixed_one_position']['cross_entropy']:.6f}。尝试更新{r['ledger']['attempted_updates']:,}次，完成路径未回退更新{r['ledger']['net_updates']:,}次，最终选中完整状态实际保留本轮{selected_updates:,}次更新（回选较早状态会舍去后续正常更新）；验证/校准等评价共{r['ledger']['evaluation_calls']}次、{r['ledger']['evaluation_samples']:,}样本。BGM导出{r['structure']['bytes']:,}字节，独立参数{r['structure']['independent_parameters']:,}；仍无参数共享映射。峰值CUDA已分配显存{r['peak_cuda_memory_bytes']:,}字节。\n")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for column, mechanism in enumerate(("magnitude", "node")):
        for kind, marker in (("old", "o"), ("adaptive", "s")):
            r = results[f"{mechanism}_{kind}"]
            stages = r["stages"]
            xs = [s["slot"] for s in stages]
            label = "旧控制" if kind == "old" else "自适应"
            axes[0, column].plot(xs, [sum(s["structure"]["nonzero_edges"])/1e6 for s in stages], marker=marker, label=label)
            axes[1, column].plot(xs, [s["validation"]["accuracy"]*100 for s in stages], marker=marker, label=label)
        axes[0, column].set_title("幅度评分" if mechanism == "magnitude" else "节点贡献评分")
        axes[0, column].set_ylabel("保留结构的真实非零边（百万）")
        axes[1, column].set_ylabel("验证准确率（%）")
        axes[1, column].axhline(93.0937519, color="gray", linestyle="--", linewidth=1, label="固定准确率门槛")
        for row in range(2):
            axes[row, column].set_xlabel("已完成父槽")
            axes[row, column].set_xticks([0, 1, 2, 3])
            axes[row, column].grid(alpha=.25)
            axes[row, column].legend()
    fig.savefig(OUT/"retained_structure_function.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, (name, r) in zip(axes.flat, results.items()):
        ax.plot(np.arange(1, len(r["rows"])+1), [p["hard_label_ce"] for p in r["rows"]], linewidth=1, alpha=.65, label="随机平移训练CE")
        calibration = [(index+1, p["fixed_training_calibration"]["cross_entropy"]) for index, p in enumerate(r["rows"]) if "fixed_training_calibration" in p]
        if calibration:
            ax.plot([p[0] for p in calibration], [p[1] for p in calibration], "o-", markersize=3, label="固定训练校准CE")
        ax.set_title(NAMES[name])
        ax.set_xlabel("已尝试训练轮数（包含回退成本）")
        ax.set_ylabel("交叉熵")
        ax.grid(alpha=.25)
        ax.legend()
    fig.savefig(OUT/"adaptation_training_curves.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    positions = np.arange(4)
    for index, (name, r) in enumerate(results.items()):
        ax.bar(positions+(index-1.5)*.2, r["structure"]["nonzero_edges"], width=.2, label=NAMES[name])
    ax.set_xticks(positions, ["第一层", "第二层", "第三层", "第四层"])
    ax.set_ylabel("最终真实非零边")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=.25)
    ax.legend()
    fig.savefig(OUT/"layer_edges.png", dpi=160)
    plt.close(fig)

    comparison = []
    layer_comparison = []
    costs_table = []
    matched = {}
    for mechanism in ("magnitude", "node"):
        old, adaptive = results[f"{mechanism}_old"], results[f"{mechanism}_adaptive"]
        old_edges, adaptive_edges = sum(old["structure"]["nonzero_edges"]), sum(adaptive["structure"]["nonzero_edges"])
        comparison.append(f"{('幅度' if mechanism == 'magnitude' else '节点贡献')}：自适应较旧控制最终少{old_edges-adaptive_edges:,}边，验证准确率差{(adaptive['validation']['accuracy']-old['validation']['accuracy'])*100:+.4f}个百分点，CE差{adaptive['validation']['cross_entropy']-old['validation']['cross_entropy']:+.6f}；墙钟比{adaptive['wall_seconds']/old['wall_seconds']:.2f}倍。")
        a = np.load(RUN/f"{mechanism}_old/slot_1_attempt_1_candidate_8.npz")
        b = np.load(RUN/f"{mechanism}_adaptive/slot_1_attempt_1_candidate_8.npz")
        matched[mechanism] = {"first_candidate_first_eight_epoch_logits_exact": bool(np.array_equal(a["logits"], b["logits"])),
                              "same_labels": bool(np.array_equal(a["labels"], b["labels"]))}
        assert all(matched[mechanism].values()), "首候选首8轮未匹配，需要解释而非假定对照成立"
    for name, r in results.items():
        edges = r["structure"]["nonzero_edges"]
        initial_layers = r["initial_closure"]["after"]["nonzero_edges"]
        layer_comparison.append(f"| {NAMES[name]} | {edges[0]:,} | {(1-edges[0]/initial_layers[0])*100:.2f}% | {edges[0]/28224:.2f}× | {edges[1]:,} | {(1-edges[1]/initial_layers[1])*100:.2f}% | {edges[1]/225792:.2f}× |")
        unassigned = r["wall_seconds"]-sum(r["costs"].values())
        costs_table.append(f"| {NAMES[name]} | {r['costs']['training_seconds']:.2f} | {r['costs']['snapshot_seconds']:.2f} | {r['costs']['validation_seconds']:.2f} | {r['costs']['calibration_seconds']:.2f} | {unassigned:.2f} |")
    (OUT/"paired_first_segment.json").write_text(json.dumps(matched, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    failures = sum(e["status"] == "completed_failure" for r in results.values() for e in r["events"])
    incompletes = sum(e["status"] == "optimization_incomplete" for r in results.values() for e in r["events"])
    report = f"""# H1自适应剪边与训练内闭合：本轮报告

2026-10-08。预登记、实现、轻检查和四臂完整实验均已完成。冻结训练提交`{registration['training_commit']}`；源SHA256`{registration['source_sha256']}`，配置SHA256`{registration['config_sha256']}`。全量机器证据见[result.json](result.json)，检查见[check.json](check.json)，协议见[执行预登记](../../../h1_adaptive_edges_v1_execution.md)。

## 功能与结构

| 实验臂 | 原真实边 | 初始闭合后边 | 最终真实边 | 验证准确率 | 验证CE | 接受操作数 | 墙钟秒 |
| --- | --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(rows)}

{chr(10).join(comparison)}

固定累计参照为旧循环密集FC94.0937519%/CE0.22852926，合格门槛93.0937519%/CE≤0.28852926。所有结构都只在首次预定验证点功能合格时接受，最后从保留完整状态中选最少边的合格状态；没有遍历48轮挑候选期最佳，没有声称找到最佳参数。操作性平台失败共{failures}次；最大适应预算内优化未完成共{incompletes}次。仅完成适应且验证也不再显著改善的失败才触发同父完整回退及减半。

两机制各自的旧/自适应首候选前8轮6400验证logits逐值完全相等，证据见[首段匹配](paired_first_segment.json)；延长阶段以后训练更新不等，控制收益与新增训练成本合并解释，不能说是同计算量优势。最终边数仍单独对照阶段520416目标，目标数字不进入剪边评分/步长。

幅度自适应最终530455边比固定静态阶段目标520416多10039（约1.93%），但低于同任务循环CNN456000×1.2=547200的数量参照。两种参照必须区分；没有新探索测试，因此只能确认旧验证上的功能门槛。第一层仍有广域连接，不能以总边数接近CNN说前两层均达到卷积规模。

| 实验臂 | 第一层边 | 相对初始闭合减少 | /循环CNN28224 | 第二层边 | 相对初始闭合减少 | /循环CNN225792 |
| --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(layer_comparison)}

![保留结构与功能](retained_structure_function.png)

图中展示逐槽完成路径；最终报告数值来自合格状态回选，因此节点旧控制选起始状态，幅度旧控制选第1槽，可能优于图中最后槽。

![分层连接数量](layer_edges.png)

## 每臂证据

{chr(10).join(events_text)}

![适应训练曲线](adaptation_training_curves.png)

## 成本、恢复与边界

设备{registration['device_name']}，Torch {registration['torch']}，FP32、batch64，评价batch512，CPU线程2；执行为稠密linear加零权重掩码。四臂内部墙钟计时合计{summary['total_arm_wall_seconds']:.2f}秒，包含臂内失败候选、训练和保存等成本；计时起点在各父checkpoint读取/恢复之后，主入口启动、父状态读取、数据核对/准备等未包含，整个进程端到端墙钟未单独实测。900秒/臂及3600秒总额按此臂内口径核算，没有按旧时间倍率否决研究。

| 实验臂 | 训练秒 | 已计时resume保存秒 | 固定验证点秒 | 固定训练校准秒 | 其余未归类秒 |
| --- | --- | --- | --- | --- | --- |
{chr(10).join(costs_table)}

JSON的评分/闭合分项也有计时。`export_seconds`字段未实际累加，其零值表示未独立测量，不是导出零成本。初始化闭合后的首次评价、候选/阶段/最终checkpoint保存、导出等部分成本留在墙钟残差，不能声称全部成本精确分类。没有为补齐计时重跑训练。无新延迟或能耗实测；边数/参数和掩码执行不能证明加速或节能。

每臂`experiments/runs/h1_adaptive_edges_v1/mnist_circular/<arm>/model_full.pt`保存选中模型的完整权重、mask、Adam、RNG及controller，同时保留被拒绝候选末态checkpoint和逐epoch恢复入口。原归档模型/checkpoint不改。只读一次CPU核对两源模型与stage5状态完全一致；闭合收益与直接剪边分别报告，全部物理节点和bias保留。

仅既有3200训练/800验证原ID，验证每ID8个固定循环位移，不是6400独立笔迹。完全复用旧位置manifest，没有读取探索测试、官方测试、封存ID或CIFAR。验证用于接受及选模，反馈次数完整记录；该旧验证反复使用及单种子限制泛化结论。固定训练单位置CE平台是有限观测下的操作判据，不能证明总体收敛或容量极限。前两层定位只使用本轮逐层曲线，未追加诊断扫描或CNN层配额。

下一方向建议优先第一层多边联合功能重构/补偿的候选机制，并做同父对照；这是下一轮登记建议，尚未实施或运行。本轮不追加第4槽或小调参来追520416。
"""
    (OUT/"report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"report": str(OUT/"report.md"), "arms": len(results), "completed_failures": failures, "optimization_incomplete": incompletes}, ensure_ascii=False))


if __name__ == "__main__":
    main()
