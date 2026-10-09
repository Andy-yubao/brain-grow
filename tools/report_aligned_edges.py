"""仅读取已冻结结果和预测，生成节点对齐实验报告；不运行模型。"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import struct

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def output_edges(path):
    with path.open("rb") as source:
        assert source.read(4) == b"BGM1"
        header = json.loads(source.read(struct.unpack("<I", source.read(4))[0]))
        for row in header["layers"]:
            bitmap = source.read(row["bitmap_bytes"])
            values = np.frombuffer(source.read(4 * row["edges"]), dtype="<f4")
            source.read(4 * row["bias_count"])
        mask = np.unpackbits(np.frombuffer(bitmap, dtype=np.uint8), bitorder="little")[:np.prod(row["shape"])].astype(bool)
        weights = np.zeros(mask.shape, dtype=np.float32)
        weights[mask] = values
        return np.count_nonzero(weights.reshape(row["shape"]), axis=1).tolist()


def main():
    folder = ROOT / "experiments/runs/h1_aligned_edges_v1/mnist_static"
    summary = json.loads((folder / "summary.json").read_text())
    results = summary["results"]
    stem = "h1_aligned_edges_v1_mnist_static_20261008"
    evidence = ROOT / "docs/reports/2026-10-08/h1_aligned_edges_v1_mnist_static" / f"{stem}_evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    cnn_summary = json.loads((ROOT / "experiments/runs/capacity_baseline_v1/mnist_static/summary.json").read_text())
    cnn = cnn_summary["results"]["cnn"]
    report = dict(summary, cnn_reused={"best_select": cnn["best_select"], "exploratory_control": cnn["exploratory_control"], "effective_expanded_edges": 433680,
                                      "source": "experiments/runs/capacity_baseline_v1/mnist_static/summary.json"}, external_fc_edge_limit=520416)
    report["external_acceptance"] = {}
    report["evaluation_accounting"] = {"saved_training_ledger_excludes_exploratory_control": True,
                                        "additional_control_calls": 5, "additional_control_samples": 5000,
                                        "by_arm": {n: {"calls_including_control": r["ledger"]["evaluation_calls"] + 1,
                                                       "samples_including_control": r["ledger"]["evaluation_samples"] + 1000}
                                                   for n, r in results.items()}}
    audit_path = ROOT / "tmp/aligned_saved_audit.json"
    if audit_path.exists():
        saved_audit = json.loads(audit_path.read_text())
        shutil.copy2(audit_path, evidence / "saved_artifact_audit.json")
        report["saved_artifact_audit"] = saved_audit
    else:
        saved_audit = {}
    for name, result in results.items():
        select = result.get("select", result.get("best", {}).get("select"))
        edges = sum(result["structure"]["nonzero_edges"])
        report["external_acceptance"][name] = {"edges": edges, "cnn_ratio": edges / 433680, "within_1_2x": edges <= 520416,
                                                "select_at_least_96_5pct": select["accuracy"] + 1e-7 >= .965,
                                                "exploratory_control_at_least_96_5pct": result["control"]["accuracy"] + 1e-7 >= .965,
                                                "output_row_nonzero_edges": output_edges(folder / name / "model.bgm")}
        for file in ("control_final.npz", "fit_final.npz"):
            shutil.copy2(folder / name / file, evidence / f"{name}_{file}")
        if name in ("magnitude", "distilled"):
            shutil.copy2(folder / name / "model.bgm", evidence / f"{name}_model.bgm")
        if name == "baseline":
            shutil.copy2(folder / name / "select_best.npz", evidence / f"{name}_select_final.npz")
        else:
            chosen = result["chosen_stage"]
            source = folder / name / ("select_stage_%d.npz" % chosen) if chosen else folder / "baseline/select_best.npz"
            shutil.copy2(source, evidence / f"{name}_select_final.npz")
            for stage in result["stages"]:
                if stage["stage"] and sum(stage["structure"]["nonzero_edges"]) <= 520416 and stage["select"]["accuracy"] >= .97:
                    shutil.copy2(folder / name / f"stage_{stage['stage']}.bgm", evidence / f"{name}_stage_{stage['stage']}.bgm")
                    shutil.copy2(folder / name / f"select_stage_{stage['stage']}.npz", evidence / f"{name}_select_stage_{stage['stage']}.npz")
            if name == "random":
                shutil.copy2(folder / name / "stage_5.bgm", evidence / "random_matched_stage_5.bgm")
                shutil.copy2(folder / name / "select_stage_5.npz", evidence / "random_matched_select_stage_5.npz")
    (ROOT / "docs/reports/2026-10-08/h1_aligned_edges_v1_mnist_static" / f"{stem}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for name in ("fixed", "magnitude", "distilled", "random"):
        rows = results[name]["stages"]
        x = [r["stage"] for r in rows]
        edges = [sum(r["structure"]["nonzero_edges"]) for r in rows]
        axes[0].semilogy(x, edges, "o-", label=name)
        axes[1].plot(x, [100 * r["select"]["accuracy"] for r in rows], "o-", label=name)
        axes[2].plot(x, [r["select"]["cross_entropy"] for r in rows], "o-", label=name)
    axes[0].axhline(520416, color="black", linestyle="--", label="外部1.2×边上限")
    for ax, title in zip(axes, ("真实非零边", "验证准确率（%）", "验证交叉熵")):
        ax.set_title(title)
        ax.set_xlabel("阶段（每阶段8轮净适应）")
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "docs/reports/2026-10-08/h1_aligned_edges_v1_mnist_static" / f"{stem}_curves.png", dpi=160)
    plt.close(fig)
    primary = results["magnitude"]
    primary_edges = sum(primary["structure"]["nonzero_edges"])
    lines = ["# H1节点对齐FC：全局剪边与充分适应实际报告", "", f"幅度主臂最终保留{primary_edges:,}条真实非零边，比新密集起点减少{1-primary_edges/7577856:.3%}、是CNN有效展开边的{primary_edges/433680:.4f}倍，达到外部≤520416要求；验证{primary['select']['accuracy']:.3%}、探索性测试{primary['control']['accuracy']:.3%}。蒸馏臂达到同一总边数，但本轮没有显示比硬标签幅度臂全面更好的效果。", "", "本轮四臂已全部冻结；表格给出预登记验证规则选择的阶段。1000历史探索性测试只在全部候选冻结后读取，未用于选择阶段。固定模型根11、数据根20261004、训练/验证/探索性测试3200/800/1000。官方测试与剩余封存ID访问均为False。", "",
             "| 分支 | 冻结阶段 | 训练准确率 | 验证准确率 / CE | 探索性测试准确率 / CE | 真实非零边 / CNN倍数 | 导出字节 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, r in results.items():
        s = r.get("select", r.get("best", {}).get("select"))
        edges = sum(r["structure"]["nonzero_edges"])
        lines.append(f"| {name} | {r.get('chosen_stage', '基线第'+str(r.get('best', {}).get('epoch', ''))+'轮')} | {r['fit']['accuracy']:.3%} | {s['accuracy']:.3%} / {s['cross_entropy']:.5f} | {r['control']['accuracy']:.3%} / {r['control']['cross_entropy']:.5f} | {edges:,} / {edges/433680:.4f}× | {r['structure']['bytes']:,} |")
    lines += ["", f"复用旧CNN证据：验证{cnn['best_select']['accuracy']:.3%}、探索性测试{cnn['exploratory_control']['accuracy']:.3%}；排除padding有效展开边433680。新FC节点固定3136/1568/128/10（4842），逐层初始边2458624/4917248/200704/1280。无节点删除、缩宽或物理清理，无权重共享，共享映射none。", "", "## 完整阶段曲线", "", f"![结构与功能曲线]({stem}_curves.png)", "", "| 分支/阶段 | 真实非零边 | 逐层边 | 验证准确率 / CE | 候选接受 |", "| --- | --- | --- | --- | --- |"]
    for name in ("fixed", "magnitude", "distilled", "random"):
        for r in results[name]["stages"]:
            s = r["select"]
            lines.append(f"| {name}/{r['stage']} | {sum(r['structure']['nonzero_edges']):,} | {r['structure']['nonzero_edges']} | {s['accuracy']:.3%} / {s['cross_entropy']:.5f} | {r['accepted']} |")
            event = r.get("event")
            if event and event.get("accepted") is False:
                cand = event["candidate_select"]
                lines.append(f"| {name}/{r['stage']}被拒候选 | {sum(event['candidate_structure']['nonzero_edges']):,} | {event['candidate_structure']['nonzero_edges']} | {cand['accuracy']:.3%} / {cand['cross_entropy']:.5f} | False、完整回退 |")
    lines += ["", "同stage5匹配对照：distilled与random逐层边数完全相同，蒸馏教师/净40轮相同，验证97.125%对90.500%，因此单种子下保留连接身份有作用。random冻结stage2来自验证相对容差，不把stage2测试96.5%当作23.7万边随机模型成绩。两主臂5/5接受，未发生实际回退；恢复机制只有必要CPU检查证据。", "", "| 分支 | 非输入节点 | 独立参数含bias | 密集分配参数含bias | 输出0至9真实入边数 |", "| --- | --- | --- | --- | --- |"]
    for name, r in results.items():
        e = report["external_acceptance"][name]
        lines.append(f"| {name} | {r['structure']['nodes']} | {r['structure']['independent_parameters']:,} | {r['structure']['allocated_parameters']:,} | {e['output_row_nonzero_edges']} |")
    if saved_audit:
        lines += ["", "主代理只读取BGM文件头、bitmap、FP32值及已保存NPZ，0模型前向，独立核对了逐层真实非零边、节点、输出入度、文件字节、验证/测试准确率/CE/混淆矩阵。按“有非零入边或bias且有非零出边”定义，有效隐藏节点如下；物理节点仍保持4842，不把这些统计记为节点删除。", "", "| 分支 | 有效隐藏节点（逐层） |", "| --- | --- |"]
        for name in results:
            lines.append(f"| {name} | {saved_audit[name]['effective_hidden_nodes']} |")
    lines += ["", "## 成本和实际形式", "", f"设备{summary['registration']['device_name']}；FP32，训练batch64、评价batch512、CPU2线程，dense Linear+零mask。总墙钟{summary['session_wall_seconds']:.3f}秒，包含基线、搜索、回退、导出、checkpoint和探索性测试；重启8轮余弦学习率。不把稀疏边数/参数数当作速度或能耗测量，本轮没有推断延迟、能耗或CSR部署测量。", "", "| 分支 | 尝试更新 / 净更新 | 训练样本暴露 | 教师样本 | 回退 | 训练秒（其中教师秒） | 验证/提议/快照/导出秒 | CUDA峰字节 |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, r in results.items():
        ledger, c = r["ledger"], r.get("costs", {})
        train_sec = c.get("training_seconds", sum(v.get("training_seconds", 0) for v in r.get("rows", [])))
        evaluation = c.get("selection_seconds", sum(v.get("evaluation_seconds", 0) for v in r.get("rows", [])))
        cost_parts = f"{evaluation:.3f}/" + ("未分项/未分项/未分项" if name == "baseline" else f"{c.get('proposal_seconds', 0):.3f}/{c.get('snapshot_seconds', 0):.3f}/{c.get('export_seconds', 0):.3f}")
        lines.append(f"| {name} | {ledger['attempted_updates']} / {ledger['net_updates']} | {ledger['optimization_samples']} | {ledger['teacher_samples']} | {ledger['rollbacks']} | {train_sec:.3f}（{c.get('teacher_seconds', 0):.3f}） | {cost_parts} | {r.get('peak_cuda_memory_bytes')} |")
    lines += ["", "教师成本是训练时间的子项，不能重复相加。分项costs不覆盖checkpoint序列化和其他开销；总墙钟是全部成本账本。baseline只有逐轮训练/评价计时，未独立拆出提议/快照/导出，不将缺失值记成0。各训练ledger含验证与末态训练评价，不含最后1000测试；实际评价总量在每臂ledger上另加1次/1000样本。随机臂采用与distilled相同教师和删量/时序，因此比较的是保留连接身份；两臂各自按验证冻结阶段可能不同，必须查看同阶段曲线，不能把不同冻结规模称严格同最终边数比较。", "", "## 证据、恢复和结论边界", "", f"训练提交`{summary['registration']['training_commit']}`；[执行登记](../../../h1_aligned_edges_v1_execution.md)、[机器可读报告]({stem}.json)、[证据目录]({stem}_evidence/)。每臂完整checkpoint与每阶段.bgm保留在`experiments/runs/h1_aligned_edges_v1/mnist_static/`。从项目根恢复`.venv/Scripts/python.exe -m tools.train_aligned_edges --resume`，已完成会仅返回summary；源/配置不匹配拒绝接续。", "", "所有混淆矩阵、真实非零与mask边、固定节点、独立参数、导出哈希、接受/回退和曲线保存在JSON及预测证据。当前只能支持这一任务、这一个种子下的连接删减证据，不能推出自主发现卷积规律、跨任务迁移、稀疏加速或首创。蒸馏温度/权重、层RMS评分、半数几何预算和类别连通保护均为先验。"]
    (ROOT / "docs/reports/2026-10-08/h1_aligned_edges_v1_mnist_static" / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report["external_acceptance"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
