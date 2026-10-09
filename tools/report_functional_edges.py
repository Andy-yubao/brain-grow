"""只读功能替代实验证据，生成中文报告和逐层连接比较图。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAMES = {"compensation": "功能补偿删边", "gates": "任务连接门控",
         "gates_control": "零数量代价对照", "gates_zero": "零数量代价对照",
         "rewire": "任务驱动净换边", "relearn": "重新学习稀疏表示"}
CNN = [28224, 225792, 200704, 1280]
PARENT = [271888, 252735, 5694, 138]
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="experiments/runs/h1_functional_edges_v1/mnist_circular")
    parser.add_argument("--out", default="docs/reports/2026-10-09/h1_functional_edges_v1")
    args = parser.parse_args()
    run, out = ROOT/args.run, ROOT/args.out
    results = {p.parent.name: json.loads(p.read_text(encoding="utf-8"))
               for p in sorted(run.glob("*/result.json"))}
    if not results:
        raise SystemExit("尚无完成结果，不能生成实验结论")
    out.mkdir(parents=True, exist_ok=True)
    for filename in ("registration.json", "data_manifest.json", "check.json", "summary.json"):
        source = run/filename
        if filename == "check.json" and not source.exists():
            source = run.parent/filename
        if source.exists():
            shutil.copyfile(source, out/filename)
    (out/"result.json").write_text(json.dumps(results, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    summary = json.loads((run/"summary.json").read_text(encoding="utf-8")) if (run/"summary.json").exists() else {}
    registration = json.loads((run/"registration.json").read_text(encoding="utf-8"))
    rows, sections = [], []
    for name, r in results.items():
        title = NAMES.get(name, name)
        edges = r["structure"]["nonzero_edges"]
        v = r["validation"]
        qualified = v["accuracy"] >= .9309375 and v["cross_entropy"] <= .28852926
        success = qualified and edges[0] <= 56448 and sum(edges) < sum(PARENT)
        wall = r.get("wall_seconds", r.get("elapsed_seconds", 0.))
        rows.append(f"| {title} | {edges[0]:,} | {edges[1]:,} | {sum(edges):,} | {v['accuracy']*100:.4f}% | {v['cross_entropy']:.6f} | {'是' if qualified else '否'} | {'是' if success else '否'} | {wall:.2f} |")
        structure = r["structure"]
        details = {key: r[key] for key in ("stop", "selected_full_checkpoint", "costs", "peak_cuda_memory_bytes", "paired_wall_seconds", "paired_costs") if key in r}
        details["ledger_measured"] = {k: v for k, v in r.get("ledger", {}).items()
                                     if k not in ("training_position_histogram", "net_updates", "teacher_samples")}
        peaks = [json.loads(p.read_text(encoding="utf-8")).get("peak_cuda_memory_bytes", 0)
                 for p in (run/name).glob("slot_*/result.json")]
        if peaks:
            details["maximum_slot_gate_peak_bytes"] = max(peaks)
        sections.append(f"### {title}\n\n各层真实非零边：`{edges}`。前两层为循环CNN的{edges[0]/CNN[0]:.3f}/{edges[1]/CNN[1]:.3f}倍。BGM导出字节：{structure.get('bytes', '未记录')}；独立参数：{structure.get('independent_parameters', '未记录')}。\n\n运行与恢复记录：\n\n```json\n{json.dumps(details, ensure_ascii=False, indent=2)}\n```\n")
        if "training_fixed_one_position" in r:
            train = r["training_fixed_one_position"]
            sections.append(f"固定单位置训练准确率{train['accuracy']*100:.4f}%、CE{train['cross_entropy']:.6f}。训练与验证混淆矩阵保存在完整结果JSON；不称为所有循环位置的训练总体性能。\n")
        if r.get("indegrees"):
            sections.append("| 层 | 平均入度 | 无入边物理节点 | 入度分位（0/10/25/50/75/90/100%） |\n| --- | --- | --- | --- |")
            for degree in r["indegrees"]:
                sections.append(f"| {degree['layer']} | {degree['mean']:.3f} | {degree['zero_input_nodes']} | `{degree['quantiles']}` |")
            sections.append("\n")
        if name == "compensation":
            first = r.get("events", [{}])[0] if r.get("events") else {}
            common = first.get("epochs")
            if common:
                vfirst = first["validation"]
                sections.append(f"首候选共同{common}轮比较（匹配原父、数据序列与Adam重置）：\n\n| 分支 | 准确率 | CE |\n| --- | --- | --- |\n| 补偿 | {vfirst['accuracy']*100:.4f}% | {vfirst['cross_entropy']:.6f} |")
            for control in ("uncompensated", "continue_training"):
                path = run/name/f"{control}_result.json"
                if path.exists():
                    shutil.copyfile(path, out/path.name)
                    c = json.loads(path.read_text(encoding="utf-8"))
                    v = c["validation"]
                    label = "同删除集合无补偿" if control == "uncompensated" else "不删边继续训练"
                    point = next((p for p in c.get("validation_points", []) if p["epoch"] == common), None)
                    if point:
                        vp = point["validation"]
                        sections.append(f"| {label} | {vp['accuracy']*100:.4f}% | {vp['cross_entropy']:.6f} |")
            sections.append("\n差别为单种子旧验证观察，不作显著性或跨任务优势判断。\n")
            if first:
                sections.append(f"首候选直接删除`{first.get('direct_deleted_per_layer')}`，闭合另删`{first.get('closure',{}).get('removed_nonzero_per_layer')}`。拟合/诊断补偿前后误差及被删权重分位保存在事件JSON。首候选保留{first.get('epochs',0)*50}次任务更新；其余失败和对照更新均计入尝试成本。\n")
            for control in ("uncompensated", "continue_training"):
                path = run/name/f"{control}_result.json"
                if path.exists():
                    c = json.loads(path.read_text(encoding="utf-8"))
                    v = c["validation"]
                    label = "同删除集合无补偿" if control == "uncompensated" else "不删边继续训练"
                    sections.append(f"{label}：运行{c['epochs']}轮，最终验证{v['accuracy']*100:.4f}%/CE{v['cross_entropy']:.6f}；全部共同验证点见[{label}记录]({path.name})。应在共同轮次比较，不能用48轮对照与提前接受候选的末态直接归因。\n")
        # 保留原事件以免把失败、继续训练和被接受的操作混为一条曲线。
        if r.get("events"):
            (out/f"{name}_events.json").write_text(json.dumps(r["events"], ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
            sections.append(f"候选、硬化和失败记录见[{title}事件]({name}_events.json)。\n")

    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    all_edges = {"循环CNN参照": CNN, "共同FC父模型": PARENT}
    all_edges.update({NAMES.get(k, k): r["structure"]["nonzero_edges"] for k, r in results.items()})
    width = .8/len(all_edges)
    pos = np.arange(4)
    for i, (label, values) in enumerate(all_edges.items()):
        ax.bar(pos-.4+width/2+i*width, values, width=width, label=label)
    ax.set_xticks(pos, ["第一层", "第二层", "第三层", "输出层"])
    ax.set_yscale("log")
    ax.set_ylabel("真实非零连接数（对数轴）")
    ax.grid(axis="y", alpha=.2)
    ax.legend(fontsize=8)
    fig.savefig(out/"layer_edges.png", dpi=160)
    plt.close(fig)

    report = f"""# H1功能补偿与任务门控：完成结果

2026-10-09。以下仅汇总已完成分支，运行状态及未完成机制以[状态](../../../status.md)为准。实验协议见[执行预登记](../../../h1_functional_edges_v1_execution.md)，完整机器证据见[result.json](result.json)。

## 性能与连接

| 分支 | 第一层边 | 第二层边 | 总边 | 验证准确率 | 验证CE | 功能合格 | 首层达标且净省边 | 端到端秒 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 共同父 | 271,888 | 252,735 | 530,455 | 93.2344% | 0.221395 | 是 | 否 | 历史成本另计 |
{chr(10).join(rows)}

第一层≤56448只是外部评价；功能门槛固定93.09375%/CE≤0.28852926。循环CNN节点与FC逐层对齐，四层展开边`{CNN}`，总456000；不要与静态CNN433680混用。节点固定4842，无物理删点，没有严格参数共享。

![各层真实连接对比](layer_edges.png)

## 本轮结论

补偿接受首槽，第二槽三次尝试全部平台失败。损伤预算5%→2.5%→1.25%仍由50%操作上限先截断，删边集合及验证轨迹相同；后两次是重复成本，不能算额外有效证据。门控32/56/80轮硬化均未跨过关闭阈值；80轮预期开启约99.405%，实际净删边为0。零数量代价对照也没有删边。

净换边三槽虽保持功能，最终新增大于删除，全部回退。重新学习32/56轮硬结构在恢复预算内仍改善，记优化未完成；80轮硬结构恢复48轮最终验证约70.4063%/CE0.925036，不合格，保留旧父。重初始化使用物理层完整fan-in，未按稀疏入度重标定，且第40轮确实在线改变支持；本轮不能证明该掩码容量不足。

最终最小合格模型来自补偿首槽，共255837边（较共同父减少51.77%），第一层仍172936、为循环CNN约6.13倍；本轮第一层目标未达到。按用户授权继续[跨节点激活功能迁移](../../../h1_activation_migration_v1_execution.md)，该新机制状态见项目状态页。

入口端到端{summary.get('end_to_end_wall_seconds', 0):.3f}秒，含首次数据准备、所有对照与失败搜索。累计尝试更新50400次，其中补偿首槽实际保留1200次新增更新；其他分支最终回退，不能把尝试总量称为最终参数保留更新。设备{registration['device_name']}，Torch {registration['torch']}，batch64/评价512，FP32任务权重/FP64重构、CPU2线程、稠密掩码执行；总体峰来自补偿2131899392字节，约2.0GiB，各阶段均未触及8GiB工程限制。

## 分支与成本

{chr(10).join(sections)}

## 证据范围

只使用既有3200训练/800验证原始ID，验证各ID8位置不是6400独立笔迹。验证用于结构接受与选模，官方测试、历史探索测试和其余封存ID关闭。没有独立泛化验证，不称全局最优或自然稀疏极限。训练采用稠密掩码执行，真实边/导出字节变化不代表实测加速或能耗。

补偿拟合和诊断样本均来自训练ID；诊断不参与候选排序，端到端任务训练使用全部3200。软门开启概率仅为优化量，实际结构以硬化及闭合后的真实非零边计数。对照、失败及恢复成本由各分支保存的事件和成本记录说明，不隐藏回退尝试。

旧基础ledger中的`net_updates`和`teacher_samples`在本入口未维护，零值不是测量结果；报告不引用这些字段作为保留更新或蒸馏样本数。使用尝试更新和具体联合/适应轮次解释成本。失败搜索秒是其他求解/训练/验证/保存成本的重叠子集，不能再次相加；分项未覆盖的读取/快照等成本保留在端到端残差。设备、精度、批量和完整来源见运行登记。

同一runner的对照/嵌套ledger可能累计之前分支，不能逐文件直接求和。升级嵌套slot在无接受时的`selected_full_checkpoint`标记沿用最初530455父路径，实际fallback是传入的补偿父；该来源标记不准确，恢复使用各臂实际`model_full.pt`及其SHA256。升级顶层显存字段是末槽门控peak；上文另列所有slot的门控峰最大值，不冒称该字段单独覆盖全部升级过程。
"""
    (out/"report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"report": str(out/"report.md"), "completed_results": list(results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
