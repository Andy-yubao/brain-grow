"""仅读取完成的边恢复实验与保存预测；生成中文报告、曲线及证据副本。"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil

from .report_pruning import check_metrics, confusion_table, digest, read_json, saved_metrics

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("fixed", "recover", "random")
COST_KEYS = ("optimization_forward_samples", "optimization_backward_samples", "attempted_updates",
             "candidate_updates", "rolled_back_updates", "fallback_updates", "evaluation_forward_samples")


def make_plot(arms, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError:
        return {"created": False, "reason": "matplotlib不可用；真实逐轮曲线保留在JSON，未安装依赖"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    fixed_bytes = {row["epoch"]: row["structure"]["export_bytes"] for row in arms["fixed"]["rows"]}
    for arm in ARMS:
        rows = arms[arm]["rows"]
        epochs = [row["epoch"] for row in rows]
        axes[0, 0].plot(epochs, [row["select"]["accuracy"] for row in rows], label=arm)
        axes[0, 1].plot(epochs, [row["select"]["loss"] for row in rows], label=arm)
        axes[1, 0].plot(epochs, [sum(row["active_counts"]) for row in rows], label=arm)
        axes[1, 1].plot(epochs, [row["structure"]["export_bytes"] / fixed_bytes[row["epoch"]]
                               for row in rows], label=arm)
    titles = (("Selection accuracy", "Accuracy"), ("Selection cross entropy", "Loss"),
              ("Active connections", "Edges"), ("Export bytes / same-epoch fixed", "Byte ratio"))
    for ax, (title, ylabel) in zip(axes.flat, titles):
        ax.set_title(title)
        ax.set_xlabel("Continuation epoch")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.25)
        ax.legend()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {"created": True, "filename": path.name, "source": "本次summary.arms[*].rows，未运行模型"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="h1_recovery_v1")
    parser.add_argument("--date", default="20261008")
    args = parser.parse_args()
    run = ROOT / "experiments" / "runs" / args.version / "mnist_static"
    summary = read_json(run / "summary.json")
    arms = summary["arms"]
    if set(arms) != set(ARMS) or not all(result["frozen"] for result in arms.values()):
        raise ValueError("仅报告全部完成并冻结的fixed/recover/random三臂")
    registration = read_json(run / "registration.json")
    if registration != summary["registration"]:
        raise ValueError("summary与registration来源不一致")
    config = registration["config"]
    training_commit = read_json(run / "training_commit.json")
    startup = read_json(run / "startup.json")
    freeze = read_json(run / "freeze.json")
    if set(freeze["arms"]) != set(ARMS):
        raise ValueError("freeze未覆盖三臂")
    parent_run = ROOT / config["source_run"]
    parent_summary = read_json(parent_run / "summary.json")
    parent_arm = parent_summary["arms"][config["source_arm"]]
    parent_structure = parent_arm["structure"]
    start_layers = parent_structure["layers"]
    start_edges = [layer["active_edges"] for layer in start_layers]
    start_bytes = parent_structure["export_bytes"]
    if (parent_run / config["source_arm"] / "final_model.bgs").stat().st_size != start_bytes:
        raise ValueError("父末态文件真实字节与父summary不一致")
    parent_edges, parent_bytes = list(start_edges), start_bytes
    stress_initial = read_json(run / "stress_initial.json") if (run / "stress_initial.json").exists() else None
    initial_costs = None
    if stress_initial is not None:
        start_edges = stress_initial["active_counts"]
        if len(start_edges) != len(start_layers) or any(type(n) is not int or n <= 0 for n in start_edges):
            raise ValueError("压力初态逐层边数非法")
        for li, (before, after) in enumerate(zip(parent_edges, start_edges)):
            if after != before - stress_initial["deleted_counts"][li]:
                raise ValueError("压力删除数量与实际初态边数不一致")
        start_layers = [{**layer, "active_edges": count} for layer, count in zip(start_layers, start_edges)]
        start_bytes = (run / "stress_initial_model.bgs").stat().st_size
        for split in ("fit", "select"):
            actual = saved_metrics(run / f"stress_initial_{split}.npz")
            check_metrics(actual, stress_initial[split], run / f"stress_initial_{split}.npz")
        initial_ledger = stress_initial["ledger"]
        initial_costs = {key: initial_ledger[key] for key in COST_KEYS}
        initial_costs.update(evaluation_calls=len(initial_ledger["evaluations"]),
                             evaluation_by_purpose=copy.deepcopy(initial_ledger["evaluations"]))
        expected_initial_samples = stress_initial["fit"]["samples"] + stress_initial["select"]["samples"]
        if (initial_costs["evaluation_calls"] != 2 or initial_costs["evaluation_forward_samples"] != expected_initial_samples
                or sum(item["samples"] for item in initial_ledger["evaluations"]) != expected_initial_samples):
            raise ValueError("压力初态正式评价账本与两份预测不一致")
    reports = ROOT / "reports"
    stem = f"{args.version}_mnist_static_{args.date}"
    evidence = reports / f"{stem}_evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    hashes, predictions, costs, layer_stats, capacity_events = {}, {}, {}, {}, {}
    for arm in ARMS:
        result = arms[arm]
        af = run / arm
        target_folder = evidence / arm
        target_folder.mkdir(exist_ok=True)
        if result["final_epoch"] != config["epochs"] or result["rows"][-1]["epoch"] != config["epochs"]:
            raise ValueError(f"{arm}未完成登记轮数")
        files = {"fit": "fit_final.npz", "select": f"select_{result['final_epoch']:03d}.npz", "confirm": "confirm_final.npz"}
        recorded = {"fit": result["fit"], "select": result["rows"][-1]["select"], "confirm": result["confirmation"]}
        predictions[arm] = {}
        for split, filename in files.items():
            actual = saved_metrics(af / filename)
            check_metrics(actual, recorded[split], af / filename)
            predictions[arm][split] = actual
        if (af / "final_model.bgs").stat().st_size != result["structure"]["export_bytes"]:
            raise ValueError(f"{arm}最终文件真实字节与summary不一致")
        for filename in ("final_model.bgs", *files.values()):
            target = target_folder / filename
            shutil.copyfile(af / filename, target)
            hashes[str(target.relative_to(reports))] = digest(target)
        ledger = result["ledger"]
        costs[arm] = {key: ledger[key] for key in COST_KEYS}
        expected_normal_updates = config["epochs"] * (predictions[arm]["fit"]["samples"] // config["batch_size"])
        costs[arm].update(net_updates=ledger["attempted_updates"] - ledger["rolled_back_updates"],
                          expected_normal_updates=expected_normal_updates,
                          extra_attempts_above_normal=ledger["attempted_updates"] - expected_normal_updates)
        if config["fallback_adaptation"] and costs[arm]["net_updates"] != expected_normal_updates:
            raise ValueError(f"{arm}净更新未达到登记的正常预算")
        evaluated = {}
        for item in ledger["evaluations"]:
            entry = evaluated.setdefault(item["purpose"], {"calls": 0, "samples": 0})
            entry["calls"] += 1
            entry["samples"] += item["samples"]
        if sum(entry["samples"] for entry in evaluated.values()) != ledger["evaluation_forward_samples"]:
            raise ValueError(f"{arm}评价样本账本不一致")
        costs[arm]["evaluation_by_purpose"] = evaluated
        costs[arm]["evaluation_calls"] = sum(entry["calls"] for entry in evaluated.values())
        capacity_events[arm] = {"rows": [{"epoch": row["epoch"],
             "mean_ordinary_fit_loss": row["mean_ordinary_fit_loss"],
             "capacity_proxy_streak": row["capacity_proxy_streak"]} for row in result["rows"]],
             "proposal_slots": len(result["events"]), "accepted": sum(bool(event["accepted"]) for event in result["events"]),
             "reasons": {}, "events": result["events"]}
        for event in result["events"]:
            reason = event["reason"]
            capacity_events[arm]["reasons"][reason] = capacity_events[arm]["reasons"].get(reason, 0) + 1
        layer_stats[arm] = []
        layers = result["structure"]["layers"]
        if len(layers) != len(start_layers):
            raise ValueError("本阶段不允许未登记的层数变化")
        for li, (layer, start) in enumerate(zip(layers, start_layers)):
            if (layer["in_features"], layer["out_features"]) != (start["in_features"], start["out_features"]):
                raise ValueError("本阶段不允许未登记的宽度变化")
            added = layer["active_edges"] - start["active_edges"]
            if added < 0 or added > int(start["active_edges"] * config["maximum_growth_fraction_start"]):
                raise ValueError("实际末态超出逐层恢复增长上限")
            layer_stats[arm].append({**layer, "layer_index": li, "input_nodes": layer["in_features"],
                 "output_nodes": layer["out_features"], "start_active_edges": start["active_edges"],
                 "added_edges": added, "active_density": layer["active_edges"] / layer["total_edges"],
                 "nonzero_density": layer["nonzero_edges"] / layer["total_edges"]})
    if costs["recover"]["net_updates"] != costs["random"]["net_updates"]:
        raise ValueError("recover/random净更新账本失配")
    for main_row, random_row in zip(arms["recover"]["rows"], arms["random"]["rows"]):
        if main_row["epoch"] != random_row["epoch"] or main_row["active_counts"] != random_row["active_counts"] or main_row["updates"] != random_row["updates"]:
            raise ValueError("真实逐轮恢复规模或尝试/回退/fallback更新失配")
    for filename in ("training_commit.json", "registration.json", "startup.json", "freeze.json", "data_manifest.json"):
        target = evidence / filename
        shutil.copyfile(run / filename, target)
        hashes[str(target.relative_to(reports))] = digest(target)
    parent_copy = evidence / "parent_summary.json"
    shutil.copyfile(parent_run / "summary.json", parent_copy)
    hashes[str(parent_copy.relative_to(reports))] = digest(parent_copy)
    if stress_initial is not None:
        for filename in ("stress_initial.json", "stress_initial_model.bgs", "stress_initial_fit.npz", "stress_initial_select.npz"):
            target = evidence / filename
            shutil.copyfile(run / filename, target)
            hashes[str(target.relative_to(reports))] = digest(target)
    differences = {arm: {split: {"accuracy_delta_vs_fixed": predictions[arm][split]["accuracy"] - predictions["fixed"][split]["accuracy"],
                        "loss_delta_vs_fixed": predictions[arm][split]["loss"] - predictions["fixed"][split]["loss"]}
                        for split in ("fit", "select", "confirm")} for arm in ARMS}
    main_random = {split: {"recover_minus_random_accuracy": predictions["recover"][split]["accuracy"] - predictions["random"][split]["accuracy"],
                            "recover_minus_random_loss": predictions["recover"][split]["loss"] - predictions["random"][split]["loss"]}
                   for split in ("fit", "select", "confirm")}
    total = {key: sum(costs[arm][key] for arm in ARMS) for key in COST_KEYS}
    total.update(net_updates=sum(costs[arm]["net_updates"] for arm in ARMS),
                 extra_attempts_above_normal=sum(costs[arm]["extra_attempts_above_normal"] for arm in ARMS),
                 evaluation_calls=sum(costs[arm]["evaluation_calls"] for arm in ARMS))
    if initial_costs is not None:
        for key in COST_KEYS:
            total[key] += initial_costs[key]
        total["evaluation_calls"] += initial_costs["evaluation_calls"]
    growth = {arm: {"start_active_edges": sum(start_edges),
                   "baseline_kind": "stress_initial" if stress_initial is not None else "parent_prune_final",
                   "final_active_edges": sum(layer["active_edges"] for layer in layer_stats[arm]),
                   "added_edges": sum(layer["added_edges"] for layer in layer_stats[arm]),
                   "added_edge_fraction_of_start": sum(layer["added_edges"] for layer in layer_stats[arm]) / sum(start_edges),
                   "source_export_bytes": start_bytes, "final_export_bytes": arms[arm]["structure"]["export_bytes"],
                   "initial_export_bytes": start_bytes,
                   "byte_change_vs_initial": arms[arm]["structure"]["export_bytes"] - start_bytes,
                   "byte_change_vs_source": arms[arm]["structure"]["export_bytes"] - start_bytes,
                   "byte_change_ratio_vs_source": arms[arm]["structure"]["export_bytes"] / start_bytes - 1,
                   "byte_change_vs_fixed": arms[arm]["structure"]["export_bytes"] - arms["fixed"]["structure"]["export_bytes"]}
              for arm in ARMS}
    commit = training_commit.get("gitHead") or training_commit.get("git_head")
    plot = make_plot(arms, reports / f"{stem}_curves.png")
    if plot["created"]:
        hashes[plot["filename"]] = digest(reports / plot["filename"])
    report = copy.deepcopy(summary)
    report.update(training_commit=training_commit, implementation_commit=commit,
                  prediction_metrics=predictions, costs=costs, costs_total=total,
                  layer_statistics=layer_stats, capacity_proxy=capacity_events,
                  differences_vs_fixed=differences, recover_minus_random=main_random,
                  growth_cost=growth, evidence_sha256=hashes, plot=plot,
                  startup_cost=startup, original_run_path=str(run),
                  parent_evidence={"run": str(parent_run), "source_arm": config["source_arm"],
                                   "source_active_counts": parent_edges, "source_export_bytes": parent_bytes},
                  experiment_initial={"kind": "fixed_pruning_stress" if stress_initial is not None else "parent_prune_final",
                                      "active_counts": start_edges, "export_bytes": start_bytes},
                  stress_initial=stress_initial, initial_costs=initial_costs,
                  report_execution="仅读summary/logits/文件长度；0模型前向、反传、更新、GPU检查")
    json_path = reports / f"{stem}.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    labels = {"fit": "拟合", "select": "结构选择", "confirm": "重复探索"}
    title = "# H1固定剪枝压力下的边恢复：实际报告" if stress_initial is not None else "# H1边恢复延续实验：实际报告"
    start_description = ("三臂从父末态构造同一压力初态后延续；" if stress_initial is not None else "三臂从同一父末态延续；")
    lines = [title, "",
             f"{args.date}；全部指标来自已完成运行与保存logits。实现提交：`{commit or '未登记；以源SHA256追溯'}`，"
             "来源为启动前training_commit.json，不使用生成报告时HEAD。", "",
             f"[完整JSON]({stem}.json)保存真实曲线、事件、混淆矩阵、成本与文件哈希。", "",
             "## 任务与机制边界", "",
             start_description + f"父来源为`{config['source_run']}/{config['source_arm']}/resume.pt`，复用源权重、Adam及RNG。"
             f"模型根{config['model_seed']}、数据根{config['data_seed']}，延续各{config['epochs']}轮，固定784→512→256→10及两层ReLU。"
             "原25轮成本不计作本阶段新增成本。", "",
             "本阶段仅使用3200拟合/800选择；1000旧control已经提供前轮探索反馈，本次全部末轮冻结后再作探索记录，"
             "不是独立确认。官方测试未访问，其他封存ID提取0；未新增种子或扩大原ID范围。", "",
             f"普通49批拟合平均损失>{config['capacity_mean_fit_loss']}连续{config['capacity_patience']}轮作为触发代理。"
             "它不能判断真正表达能力不足，也不能排除优化困难。recover按inactive边展开梯度EMA(g²)排序，"
             "恢复identity连接、theta初始0、新槽Adam矩清零；一批适应后用选择集评价。未实现节点增删或线性无关组简化。", "",
             f"每层候选至多原始边数{config['restore_fraction_original']:.2%}，累计至多起始活跃边数{config['maximum_growth_fraction_start']:.0%}；"
             f"至多{config['max_proposals']}slots。候选须满足同轮fixed的1pp/0.05容差，并相对自身候选前改善，且准确率不下降、损失不增加。"
             "恢复允许文件增长，不能以单步压缩判断成功。", "",
             "拒绝先完整回退图、权重、Adam、梯度统计与RNG，再对原图同批适应；额外候选尝试成本保留。"
             "random无条件匹配recover的接受时序和逐层恢复数，仅随机恢复身份；主拒绝slot也先适应回退再fallback。"
             "随机臂的选择集成绩不决定接受，比较的是连接身份而非独立随机搜索。", "",
             "## 实际任务性能", "", "| 臂 | 集合 | 样本 | 准确率 | 交叉熵 |", "| --- | --- | --- | --- | --- |"]
    if stress_initial is not None:
        index = lines.index("## 实际任务性能")
        lines[index:index] = ["## 预登记人为压力初态", "",
            f"固定按父拟合敏感度额外删除各层活跃边{config['additional_prune_fraction']:.0%}，"
            "这是人为压力条件，不是自然发生的容量不足；不按新选择或探索成绩挑选强度，不修改自然恢复轮结果。", "",
            f"父末态{sum(parent_edges)}边→压力初态{sum(start_edges)}边；恢复累计增长上限按压力初态逐层计算，"
            f"不是按原父图计算。压力初态真实文件{start_bytes}字节。", "",
            "| 初态集合 | 样本 | 准确率 | 交叉熵 |", "| --- | --- | --- | --- |",
            *[f"| {labels[split]} | {stress_initial[split]['samples']} | {stress_initial[split]['accuracy']:.4%} | {stress_initial[split]['loss']:.6f} |"
              for split in ("fit", "select")], "",
            f"压力初态正式评价{initial_costs['evaluation_calls']}次/{initial_costs['evaluation_forward_samples']}样本已加入总评价成本，"
            "单列在initial_costs，不漏入各臂账本之外；没有额外优化更新，也不是GPU诊断。", ""]
    for arm in ARMS:
        for split in labels:
            m = predictions[arm][split]
            lines.append(f"| {arm} | {labels[split]} | {m['samples']} | {m['accuracy']:.4%} | {m['loss']:.6f} |")
    lines.extend(["", "采用登记末轮，未用探索集挑选最佳轮；保存预测的准确率、损失、混淆矩阵已与summary核对。", "",
                  "## 逐层结构与增长代价", "",
                  "| 臂 | 层 | 输入→输出节点 | 起始活跃边 | 末态活跃边 | 新增边 | 非零边 | 活跃密度 | 自由参数含偏置 | 分配槽 |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"])
    for arm in ARMS:
        for layer in layer_stats[arm]:
            lines.append(f"| {arm} | {layer['layer_index']+1} | {layer['input_nodes']}→{layer['output_nodes']} | "
                         f"{layer['start_active_edges']} | {layer['active_edges']} | {layer['added_edges']} | {layer['nonzero_edges']} | "
                         f"{layer['active_density']:.4%} | {layer['independent_parameters']} | {layer['allocated_parameter_slots']} |")
    lines.extend(["", "自由参数由活跃独立权重与偏置计数；新增但仍为零的边单独反映在非零边统计。"
                  "隐藏节点、输出节点及总计算节点均保持768/10/778；本轮只恢复边，不能称物理删节点。", "",
                  "| 臂 | 新增边总数 | 相对起始边增长 | 实际文件字节 | 相对本轮初态增减字节 | 相对fixed增减字节 | 臂计时秒 |",
                  "| --- | --- | --- | --- | --- | --- | --- |"])
    for arm in ARMS:
        g = growth[arm]
        lines.append(f"| {arm} | {g['added_edges']} | {g['added_edge_fraction_of_start']:.3%} | {g['final_export_bytes']} | "
                     f"{g['byte_change_vs_source']:+d} | {g['byte_change_vs_fixed']:+d} | {arms[arm]['wall_seconds']:.3f} |")
    lines.extend(["", f"本轮初态活跃边{sum(start_edges)}，初态实际文件{start_bytes}字节。文件同为FP32无损编码，"
                  "包含映射、索引、数值与全部元数据；字节变化也受参数数值与压缩影响，不能全归因于新增边。"
                  "theta和Adam仍分配原535818槽并执行稠密gather矩阵乘，不称推断加速、内存节约、卷积复用或实测能耗改善。", ""])
    if plot["created"]:
        lines.extend([f"![实际逐轮曲线]({plot['filename']})", "",
                      "准确率、损失、活跃边与对应轮fixed字节比均来自本阶段真实记录，延续轮从1计数，不混入父实验25轮。", ""])
    else:
        lines.extend([plot["reason"] + "。", ""])
    lines.extend(["## 实际更新、评价与容量代理", "",
                  "| 臂 | 优化前向/反传样本 | 尝试更新 | 候选适应 | 回退 | fallback | 净更新 | 超正常预算尝试 | 评价调用/样本 |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"])
    for arm in ARMS:
        c = costs[arm]
        lines.append(f"| {arm} | {c['optimization_forward_samples']}/{c['optimization_backward_samples']} | {c['attempted_updates']} | "
                     f"{c['candidate_updates']} | {c['rolled_back_updates']} | {c['fallback_updates']} | {c['net_updates']} | "
                     f"{c['extra_attempts_above_normal']} | {c['evaluation_calls']}/{c['evaluation_forward_samples']} |")
    if initial_costs is not None:
        c = initial_costs
        lines.append(f"| 压力初态（臂外） | {c['optimization_forward_samples']}/{c['optimization_backward_samples']} | {c['attempted_updates']} | "
                     f"{c['candidate_updates']} | {c['rolled_back_updates']} | {c['fallback_updates']} | 0 | 0 | "
                     f"{c['evaluation_calls']}/{c['evaluation_forward_samples']} |")
    lines.extend(["", f"合计新增优化前向/反传{total['optimization_forward_samples']}/{total['optimization_backward_samples']}样本，"
                  f"尝试{total['attempted_updates']}，回退{total['rolled_back_updates']}，fallback {total['fallback_updates']}，"
                  f"净更新{total['net_updates']}；各臂正常预算500净更新。额外候选尝试与回退不从实际成本删除。"
                  f"总评价{total['evaluation_calls']}次/{total['evaluation_forward_samples']}样本，含全部臂及已发生的压力初态评价。", "",
                  "| 臂 | 恢复slots | 接受数 | 实际理由计数 |", "| --- | --- | --- | --- |"])
    for arm in ARMS:
        c = capacity_events[arm]
        reasons = "；".join(f"{key}: {value}" for key, value in c["reasons"].items()) or "无恢复候选"
        lines.append(f"| {arm} | {c['proposal_slots']} | {c['accepted']} | {reasons} |")
    lines.extend(["", "完整JSON保留每轮普通拟合平均损失、容量代理连续计数、候选前后成绩及拒绝原因。未触发或未接受均是实际结果，"
                  "不能写成恢复机制有效；候选前后的改善还包含一批适应本身，必须结合fixed/random差异判断。", "",
                  "| 臂 | 评价用途 | 调用次数 | 样本 |", "| --- | --- | --- | --- |"])
    for arm in ARMS:
        for purpose, entry in costs[arm]["evaluation_by_purpose"].items():
            lines.append(f"| {arm} | {purpose} | {entry['calls']} | {entry['samples']} |")
    if initial_costs is not None:
        for item in initial_costs["evaluation_by_purpose"]:
            lines.append(f"| 压力初态（臂外） | {item['purpose']} | 1 | {item['samples']} |")
    lines.extend(["", f"本次启动成本前向{startup['forward_samples']}、反传{startup['backward_samples']}、更新{startup['updates']}；"
                  "复用既有同尺寸启动证据，未重跑8样本GPU检查。报告生成也没有模型前向、GPU诊断或恢复重放。", "",
                  f"设备{registration['device']}，PyTorch {registration['torch']}，FP32，batch={config['batch_size']}，"
                  f"评价batch={config['evaluation_batch_size']}，CPU线程2；CUDA整次峰值分配{summary.get('peak_cuda_memory_bytes')}字节。", "",
                  f"实际完整本次会话墙钟{summary['current_session_wall_seconds']:.3f}秒。臂计时覆盖每轮普通训练、候选事务/评价及编码，"
                  "排除checkpoint/预测写入、末态拟合及探索评价；不是完整端到端推断计时。事务计时仅按真实候选事件字段保留在JSON，"
                  "未分项计时的开销不得归因具体瓶颈。若经历多个恢复会话，该字段只表示完成时会话，不冒称跨会话总耗时。", "",
                  "runtime_ratio不是本阶段接受或成功门槛，较长搜索时间按成本记录。阶段墙钟预算与停止记录以实际登记为准，"
                  "不以旧3倍门槛否决新机制。", "", "## 效果增量与谨慎判定", "",
                  "| 臂 | 集合 | 相对fixed准确率变化（百分点） | 相对fixed交叉熵变化 |", "| --- | --- | --- | --- |"])
    for arm in ARMS:
        for split in labels:
            delta = differences[arm][split]
            lines.append(f"| {arm} | {labels[split]} | {100*delta['accuracy_delta_vs_fixed']:+.4f} | {delta['loss_delta_vs_fixed']:+.6f} |")
    lines.extend(["", "| 集合 | recover−random准确率（百分点） | recover−random交叉熵 |", "| --- | --- | --- |"])
    for split in labels:
        delta = main_random[split]
        lines.append(f"| {labels[split]} | {100*delta['recover_minus_random_accuracy']:+.4f} | {delta['recover_minus_random_loss']:+.6f} |")
    accepted = capacity_events["recover"]["accepted"]
    delta = differences["recover"]["select"]
    random_delta = main_random["select"]
    if not capacity_events["recover"]["proposal_slots"]:
        conclusion = "本轮未触发恢复候选，不能据此检验恢复边效果或断言容量充足；本次主要是已有剪枝图继续训练的证据。"
    elif not accepted:
        conclusion = "本轮恢复候选均未接受，未获得满足登记规则的恢复增量；不把失败候选或原图适应成果归因于边恢复。"
    elif delta["accuracy_delta_vs_fixed"] >= 0 and delta["loss_delta_vs_fixed"] < 0 and random_delta["recover_minus_random_accuracy"] >= 0 and random_delta["recover_minus_random_loss"] < 0:
        conclusion = "本轮任务恢复在选择损失上优于fixed及匹配random，且准确率不低，是连接身份恢复可能有增量的单根信号；需同时衡量新增边与文件代价。"
    else:
        conclusion = "本轮有恢复接受，但末态效果未同时一致优于fixed与匹配random；应报告完整权衡，不能概括为容量恢复已验证有效。"
    if stress_initial is not None:
        parent_accuracy = parent_arm["confirmation"]["accuracy"]
        restored_accuracy = predictions["recover"]["confirm"]["accuracy"]
        conclusion += (f"本轮重复探索准确率{restored_accuracy:.2%}，父prune末态为{parent_accuracy:.2%}；"
                       "仍低于父表现，未恢复到父性能，不能称恢复成功。" if restored_accuracy < parent_accuracy else
                       f"本轮重复探索{restored_accuracy:.2%}，父末态{parent_accuracy:.2%}；这仍是人为压力与重复探索比较。")
    lines.extend(["", conclusion, "",
                  "选择集反复用于候选决策，旧control也已提供探索反馈；唯一根11不支持统计显著性、稳健性或普适性结论。"
                  "不声称发现卷积、节点基组或能耗改善。", "", "## 证据与恢复", "",
                  f"[重要证据索引]({stem}_evidence/README.md)保存真实模型、三类最终预测、training_commit、registration、"
                  "复用启动记录、freeze、数据清单和父summary。文件哈希见完整JSON。", "",
                  f"原完整轮恢复checkpoint、日志及逐轮预测保留于`{run}`；大型resume.pt未复制进报告证据。"
                  "源码与配置哈希一致时可恢复；已完成入口不会重复训练或确认。", "", "```powershell",
                  ".venv/Scripts/python.exe -u -B -m tools.train_recovery_stress --resume" if stress_initial is not None else
                  ".venv/Scripts/python.exe -u -B -m tools.train_recovery --resume", "```", "",
                  "## 混淆矩阵", ""])
    if stress_initial is not None:
        for split in ("fit", "select"):
            lines.extend([f"### 压力初态：{labels[split]}", "", *confusion_table(stress_initial[split]["confusion_matrix"]), ""])
    for arm in ARMS:
        for split in labels:
            lines.extend([f"### {arm}：{labels[split]}", "", *confusion_table(predictions[arm][split]["confusion_matrix"]), ""])
    md_path = reports / f"{stem}.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    (evidence / "README.md").write_text(
        "# H1边恢复实际证据\n\n三臂final_model.bgs、fit_final.npz、最后一轮select及confirm_final.npz按原字节复制，"
        "SHA256在上级完整JSON。FP32共享图编码包含参数和全部结构映射；本阶段未变层宽。\n\n"
        "training_commit为启动前实际实现提交与来源；registration为源/config/父checkpoint哈希；"
        "startup为复用记录，新增启动成本0；freeze证明三臂先冻结；manifest为授权原ID与来源追溯。"
        "parent_summary保留父实验真实结构与字节，父实验成本不是本次新增成本。\n\n"
        + ("stress_initial.json、stress_initial_model.bgs及初态fit/select预测保留实际压力基准；"
           "逐层恢复增长与字节代价相对此初态计算，不相对父prune计算。两次/4000样本正式初态评价已纳入总成本。"
           "压力为预登记固定额外90%剪枝，非自然容量不足，不按成绩选择强度。\n\n" if stress_initial is not None else "") +
        "1000旧control已在前轮使用，本次只作重复探索，非独立确认；官方测试与其他封存ID没有解封。"
        "报告只读保存预测，不运行模型，也未读取训练checkpoint。\n\n"
        f"原日志、逐轮预测与完整恢复checkpoint保留在`{run}`。\n", encoding="utf-8")
    print(json.dumps({"report": str(md_path), "json": str(json_path), "plot": plot,
                      "costs": total, "accepted": accepted}, ensure_ascii=False))


if __name__ == "__main__":
    main()
