"""读取多节点简化末态预测、事件与文件，生成中文报告；不加载模型或checkpoint。"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import struct

import numpy as np

from .report_pruning import check_metrics, confusion_table, digest, read_json, saved_metrics

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("fixed", "reduce", "random")
COST_KEYS = ("optimization_forward_samples", "optimization_backward_samples", "attempted_updates",
             "candidate_updates", "rolled_back_updates", "fallback_updates", "evaluation_forward_samples",
             "activation_forward_samples")
TIMING_KEYS = ("search_wall_seconds", "search_cpu_process_seconds", "holdout_wall_seconds",
               "construction_encoding_decode_wall_seconds", "candidate_adaptation_evaluation_wall_seconds")


def file_widths(path):
    """读取实际BGS1文件架构头；不构造模型、不解码权重、不执行前向。"""
    with Path(path).open("rb") as source:
        prefix = source.read(8)
        if len(prefix) != 8 or prefix[:4] != b"BGS1":
            raise ValueError(f"不是BGS1导出：{path}")
        header_size = struct.unpack("<I", prefix[4:])[0]
        if header_size > Path(path).stat().st_size - 8:
            raise ValueError("导出头长度非法")
        header = json.loads(source.read(header_size).decode("utf-8"))
    return header["widths"]


def coefficient_evidence(event):
    proposal = event.get("node_proposal")
    if proposal is None:
        if event["accepted"]:
            raise ValueError("已接受节点简化缺少实际系数证据")
        return None
    coefficients = np.asarray(proposal["coefficients"], dtype=np.float64)
    kept, dropped = proposal["kept_indices"], proposal["dropped_indices"]
    if coefficients.shape != (len(kept), len(dropped)) or not np.isfinite(coefficients).all():
        raise ValueError("保存的节点系数矩阵非法")
    if not np.array_equal(coefficients, coefficients.astype(np.float32).astype(np.float64)):
        raise ValueError("系数不符合已登记固定C32协议")
    exact_counts = np.count_nonzero(coefficients, axis=0)
    significant_counts = np.count_nonzero(np.abs(coefficients) > 1e-8, axis=0)
    holdout_abs = event.get("holdout_absolute_residuals", [])
    holdout_rel = event.get("holdout_relative_residuals_fixed_float32_coefficients", [])
    fit_abs = event.get("fit_absolute_residuals_float64_lstsq", proposal["absolute_residuals"])
    fit_rel = event.get("fit_relative_residuals_float64_lstsq", proposal["relative_residuals"])
    columns = []
    for column, node in enumerate(dropped):
        nz = np.flatnonzero(coefficients[:, column] != 0)
        # C32全零且留出误差范数为零，才能确认该留出集合上的零激活；不推出全域零。
        zero_on_holdout = bool(exact_counts[column] == 0 and len(holdout_abs) == len(dropped)
                               and holdout_abs[column] == 0)
        columns.append({"dropped_node_index": node, "nonzero_coefficients": int(exact_counts[column]),
                        "coefficients_abs_above_1e_minus_8": int(significant_counts[column]),
                        "nonzero_kept_node_indices": [kept[int(index)] for index in nz],
                        "coefficient_column_all_zero": bool(exact_counts[column] == 0),
                        "zero_activation_verified_on_640_holdout": zero_on_holdout,
                        "fit_absolute_residual_float64_lstsq": fit_abs[column],
                        "fit_relative_residual_float64_lstsq": fit_rel[column],
                        "holdout_absolute_residual_fixed_C32": holdout_abs[column] if len(holdout_abs) == len(dropped) else None,
                        "holdout_relative_residual_fixed_C32": holdout_rel[column] if len(holdout_rel) == len(dropped) else None})
    return {"epoch": event["epoch"], "proposal": event["proposal"], "layer_index": event["layer_index"],
            "accepted": event["accepted"], "basis_indices": proposal["basis_indices"],
            "basis_count": len(proposal["basis_indices"]), "kept_count": len(kept), "dropped_count": len(dropped),
            "coefficient_shape": list(coefficients.shape), "columns": columns,
            "multi_nonzero_coefficient_columns": int(np.sum(exact_counts >= 2)),
            "multi_coefficients_above_1e_minus_8_columns": int(np.sum(significant_counts >= 2)),
            "all_zero_coefficient_columns": int(np.sum(exact_counts == 0)),
            "zero_activation_verified_on_holdout_columns": sum(column["zero_activation_verified_on_640_holdout"] for column in columns),
            "fixed_precision": proposal.get("coefficients_fixed_precision"),
            "exact_nonzero_rule": "系数!=0；额外abs>1e-8统计仅为描述，不是秩或功能独立性证明",
            "scope": "仅保存拟合/640留出证据；全零系数不自动证明全输入域死节点"}


def make_plot(arms, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError:
        return {"created": False, "reason": "matplotlib不可用，完整真实曲线保留JSON，未安装依赖"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    fixed_bytes = {row["epoch"]: row["structure"]["export_bytes"] for row in arms["fixed"]["rows"]}
    for arm in ARMS:
        rows = arms[arm]["rows"]
        epochs = [row["epoch"] for row in rows]
        axes[0, 0].plot(epochs, [row["select"]["accuracy"] for row in rows], label=arm)
        axes[0, 1].plot(epochs, [row["select"]["loss"] for row in rows], label=arm)
        axes[1, 0].plot(epochs, [sum(row["widths"][1:-1]) for row in rows], label=arm)
        axes[1, 1].plot(epochs, [row["structure"]["export_bytes"] / fixed_bytes[row["epoch"]] for row in rows], label=arm)
    for ax, (title, ylabel) in zip(axes.flat, (("Selection accuracy", "Accuracy"), ("Selection cross entropy", "Loss"),
                                             ("Physical hidden width", "Hidden nodes"), ("Export / same-epoch fixed", "Byte ratio"))):
        ax.set_title(title)
        ax.set_xlabel("Continuation epoch")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.25)
        ax.legend()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {"created": True, "filename": path.name, "source": "本次summary.arms[*].rows；未执行模型"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="h1_nodes_v1")
    parser.add_argument("--date", default="20261008")
    args = parser.parse_args()
    run = ROOT / "experiments/runs" / args.version / "mnist_static"
    summary = read_json(run / "summary.json")
    arms = summary["arms"]
    registration = read_json(run / "registration.json")
    if registration != summary["registration"] or set(arms) != set(ARMS) or not all(result["frozen"] for result in arms.values()):
        raise ValueError("要求来源一致且三臂全部冻结完成的节点实验")
    config = registration["config"]
    initial_widths = registration["initial_widths"]
    training_commit = read_json(run / "training_commit.json")
    startup = read_json(run / "startup.json")
    freeze = read_json(run / "freeze.json")
    if set(freeze["arms"]) != set(ARMS):
        raise ValueError("freeze未覆盖三臂")
    reports = ROOT / "reports"
    stem = f"{args.version}_mnist_static_{args.date}"
    evidence = reports / f"{stem}_evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    predictions, costs, structures, reconstruction, hashes = {}, {}, {}, {}, {}
    for arm in ARMS:
        result = arms[arm]
        af, destination = run / arm, evidence / arm
        destination.mkdir(exist_ok=True)
        if result["final_epoch"] != config["epochs"] or len(result["rows"]) != config["epochs"]:
            raise ValueError(f"{arm}未完成预算轮数")
        files = {"fit": "fit_final.npz", "select": f"select_{result['final_epoch']:03d}.npz", "confirm": "confirm_final.npz"}
        recorded = {"fit": result["fit"], "select": result["rows"][-1]["select"], "confirm": result["confirmation"]}
        predictions[arm] = {}
        for split, filename in files.items():
            actual = saved_metrics(af / filename)
            check_metrics(actual, recorded[split], af / filename)
            predictions[arm][split] = actual
        model_path = af / "final_model.bgs"
        widths = file_widths(model_path)
        if widths != result["widths"] or widths != result["rows"][-1]["widths"]:
            raise ValueError("实际模型头与末态物理宽度记录失配")
        structure = result["structure"]
        if model_path.stat().st_size != structure["export_bytes"]:
            raise ValueError("实际导出文件长度与summary不一致")
        slots = sum(a*b+b for a, b in zip(widths, widths[1:]))
        if slots != structure["model_total"]["allocated_parameter_slots"]:
            raise ValueError("物理宽度与分配参数槽计数失配")
        layers = []
        for li, layer in enumerate(structure["layers"]):
            if (layer["in_features"], layer["out_features"]) != (widths[li], widths[li+1]):
                raise ValueError("逐层结构与物理宽度不一致")
            layers.append({**layer, "layer_index": li, "active_density": layer["active_edges"] / layer["total_edges"],
                           "nonzero_density": layer["nonzero_edges"] / layer["total_edges"]})
        structures[arm] = {"widths": widths, "initial_widths": initial_widths,
                           "hidden_nodes": sum(widths[1:-1]), "output_nodes": widths[-1],
                           "computing_nodes": sum(widths[1:]), "hidden_nodes_removed": sum(initial_widths[1:-1])-sum(widths[1:-1]),
                           "allocated_parameter_slots": slots, "layers": layers, "model_total": structure["model_total"],
                           "export_bytes": structure["export_bytes"]}
        for filename in ("final_model.bgs", *files.values()):
            target = destination / filename
            shutil.copyfile(af / filename, target)
            hashes[str(target.relative_to(reports))] = digest(target)
        ledger = result["ledger"]
        costs[arm] = {key: ledger[key] for key in COST_KEYS}
        net = ledger["attempted_updates"] - ledger["rolled_back_updates"]
        expected = config["epochs"] * (predictions[arm]["fit"]["samples"] // config["batch_size"])
        if net != expected:
            raise ValueError("正常净更新预算失配")
        evaluations = {}
        for item in ledger["evaluations"]:
            entry = evaluations.setdefault(item["purpose"], {"calls": 0, "samples": 0})
            entry["calls"] += 1
            entry["samples"] += item["samples"]
        extractions = ledger["activation_extractions"]
        if (sum(item["samples"] for item in ledger["evaluations"]) != ledger["evaluation_forward_samples"]
                or sum(item["samples"] for item in extractions) != ledger["activation_forward_samples"]):
            raise ValueError("评价或激活提取样本账本不一致")
        costs[arm].update(net_updates=net, expected_normal_updates=expected,
                          extra_attempts=ledger["attempted_updates"]-expected,
                          evaluation_calls=len(ledger["evaluations"]), evaluation_by_purpose=evaluations,
                          activation_extraction_calls=len(extractions), activation_extractions=extractions,
                          adam_reset_epochs=ledger["adam_reset_epochs"],
                          timings={key: sum(event.get(key, 0) for event in result["events"]) for key in TIMING_KEYS})
        reconstruction[arm] = [details for event in result["events"] if (details := coefficient_evidence(event)) is not None]
    for main_row, random_row in zip(arms["reduce"]["rows"], arms["random"]["rows"]):
        if (main_row["epoch"] != random_row["epoch"] or main_row["widths"] != random_row["widths"]
                or main_row["updates"] != random_row["updates"] or main_row["adam_reset_epochs"] != random_row["adam_reset_epochs"]):
            raise ValueError("随机臂实际逐轮宽度、尝试/回退/fallback或Adam重置失配")
    if any(costs[arm]["adam_reset_epochs"] != costs["fixed"]["adam_reset_epochs"] for arm in ARMS):
        raise ValueError("三臂Adam重置日程失配")
    for filename in ("training_commit.json", "registration.json", "startup.json", "freeze.json", "data_manifest.json"):
        target = evidence / filename
        shutil.copyfile(run / filename, target)
        hashes[str(target.relative_to(reports))] = digest(target)
    if (run / "session_cost.json").exists():
        target = evidence / "session_cost.json"
        shutil.copyfile(run / "session_cost.json", target)
        hashes[str(target.relative_to(reports))] = digest(target)
    differences = {arm: {split: {"accuracy_delta_vs_fixed": predictions[arm][split]["accuracy"]-predictions["fixed"][split]["accuracy"],
                                "loss_delta_vs_fixed": predictions[arm][split]["loss"]-predictions["fixed"][split]["loss"]}
                        for split in ("fit", "select", "confirm")} for arm in ARMS}
    random_difference = {split: {"reduce_minus_random_accuracy": predictions["reduce"][split]["accuracy"]-predictions["random"][split]["accuracy"],
                                "reduce_minus_random_loss": predictions["reduce"][split]["loss"]-predictions["random"][split]["loss"]}
                         for split in ("fit", "select", "confirm")}
    total = {key: sum(costs[arm][key] for arm in ARMS) for key in COST_KEYS}
    total.update({key: sum(costs[arm][key] for arm in ARMS) for key in
                  ("net_updates", "extra_attempts", "evaluation_calls", "activation_extraction_calls")})
    evidence_totals = {}
    for arm in ARMS:
        accepted = [event for event in reconstruction[arm] if event["accepted"]]
        evidence_totals[arm] = {"accepted_slots": sum(event["accepted"] for event in arms[arm]["events"]),
            "accepted_coefficient_columns": sum(event["dropped_count"] for event in accepted),
            "accepted_multi_nonzero_coefficient_columns": sum(event["multi_nonzero_coefficient_columns"] for event in accepted),
            "accepted_multi_coefficients_above_1e_minus_8_columns": sum(event["multi_coefficients_above_1e_minus_8_columns"] for event in accepted),
            "accepted_all_zero_coefficient_columns": sum(event["all_zero_coefficient_columns"] for event in accepted),
            "accepted_zero_activation_verified_on_holdout_columns": sum(event["zero_activation_verified_on_holdout_columns"] for event in accepted)}
    simplification = {arm: {"physical_hidden_nodes_removed": structures[arm]["hidden_nodes_removed"],
        "slots_reduction_vs_fixed": structures["fixed"]["allocated_parameter_slots"]-structures[arm]["allocated_parameter_slots"],
        "storage_reduction_vs_fixed": 1-structures[arm]["export_bytes"]/structures["fixed"]["export_bytes"],
        "task_preserved_vs_fixed_select": differences[arm]["select"]["accuracy_delta_vs_fixed"] >= -config["accuracy_tolerance"]
          and differences[arm]["select"]["loss_delta_vs_fixed"] <= config["loss_tolerance"]} for arm in ARMS}
    commit = training_commit.get("gitHead") or training_commit.get("git_head")
    plot = make_plot(arms, reports / f"{stem}_curves.png")
    if plot["created"]:
        hashes[plot["filename"]] = digest(reports / plot["filename"])
    report = copy.deepcopy(summary)
    report.update(training_commit=training_commit, implementation_commit=commit, prediction_metrics=predictions,
        actual_structure=structures, costs=costs, costs_total=total, reconstruction_evidence=reconstruction,
        reconstruction_totals=evidence_totals, simplification=simplification, differences_vs_fixed=differences,
        reduce_minus_random=random_difference, plot=plot, evidence_sha256=hashes, startup_cost=startup,
        original_run_path=str(run), report_execution="保存logits/模型头/实际字节/账本只读；0模型前向反传更新，无checkpoint加载")
    json_path = reports / f"{stem}.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    labels = {"fit": "拟合", "select": "结构选择", "confirm": "重复探索"}
    lines = ["# H1多节点激活基组简化：实际报告", "",
        f"{args.date}；真实结果来自本阶段保存预测、事件和模型文件。启动前实现提交`{commit or '未登记；以源SHA256追溯'}`，"
        "不使用报告生成时HEAD。", "", f"[完整JSON]({stem}.json)保留曲线、事件、系数、成本与文件哈希。", "",
        "## 任务与方法边界", "",
        f"从`{config['source_run']}/{config['source_arm']}/resume.pt`的普通FC末态延续，非剪枝末态；"
        f"初始{'→'.join(map(str, initial_widths))}，模型根{config['model_seed']}、数据根{config['data_seed']}。"
        f"fixed/reduce/random各{config['epochs']}轮，同父权重/Adam/RNG及逐轮批次；不重训父25轮。", "",
        "3200授权fit内每数字前256样本用于系数拟合、后64留出，合计2560/640，索引在登记及数据清单；"
        "640留出只验证固定系数、不重新求解。800结构选择反复用于接受，1000旧control已有探索反馈，"
        "全部臂末轮冻结后仅重复探索；官方测试与其他封存ID未解封，无新增种子。", "",
        "reduce在ReLU激活上用CPU float64贪心选择基列、无截距lstsq估计多节点重构；拟合后一次固定为C32，"
        "出边补偿与640留出残差均使用固定C32。矩阵存为float64仅保存这些FP32值，不表示重新高精度拟合。"
        "不是严格极大线性无关组，也不证明全输入域等价。", "",
        "random继承reduce接受slot、层及实际删除数，无条件均匀选择节点身份；使用所有保留列进行自己的lstsq拟合补偿，"
        "不采用主臂贪心基组限制，留出残差只报告。这是带拟合补偿的随机身份对照，求解空间与主臂不同；"
        "不能将差异单独归因于节点选择，更不能称无补偿随机删节点。", "",
        f"拟合与固定C32留出相对残差阈值{config['max_relative_residual']}；每slot最多{config['max_remove']}节点。"
        "主候选需物理缩宽、完整实际文件减少、同轮fixed效果容差保持；不要求优于fixed。"
        "新尺寸图使用新Adam，三臂同候选轮统一重置；已适应拒绝再原图fallback，额外成本不回退。", "",
        "## 真实任务成绩", "", "| 臂 | 集合 | 样本 | 准确率 | 交叉熵 |", "| --- | --- | --- | --- | --- |"]
    for arm in ARMS:
        for split in labels:
            m = predictions[arm][split]
            lines.append(f"| {arm} | {labels[split]} | {m['samples']} | {m['accuracy']:.4%} | {m['loss']:.6f} |")
    lines.extend(["", "所有准确率、损失、混淆矩阵由保存logits重算并核对summary，采用登记末轮，不按探索成绩挑轮。", "",
        "## 物理宽度、节点与实际文件", "",
        "| 臂 | 实际导出宽度 | 隐藏/输出/总计算节点 | 删除隐藏节点 | 自由参数含偏置 | 分配参数槽 | 文件字节 | 相对fixed文件减少 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |"])
    for arm in ARMS:
        s = structures[arm]
        lines.append(f"| {arm} | {'→'.join(map(str, s['widths']))} | {s['hidden_nodes']}/{s['output_nodes']}/{s['computing_nodes']} | "
            f"{s['hidden_nodes_removed']} | {s['model_total']['independent_parameters']} | {s['allocated_parameter_slots']} | "
            f"{s['export_bytes']} | {simplification[arm]['storage_reduction_vs_fixed']:.3%} |")
    lines.extend(["", "已读取真实BGS1文件架构头核对物理宽度与槽数；没有构造模型或运行前向。"
        "同为FP32无损编码，字节含参数、映射、索引与元数据，文件长度来自实际文件。", "",
        "| 臂 | 层 | 输入→输出 | 活跃边 | 非零边 | 活跃/非零密度 | 自由参数含偏置 | 分配槽 | 非零共享组 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"])
    for arm in ARMS:
        for layer in structures[arm]["layers"]:
            lines.append(f"| {arm} | {layer['layer_index']+1} | {layer['in_features']}→{layer['out_features']} | {layer['active_edges']} | "
                f"{layer['nonzero_edges']} | {layer['active_density']:.3%}/{layer['nonzero_density']:.3%} | "
                f"{layer['independent_parameters']} | {layer['allocated_parameter_slots']} | {layer['nonzero_shared_groups']} |")
    lines.extend(["", "补偿可能新增或归零出边，事件中的compensation记录具体变化；删除节点与减少边数分别报告。"
        "物理宽度/参数槽缩小仍使用稠密gather矩阵乘，本轮没有实测推断延迟或能耗，不能将参数数、边数当作速度/能耗收益。", "",
        "## 多节点重构还是零激活删除", "",
        "| 臂 | 已接受slot | 删除系数列 | ≥2非零系数列 | ≥2个abs>1e-8系数列 | 全零系数列 | 640留出确认零激活列 |",
        "| --- | --- | --- | --- | --- | --- | --- |"])
    for arm in ARMS:
        t = evidence_totals[arm]
        lines.append(f"| {arm} | {t['accepted_slots']} | {t['accepted_coefficient_columns']} | {t['accepted_multi_nonzero_coefficient_columns']} | "
            f"{t['accepted_multi_coefficients_above_1e_minus_8_columns']} | {t['accepted_all_zero_coefficient_columns']} | "
            f"{t['accepted_zero_activation_verified_on_holdout_columns']} |")
    lines.extend(["", "每个dropped列按固定C32的coef!=0统计；abs>1e-8只是额外描述阈值，不能证明秩或功能独立。"
        "C32全零且640留出绝对残差为0，可确认该留出上的零激活；单凭全零系数不能证明全域死节点。"
        "原拟合残差为float64求解结果，留出残差则来自固定C32，两者不能混写。", "",
        "| 臂 | 轮/slot | 层 | 接受 | 基列/保留/删除数 | dropped节点 | 非零coef数 | 拟合相对残差64 | 留出相对残差C32 | 留出绝对残差 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"])
    for arm in ARMS:
        for event in reconstruction[arm]:
            for column in event["columns"]:
                lines.append(f"| {arm} | {event['epoch']}/{event['proposal']} | {event['layer_index']} | {event['accepted']} | "
                    f"{event['basis_count']}/{event['kept_count']}/{event['dropped_count']} | {column['dropped_node_index']} | "
                    f"{column['nonzero_coefficients']} | {column['fit_relative_residual_float64_lstsq']} | "
                    f"{column['holdout_relative_residual_fixed_C32']} | {column['holdout_absolute_residual_fixed_C32']} |")
    lines.extend(["", "完整实际系数矩阵、逐列保留节点ID、绝对残差、系数范数与补偿新增边都保留JSON；未接受提议单列，不能当有效简化。", ""])
    if plot["created"]:
        lines.extend([f"![真实逐轮变化]({plot['filename']})", "", "曲线为本阶段选择准确率/损失、物理隐藏节点及对应轮fixed文件比，未混入父25轮。", ""])
    else:
        lines.extend([plot["reason"]+"。", ""])
    lines.extend(["## 实际成本与状态匹配", "",
        "| 臂 | 优化前向/反传样本 | 尝试/候选/回退/fallback | 净更新/额外尝试 | 激活提取次数/样本 | 评价次数/样本 | Adam重置轮 | 臂计时秒 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |"])
    for arm in ARMS:
        c = costs[arm]
        lines.append(f"| {arm} | {c['optimization_forward_samples']}/{c['optimization_backward_samples']} | "
            f"{c['attempted_updates']}/{c['candidate_updates']}/{c['rolled_back_updates']}/{c['fallback_updates']} | "
            f"{c['net_updates']}/{c['extra_attempts']} | {c['activation_extraction_calls']}/{c['activation_forward_samples']} | "
            f"{c['evaluation_calls']}/{c['evaluation_forward_samples']} | {c['adam_reset_epochs']} | {arms[arm]['wall_seconds']:.3f} |")
    lines.extend(["", f"总新增优化前向/反传{total['optimization_forward_samples']}/{total['optimization_backward_samples']}样本，"
        f"尝试{total['attempted_updates']}、回退{total['rolled_back_updates']}、fallback {total['fallback_updates']}、净更新{total['net_updates']}。"
        f"激活提取{total['activation_extraction_calls']}次/{total['activation_forward_samples']}样本，评价{total['evaluation_calls']}次/{total['evaluation_forward_samples']}样本。"
        "激活提取只运行到目标隐藏层，与完整分类评价分别计数；已回退尝试仍是实际成本。", "",
        "random真实逐轮物理宽度、尝试/候选/回退/fallback及Adam重置时序已与reduce记录核对一致；正常净更新预算亦由账本核对。", "",
        "| 臂 | 提取/评价 | 用途 | 调用次数 | 样本 |", "| --- | --- | --- | --- | --- |"])
    for arm in ARMS:
        for purpose, entry in costs[arm]["evaluation_by_purpose"].items():
            lines.append(f"| {arm} | 分类评价 | {purpose} | {entry['calls']} | {entry['samples']} |")
        extraction_groups = {}
        for extraction in costs[arm]["activation_extractions"]:
            entry = extraction_groups.setdefault(extraction["purpose"], {"calls": 0, "samples": 0})
            entry["calls"] += 1
            entry["samples"] += extraction["samples"]
        for purpose, entry in extraction_groups.items():
            lines.append(f"| {arm} | 隐藏激活 | {purpose} | {entry['calls']} | {entry['samples']} |")
    lines.extend(["", "| 臂 | 搜索wall秒 | 搜索CPU process秒 | 留出wall秒 | 构造编码解码wall秒 | 候选适应评价wall秒 |",
        "| --- | --- | --- | --- | --- | --- |"])
    for arm in ARMS:
        t = costs[arm]["timings"]
        lines.append(f"| {arm} | " + " | ".join(f"{t[key]:.6f}" for key in TIMING_KEYS) + " |")
    lines.extend(["", "搜索wall包括激活提取/传输及CPU求解，不能称纯CPU墙钟；CPU process另外报告。"
        "未记录字段的汇总0只表示无已记录该项，不保证物理成本为0；这些计时已包含于臂计时，不能再相加作为总墙钟。", "",
        f"实际完成会话墙钟{summary['current_session_wall_seconds']:.3f}秒；跨恢复会话累计墙钟{summary['cumulative_wall_seconds']:.3f}秒。"
        "臂计时排除checkpoint/预测写入及末态拟合/探索评价；累计墙钟为本阶段，不含父25轮。运行时间倍率不是否决门槛。", "",
        f"设备{registration['device']}，PyTorch {registration['torch']}，FP32，batch{config['batch_size']}、评价batch{config['evaluation_batch_size']}、CPU线程2；"
        f"整次CUDA峰值分配{summary.get('peak_cuda_memory_bytes')}字节，不是逐臂内存比较。", "",
        f"启动证据复用：新增前向{startup['forward_samples']}、反传{startup['backward_samples']}、更新{startup['updates']}。"
        "报告0模型前向/反传/更新，未加载checkpoint，未运行GPU诊断或测试。", "", "## 效果与结论边界", "",
        "| 臂 | 集合 | 相对fixed准确率（百分点） | 相对fixed交叉熵 |", "| --- | --- | --- | --- |"])
    for arm in ARMS:
        for split in labels:
            d = differences[arm][split]
            lines.append(f"| {arm} | {labels[split]} | {100*d['accuracy_delta_vs_fixed']:+.4f} | {d['loss_delta_vs_fixed']:+.6f} |")
    lines.extend(["", "| 集合 | reduce−random准确率（百分点） | reduce−random交叉熵 |", "| --- | --- | --- |"])
    for split in labels:
        d = random_difference[split]
        lines.append(f"| {labels[split]} | {100*d['reduce_minus_random_accuracy']:+.4f} | {d['reduce_minus_random_loss']:+.6f} |")
    physical = simplification["reduce"]
    e = evidence_totals["reduce"]
    if physical["physical_hidden_nodes_removed"] and physical["storage_reduction_vs_fixed"] > 0 and physical["task_preserved_vs_fixed_select"]:
        finding = "本轮主臂实际物理缩宽、减少参数槽与完整文件，并在登记选择容差内保持效果。"
    else:
        finding = "本轮尚未同时获得物理缩宽、实际文件减少与登记效果保持，应分别保留失败项。"
    if e["accepted_multi_nonzero_coefficient_columns"]:
        mechanism = f"已接受删除列中{e['accepted_multi_nonzero_coefficient_columns']}列使用至少两个非零固定系数，支持本轮确有多列重构操作；这不等于严格线性独立基或全域等价。"
    elif e["accepted_coefficient_columns"] and e["accepted_zero_activation_verified_on_holdout_columns"] == e["accepted_coefficient_columns"]:
        mechanism = "已接受删除列均为全零固定系数且640留出绝对残差为0，当前简化证据主要是该留出上的零激活清理，未验证有用非零节点的多列重构。"
    else:
        mechanism = "尚无已接受非零多基列重构的证据；单列重构、零系数及未接受提议须分开解释。"
    lines.extend(["", finding+mechanism, "",
        "唯一模型根、反复使用的选择集与旧control探索不足以声称显著性、稳健性或跨任务通用性；主/随机求解空间差别限制归因。"
        "没有卷积发现、全域函数等价、推断速度或能耗收益证据。", "", "## 证据与恢复", "",
        f"[证据索引]({stem}_evidence/README.md)保存真实末态模型、fit/select/confirm预测、training_commit、registration、"
        "startup、freeze与manifest，文件哈希见JSON。完整节点系数与事件保留在报告JSON。", "",
        f"原逐轮日志/预测及完整动态宽度恢复checkpoint保留于`{run}`，未复制大型checkpoint。", "", "```powershell",
        ".venv/Scripts/python.exe -u -B -m tools.train_nodes --resume", "```", "", "## 混淆矩阵", ""])
    for arm in ARMS:
        for split in labels:
            lines.extend([f"### {arm}：{labels[split]}", "", *confusion_table(predictions[arm][split]["confusion_matrix"]), ""])
    md_path = reports / f"{stem}.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    (evidence / "README.md").write_text(
        "# 多节点简化实际证据\n\n三臂真实final_model.bgs、末态fit/select/confirm预测按原字节复制；哈希见上级完整JSON。"
        "模型架构头仅用于核对物理宽度，不构造模型、不执行前向；没有读取checkpoint。\n\n"
        "training_commit保存启动前提交，registration包含源/config/父checkpoint哈希与2560/640固定激活划分，"
        "startup为既有启动复用0新增成本，freeze记录全部末轮冻结，manifest追溯授权ID。"
        "可用时session_cost保存跨恢复会话累计成本。\n\n"
        "完整C32系数、逐dropped列非零系数统计、float64拟合残差、固定C32留出残差、补偿边变化与成本保留于报告JSON。"
        "主臂贪心基组与random所有kept列最小二乘的空间不同，不能把全部差异归因节点身份；有限样本重构不保证全域等价。\n\n"
        "旧1000control只重复探索，不是独立确认；官方测试及其他封存ID未解封。"
        f"原日志、逐轮预测与动态宽度恢复checkpoint保留于`{run}`。\n", encoding="utf-8")
    print(json.dumps({"report": str(md_path), "json": str(json_path), "plot": plot,
                      "costs": total, "simplification": simplification, "reconstruction": evidence_totals}, ensure_ascii=False))


if __name__ == "__main__":
    main()
