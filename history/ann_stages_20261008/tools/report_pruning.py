"""仅读取已完成 H1 运行的账本和保存预测，生成报告与重要证据；不运行模型。"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("fc", "prune", "random", "cnn")
COST_KEYS = ("optimization_forward_samples", "optimization_backward_samples", "attempted_updates",
             "candidate_updates", "rolled_back_updates", "evaluation_forward_samples")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def saved_metrics(path):
    with np.load(path, allow_pickle=False) as archive:
        scores = np.asarray(archive["logits"], dtype=np.float64)
        labels = archive["labels"]
    if (scores.ndim != 2 or scores.shape[1] != 10 or labels.shape != (len(scores),)
            or not len(scores) or not np.isfinite(scores).all()
            or not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0) or np.any(labels >= 10)):
        raise ValueError(f"保存预测非法：{path}")
    shifted = scores - scores.max(axis=1, keepdims=True)
    loss = np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(len(labels)), labels]
    predicted = scores.argmax(axis=1)
    confusion = np.zeros((10, 10), dtype=np.int64)
    np.add.at(confusion, (labels, predicted), 1)
    return {"samples": len(labels), "accuracy": float(np.mean(predicted == labels)),
            "loss": float(loss.mean()), "confusion_matrix": confusion.tolist()}


def check_metrics(actual, recorded, path):
    if (actual["samples"] != recorded["samples"]
            or abs(actual["accuracy"] - recorded["accuracy"]) > 1e-9
            or abs(actual["loss"] - recorded["loss"]) > 1e-7
            or actual["confusion_matrix"] != recorded["confusion_matrix"]):
        raise ValueError(f"保存预测与summary成绩不一致，停止生成报告：{path}")


def confusion_table(matrix):
    lines = ["| 真值\\预测 | " + " | ".join(map(str, range(10))) + " |",
             "| --- | " + " | ".join(["---"] * 10) + " |"]
    lines.extend("| " + str(label) + " | " + " | ".join(map(str, row)) + " |"
                 for label, row in enumerate(matrix))
    return lines


def make_plot(arms, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError:
        return {"created": False, "reason": "matplotlib不可用；完整真实曲线保留于JSON，未安装依赖"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    fc_bytes = {row["epoch"]: row["structure"]["export_bytes"] for row in arms["fc"]["rows"]}
    for arm in ARMS:
        rows = arms[arm]["rows"]
        epochs = [row["epoch"] for row in rows]
        axes[0, 0].plot(epochs, [row["select"]["accuracy"] for row in rows], label=arm)
        axes[0, 1].plot(epochs, [row["select"]["loss"] for row in rows], label=arm)
        if arm != "cnn":
            axes[1, 0].plot(epochs, [row["structure"]["model_total"]["active_edges"] /
                                   row["structure"]["model_total"]["total_edges"] for row in rows], label=arm)
            axes[1, 1].plot(epochs, [row["structure"]["export_bytes"] / fc_bytes[row["epoch"]]
                                   for row in rows], label=arm)
    titles = (("Selection accuracy", "Accuracy"), ("Selection cross entropy", "Loss"),
              ("Active edge fraction (FC arms)", "Fraction of initial edges"),
              ("Export bytes / same-epoch FC (same format)", "Byte ratio"))
    for ax, (title, ylabel) in zip(axes.flat, titles):
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.25)
        ax.legend()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {"created": True, "filename": path.name,
            "data_source": "summary.arms[*].rows；CNN仅出现在准确率与损失面板"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="mnist_static", choices=["mnist_static"])
    parser.add_argument("--version", default="pruning_v1")
    parser.add_argument("--date", default="20261007")
    args = parser.parse_args()
    run = ROOT / "experiments" / "runs" / args.version / args.task
    summary = read_json(run / "summary.json")
    if set(summary["arms"]) != set(ARMS) or not all(result.get("frozen") for result in summary["arms"].values()):
        raise ValueError("报告只接受全部冻结且完成确认的四臂运行")
    registration = read_json(run / "registration.json")
    if registration != summary["registration"]:
        raise ValueError("registration与summary来源不一致")
    startup = read_json(run / "startup.json")
    freeze = read_json(run / "freeze.json")
    manifest = read_json(run / "data_manifest.json")
    training_commit = read_json(run / "training_commit.json")
    if set(freeze["arms"]) != set(ARMS):
        raise ValueError("freeze记录未覆盖四臂")
    reports = ROOT / "reports"
    stem = f"{args.version}_{args.task}_{args.date}"
    evidence = reports / f"{stem}_evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    hashes, predictions = {}, {}
    for arm in ARMS:
        result = summary["arms"][arm]
        arm_path = run / arm
        target_path = evidence / arm
        target_path.mkdir(exist_ok=True)
        files = {"fit": "fit_final.npz", "select": f"select_{result['final_epoch']:03d}.npz",
                 "confirm": "confirm_final.npz"}
        predictions[arm] = {}
        recorded = {"fit": result["fit"], "select": result["rows"][-1]["select"], "confirm": result["confirmation"]}
        for split, filename in files.items():
            actual = saved_metrics(arm_path / filename)
            check_metrics(actual, recorded[split], arm_path / filename)
            predictions[arm][split] = actual
        model_path = arm_path / "final_model.bgs"
        if model_path.stat().st_size != result["structure"]["export_bytes"]:
            raise ValueError(f"实际模型文件字节与summary不一致：{model_path}")
        for filename in ("final_model.bgs", *files.values()):
            target = target_path / filename
            shutil.copyfile(arm_path / filename, target)
            hashes[str(target.relative_to(reports))] = digest(target)
    for filename in ("registration.json", "training_commit.json", "startup.json", "freeze.json", "data_manifest.json"):
        target = evidence / filename
        shutil.copyfile(run / filename, target)
        hashes[str(target.relative_to(reports))] = digest(target)
    config = registration["config"]
    spec = config["tasks"][args.task]
    arms = summary["arms"]
    total = {key: sum(result["ledger"][key] for result in arms.values()) for key in COST_KEYS}
    total["net_updates"] = total["attempted_updates"] - total["rolled_back_updates"]
    evaluated = {}
    for result in arms.values():
        for item in result["ledger"]["evaluations"]:
            cell = evaluated.setdefault(item["purpose"], {"calls": 0, "samples": 0})
            cell["calls"] += 1
            cell["samples"] += item["samples"]
    differences = {split: {"prune_minus_random_accuracy": predictions["prune"][split]["accuracy"] - predictions["random"][split]["accuracy"],
                           "prune_minus_random_loss": predictions["prune"][split]["loss"] - predictions["random"][split]["loss"]}
                   for split in ("fit", "select", "confirm")}
    commit = training_commit.get("gitHead") or training_commit.get("git_head")
    plot = make_plot(arms, reports / f"{stem}_curves.png")
    if plot["created"]:
        hashes[plot["filename"]] = digest(reports / plot["filename"])
    report_json = copy.deepcopy(summary)
    report_json.update(prediction_metrics=predictions, evidence_sha256=hashes, implementation_commit=commit,
                       prediction_source="本次运行保存logits，仅重算指标；0模型前向/反传/更新",
                       costs_total=total, evaluation_costs_by_purpose=evaluated,
                       prune_random_difference=differences, plot=plot, original_run_path=str(run),
                       startup_cost=startup, training_commit=training_commit)
    json_path = reports / f"{stem}.json"
    json_path.write_text(json.dumps(report_json, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    commit_text = f"`{commit}`（启动前training_commit.json登记）" if commit else "training_commit未登记Git提交；实现以registration中的训练时源文件SHA256追溯"
    lines = ["# H1任务保持剪枝：静态MNIST实际报告", "",
             f"{args.date}；来自实际完成的四臂运行，报告只读取保存预测、模型文件与账本。", "",
             f"实现提交：{commit_text}。配置、源哈希、完整曲线和分项账本见[完整JSON]({stem}.json)。", "",
             "## 任务、边界与对照", "",
             f"模型根{config['model_seed']}，数据根{config['data_seed']}；仅授权5000原ID，分为3200参数拟合、800结构选择、1000冻结后探索性确认。"
             "原control曾用于历史探索，不是全新独立最终测试；未解封剩余原ID，未访问官方测试。", "",
             f"FC/剪枝/随机臂骨架为{'→'.join(map(str, spec['widths']))}、两层ReLU，同初始化与逐轮批次顺序。"
             "CNN为人工预置8通道5×5有效卷积、ReLU、展平线性10输出；容量与执行形式不同，仅作参照。", "",
             f"四臂各{spec['epochs']}轮，AdamW初始学习率{config['learning_rate']}，weight decay={config['weight_decay']}，"
             f"批量{config['batch_size']}；前{spec['warmup_epochs']}轮预热，每个主候选使用一个原预算批次适应，最多{spec['max_proposals']}次提议。"
             "采用最终轮，不根据确认集选择轮次。", "",
             "prune使用拟合梯度与权重敏感度提议删边，固定256样本筛选集及完整800样本选择集作接受依据，接受还须实际文件缩小。"
             "random按主剪枝接受的时间、slot和每层删边数，无条件均匀随机删除；主拒绝slot同样适应后回退，以匹配净保留更新。"
             "随机臂的选择评价不决定接受，比较的是相同接受日程下删除连接的身份，不是独立随机搜索方法。", "",
             "## 实际性能", "",
             "| 臂 | 集合 | 样本 | 准确率 | 交叉熵 |", "| --- | --- | --- | --- | --- |"]
    labels = {"fit": "拟合", "select": "结构选择", "confirm": "探索确认"}
    for arm in ARMS:
        for split in labels:
            m = predictions[arm][split]
            lines.append(f"| {arm} | {labels[split]} | {m['samples']} | {m['accuracy']:.4%} | {m['loss']:.6f} |")
    lines.extend(["", "混淆矩阵见下方：行是真值、列是预测；所有指标由本次保存logits重算并核对summary。", "",
                  "## 结构与实际文件", "",
                  "| 臂 | 活跃边 | 非零边 | 独立参数含偏置 | 分配参数槽 | 实际导出字节 | 计时边界内墙钟秒 | 接受/slot |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"])
    for arm in ARMS:
        result, structure = arms[arm], arms[arm]["structure"]
        s = structure.get("model_total", {})
        parameters = s.get("independent_parameters", structure.get("independent_parameters"))
        count = sum(bool(event["accepted"]) for event in result["events"])
        lines.append(f"| {arm} | {s.get('active_edges', '人工卷积参照')} | {s.get('nonzero_edges', '未按展开边计数')} | "
                     f"{parameters} | {s.get('allocated_parameter_slots', parameters)} | {structure['export_bytes']} | "
                     f"{result['wall_seconds']:.3f} | {count}/{len(result['events'])} |")
    lines.extend(["", "FC/剪枝/随机臂隐藏单元768、输出10、总计算单元778（不含输入784），本轮不删除单元。", "",
                  "| 臂 | 层 | 活跃边/初始边 | 密度 | 独立参数含偏置 |",
                  "| --- | --- | --- | --- | --- |"])
    for arm in ("fc", "prune", "random"):
        for index, layer in enumerate(arms[arm]["structure"]["layers"]):
            lines.append(f"| {arm} | {spec['widths'][index]}→{spec['widths'][index+1]} | "
                         f"{layer['active_edges']}/{layer['total_edges']} | "
                         f"{layer['active_edges']/layer['total_edges']:.4%} | {layer['independent_parameters']} |")
    lines.extend(["", "FC/剪枝/随机臂使用同一FP32无损编码，实际字节包含数值、掩码、索引和全部元数据。"
                  "CNN使用含架构名称的torch state文件，格式不同，不据文件大小直接比较架构效率。"
                  "theta槽固定分配，前向仍是稠密gather及矩阵乘；存储减少不代表推断加速、内存减少、计算复用或实测能耗下降。", ""])
    if plot["created"]:
        lines.extend([f"![真实逐轮曲线]({plot['filename']})", "",
                      "图中准确率/损失展示四臂；边比例/字节比仅展示同骨架同编码的FC臂。字节比参照对应轮FC，CNN未混入该曲线。", ""])
    else:
        lines.extend([plot["reason"] + "。", ""])
    lines.extend(["## 成本与检查边界", "",
                  "| 臂 | 优化前向样本 | 优化反传样本 | 尝试更新 | 候选适应 | 回退更新 | 净保留更新 | 评价前向样本 |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"])
    for arm in ARMS:
        ledger = arms[arm]["ledger"]
        lines.append(f"| {arm} | {ledger['optimization_forward_samples']} | {ledger['optimization_backward_samples']} | "
                     f"{ledger['attempted_updates']} | {ledger['candidate_updates']} | {ledger['rolled_back_updates']} | "
                     f"{ledger['attempted_updates']-ledger['rolled_back_updates']} | {ledger['evaluation_forward_samples']} |")
    lines.extend(["", f"合计优化前向/反传样本{total['optimization_forward_samples']}/{total['optimization_backward_samples']}；"
                  f"尝试更新{total['attempted_updates']}，回退{total['rolled_back_updates']}，净保留{total['net_updates']}。"
                  f"评价前向合计{total['evaluation_forward_samples']}样本。"
                  "被回退适应仍消耗样本和搜索成本。每种评价用途的调用与样本账本见JSON。", "",
                  f"唯一启动检查为{startup['forward_samples']}样本前向/{startup['backward_samples']}样本反传、{startup['updates']}更新，单列且未重复。"
                  "本报告生成0模型前向/反传/更新，未启动GPU测试或恢复重放。", "",
                  f"设备{registration['device']}，PyTorch {registration['torch']}；FP32、训练批量{config['batch_size']}、"
                  f"评价批量{config['evaluation_batch_size']}、CPU线程2。CUDA峰值分配{summary.get('peak_cuda_memory_bytes')}字节，"
                  "是整次运行峰值，不能作为逐臂内存比较。", "",
                  f"本次会话实际墙钟{summary.get('current_session_wall_seconds', '未记录')}秒。"
                  "臂间墙钟覆盖每轮训练、候选搜索、选择评价和编码，排除checkpoint/预测写入及最终拟合/确认评价；"
                  "不是完整端到端训练或推断计时，同epoch也不是同计算预算。", "", "## H1判定与随机比较", ""])
    gate = summary["gate"]
    lines.append(f"主剪枝相对FC文件减少{gate['storage_reduction']:.2%}，计时边界内墙钟倍数{gate['runtime_ratio']:.3f}；"
                 f"FC学会={gate['fc_learned']}，效果保持={gate['task_preserved']}，存储门槛={gate['storage_passed']}，"
                 f"运行成本门槛={gate['runtime_passed']}，继续门槛={gate['continue']}。共享不是该H1门槛。")
    lines.extend(["", "| 集合 | prune−random准确率（百分点） | prune−random交叉熵 |", "| --- | --- | --- |"])
    for split, delta in differences.items():
        lines.append(f"| {labels[split]} | {100*delta['prune_minus_random_accuracy']:+.4f} | {delta['prune_minus_random_loss']:+.6f} |")
    selected_delta = differences["select"]
    if selected_delta["prune_minus_random_accuracy"] > 0 and selected_delta["prune_minus_random_loss"] < 0:
        interpretation = "本次主剪枝的结构选择准确率和损失均优于匹配随机删除，是任务敏感删边可能有用的单根信号。"
    elif selected_delta["prune_minus_random_accuracy"] <= 0 and selected_delta["prune_minus_random_loss"] >= 0:
        interpretation = "本次主剪枝未在结构选择准确率和损失上同时胜过匹配随机删除，尚不能说明连接身份选择优于随机。"
    else:
        interpretation = "主剪枝与匹配随机删除在结构选择准确率和损失上存在权衡，不能归纳为一致优势。"
    lines.extend(["", interpretation + "随机臂依赖主接受日程，选择集被多次用于主剪枝决策，旧control也是探索确认；"
                  "单一模型根不能证明稳健性、统计显著性或通用性。"])
    lines.append("本轮通过预登记的H1扩大门槛，仅支持继续剪枝诊断。" if gate["continue"] else
                 "本轮未通过预登记扩大门槛，应保留失败项并停止相同条件预算扩大；不以压缩单项宣布成功。")
    if not gate["runtime_passed"]:
        lines.append("下一步先定位候选编码、状态复制、图编辑及评价的成本；本轮未分别计时，不能断言具体瓶颈。"
                     "性能保持/存储缩小与搜索效率失败分别记录，不改动原门槛，也不追加训练以改善判定。")
    lines.extend(["不要求形成共享或卷积，也没有空间局部性、卷积计算复用及跨任务普适性证据。", "",
                  "## 证据与恢复", "",
                  f"[重要证据索引]({stem}_evidence/README.md)保留四臂最终模型、fit/select/confirm预测和运行登记。"
                  "所有复制文件SHA256在完整JSON，未复制大型恢复checkpoint。", "",
                  f"原运行日志、逐轮预测与resume.pt路径：`{run}`。源码与配置哈希一致时恢复；"
                  "已完成运行的--resume仅返回完成判定，不重训或重复确认。", "", "```powershell",
                  f".venv/Scripts/python.exe -u -B -m tools.train_pruning --task {args.task} --resume", "```", "",
                  "## 混淆矩阵", ""])
    for arm in ARMS:
        for split in labels:
            lines.extend([f"### {arm}：{labels[split]}", "", *confusion_table(predictions[arm][split]["confusion_matrix"]), ""])
    md_path = reports / f"{stem}.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    permissions = manifest.get("permissions_used", {})
    (evidence / "README.md").write_text(
        "# H1静态MNIST重要证据\n\n"
        "四臂final_model.bgs及最终fit/select/confirm logits按原字节复制；SHA256在上级完整JSON。"
        "CNN虽沿用bgs扩展名，内容是带架构说明的torch state格式，不能用共享图解码器读取。\n\n"
        "registration保存配置与训练实现源哈希；training_commit保存启动前Git提交及来源提交；freeze记录全部臂先冻结；startup记录唯一8样本启动；"
        "data_manifest仅用于授权ID及来源追溯，不传入模型。原control为探索性确认。\n\n"
        f"运行数据清单中的许可记录：`{json.dumps(permissions, ensure_ascii=False)}`。"
        "该清单在训练前保存，可能仍记确认像素提取0；本次确认确实发生于freeze之后，"
        "最终确认预测保留1000授权原ID的成绩，不意味着其他封存ID或官方测试被访问。\n\n"
        f"完整恢复checkpoint及原日志仍在`{run}`。本报告没有读取checkpoint或运行模型。\n",
        encoding="utf-8")
    print(json.dumps({"report": str(md_path), "json": str(json_path), "gate": gate,
                      "plot": plot, "costs": total}, ensure_ascii=False))


if __name__ == "__main__":
    main()
