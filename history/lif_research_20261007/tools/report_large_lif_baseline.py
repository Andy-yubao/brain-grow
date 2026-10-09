"""仅读取已停止的三层LIF实验，复算保存预测并导出报告/曲线；不运行模型。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch

from braingrow.evaluation.metrics import evaluate
from tools import train_large_baseline as base
from tools import train_adaptive_baseline as adaptive


PREFIX = Path("reports/large_lif_baseline_training_20261005")
TASKS = ("digit_accuracy", "motion_accuracy", "joint_accuracy")
MOTIONS = ("静止", "上移", "下移", "左移", "右移", "顺时针", "逆时针")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def verify_predictions(outdir, epoch, pool, expected):
    path = outdir / "evaluations" / f"{pool}_{epoch:03d}.npz"
    with np.load(path, allow_pickle=False) as saved:
        observed = evaluate(saved["potential"], saved["y"])
        predictions = np.column_stack((saved["potential"][:, :10].argmax(1),
                                       saved["potential"][:, 10:].argmax(1)))
        truth = saved["y"]
        joint = np.zeros((70, 70), dtype=np.int64)
        np.add.at(joint, (truth[:, 0] * 7 + truth[:, 1],
                         predictions[:, 0] * 7 + predictions[:, 1]), 1)
        assert len(truth) == expected["videos"]
        assert int(joint.trace()) / len(truth) == observed["joint_accuracy"]
        raw_ids = np.unique(saved["raw_ids"]).tolist()
    for key in (*TASKS, "loss"):
        assert math.isclose(observed[key], expected[key], rel_tol=1e-12, abs_tol=1e-12), (pool, epoch, key)
    for key in ("digit_confusion", "motion_confusion"):
        assert observed[key] == expected[key], (pool, epoch, key)
    return dict(epoch=epoch, pool=pool, metrics=observed, joint_confusion=joint.tolist(),
                joint_class_order="数字×7+运动；运动顺序静止/上/下/左/右/顺时针/逆时针",
                original_ids=raw_ids, source=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def plot_curves(rows, current_epoch, current_lr, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font = Path("C:/Windows/Fonts/msyh.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 10
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.0), layout="constrained")
    control = [row for row in rows if row["control"] is not None]
    fit = [row for row in rows if row["fit"] is not None]
    ax = axes[0, 0]
    ax.plot([r["epoch"] for r in fit], [100 * r["fit"]["joint_accuracy"] for r in fit], "o-", label="拟合池联合")
    ax.plot([r["epoch"] for r in control], [100 * r["control"]["joint_accuracy"] for r in control], "o-", label="控制池联合")
    ax.set(title="拟合已达100%，跨ID控制约27%", ylabel="准确率（%）", ylim=(0, 105))
    ax.legend()
    ax = axes[0, 1]
    ax.semilogy([r["epoch"] for r in rows], [r["optimizer_mean_loss"] for r in rows], label="优化过程批次平均")
    ax.semilogy([r["epoch"] for r in fit], [r["fit"]["loss"] for r in fit], "o-", label="冻结模型拟合池")
    ax.semilogy([r["epoch"] for r in control], [r["control"]["loss"] for r in control], "o-", label="冻结模型控制池")
    ax.set(title="拟合损失下降，控制损失持续上升", ylabel="数字＋运动交叉熵（对数轴）")
    ax.legend()
    ax = axes[1, 0]
    ax.step([r["epoch"] for r in rows] + [current_epoch],
            [r["learning_rate"] for r in rows] + [current_lr], where="post", color="#6d4ba5")
    ax.scatter([current_epoch], [current_lr], marker="^", color="#6d4ba5", label=f"第{current_epoch}遍部分训练")
    ax.set_yscale("log")
    ax.axvline(16, color="gray", linestyle="--", alpha=0.65)
    ax.set(title="实际学习率；第16遍开始自适应接续", ylabel="学习率（对数轴）")
    ax.legend()
    ax = axes[1, 1]
    for key, label in (("digit_accuracy", "控制数字"), ("motion_accuracy", "控制运动"),
                       ("joint_accuracy", "控制联合")):
        ax.plot([r["epoch"] for r in control], [100 * r["control"][key] for r in control], "o-", label=label)
    ax.set(title="控制池三项准确率", ylabel="准确率（%）", ylim=(0, 105))
    ax.legend()
    for ax in axes.flat:
        ax.set_xlabel("训练遍数")
        ax.grid(alpha=0.2)
    fig.suptitle("5000 / 3000 / 2000 LIF固定图：训练已由用户暂停", fontsize=15)
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def collect(outdir):
    summary = read_json(outdir / "summary.json")
    assert summary["status"].startswith("user_interrupt"), "报告入口要求用户已安全暂停"
    if (outdir / "running.json").exists():
        owner = read_json(outdir / "running.json")
        assert not adaptive.process_alive(owner["pid"]), "训练仍在运行"
    payload, checkpoint_path = base.load_resume(outdir)
    state = payload["state"]
    assert state["completed_epochs"] == summary["completed_epochs"]
    assert state["costs"] == summary["costs"]
    config = read_json(outdir / "config.json")
    assert config["source_sha256"] == adaptive.source_hashes(), "训练源码与记录不一致"
    parent = Path(config["parent_experiment"]["parent_directory"])
    rows = []
    for directory in (parent, outdir):
        rows.extend(read_json(path) for path in sorted(directory.glob("epoch_*.json"))
                    if int(path.stem.split("_")[-1]) <= state["completed_epochs"])
    rows.sort(key=lambda row: row["epoch"])
    assert [r["epoch"] for r in rows] == list(range(1, state["completed_epochs"] + 1))
    control_rows = [r for r in rows if r["control"] is not None]
    fit_rows = [r for r in rows if r["fit"] is not None]
    best_joint = max(control_rows, key=lambda row: row["control"]["joint_accuracy"])
    best_selected = max(control_rows, key=lambda row: sum(row["control"][key] for key in TASKS))
    best_payload = torch.load(outdir / "best.pt", map_location="cpu", weights_only=False)
    assert best_payload["state"]["tracker"]["best_control"]["epoch"] == best_selected["epoch"]
    del best_payload
    checked = [verify_predictions(outdir, row["epoch"], pool, row[pool]) for row, pool in
               ((best_joint, "control"), (control_rows[-1], "control"), (fit_rows[-1], "fit"))]
    assert set(checked[1]["original_ids"]).isdisjoint(checked[2]["original_ids"])
    progress = state["progress"]
    partial = 0 if progress is None else progress["next_sample"]
    updates = [int(value["step"].item()) for value in payload["optimizer"]["state"].values()]
    assert len(set(updates)) == 1
    expected_updates = state["completed_epochs"] * 35 + partial // config["effective_batch"]
    assert updates[0] == expected_updates
    costs = state["costs"]
    assert costs["train_forward_videos"] == costs["train_backward_videos"] == state["completed_epochs"] * 4480 + partial
    assert costs["control_metric_evaluations"] == len(control_rows)
    assert costs["fit_metric_evaluations"] == len(fit_rows)
    assert costs["control_forward_videos"] == len(control_rows) * 1120
    assert costs["fit_evaluation_forward_videos"] == len(fit_rows) * 4480
    events = [json.loads(line) for line in (outdir / "learning_rate_events.jsonl").read_text(encoding="utf-8").splitlines()]
    parent_summary = read_json(parent / "summary.json")
    return dict(generated_at=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        status=summary["status"], completed_epochs=state["completed_epochs"], maximum_epochs=state["maximum_epochs"],
        progress=None if progress is None else {k: v for k, v in progress.items() if k != "order"},
        completed_optimizer_updates=updates[0], current_learning_rate=payload["optimizer"]["param_groups"][0]["lr"],
        checkpoint=str(checkpoint_path.resolve()), best_checkpoint=str((outdir / "best.pt").resolve()),
        checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        best_checkpoint_sha256=hashlib.sha256((outdir / "best.pt").read_bytes()).hexdigest(),
        best_joint_control_epoch=best_joint["epoch"], best_joint_control=best_joint["control"],
        selected_best_epoch=best_selected["epoch"], selected_best_control=best_selected["control"],
        latest_control_epoch=control_rows[-1]["epoch"], latest_control=control_rows[-1]["control"],
        latest_fit_epoch=fit_rows[-1]["epoch"], latest_fit=fit_rows[-1]["fit"],
        epoch_rows=rows, learning_rate_events=events, prediction_rechecks=checked,
        costs=costs, resources=control_rows[-1]["control"]["resources"],
        parent_session_wall_seconds=parent_summary["wall_seconds_this_session"],
        adaptive_session_wall_seconds=summary["wall_seconds_this_session"],
        combined_training_session_wall_seconds=parent_summary["wall_seconds_this_session"] + summary["wall_seconds_this_session"],
        gpu_peak_allocated_bytes=summary["gpu_peak_allocated_bytes"],
        process_peak_working_set_bytes=summary["process_peak_working_set_bytes"],
        config=config, source_guard_matches=True, saved_prediction_metrics_match=True,
        cost_and_optimizer_steps_match=True, report_model_forward_videos=0,
        report_model_backward_videos=0, report_optimizer_updates=0,
        validation_forward_evaluations=0, sealed_test_reads=0,
        stderr_bytes=(outdir / "console_error.log").stat().st_size)


def markdown(result, prefix):
    control, fit = result["latest_control"], result["latest_fit"]
    partial = result["progress"]
    lines = ["# 大容量三层LIF基线训练报告", "", f"生成时间：{result['generated_at']}。", "",
        "## 1. 训练状态与主要结论", "",
        f"用户已安全暂停，已完成{result['completed_epochs']}遍；第{partial['epoch']}遍额外完成{partial['next_sample']}/4480视频（4个有效批次）。原预算100遍未跑满，停止原因是用户中断，不是预算结束或自动收敛。没有活动训练进程。", "",
        "拟合数据已完全学会，跨原始ID表现明显较差。第40遍起完整拟合评估的数字、运动、联合准确率均为100%；控制池联合最高27.50%，最近27.41%。这次结果支持主要瓶颈是泛化差距，不能将低控制准确率简单归因于隐藏节点不足。完全拟合当前有限数据也不证明模型已形成正确的任务表征。", "",
        "## 2. 准确率与损失", "",
        "| 评估 | 遍数 | 数字 | 运动 | 联合 | 数字＋运动损失 |", "|---|---:|---:|---:|---:|---:|"]
    old = next(row for row in result["epoch_rows"] if row["epoch"] == 15)
    for label, epoch, metrics in (("降率前控制", 15, old["control"]),
                                 ("控制联合最高", result["best_joint_control_epoch"], result["best_joint_control"]),
                                 ("最近完整控制", result["latest_control_epoch"], control),
                                 ("最近完整拟合", result["latest_fit_epoch"], fit)):
        lines.append(f"| {label} | {epoch} | {metrics['digit_accuracy']:.3%} | {metrics['motion_accuracy']:.3%} | {metrics['joint_accuracy']:.3%} | {metrics['loss']:.6f} |")
    lines += ["", "拟合池4480视频/160原始ID；控制池1120视频/40不同原始ID。控制池用于学习率与检查点选择，结果属于探索反馈，不能称独立验证或最终测试成绩。", "",
        f"程序的best.pt按数字＋运动＋联合准确率之和选择，实际选择第{result['selected_best_epoch']}遍；联合单项最高在第{result['best_joint_control_epoch']}遍。两个标准须区分。当前resume.pt在第76遍又更新了4批，未重新评价；表中第75遍成绩对应最近完成的控制评价，不冒称暂停瞬间权重的精确成绩。", "",
        "![训练曲线](" + prefix.name + "_curves.png)", "",
        "从第35遍之后，控制联合约在27%附近平台，而控制交叉熵持续上升到7.929。这符合明显过拟合的表现。未进行跨种子复现或机制消融，不能把泛化问题唯一归因于Fire、LIF、时间表达或模型容量。", "",
        "## 3. 学习率变化", "", "| 生效遍数 | 原学习率 | 新学习率 |", "|---:|---:|---:|"]
    for event in result["learning_rate_events"]:
        if event.get("reduced"):
            lines.append(f"| {event['effective_epoch']} | {event['learning_rates_before'][0]:g} | {event['learning_rates_after'][0]:g} |")
    lines += ["", "每5遍完整控制评估；损失相对历史有效最好值改善超过1%才重置平台，连续2次未改善则减半，冷却1次评价，下限1e-5。第76遍已按3.125e-5实际更新4批。下降机制确实执行，但没有同预算固定学习率对照，不能把后续改善全部归因于它。", "",
        "## 4. 模型规模与实际发放", "",
        "三层始终5000/3000/2000 LIF，总10000隐藏LIF；17输出为不Fire、不leak的累积单元，总计算单元10017。24,954,000相邻层边权保持全连接，输入3,920,000、内部21,000,000、输出34,000，密度均1。只训练边权；泄漏exp(-5/20)、阈值0.5固定，未生长、拆分或删减。每视频8帧，每帧4个5ms神经步，共32步/160ms；实际视频播放时长未规定。视频边界全部状态清零。", "",
        f"以下活动统计来自第{result['latest_control_epoch']}遍完整控制池；静默指该池全部视频均无Fire，不是永久失能。", "",
        "| 隐藏层 | LIF总数 | 完全静默 | 至少Fire一次 | 每LIF每步Fire率 |", "|---|---:|---:|---:|---:|"]
    for index, width in enumerate(control["resources"]["hidden_layers"]):
        silent = control["silent_lif_by_layer"][index]
        lines.append(f"| {index + 1} | {width} | {silent} | {width - silent} | {control['spikes_per_lif_step_by_layer'][index]:.5%} |")
    lines += ["", f"完整控制池实际Fire共{control['hidden_spikes']:,}次，平均每视频{control['mean_spikes_per_video']:.2f}、每帧{control['mean_spikes_per_frame']:.2f}；输出Fire为0。Fire与连接数是资源代理，没有实测能耗。", "",
        "## 5. 错误分布", "", "第75遍各数字召回率：", "",
        "| 数字 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    digit = np.asarray(control["digit_confusion"])
    lines.append("| 召回率 | " + " | ".join(f"{100 * digit[i, i] / digit[i].sum():.2f}%" for i in range(10)) + " |")
    lines += ["", "| 运动 | 召回率 |", "|---|---:|"]
    motion = np.asarray(control["motion_confusion"])
    for index, name in enumerate(MOTIONS):
        lines.append(f"| {name} | {motion[index, index] / motion[index].sum():.3%} |")
    lines += ["", "完整数字、运动与70类联合混淆矩阵在复核JSON中。70类联合按数字×7＋运动编号。标签与输出均来自已保存的评估证据，本次报告没有新模型前向。", "",
        "## 6. 实际成本与保存位置", "",
        f"实际优化器更新{result['completed_optimizer_updates']:,}次＝75×35＋4；有效批量128，微批量32，AdamW、FP32、梯度范数裁剪1。模型唯一根11、数据根20261004、v5数据与原始ID划分保持。", "",
        "| 工作 | 前向视频次 | 反传视频次 |", "|---|---:|---:|"]
    cost = result["costs"]
    for label, forward, backward in (("参数优化", cost["train_forward_videos"], cost["train_backward_videos"]),
                                     ("初始无标签校准", cost["calibration_forward_videos"], 0),
                                     ("初始梯度诊断", cost["diagnostic_forward_videos"], cost["diagnostic_backward_videos"]),
                                     ("控制评估（15次）", cost["control_forward_videos"], 0),
                                     ("拟合评估（7次）", cost["fit_evaluation_forward_videos"], 0)):
        lines.append(f"| {label} | {forward:,} | {backward:,} |")
    lines += ["", f"训练两会话墙钟合计{result['combined_training_session_wall_seconds'] / 3600:.3f}小时：原会话{result['parent_session_wall_seconds'] / 3600:.3f}小时，接续{result['adaptive_session_wall_seconds'] / 3600:.3f}小时。含共享GPU及第65→66遍约5小时日志推进间隔，原因未由日志确认，不等同于纯计算时间。初始化校准/诊断墙钟另计约20.88秒，继承核对与报告复算另计工程工作。", "",
        f"设备RTX 5080 Laptop GPU；训练记录GPU分配峰值{result['gpu_peak_allocated_bytes']:,}字节（{result['gpu_peak_allocated_bytes'] / 2**20:.2f} MiB），进程工作集峰值{result['process_peak_working_set_bytes']:,}字节。没有OOM重试或数值错误，stderr为空。未修改ONN环境。", "",
        "原运行：experiments/runs/large_lif_baseline_v1/；续训：experiments/runs/large_lif_baseline_adaptive_v1/。resume.pt保存第76遍512视频后的完整状态，best.pt保存第75遍选择模型，checkpoints/epoch_045_model.pt保存联合最高模型。报告JSON记录源哈希、检查点SHA256、完整曲线、混淆矩阵、活动与成本。", "",
        "## 7. 复核与结论范围", "",
        "读取最终检查点，四组Adam步数与实际视频账相符；全部75遍记录连续。保存的最佳联合控制、最新控制与最新拟合预测，复算损失、三项准确率及数字/运动混淆矩阵全部一致，并生成联合混淆矩阵。拟合与控制原始ID无交集，训练源码哈希吻合。报告生成新增模型前向/反传/优化更新均0；原验证与封存测试未读取。", "",
        "当前应优先处理跨ID泛化、训练原始笔迹覆盖与表征形成，再决定是否扩容或剪减。160个拟合原始ID生成的4480视频不能替代更多独立笔迹。该结果不支持将大规模删减本身当作成功，也不足以证明某一种LIF机制必然失败。用户已暂停，本报告不追加训练。", "",
        f"完整复核：[JSON证据]({prefix.name}.json)。"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=adaptive.RUN_DIR)
    parser.add_argument("--prefix", type=Path, default=PREFIX)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = collect(args.outdir)
    args.prefix.parent.mkdir(parents=True, exist_ok=True)
    plot_curves(result["epoch_rows"], result["progress"]["epoch"], result["current_learning_rate"],
                args.prefix.with_name(args.prefix.name + "_curves.png"))
    # PyTorch调度状态包含Infinity；导出报告时转成字符串，保持严格JSON格式。
    def finite_json(value):
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        if isinstance(value, dict):
            return {key: finite_json(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [finite_json(item) for item in value]
        return value
    args.prefix.with_suffix(".json").write_text(json.dumps(finite_json(result), ensure_ascii=False,
                                                        indent=2, allow_nan=False) + "\n", encoding="utf-8")
    args.prefix.with_suffix(".md").write_text(markdown(result, args.prefix), encoding="utf-8")
    print(json.dumps(dict(status=result["status"], completed_epochs=result["completed_epochs"],
        partial_epoch=result["progress"]["epoch"], partial_videos=result["progress"]["next_sample"],
        best_joint_epoch=result["best_joint_control_epoch"], selected_best_epoch=result["selected_best_epoch"],
        current_lr=result["current_learning_rate"], optimizer_updates=result["completed_optimizer_updates"],
        checks_ok=True, report=str(args.prefix.with_suffix(".md"))), ensure_ascii=False))


if __name__ == "__main__":
    main()
