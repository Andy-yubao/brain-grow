"""汇总实际完成的同数据对照与预算修订，复核输出/成本并生成中文报告。"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from braingrow.evaluation.metrics import evaluate
from braingrow.network.ann_controls import ARMS
from tools.run_layered_gradient import write_json


ROOT = Path("experiments/runs/ann_comparison_v1")
PREFIX = Path("reports/ann_comparison_20261005")
NAMES = dict(video_mlp="视频MLP", stacked_cnn="8帧通道CNN", cnn_gru="逐帧CNN＋GRU")
MACS = dict(video_mlp=24714000, stacked_cnn=10641664, cnn_gru=68395264)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def accuracy_text(metrics):
    return " / ".join(f"{100 * metrics[key]:.2f}%"
                      for key in ("digit_accuracy", "motion_accuracy", "joint_accuracy"))


def verify_output(path, metadata):
    with np.load(path) as arrays:
        output = arrays["logits"] if "logits" in arrays else arrays["potential"]
        target = arrays["y"]
    recalculated = evaluate(output, target)
    for key in recalculated:
        if isinstance(recalculated[key], list):
            assert recalculated[key] == metadata[key], (path, key)
        else:
            assert abs(recalculated[key] - metadata[key]) < 1e-10, (path, key)
    prediction = np.column_stack((output[:, :10].argmax(1), output[:, 10:].argmax(1)))
    joint_confusion = np.zeros((70, 70), dtype=np.int64)
    np.add.at(joint_confusion, (target[:, 0] * 7 + target[:, 1],
                              prediction[:, 0] * 7 + prediction[:, 1]), 1)
    return prediction, target, joint_confusion


def recall(metrics, key):
    cm = np.asarray(metrics[key])
    return (cm.diagonal() / cm.sum(1)).tolist()


def paired_id_analysis(baseline, candidate, target, ids):
    first, second = ((p == target).all(1) for p in (baseline, candidate))
    per_id = []
    for raw_id in np.unique(ids):
        mask = ids == raw_id
        per_id.append(dict(raw_id=int(raw_id), videos=int(mask.sum()),
            snn_joint=float(first[mask].mean()), ann_joint=float(second[mask].mean()),
            joint_delta=float(second[mask].mean() - first[mask].mean())))
    delta = np.asarray([row["joint_delta"] for row in per_id])
    return dict(both_correct=int((first & second).sum()),
        snn_only_correct=int((first & ~second).sum()), ann_only_correct=int((~first & second).sum()),
        both_wrong=int((~first & ~second).sum()), raw_ids_improved=int((delta > 0).sum()),
        raw_ids_tied=int((delta == 0).sum()), raw_ids_worse=int((delta < 0).sum()),
        per_raw_id=per_id,
        note="同ID的28视频相关；报告40个原始ID的配对差值，不把1120视频当独立统计重复。")


def plot_figures(evidence, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    colors = dict(video_mlp="#5668c8", stacked_cnn="#178a6e", cnn_gru="#c77c28")
    for arm in ARMS:
        epoch_rows = rows[arm]
        controls = [r for r in epoch_rows if r["control"] is not None]
        for index, key in enumerate(("digit_accuracy", "motion_accuracy", "joint_accuracy")):
            axes[index].plot([r["epoch"] for r in controls],
                [100 * r["control"][key] for r in controls], marker="o", markersize=3,
                label=NAMES[arm], color=colors[arm])
    for index, key in enumerate(("digit_accuracy", "motion_accuracy", "joint_accuracy")):
        value = 100 * evidence["snn_epoch10"]["control"][key]
        axes[index].scatter([10], [value], marker="*", s=130, color="#b33a40", zorder=5,
                            label="SNN 第10遍")
        axes[index].set(title=("数字", "运动", "联合")[index] + "控制准确率", xlabel="训练遍数", ylabel="准确率 (%)", ylim=(0, 102))
        axes[index].grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("相同4480拟合 / 1120控制视频；根11；控制结果属于探索证据")
    fig.savefig(PREFIX.with_name(PREFIX.name + "_curves.png"), dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(4, 2, figsize=(11, 17), constrained_layout=True)
    models = [("SNN第10遍", evidence["snn_epoch10"]["control"])]
    models += [(NAMES[arm] + f" 最佳第{evidence['arms'][arm]['best']['epoch']}遍",
                evidence["arms"][arm]["best"]["metrics"]) for arm in ARMS]
    for row, (name, metrics) in enumerate(models):
        for col, key in enumerate(("digit_confusion", "motion_confusion")):
            cm = np.asarray(metrics[key])
            normalized = cm / cm.sum(1, keepdims=True)
            ax = axes[row, col]
            ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
            labels = list(range(10)) if col == 0 else ["静", "上", "下", "左", "右", "顺", "逆"]
            ax.set(xticks=range(len(labels)), yticks=range(len(labels)),
                   xticklabels=labels, yticklabels=labels, xlabel="预测类别", ylabel="真实类别",
                   title=name + (" · 数字" if col == 0 else " · 运动"))
            for i in range(len(cm)):
                for j in range(len(cm)):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7,
                            color="white" if normalized[i, j] > 0.5 else "#222222")
    fig.suptitle("控制集混淆矩阵：格内为视频数，颜色为每真实类别比例；最佳检查点参与了控制选择")
    fig.savefig(PREFIX.with_name(PREFIX.name + "_confusions.png"), dpi=160)
    plt.close(fig)


def main():
    summary = read(ROOT / "summary.json")
    assert summary["completed"]
    split = np.load(ROOT / "split_evidence.npz")
    ids = split["control_ids"]
    target = split["control_y"]
    assert len(ids) == 1120 and len(np.unique(ids)) == 40
    assert len(set(split["fit_ids"]) & set(ids)) == 0
    baseline = dict(control=read(ROOT / "snn_reference/evaluations/control_010.json"),
                    fit=read(ROOT / "snn_reference/evaluations/fit_010.json"),
                    epoch=read(ROOT / "snn_reference/epoch_010.json"),
                    config=read(ROOT / "snn_reference/config.json"),
                    provenance=read(ROOT / "snn_reference/provenance.json"))
    snn_prediction, snn_target, snn_joint = verify_output(
        ROOT / "snn_reference/evaluations/control_010.npz", baseline["control"])
    verify_output(ROOT / "snn_reference/evaluations/fit_010.npz", baseline["fit"])
    np.testing.assert_array_equal(snn_target, target)
    evidence = dict(completed=True, data=summary["data"], snn_epoch10=baseline, arms={},
        validation_forward_evaluations=0, sealed_test_reads=0,
        audit=dict(outputs_and_confusions_verified=True, cost_ledger_verified=True,
                   fit_control_original_ids_disjoint=True, extra_model_forward_evaluations=0))
    matrices = dict(snn_epoch10_joint_confusion=snn_joint,
        joint_label_order_digit_motion=np.column_stack((np.repeat(np.arange(10), 7), np.tile(np.arange(7), 10))))
    rows = {}
    for arm in ARMS:
        directory = ROOT / arm
        result = read(directory / "result.json")
        config = read(directory / "config.json")
        diag = read(directory / "diagnostic.json")
        assert result["status"] == "completed" and result["completed_epochs"] == config["maximum_epochs"]
        completed = result["completed_epochs"]
        assert completed >= 10
        assert config["data"] == summary["data"] and config["model_seed"] == 11
        assert diag["ok"] and diag["full_optimizer_and_rng_replay_exact"]
        epoch10 = read(directory / "epoch_010.json")
        budget_end = read(directory / f"epoch_{completed:03d}.json")
        epoch100 = budget_end if completed == 100 else None
        best = result["best_control"]
        rows[arm] = [read(directory / f"epoch_{e:03d}.json") for e in range(1, completed + 1)]
        for row in rows[arm]:
            for pool in ("control", "fit"):
                if row[pool] is not None:
                    prediction, stored_target, joint = verify_output(
                        directory / "evaluations" / f"{pool}_{row['epoch']:03d}.npz", row[pool])
                    if pool == "control":
                        np.testing.assert_array_equal(stored_target, target)
                    if pool == "control" and row["epoch"] in (10, completed, best["epoch"]):
                        matrices[f"{arm}_control_epoch{row['epoch']}_joint_confusion"] = joint
        verify_output(directory / "evaluations" / f"selected_fit_{best['epoch']:03d}.npz",
                      result["best_control_fit"])
        prediction10, _, _ = verify_output(directory / "evaluations/control_010.npz", epoch10["control"])
        prediction_best, _, _ = verify_output(directory / "evaluations" / f"control_{best['epoch']:03d}.npz", best["metrics"])
        last_journal = json.loads((directory / "cost_journal.jsonl").read_text(encoding="utf-8").splitlines()[-1])
        assert result["costs"] == last_journal["cumulative_costs"]
        expected = dict(train_forward_videos=completed * 4480, train_backward_videos=completed * 4480,
            optimizer_updates=completed * 35, control_forward_videos=(completed // 5) * 1120,
            fit_forward_videos=(completed // 10) * 4480,
            control_metric_evaluations=completed // 5, fit_metric_evaluations=completed // 10,
            selected_fit_forward_videos=4480, selected_fit_metric_evaluations=1,
            diagnostic_forward_videos=160, diagnostic_backward_videos=96, diagnostic_optimizer_updates=3)
        for key, value in expected.items():
            assert result["costs"][key] >= value, (arm, key, result["costs"][key], value)
            if key != "train_forward_videos" and key != "train_backward_videos" and key != "optimizer_updates":
                assert result["costs"][key] == value, (arm, key)
        evidence["arms"][arm] = dict(config=config, diagnostic=diag, result=result,
            epoch10=epoch10, epoch100=epoch100, budget_end=budget_end, completed_budget_epochs=completed, best=best,
            paired_epoch10=paired_id_analysis(snn_prediction, prediction10, target, ids),
            paired_best=paired_id_analysis(snn_prediction, prediction_best, target, ids),
            digit_recall_best=recall(best["metrics"], "digit_confusion"),
            motion_recall_best=recall(best["metrics"], "motion_confusion"),
            analytic_macs_per_video=MACS[arm],
            checkpoint_sha256={name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                               for name in ("best.pt", "epoch_010_model.pt", "resume.pt")})
        frozen = PREFIX.with_name(PREFIX.name + "_evidence") / arm
        frozen.mkdir(parents=True, exist_ok=True)
        for name in ("config.json", "diagnostic.json", "result.json", "best_meta.json", "epoch_010.json", f"epoch_{completed:03d}.json"):
            shutil.copy2(directory / name, frozen / name)
        for pool, epoch in (("control", 10), ("fit", 10), ("control", completed), ("fit", completed),
                            ("control", best["epoch"]), ("selected_fit", best["epoch"])):
            for ext in ("json", "npz"):
                shutil.copy2(directory / "evaluations" / f"{pool}_{epoch:03d}.{ext}", frozen / f"{pool}_{epoch:03d}.{ext}")
        write_json(frozen / "curves.json", [dict(epoch=r["epoch"],
            optimizer_mean_loss=r["optimizer_mean_loss"], control=r["control"], fit=r["fit"],
            optimizer_training_wall_seconds=r["optimizer_training_wall_seconds"]) for r in rows[arm]])
        for name in ("microbatch_probe.json", "budget_cap.json", "config_before_budget_cap.json",
                     "result_at_engineering_stop.json", "result_before_budget_cap.json"):
            if (directory / name).exists():
                shutil.copy2(directory / name, frozen / name)
                evidence["arms"][arm][name.removesuffix(".json")] = read(directory / name)
    totals = {key: sum(evidence["arms"][arm]["result"]["costs"][key] for arm in ARMS)
              for key in evidence["arms"][ARMS[0]]["result"]["costs"]}
    evidence["total_costs"] = totals
    engineering_probes = [info["microbatch_probe"] for info in evidence["arms"].values()
                          if "microbatch_probe" in info]
    evidence["additional_engineering_costs"] = dict(
        forward_videos=sum(p["forward_videos"] for p in engineering_probes),
        backward_videos=sum(p["backward_videos"] for p in engineering_probes), optimizer_updates=0)
    combined_summary = dict(completed=True, arms=list(ARMS), data=summary["data"],
        common_comparison_epoch=10,
        completed_budget_epochs={arm: evidence["arms"][arm]["completed_budget_epochs"] for arm in ARMS},
        results={arm: evidence["arms"][arm]["result"] for arm in ARMS},
        total_costs=totals, additional_engineering_costs=evidence["additional_engineering_costs"],
        validation_forward_evaluations=0, sealed_test_reads=0,
        resume_commands=[
            ".\\.venv\\Scripts\\python.exe -u -m tools.train_ann_comparison --resume --arms video_mlp stacked_cnn --epochs 100",
            ".\\.venv\\Scripts\\python.exe -u -m tools.train_ann_comparison --resume --arms cnn_gru --epochs 10"],
        budget_note="GRU在查看控制成绩前收束本轮为10遍；原100遍扩展未执行。")
    write_json(ROOT / "summary.json", combined_summary)
    write_json(PREFIX.with_name(PREFIX.name + "_evidence") / "summary.json", combined_summary)
    write_json(PREFIX.with_suffix(".json"), evidence)
    np.savez_compressed(PREFIX.with_name(PREFIX.name + "_joint_confusions.npz"), **matrices)
    frozen_ref = PREFIX.with_name(PREFIX.name + "_evidence") / "snn_epoch10"
    shutil.copytree(ROOT / "snn_reference", frozen_ref, dirs_exist_ok=True)
    shutil.copy2(ROOT / "split_evidence.npz", frozen_ref.parent / "split_evidence.npz")
    plot_figures(evidence, rows)
    strongest10 = max(ARMS, key=lambda arm: evidence["arms"][arm]["epoch10"]["control"]["joint_accuracy"])
    strongest_full = max(ARMS, key=lambda arm: evidence["arms"][arm]["best"]["metrics"]["joint_accuracy"])
    leading10 = evidence["arms"][strongest10]["epoch10"]["control"]
    delta10 = 100 * (leading10["joint_accuracy"] - baseline["control"]["joint_accuracy"])
    lines = ["# 同数据非SNN对照：共同10遍及扩展结果", "", "2026-10-05。本报告只汇总真实完成工作，所有数字按保存logit重新计算并复核混淆矩阵和成本账；汇总没有增加模型前向。", "",
        f"同第10遍，对照中控制联合最高的是{NAMES[strongest10]}：{100*leading10['joint_accuracy']:.2f}%，相对SNN第10遍为{delta10:+.2f}个百分点。各臂实际预算内最高探索控制联合为{100*evidence['arms'][strongest_full]['best']['metrics']['joint_accuracy']:.2f}%（{NAMES[strongest_full]}）；后者涉及不同遍数，不能当作与10遍SNN等预算比较。", "",
        "实际预算：视频MLP和8帧通道CNN各完成100遍；CNN＋GRU原计划100遍，因用户后台GPU任务持续占用且不能干预，在查看任何GRU控制结果前将本轮收束为10遍。它的90遍扩展未执行，不称失败或收敛。三模型第10遍是本轮主要共同预算证据。", "",
        "数据与SNN完全一致：v5、根20261004、8帧×28×28原灰度，4480拟合视频/160原始ID、1120控制视频/40原始ID，原始ID隔离。三臂唯一模型根11、从零训练，无外部数据、预训练或新增增强。原验证/封存测试均未解包或前向。", "",
        "## 第10遍：同遍数、同44800拟合视频次", "",
        "三项顺序均为数字 / 运动 / 联合。SNN沿用用户指定第10遍冻结证据；三臂同批量128、交叉熵目标、AdamW与学习率计划，但网络时间结构不同，不是单变量LIF替换消融。", "",
        "| 模型 | 可训练参数 | 控制准确率 | 拟合准确率 |", "|---|---:|---|---|",
        f"| SNN 5000/3000/2000 | 24,954,000 | {accuracy_text(baseline['control'])} | {accuracy_text(baseline['fit'])} |"]
    for arm in ARMS:
        info = evidence["arms"][arm]
        lines.append(f"| {NAMES[arm]} | {info['config']['resources']['trainable_parameters']:,} | {accuracy_text(info['epoch10']['control'])} | {accuracy_text(info['epoch10']['fit'])} |")
    lines += ["", "## 各臂实际预算末态与控制最佳检查点", "",
        "100遍和10遍结果不可直接称为等预算胜负。最佳检查点按最高控制联合准确率选择，是反复查看控制后的探索成绩；最终测试仍封存。", "",
        "| 模型 | 完成遍数 | 预算末态控制 | 预算末态拟合 | 最佳遍数 | 最佳控制 | 同检查点拟合 |", "|---|---:|---|---|---:|---|---|"]
    for arm in ARMS:
        info = evidence["arms"][arm]
        lines.append(f"| {NAMES[arm]} | {info['completed_budget_epochs']} | {accuracy_text(info['budget_end']['control'])} | {accuracy_text(info['budget_end']['fit'])} | {info['best']['epoch']} | {accuracy_text(info['best']['metrics'])} | {accuracy_text(info['result']['best_control_fit'])} |")
    mlp = evidence["arms"]["video_mlp"]
    gru = evidence["arms"]["cnn_gru"]
    lines += ["", "## 结论的适用范围", "",
        f"视频MLP参数比SNN少{100*(1-mlp['config']['resources']['trainable_parameters']/24954000):.2f}%，属于接近参数量的对照。第100遍拟合联合{100*mlp['epoch100']['fit']['joint_accuracy']:.2f}%，控制联合{100*mlp['epoch100']['control']['joint_accuracy']:.2f}%；这说明本轮充分记忆拟合视频与跨原始ID识别是不同问题，增加密集参数不能自动解决泛化。", "",
        f"CNN＋GRU第10遍拟合联合仅{100*gru['epoch10']['fit']['joint_accuracy']:.2f}%，控制联合{100*gru['epoch10']['control']['joint_accuracy']:.2f}%，仍明显欠拟合；它未完成原100遍扩展，不能根据本轮10遍成绩断言充分训练后的性能上限或判定GRU架构普遍不适合。", "",
        "CNN/GRU允许卷积共享、池化、ReLU、GRU和不同时间读出；这些变化合在一起比较常规架构与当前SNN，不能把差值归因于单独的Fire或LIF机制。三模型只使用同样原始像素，无额外教师/特征；初始化和网络结构差异完整披露。", "",
        "这是根11、40个控制原始ID上的有限对照。控制集曾被旧研究查看，又用于选择本轮最佳检查点，所以最佳分数只表示探索结果。没有冻结后新验证或最终测试；没有证明某一模型家族的普遍优势，也没有把运动准确率等同于已证实的运动计算机制。"]
    lines += ["", "## 配对错误与类别表现", "",
        "同一原始ID的28条视频相关，以下按40个ID报告联合准确率改善/持平/下降；不提供把1120条视频假作独立样本的显著性结论。", "",
        "| 对照第10遍 vs SNN第10遍 | 联合差值（百分点） | ID改善/持平/下降 | 仅ANN正确 / 仅SNN正确 |", "|---|---:|---|---|"]
    for arm in ARMS:
        info = evidence["arms"][arm]
        pair = info["paired_epoch10"]
        delta = 100 * (info["epoch10"]["control"]["joint_accuracy"] - baseline["control"]["joint_accuracy"])
        lines.append(f"| {NAMES[arm]} | {delta:+.2f} | {pair['raw_ids_improved']}/{pair['raw_ids_tied']}/{pair['raw_ids_worse']} | {pair['ann_only_correct']} / {pair['snn_only_correct']} |")
    for arm in ARMS:
        info = evidence["arms"][arm]
        digits = ", ".join(f"{i}:{100*r:.1f}%" for i, r in enumerate(info["digit_recall_best"]))
        motions = ", ".join(f"{label}:{100*r:.1f}%" for label, r in zip(["静止", "上", "下", "左", "右", "顺", "逆"], info["motion_recall_best"]))
        lines += ["", f"{NAMES[arm]}控制最佳：逐数字召回 {digits}；逐运动召回 {motions}。"]
    lines += ["", "## 成本、诊断与保存", "",
        "设备为本地RTX 5080 Laptop GPU、PyTorch 2.13.0+cu130、FP32最高精度、确定性算法、TF32关闭。用户SNN同时运行，因此墙钟包含共享GPU负载，不能当独占速度或能耗。Fire、神经仿真步及生长曲线对这些固定非LIF模型不适用；配置中另列参数分组与每视频逻辑激活位置，不能与SNN物理单元直接等同。", "",
        "三臂均完成真实32拟合视频诊断：所有权重梯度有效、模型/优化器/RNG重放完全一致、视频批次隔离及帧序敏感检查通过。每臂诊断160前向/96反传视频次、3优化器更新（含重放），随后恢复第0遍；不继承诊断更新。", "",
        "| 模型 | 已记录墙钟（秒，含诊断） | 优化训练墙钟（秒） | 最终会话GPU分配峰值（MiB） | 最终会话进程工作集峰值（MiB） | 理论MAC/视频（百万） |", "|---|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        info = evidence["arms"][arm]
        result = info["result"]
        lines.append(f"| {NAMES[arm]} | {result['wall_seconds_including_diagnostic']:.1f} | {result['optimizer_training_wall_seconds']:.1f} | {result['gpu_peak_allocated_bytes']/2**20:.1f} | {result['process_peak_working_set_bytes']/2**20:.1f} | {MACS[arm]/1e6:.2f} |")
    lines += ["", "已记录墙钟为各臂诊断和正式训练阶段之和，不含共同数据加载、最初模型创建及报告生成。内存表取最终运行会话，GPU峰值在各臂正式运行前重置，进程工作集峰值包含同一进程此前阶段；GRU跨会话的原始停止记录另存在证据目录，不能把表内进程峰值当纯模型内存。准备诊断中的GPU计数器没有按臂重置，其原始数值只能表示截至该阶段的累计峰值，不能直接用于比较三个模型。理论MAC仅计卷积、线性和GRU矩阵乘，含共享权重的重复使用，未计激活/池化/门逐元素计算、反传和优化器；不是实测运算时间或能耗。", "",
        f"三臂总实际成本：训练前向{totals['train_forward_videos']:,}、训练反传{totals['train_backward_videos']:,}、正式更新{totals['optimizer_updates']:,}；控制{totals['control_forward_videos']:,}视频次/{totals['control_metric_evaluations']}次评价；拟合{totals['fit_forward_videos']:,}视频次/{totals['fit_metric_evaluations']}次评价；冻结最佳后的额外拟合{totals['selected_fit_forward_videos']:,}视频次/{totals['selected_fit_metric_evaluations']}次。诊断另为前向{totals['diagnostic_forward_videos']}、反传{totals['diagnostic_backward_videos']}、更新{totals['diagnostic_optimizer_updates']}。", "",
        "证据索引：[完整JSON](ann_comparison_20261005.json)、[重要评价输出和配置](ann_comparison_20261005_evidence/)、[联合70类混淆矩阵](ann_comparison_20261005_joint_confusions.npz)、[控制曲线](ann_comparison_20261005_curves.png)、[数字/运动混淆矩阵图](ann_comparison_20261005_confusions.png)。联合类别按digit×7＋motion排列，NPZ同时保存标签顺序。完整检查点/逐批成本/源码留在`experiments/runs/ann_comparison_v1/`；模型哈希在JSON中。", "",
        "额外工程核对：CNN＋GRU第3遍检查点上比较32分批与128直接计算，另发生256前向/256反传视频次、0更新；损失与梯度相符，但共享GPU下没有证实分批加速，所以未采用。训练源码哈希恢复原值，继续原128执行。原配置/管理预算修订和原检查点均保存，后台另一个GPU任务没有被停止、修改或调优。", "",
        "已完成目录恢复不重复模型前向或更新。MLP/CNN命令：`.\\.venv\\Scripts\\python.exe -u -m tools.train_ann_comparison --resume --arms video_mlp stacked_cnn --epochs 100`；GRU命令：`.\\.venv\\Scripts\\python.exe -u -m tools.train_ann_comparison --resume --arms cnn_gru --epochs 10`。实现与预算见[登记](../docs/ann_comparison_execution.md)。没有新增依赖或支出，不修改ONN；现有SNN进程/检查点保持。", ""]
    PREFIX.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(dict(ok=True, report=str(PREFIX.with_suffix('.md')), total_costs=totals), ensure_ascii=False))


if __name__ == "__main__":
    main()
