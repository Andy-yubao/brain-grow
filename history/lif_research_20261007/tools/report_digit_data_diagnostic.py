"""从保存预测复核四组数据诊断；报告不执行模型前向或更新。"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import torch

from braingrow.evaluation.metrics import evaluate
from tools.run_layered_gradient import write_json
from tools.train_ann_comparison import nested_equal, sha
from tools.train_digit_data_diagnostic import ARMS, ROOT, read


PREFIX = Path("reports/digit_data_diagnostic_20261005")
NAMES = dict(small_full="小数字＋全圆初始角", small_narrow="小数字＋窄初始角",
             large_full="大数字＋全圆初始角", large_narrow="大数字＋窄初始角")


def recheck(path, expected):
    with np.load(path) as arrays:
        scores, targets = arrays["logits"], arrays["y"]
    metrics = evaluate(scores, targets)
    for key, value in metrics.items():
        if isinstance(value, list):
            assert value == expected[key], (path, key)
        else:
            assert abs(value - expected[key]) < 1e-10, (path, key)
    prediction = np.column_stack((scores[:, :10].argmax(1), scores[:, 10:].argmax(1)))
    cm = np.zeros((70, 70), dtype=np.int64)
    np.add.at(cm, (targets[:, 0] * 7 + targets[:, 1], prediction[:, 0] * 7 + prediction[:, 1]), 1)
    return prediction, targets, cm


def figures(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    colors = dict(small_full="#616c80", small_narrow="#2873b8", large_full="#df8a29", large_narrow="#268064")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for arm in ARMS:
        for ax, pool in zip(axes, ["control", "fit"]):
            observed = [r for r in rows[arm] if r[pool] is not None]
            ax.plot([r["epoch"] for r in observed], [100 * r[pool]["digit_accuracy"] for r in observed],
                    marker="o", color=colors[arm], label=NAMES[arm])
            ax.set(xlabel="训练遍数", ylabel="数字准确率 (%)", ylim=(0, 102),
                   title="控制：40个新笔迹" if pool == "control" else "拟合：160个训练笔迹")
            ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("同CNN / 同初始权重 / 每组20遍；四组共同短位移和安全位置，属于独立诊断")
    fig.savefig(PREFIX.with_name(PREFIX.name + "_curves.png"), dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(4, 10, figsize=(15, 8.3), constrained_layout=True)
    for row, arm in enumerate(ARMS):
        with np.load(ROOT / f"{arm}_data.npz") as arrays:
            mask = ~arrays["fit_mask"]
            ids, targets, variants, poses = arrays["ids"][mask], arrays["y"][mask], arrays["variants"][mask], arrays["poses"][mask]
            pixels = arrays["x"][mask]
        for digit in range(10):
            raw_id = int(ids[targets[:, 0] == digit][0])
            index = int(np.flatnonzero((ids == raw_id) & (variants == 0) & (targets[:, 1] == 0))[0])
            axes[row, digit].imshow(pixels[index, 0].reshape(28, 28), cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            axes[row, digit].set_title(f"{digit} · {poses[index, 0, 2]:.0f}°", fontsize=9)
            axes[row, digit].set_xticks([])
            axes[row, digit].set_yticks([])
            if digit == 0:
                axes[row, digit].set_ylabel(NAMES[arm], fontsize=8)
    fig.suptitle("实际配对控制像素：每类首个控制ID、variant 0、静止首帧；没有挑选误判样本")
    fig.savefig(PREFIX.with_name(PREFIX.name + "_examples.png"), dpi=170)
    plt.close(fig)


def main():
    summary, config = read(ROOT / "summary.json"), read(ROOT / "config.json")
    assert summary["completed"] and config["maximum_epochs"] == 20
    evidence = PREFIX.with_name(PREFIX.name + "_evidence")
    evidence.mkdir(exist_ok=True)
    (evidence / ".gitattributes").write_text("*.json -text\n**/*.json -text\n", encoding="utf-8")
    for name in ["config.json", "summary.json", "data_audit.json", "data_pilot.json", "gpu_budget.json", "preparation_attempts.json"]:
        shutil.copy2(ROOT / name, evidence / name)
    shutil.copy2(ROOT / "diagnostic/diagnostic.json", evidence / "diagnostic.json")
    initial = torch.load(ROOT / "initial_model.pt", map_location="cpu", weights_only=False)
    arms, rows, matrices, predictions = {}, {}, {}, {}
    target_reference, id_reference = None, None
    expected_rng = np.random.default_rng(11)
    for _ in range(20):
        expected_rng.permutation(4480)
    for arm in ARMS:
        directory, frozen = ROOT / arm, evidence / arm
        frozen.mkdir(exist_ok=True)
        result = read(directory / "result.json")
        assert result["status"] == "completed" and result["completed_epochs"] == 20
        curve = [read(directory / f"epoch_{epoch:03d}.json") for epoch in range(1, 21)]
        rows[arm] = curve
        expected_costs = dict(train_forward_videos=89600, train_backward_videos=89600,
                              optimizer_updates=700, control_forward_videos=4480, control_metric_evaluations=4,
                              fit_forward_videos=8960, fit_metric_evaluations=2)
        costs = result["costs"]
        for key, value in costs.items():
            assert value == expected_costs.get(key, 0), (arm, key, value)
        ledger_last = json.loads((directory / "cost_journal.jsonl").read_text(encoding="utf-8").splitlines()[-1])
        assert ledger_last["cumulative_costs"] == costs
        saved = torch.load(directory / "resume.pt", map_location="cpu", weights_only=False)
        assert saved["state"]["completed_epochs"] == 20 and saved["state"]["costs"] == costs
        assert saved["numpy_rng"] == expected_rng.bit_generator.state
        assert saved["state"]["model_initialization_equal"]
        steps = [int(v["step"]) for v in saved["optimizer"]["state"].values()]
        assert min(steps) == max(steps) == 700 and not nested_equal(initial["model"], saved["model"])
        assert sha(directory / "resume.pt") == result["resume_sha256"]
        for row in curve:
            for pool in ["control", "fit"]:
                if row[pool] is None:
                    continue
                path = directory / "evaluations" / f"{pool}_{row['epoch']:03d}.npz"
                prediction, target, matrix = recheck(path, row[pool])
                if pool == "control":
                    if target_reference is None:
                        target_reference = target
                    np.testing.assert_array_equal(target_reference, target)
                    if row["epoch"] == 20:
                        predictions[arm], matrices[arm + "_control_joint_confusion"] = prediction, matrix
                for ext in ["npz", "json"]:
                    shutil.copy2(path.with_suffix("." + ext), frozen / path.with_suffix("." + ext).name)
        with np.load(directory / "split_evidence.npz") as split:
            ids = split["control_ids"]
            assert len(np.unique(ids)) == 40 and not set(ids.tolist()) & set(split["fit_ids"].tolist())
            if id_reference is None:
                id_reference = ids
            np.testing.assert_array_equal(id_reference, ids)
        shutil.copy2(directory / "split_evidence.npz", frozen / "split_evidence.npz")
        shutil.copy2(directory / "result.json", frozen / "result.json")
        write_json(frozen / "curves.json", curve)
        arms[arm] = dict(result=result, final_control=curve[-1]["control"], final_fit=curve[-1]["fit"],
                         control_digit_mean_epochs_15_20=float(np.mean([curve[e - 1]["control"]["digit_accuracy"] for e in [15, 20]])),
                         checkpoint_audit=dict(completed_epochs=20, optimizer_steps_min=700, optimizer_steps_max=700,
                                               weights_changed=True, extra_model_forward_videos=0))
        del saved
    contrasts = {}
    for name, first, second in [("窄角收益_小数字", "small_full", "small_narrow"),
                                ("窄角收益_大数字", "large_full", "large_narrow"),
                                ("放大收益_全圆", "small_full", "large_full"),
                                ("放大收益_窄角", "small_narrow", "large_narrow")]:
        mean_delta = arms[second]["control_digit_mean_epochs_15_20"] - arms[first]["control_digit_mean_epochs_15_20"]
        final_delta = arms[second]["final_control"]["digit_accuracy"] - arms[first]["final_control"]["digit_accuracy"]
        per_id = [dict(raw_id=int(raw_id), delta=float((predictions[second][id_reference == raw_id, 0] == target_reference[id_reference == raw_id, 0]).mean()
                                                     - (predictions[first][id_reference == raw_id, 0] == target_reference[id_reference == raw_id, 0]).mean()))
                  for raw_id in np.unique(id_reference)]
        contrasts[name] = dict(first=first, second=second, mean_delta_percentage_points=100 * mean_delta,
                              final_delta_percentage_points=100 * final_delta,
                              practical_support=bool(mean_delta >= .05 and final_delta > 0), per_original_id=per_id,
                              raw_ids_improved=sum(r["delta"] > 0 for r in per_id),
                              raw_ids_tied=sum(r["delta"] == 0 for r in per_id),
                              raw_ids_worse=sum(r["delta"] < 0 for r in per_id),
                              lower_score_arm_fit_below_70=bool(arms[first]["final_fit"]["digit_accuracy"] < .7))
    diagnostic = summary["common_diagnostic"]
    assert diagnostic["full_optimizer_and_rng_replay_exact"] and diagnostic["restored_to_original_model_optimizer_rng"]
    for key, value in [("diagnostic_forward_videos", 160), ("diagnostic_backward_videos", 96), ("diagnostic_optimizer_updates", 3)]:
        assert diagnostic["costs"][key] == value
    report = dict(completed=True, config=config, arms=arms, contrasts=contrasts,
                  interaction_percentage_points=contrasts["放大收益_窄角"]["mean_delta_percentage_points"] - contrasts["放大收益_全圆"]["mean_delta_percentage_points"],
                  common_diagnostic=diagnostic, total_costs={key: sum(arms[a]["result"]["costs"][key] for a in ARMS) for key in arms[ARMS[0]]["result"]["costs"]},
                  gpu_wall_seconds=summary["gpu_wall_seconds"], data_audit=read(ROOT / "data_audit.json"),
                  preparation_attempts=read(ROOT / "preparation_attempts.json"),
                  report_model_forward_videos=0, report_model_backward_videos=0, report_optimizer_updates=0,
                  validation_forward_evaluations=0, sealed_test_reads=0,
                  extra_data_audit_render_calls=5040, failed_preparation_render_calls=58,
                  audit=dict(saved_predictions_verified=True, cost_ledger_verified=True,
                             optimizer_step_and_checkpoint_verified=True, common_shuffle_rng_verified=True,
                             control_order_and_raw_id_isolation_verified=True))
    write_json(PREFIX.with_suffix(".json"), report)
    matrices["joint_label_order_digit_motion"] = np.column_stack((np.repeat(np.arange(10), 7), np.tile(np.arange(7), 10)))
    np.savez_compressed(PREFIX.with_name(PREFIX.name + "_joint_confusions.npz"), **matrices)
    figures(rows)
    write_markdown(report)
    print(json.dumps({"completed": True, "control": {a: arms[a]["final_control"]["digit_accuracy"] for a in ARMS},
                      "contrasts": {k: v["mean_delta_percentage_points"] for k, v in contrasts.items()},
                      "model_forward_in_report": 0}, ensure_ascii=False))


def write_markdown(report):
    lines = ["# 数字数据轻量诊断：尺寸×初始角", "",
             "四组已完成相同20遍预算；同CNN1,704,945参数、同种子11初始权重、同160拟合/40控制笔迹，模型读取28×28×8原灰度。训练目标为数字CE＋运动CE，AdamW、恒lr0.001、批量128。没有预训练或新笔迹。", "",
             "四组共同短位移4.1–4.4像素、按大前景确定的相同安全中心；净旋转仍±105°。窄初始角±15°只限制起始姿态。尺寸和角度各自逐项配对，原v5缓存/模型保持。原14像素平移任务的绝对成绩不能与本轮直接比较。", "",
             "## 相同20遍末态", "", "| 组 | 拟合数字/运动/联合 | 控制数字/运动/联合 | 控制数字第15/20遍均值 |", "|---|---|---|---:|"]
    for arm in ARMS:
        info = report["arms"][arm]
        text = lambda m: "/".join(f"{m[key]:.2%}" for key in ["digit_accuracy", "motion_accuracy", "joint_accuracy"])
        lines.append(f"| {NAMES[arm]} | {text(info['final_fit'])} | {text(info['final_control'])} | {info['control_digit_mean_epochs_15_20']:.2%} |")
    lines += ["", "本轮有限20遍下，初始角范围收窄与数字放大都通过预设收益门槛；初始角收窄的增益更大。支持小尺寸和全圆初始姿态是当前CNN学习的重要障碍，不能据此认定v5渲染损坏或排除模型因素。小数字全圆组拟合仍仅60.13%，因此比较同时反映有限预算下的可学习性，不能当作收敛性能上限。",
              "", "![数字曲线](digit_data_diagnostic_20261005_curves.png)", "", "## 预登记因素对比", "",
              "均值增益≥5个百分点且末态同向才称初步支持；控制参与判读，非独立测试，不作显著性结论。", "",
              "| 因素 | 第15/20遍平均增益（百分点） | 末态增益 | 控制笔迹改善/持平/下降 | 预设门槛 |", "|---|---:|---:|---|---|"]
    for name, info in report["contrasts"].items():
        lines.append(f"| {name} | {info['mean_delta_percentage_points']:+.2f} | {info['final_delta_percentage_points']:+.2f} | {info['raw_ids_improved']}/{info['raw_ids_tied']}/{info['raw_ids_worse']} | {'通过' if info['practical_support'] else '未通过'} |")
    lines += ["", f"交互项（窄角放大收益−全圆放大收益）：{report['interaction_percentage_points']:+.2f}个百分点。只描述这四个有限预算探针，不是总体因果贡献比例。",
              "低分组拟合低于70%时，未见收益不能排除该因素。没有扩大训练、挑选最佳检查点或换种子。本轮没有增加独立笔迹，不能确定更多笔迹的增益；CNN探针结果不能直接视为原SNN在新数据上的成绩，也未证明理论准确率上限。", "",
              "## 实际配对像素", "", "![四组实际控制首帧](digit_data_diagnostic_20261005_examples.png)", "",
              "每类同一首个控制ID、variant0、静止首帧，未按模型错误选图。保持共同中心可直接看到尺度与角度差异。", "",
              "## 审计、成本与恢复", "",
              "原v5小样本560帧逐像素复现；四组2240帧与独立大画布逐像素比较、画布外无前景；全量179200帧非空、几何边距≥1像素且纯因素配对有效。几何审计不证明无视觉歧义或捷径。准备阶段只读图像缓冲区错误已修复，失败发生在训练前，另计58次渲染调用；正式审计另计5040次渲染调用。",
              "真实32视频诊断通过梯度、视频隔离与模型/优化器/RNG更新重放，随后撤回全部更新。正式每组检查点的Adam步数均700，末态NumPy RNG与同根20次相同4480样本置乱相符；预测、三种混淆矩阵与成本日志复算相符。",
              f"正式总优化前向/反传358400/358400、2800更新；控制17920视频/16次、拟合35840视频/8次；共同诊断另计160前向/96反传/3更新含重放。GPU阶段墙钟{report['gpu_wall_seconds']:.1f}秒，数据生成{report['data_audit']['generation_wall_seconds']:.1f}秒；没有新验证或封存测试前向。报告生成0模型前向/0反传/0更新。", "",
              "| 组 | 模型阶段墙钟（秒） | 优化墙钟（秒） | GPU分配峰值（MiB） | 进程工作集峰值（MiB） |", "|---|---:|---:|---:|---:|"]
    for arm in ARMS:
        info = report["arms"][arm]["result"]
        lines.append(f"| {NAMES[arm]} | {info['wall_seconds']:.1f} | {info['optimizer_wall_seconds']:.1f} | {info['gpu_peak_allocated_bytes']/2**20:.1f} | {info['process_peak_working_set_bytes']/2**20:.1f} |")
    lines += ["", "设备RTX 5080 Laptop GPU，PyTorch版本见JSON；FP32确定性、TF32关闭。固定非LIF网络的Fire、神经仿真与生长不适用；未测能耗。每视频逻辑激活位置按原CNN配置披露，位置数不等于SNN物理单元。进程峰值包含同会话之前的数据准备/诊断，不能视为各模型独占内存。", "",
              "[完整JSON](digit_data_diagnostic_20261005.json)、[重要配置和预测证据](digit_data_diagnostic_20261005_evidence/)、[70类联合混淆矩阵](digit_data_diagnostic_20261005_joint_confusions.npz)、[事前登记](../docs/digit_data_diagnostic_execution.md)。数字/运动混淆矩阵保存在每份评价JSON，联合类按digit×7＋motion。",
              "全部缓存、源快照、初始模型、按遍恢复检查点/优化器/RNG与成本日志留在`experiments/runs/digit_data_diagnostic_v1/`。恢复命令：项目根目录`.venv\\Scripts\\python.exe -u -B -m tools.train_digit_data_diagnostic --resume`；已完成20遍会跳过训练。复核报告：`.venv\\Scripts\\python.exe -B -m tools.report_digit_data_diagnostic`。"]
    PREFIX.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
