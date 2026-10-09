"""只用CPU复核既有预测与控制像素；不加载模型、不训练、不解包验证或封存成员。"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np


PREFIX = Path("reports/digit_difficulty_20261005")
EVIDENCE = Path("reports/ann_comparison_20261005_evidence")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prediction_statistics(path, ids, target, poses, expected):
    with np.load(path) as arrays:
        scores = arrays["logits"] if "logits" in arrays.files else arrays["potential"]
        np.testing.assert_array_equal(arrays["y"], target)
        if "ids" in arrays.files:
            np.testing.assert_array_equal(arrays["ids"], ids)
    prediction = np.column_stack((scores[:, :10].argmax(1), scores[:, 10:].argmax(1)))
    correct = prediction == target
    for key, value in (("digit_accuracy", correct[:, 0].mean()),
                       ("motion_accuracy", correct[:, 1].mean()),
                       ("joint_accuracy", correct.all(1).mean())):
        assert abs(float(value) - expected[key]) < 1e-10, (path, key)
    cm = np.zeros((10, 10), dtype=np.int64)
    np.add.at(cm, (target[:, 0], prediction[:, 0]), 1)
    np.testing.assert_array_equal(cm, expected["digit_confusion"])
    pair_errors = int(cm[6, 9] + cm[9, 6])
    total_errors = int((~correct[:, 0]).sum())
    non69 = ~np.isin(target[:, 0], [6, 9])
    per_id = [dict(raw_id=int(raw_id), digit=int(target[ids == raw_id, 0][0]),
                   digit_accuracy=float(correct[ids == raw_id, 0].mean()))
              for raw_id in dict.fromkeys(ids.tolist())]
    per_motion = [dict(motion=motion, samples=int((target[:, 1] == motion).sum()),
                       digit_accuracy=float(correct[target[:, 1] == motion, 0].mean()))
                  for motion in range(7)]
    angles = (poses[:, 0, 2] + 180) % 360 - 180
    angle_bins = []
    for low in range(-180, 180, 45):
        mask = (angles >= low) & (angles < low + 45)
        angle_bins.append(dict(degrees=[low, low + 45], videos=int(mask.sum()),
                               digit_accuracy=float(correct[mask, 0].mean())))
    return dict(prediction_source=str(path), prediction_sha256=sha(path),
                digit_accuracy=float(correct[:, 0].mean()),
                motion_accuracy=float(correct[:, 1].mean()),
                joint_accuracy=float(correct.all(1).mean()), digit_confusion=cm.tolist(),
                digit_recall=(cm.diagonal() / cm.sum(1)).tolist(),
                digit_errors=total_errors, pure_6_9_swap_errors=pair_errors,
                pure_6_9_share_of_all_digit_errors=pair_errors / total_errors,
                digit_accuracy_excluding_true_6_9=float(correct[non69, 0].mean()),
                samples_excluding_true_6_9=int(non69.sum()), per_motion=per_motion,
                per_raw_id=per_id, initial_angle_bins=angle_bins,
                caveat="按运动/初始角的分组是相关性描述，不是控制笔迹和位置后的因果消融；每ID的28视频相关。")


def example_figure(x, y, ids, variants, poses):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 10, figsize=(15, 4.8), constrained_layout=True)
    selected = []
    for digit in range(10):
        raw_ids = list(dict.fromkeys(ids[y[:, 0] == digit].tolist()))[:2]
        for row, raw_id in enumerate(raw_ids):
            index = int(np.flatnonzero((ids == raw_id) & (variants == 0) & (y[:, 1] == 0))[0])
            axes[row, digit].imshow(x[index, 0].reshape(28, 28), cmap="gray", vmin=0,
                                    vmax=1, interpolation="nearest")
            angle = float(poses[index, 0, 2])
            axes[row, digit].set_title(f"原标签 {digit}\n初始角 {angle:.0f}°", fontsize=9)
            axes[row, digit].set_xlabel(f"ID {raw_id}", fontsize=8)
            axes[row, digit].set_xticks([])
            axes[row, digit].set_yticks([])
            selected.append(dict(digit=digit, raw_id=raw_id, control_video_index=index,
                                 variant=0, motion=0, frame=0, angle=angle))
    fig.suptitle("实际控制像素：每类按既有顺序取前两个ID；variant 0、静止、第1帧\n"
                 "放大显示原始28×28像素；没有旋正、裁剪或挑选误判样本", fontsize=12)
    fig.savefig(PREFIX.with_name(PREFIX.name + "_examples.png"), dpi=160)
    plt.close(fig)
    return selected


def main():
    started = time.perf_counter()
    manifest_path = Path("data/manifest_v5.json")
    cache_path = Path("data/dynamic_v5.npz")
    manifest = read(manifest_path)
    assert sha(cache_path) == manifest["cache_sha256"]
    with np.load(EVIDENCE / "split_evidence.npz") as split:
        ids, target, fit_ids = split["control_ids"], split["control_y"], split["fit_ids"]
    # 只访问train成员；验证和sealed_test不读取。
    with np.load(cache_path) as cache:
        all_ids, all_y = cache["train_ids"], cache["train_y"]
        mask = np.isin(all_ids, np.unique(ids))
        x, poses, variants = (cache[key][mask] for key in ("train_x", "train_poses", "train_variants"))
    np.testing.assert_array_equal(all_ids[mask], ids)
    np.testing.assert_array_equal(all_y[mask], target)
    assert len(np.unique(ids)) == 40 and len(np.unique(fit_ids)) == 160
    assert not (set(ids.tolist()) & set(fit_ids.tolist()))
    ann_path = Path("reports/ann_comparison_20261005.json")
    snn_path = Path("reports/large_lif_baseline_training_20261005.json")
    tuning_path = Path("reports/lif_generalization_tuning_20261005.json")
    ann, snn, tuning = read(ann_path), read(snn_path), read(tuning_path)
    cases = {
        "SNN第10遍": (EVIDENCE / "snn_epoch10/evaluations/control_010.npz", ann["snn_epoch10"]["control"]),
        "CNN第10遍": (EVIDENCE / "stacked_cnn/control_010.npz", ann["arms"]["stacked_cnn"]["epoch10"]["control"]),
        "CNN第75遍": (EVIDENCE / "stacked_cnn/control_075.npz", ann["arms"]["stacked_cnn"]["best"]["metrics"]),
        "CNN第100遍": (EVIDENCE / "stacked_cnn/control_100.npz", ann["arms"]["stacked_cnn"]["epoch100"]["control"]),
    }
    saved_snn = next(row for row in snn["prediction_rechecks"]
                     if row["epoch"] == 75 and row["pool"] == "control")
    assert sha(saved_snn["source"]) == saved_snn["source_sha256"]
    cases["SNN第75遍"] = (Path(saved_snn["source"]), saved_snn["metrics"])
    statistics = {name: prediction_statistics(path, ids, target, poses, expected)
                  for name, (path, expected) in cases.items()}
    stationary = x[target[:, 1] == 0, 0].reshape(-1, 28, 28)
    ink_pixels = (stationary > 0.1).sum((1, 2))
    records = [row for row in manifest["samples"]["train"] if row["raw_id"] in set(ids.tolist())]
    geometric_sizes = np.array([[(row["foreground_bbox_xyxy"][2] - row["foreground_bbox_xyxy"][0]) * row["scale"],
                                 (row["foreground_bbox_xyxy"][3] - row["foreground_bbox_xyxy"][1]) * row["scale"]]
                                for row in records])
    results = dict(
        mode="仅CPU事后描述性分析", model_forward_videos=0, model_backward_videos=0,
        optimizer_updates=0, newly_evaluated_validation_videos=0, sealed_test_reads=0,
        seed_change=False, numpy_version=np.__version__, fit_original_ids=160,
        fit_original_ids_per_digit=16, control_original_ids=40, control_original_ids_per_digit=4,
        videos_per_id=28, control_videos=len(ids), prediction_rows_rechecked=len(cases) * len(ids),
        input_sources={str(path): sha(path) for path in (manifest_path, cache_path, ann_path, snn_path, tuning_path,
                                                       EVIDENCE / "split_evidence.npz")},
        control_geometric_size_median_xy=np.median(geometric_sizes, axis=0).tolist(),
        stationary_first_frame_ink_pixels_threshold_0_1_median=float(np.median(ink_pixels)),
        statistics=statistics, selected_examples=example_figure(x, target, ids, variants, poses),
        trajectory_refresh_digit_comparison={arm: next(row["control"]["digit_accuracy"]
                                                     for row in tuning["arms"][arm]["rows"] if row["epoch"] == 6)
                                             for arm in ("fixed", "refresh", "refresh_smooth")},
        caveats=["仅根11、40个控制笔迹的探索证据，不作独立测试或总体显著性结论。",
                 "小图、全圆旋转、笔迹覆盖、架构与优化因素未作单变量消融，不能分配因果贡献或给出任务准确率上限。",
                 "去掉真实6/9仅复算剩余八类准确率；没有重新训练，也没有把它当新十类成绩。"])
    results["analysis_wall_seconds"] = time.perf_counter() - started
    PREFIX.with_suffix(".json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# CNN与SNN数字识别困难：已有证据分析", "",
             "本报告只在CPU读取既有预测、训练缓存中的控制像素与已有报告；0模型前向、0反传、0更新。后台GPU任务未操作，验证和最终封存数据未读取。", "",
             "## 主要判断", "",
             "最强证据指向跨笔迹泛化不足，并叠加小尺寸和全圆姿态的表征难度。第10遍CNN还有明显欠拟合；后期固定视频模型能够记住训练视频，却没有取得相近控制准确率。不能把共同低准确率唯一归于LIF或Fire。", "",
             "## 实际数据难度", "",
             "4480拟合视频来自160个原始笔迹，每类仅16个；每ID的4轨迹×7运动不等于28个新笔迹。控制1120视频仅40个笔迹，每类4个，同ID视频相关。每帧28×28画布中的前景对角线仅10–11.5像素，起始角在全圆均匀抽取；6/9原标签不随旋转改变。", "",
             f"控制ID的160条轨迹：旋转前前景宽/高中位数{results['control_geometric_size_median_xy'][0]:.2f}/{results['control_geometric_size_median_xy'][1]:.2f}像素。静止视频首帧160张中，灰度>0.1的像素数中位数为{np.median(ink_pixels):.0f}；这不是信息量测量，也不代表不可识别。", "",
             "![实际控制像素](digit_difficulty_20261005_examples.png)", "",
             "## 6/9不是全部解释", "",
             "| 保存模型 | 数字准确率 | 排除真实6/9后八类准确率 | 单纯6↔9互认 / 所有数字错误 |", "|---|---:|---:|---:|"]
    for name, info in statistics.items():
        lines.append(f"| {name} | {info['digit_accuracy']:.2%} | {info['digit_accuracy_excluding_true_6_9']:.2%} | {info['pure_6_9_swap_errors']} / {info['digit_errors']} |")
    lines += ["", "6/9倒置存在任务定义上的歧义风险，但数据未证明两类分布完全相同；不能据此宣称不可学或给出上限。即使排除这两类，其余类仍弱，错误并非主要只发生在6和9之间。", "",
              "## 训练能力与泛化分开判断", "",
              "CNN第10遍数字拟合33.82%、控制27.95%，存在训练不足；第100遍数字拟合100%、控制40.71%，此时固定视频的主要差距是泛化。CNN第75遍同权重数字拟合99.62%、控制41.16%。SNN第10遍数字拟合79.31%、控制32.05%；旧基线第70遍完整拟合100%、第75遍控制36.25%，这两项是相邻但不同检查点，不能拼成同权重差距。", "",
              "已有同起点同6轮SNN实验：固定视频控制数字36.964%，刷新轨迹42.411%，刷新加平滑42.500%；轨迹多样性确有探索性收益。每轮刷新仍只有原160笔迹；最终平滑第12轮动态拟合数字56.607%、控制43.750%，说明新轨迹仍有拟合不足，剩余问题不能全部称过拟合。", "",
              "## 结构机制：依据源码的解释与尚未验证的假设", "",
              "CNN局部权重共享有利于在不同位置复用图形，但当前实现没有显式旋转等变/不变机制；flatten＋全连接读出仍可依赖绝对位置。两次2倍池化累计4倍降采样，10–11.5像素前景对角线只相当于末级网格约2.5–2.9个间距，可能削弱细笔画；卷积通道也能保留信息，尚未证明池化是主要损失原因。8帧通道直接联合处理，没有显式把笔迹身份与运动位姿分离。", "",
              "SNN全连接输入没有预置空间权重共享，须从有限笔迹学习位置/方向泛化；LIF阈值与脉冲可能影响弱细节，但没有隔离这一因素的对照。后期拟合能力和非SNN同样低的控制数字准确率，都不支持仅凭当前结果断言LIF学习失败或容量不足。", "",
              "运动标签能利用大幅14.1–14.4像素净平移的方向/端点，数字标签则要识别小图的细笔画并跨任意姿态和新笔迹。运动98%不等于已经学会稳健数字表征；端点捷径仍未排除。两个任务共享隐藏层且交叉熵相加，是否存在梯度竞争需要同数据数字单任务对照，目前没有证据判定。", "",
              "## 下一步最有区分力的诊断（尚未运行）", "",
              "1. 固定原始ID、根种子和模型，成对限制初始旋转，判断全圆姿态与身份歧义的贡献；改变任务难度，须另登记为诊断协议。",
              "2. 单独降低CNN降采样，或在另行登记的诊断中放大前景，判断细节损失；放大前景需同步调整14像素平移以避免裁切，不能把合并改动当单变量证据。",
              "3. 同原始ID和视频数比较固定轨迹与在线刷新，区分轨迹记忆；已有SNN证据，CNN还未做。",
              "4. 同数据、同预算比较数字单任务，判断共享任务竞争。",
              "扩大独立笔迹需要重新登记可比较的数据协议与隔离划分，不直接解封当前验证/最终测试。优先前述小诊断，不继续用增加层宽或重复训练代替原因区分。", "",
              "## 复核与来源", "",
              "数字混淆矩阵、按运动/角度/原始ID的描述性统计、哈希和成本在[JSON](digit_difficulty_20261005.json)。角度分组未控制其他变量，不能视为角度的因果影响。",
              "依据[数据协议](../docs/data_protocol.md)、[渲染源码](../src/braingrow/data/moving_mnist_v5.py)、[CNN源码](../src/braingrow/network/ann_controls.py)、[对照报告](ann_comparison_20261005.md)、[旧SNN训练报告](large_lif_baseline_training_20261005.md)和[最新SNN泛化报告](lif_generalization_tuning_20261005.md)。",
              "复现入口：项目根目录运行 `.venv\\Scripts\\python.exe -B -m tools.report_digit_difficulty`。无需GPU，不读取检查点，不启动训练。"]
    PREFIX.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(PREFIX.with_suffix('.md')), "wall_seconds": results["analysis_wall_seconds"],
                      "model_forward_videos": 0, "statistics": {name: {key: info[key] for key in
                      ("digit_accuracy", "digit_accuracy_excluding_true_6_9", "pure_6_9_swap_errors", "digit_errors", "per_motion")}
                      for name, info in statistics.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
