"""从已保存预测和逐轮指标绘制CNN扩充结果；无模型前向。"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


def main():
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    report = json.loads(Path("reports/cnn_5000_20261005.json").read_text(encoding="utf-8"))
    rows = json.loads(Path("reports/cnn_5000_20261005_evidence/new_epochs.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.5), layout="constrained")
    names = ("digit_accuracy", "motion_accuracy", "joint_accuracy")
    labels = ("数字", "运动", "联合")
    x = np.arange(3)
    for offset, key, color, legend in ((-.19, "old", "#8696ad", "旧CNN：160个拟合笔迹"),
                                      (.19, "new", "#167d91", "新CNN：4000个拟合笔迹")):
        values = [100 * report["expanded_control"][key][name] for name in names]
        bars = axes[0].bar(x+offset, values, .36, color=color, label=legend)
        axes[0].bar_label(bars, labels=[f"{v:.2f}%" for v in values], fontsize=9, padding=3)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 112)
    axes[0].set_ylabel("准确率（%）")
    axes[0].set_title("同一1000控制笔迹 / 28000视频")
    axes[0].legend(loc="lower left", fontsize=9)
    for pool, color, legend in (("control", "#167d91", "控制联合（28000视频）"),
                                 ("fit", "#ce8234", "拟合联合（112000视频）")):
        points = [r for r in rows if r[pool] is not None]
        axes[1].plot([r["epoch"] for r in points], [100*r[pool]["joint_accuracy"] for r in points],
                     marker="o", linewidth=2, color=color, label=legend)
    axes[1].axvline(14, color="#a9454c", linestyle="--", alpha=.6)
    axes[1].set_xlim(1, 15)
    axes[1].set_ylim(90, 101)
    axes[1].set_xlabel("已完成epoch")
    axes[1].set_ylabel("准确率（%）")
    axes[1].set_title("新CNN实际学习曲线：第14轮早停")
    axes[1].legend(loc="lower right", fontsize=9)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=.15)
        axis.set_axisbelow(True)
    fig.suptitle("5000原图扩充：架构、数据渲染及根11保持，优化量同时增加", fontsize=13)
    fig.savefig("reports/cnn_5000_20261005_curves.png", dpi=160)
    plt.close(fig)
    print("真实结果曲线已保存；模型前向/更新均0。")


if __name__ == "__main__":
    main()
