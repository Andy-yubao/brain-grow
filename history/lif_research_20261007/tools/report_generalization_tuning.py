"""读取泛化调优的已保存结果；复算预测、同轮拟合差距与实际成本，无模型运行。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from braingrow.evaluation.metrics import evaluate
from tools import train_generalization_tuning as tuning
from tools import train_adaptive_baseline as adaptive

PREFIX = Path("reports/lif_generalization_tuning_20261005")
LABELS = {"fixed": "固定视频", "refresh": "刷新轨迹", "refresh_smooth": "刷新＋平滑"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def observed_costs(directory, fallback):
    """只读取最后完整事件，不修复/修改正在写入的日志。"""
    path = directory / "cost_journal.jsonl"
    if path.exists():
        for line in reversed(path.read_text(encoding="utf-8").splitlines()):
            try:
                return json.loads(line)["cumulative_costs"]
            except json.JSONDecodeError:
                continue
    return fallback.copy()


def verify(path, expected):
    with np.load(path, allow_pickle=False) as saved:
        actual = evaluate(saved["potential"], saved["y"])
        confidence = {}
        for start, stop, column, task in ((0, 10, 0, "digit"), (10, 17, 1, "motion")):
            logits = saved["potential"][:, start:stop].astype(np.float64)
            logits -= logits.max(axis=1, keepdims=True)
            normalized = np.exp(logits)
            normalized /= normalized.sum(axis=1, keepdims=True)
            maximum = normalized.max(axis=1)
            wrong = logits.argmax(axis=1) != saved["y"][:, column]
            confidence[task] = dict(incorrect_videos=int(wrong.sum()),
                mean_maximum_normalized_score=float(maximum.mean()),
                mean_maximum_score_on_errors=float(maximum[wrong].mean()) if wrong.any() else None,
                note="保存电位的softmax描述统计；不改变预测，不等同于已验证的概率校准")
        pred = saved["potential"][:, :10].argmax(1) * 7 + saved["potential"][:, 10:].argmax(1)
        truth = saved["y"][:, 0] * 7 + saved["y"][:, 1]
        cm = np.zeros((70, 70), dtype=np.int64)
        np.add.at(cm, (truth, pred), 1)
        ids = sorted(set(saved["raw_ids"].tolist()))
        assert len(truth) == expected["videos"]
        assert expected["hidden_spikes"] == int(saved["video_fire_by_layer"].sum(dtype=np.float64))
        for i in range(3):
            counts = saved[f"per_unit_fire_layer{i+1}"]
            assert int((counts == 0).sum()) == expected["silent_lif_by_layer"][i]
            assert math.isclose(float(counts.sum() / (len(truth) * 32 * len(counts))),
                expected["spikes_per_lif_step_by_layer"][i], abs_tol=1e-8)
    for key in (*tuning.TASKS, "loss"):
        assert math.isclose(actual[key], expected[key], rel_tol=1e-12, abs_tol=1e-12), (path, key)
    for key in ("digit_confusion", "motion_confusion"):
        assert actual[key] == expected[key], (path, key)
    assert cm.tolist() == expected["joint_confusion"], path
    return dict(prediction_path=str(path.resolve()), prediction_sha256=sha(path),
                original_ids=ids, metrics_and_joint_confusion_verified=True, confidence=confidence)


def collect(root, partial=False):
    suite, config = read(root / "suite.json"), read(root / "config.json")
    if not partial:
        assert suite["phase"] == "complete" and suite["status"] == "complete_registered_round", "本轮尚未完成"
        if (root / "running.json").exists():
            assert not adaptive.process_alive(read(root / "running.json")["pid"]), "训练仍在收尾"
    assert config["source_sha256"] == tuning.hashes(), "训练源码与登记不一致"
    arms, evidence = {}, []
    for name in tuning.ARMS:
        directory = root / name
        if not directory.exists():
            continue
        if (directory / "summary.json").exists():
            summary = read(directory / "summary.json")
        else:
            assert partial, "完成轮缺少按臂汇总"
            summary = dict(status="running_no_complete_epoch", completed_epochs=0,
                           best_control=None, costs=tuning.costs())
        rows = []
        for path in sorted(directory.glob("epoch_*.json")):
            row = read(path)
            if row["epoch"] > summary["completed_epochs"]:
                continue
            for pool in ("control", "fit"):
                if row[pool] is not None:
                    proof = verify(directory / "evaluations" / f"{pool}_{row['epoch']:03d}.npz", row[pool])
                    proof.update(arm=name, epoch=row["epoch"], pool=pool)
                    evidence.append(proof)
            row["matched_epoch_gap"] = ({key: row["fit"][key] - row["control"][key] for key in tuning.TASKS}
                if row["fit"] is not None and row["control"] is not None else None)
            rows.append(row)
        cost = observed_costs(directory, summary["costs"])
        if not partial:
            assert cost == summary["costs"], (name, "成本账不符")
        arms[name] = dict(summary=summary, rows=rows, actual_observed_costs=cost,
                         matched_epoch_gaps=[dict(epoch=r["epoch"], **r["matched_epoch_gap"])
                                            for r in rows if r["matched_epoch_gap"] is not None])
    validation = {}
    for name in suite["validation_done"]:
        saved = read(root / f"validation_{name}.json")
        saved["tuning_epochs_at_frozen_model"] = arms[name]["summary"]["best_control"]["epoch"]
        evidence.append(dict(arm=name, pool="validation", **verify(root / f"validation_{name}.npz", saved["metrics"])))
        validation[name] = saved
    if suite.get("comparison"):
        pilot_summaries = {name: arms[name]["summary"] for name in tuning.ARMS}
        assert tuning.paired_comparison(pilot_summaries) == suite["comparison"]
    common = observed_costs(root, suite["common_costs"])
    total = common.copy()
    for arm in arms.values():
        for key, value in arm["actual_observed_costs"].items():
            total[key] += value
    online = None
    diagnostic_directory = root / "online_fit_diagnostic"
    if (diagnostic_directory / "result.json").exists():
        online = read(diagnostic_directory / "result.json")
        assert online["config"]["frozen"] == suite["frozen"][suite["winner"]]
        assert online["config"]["epoch"] == arms[suite["winner"]]["summary"]["best_control"]["epoch"]
        assert online["costs"] == observed_costs(diagnostic_directory, {})
        assert online["optimizer_updates"] == online["backward_videos"] == online["validation_forward_videos"] == 0
        proof = verify(diagnostic_directory / "predictions.npz", online["metrics"])
        evidence.append(dict(arm=suite["winner"], pool="online_fit", epoch=online["config"]["epoch"], **proof))
        for key, value in online["costs"].items():
            total[key] += value
    augmentation = [dict(epoch=m["epoch"], videos=m["videos"], variants=m["variants"],
        wall_seconds=m["wall_seconds"], sha256=m["sha256"]) for m in
        (read(p) for p in sorted((root / "augmentation").glob("epoch_*.json")))]
    origin_path = Path(config["warm_checkpoint"]).parent.parent / "epoch_020.json"
    origin = (dict(path=str(origin_path.resolve()), sha256=sha(origin_path), costs=read(origin_path)["costs"],
                   role="共同第20轮模型的历史累计成本，属于旧基线已计账前缀；与本轮新增成本分列")
              if origin_path.exists() else None)
    return dict(timestamp=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        partial_report=partial, status=suite["status"], phase=suite["phase"], active_arm=suite["active_arm"],
        run_directory=str(root.resolve()), config=config, comparison=suite.get("comparison"),
        winner=suite["winner"], arms=arms, validation=validation, evidence=evidence,
        post_freeze_online_fit_diagnostic=online,
        common_costs=common, total_actual_costs=total, augmentation=augmentation,
        augmentation_wall_seconds=sum(m["wall_seconds"] for m in augmentation),
        reused_origin=origin,
        elapsed_wall_seconds_since_prepare=datetime.now().timestamp() - suite["started_timestamp"],
        new_model_forward_videos=0, new_model_backward_videos=0, new_optimizer_updates=0,
        sealed_test_reads=0, validation_role="冻结后探索性确认；旧验证池曾被探索，不是最终测试",
        resources_note="边权与Fire是资源代理指标；未实测能耗，墙钟可能含等待或共享GPU影响")


def curves(result, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.ticker import MaxNLocator
    font = Path("C:/Windows/Fonts/msyh.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    for name, arm in result["arms"].items():
        rows = [r for r in arm["rows"] if r["control"] is not None]
        for ax, key, title in ((axes[0,0], "joint_accuracy", "控制联合准确率"),
                              (axes[1,0], "digit_accuracy", "控制数字准确率"),
                              (axes[1,1], "motion_accuracy", "控制运动准确率")):
            ax.plot([r["epoch"] for r in rows], [100*r["control"][key] for r in rows], "o-", label=LABELS[name])
            ax.set(title=title, ylabel="准确率（%）")
        axes[0,1].plot([r["epoch"] for r in rows], [r["control"]["loss"] for r in rows], "o-", label=LABELS[name])
    axes[0,1].set(title="控制硬标签损失（平滑臂也统一硬标签评价）", ylabel="两分组交叉熵")
    maximum = max([6] + [r["epoch"] for a in result["arms"].values() for r in a["rows"]])
    for ax in axes.flat:
        ax.set_xlabel("调优轮数（共同起点为原第20轮）")
        ax.set_xlim(0, maximum + .5)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.axvline(6, color="gray", linestyle="--", alpha=.5)
        ax.grid(alpha=.2)
        if ax.lines:
            ax.legend()
    fig.suptitle("纯LIF泛化调优：" + ("阶段进展，尚未完成" if result["partial_report"] else "同起点配对结果与候选扩展") + "（虚线为共同6轮预算）", fontsize=15)
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def markdown(result, prefix):
    lines = ["# 纯LIF泛化调优报告", "", f"状态：{result['status']}；阶段：{result['phase']}。",
        "阶段报告，本轮尚未完成。" if result["partial_report"] else "已按登记规则结束本轮，轮数与停止原因见各臂汇总；轮次结束不等于问题已解决。", "",
        "5000/3000/2000隐藏LIF、17累积输出、24954000边权；共同原第20轮权重，新AdamW、初始lr=0.0003。",
        "每视频状态清零；8帧各4步、每步5ms，神经时间32步/160ms。生成运动用8个归一化时间点0–1，未定义物理帧率，160ms不是视频播放时长。",
        "仅160拟合ID优化；40控制ID选择。增强不增加独立笔迹；每轮更新原v5轨迹，平滑系数0.1。", "",
        "## 已保存的控制结果", "", "| 臂 | 完成轮 | 最佳控制轮 | 数字 | 运动 | 联合 |", "|---|---:|---:|---:|---:|---:|"]
    for name, arm in result["arms"].items():
        summary, best = arm["summary"], arm["summary"]["best_control"]
        if best:
            lines.append(f"| {LABELS[name]} | {summary['completed_epochs']} | {best['epoch']} | {best['digit_accuracy']:.3%} | "
                         f"{best['motion_accuracy']:.3%} | {best['joint_accuracy']:.3%} |")
    lines += ["", "各臂结束/进展：" + "；".join(f"{LABELS[name]}={arm['summary']['status']}" for name, arm in result["arms"].items()) + "。"]
    lines += ["", "## 同轮原固定拟合与控制", "", "这里的拟合是原固定4480视频，不是增强臂每轮新轨迹。两侧采用同一轮相同权重，评价均为硬标签；不能将不同轮最佳控制与末轮拟合拼成差距。", ""]
    if any(arm["matched_epoch_gaps"] for arm in result["arms"].values()):
        lines += ["| 臂 | 调优轮 | 原固定拟合数字/运动/联合 | 控制数字/运动/联合 | 联合差距（百分点） | 原固定拟合/控制损失 |",
                  "|---|---:|---|---|---:|---|"]
    else:
        lines.append("尚未完成同轮拟合与控制的成对评价，不报告泛化差距。")
    for name, arm in result["arms"].items():
        for row in arm["rows"]:
            if row["matched_epoch_gap"] is None:
                continue
            fmt = lambda m: "/".join(f"{100*m[k]:.2f}%" for k in tuning.TASKS)
            lines.append(f"| {LABELS[name]} | {row['epoch']} | {fmt(row['fit'])} | {fmt(row['control'])} | "
                f"{100*row['matched_epoch_gap']['joint_accuracy']:.2f} | {row['fit']['loss']:.4f}/{row['control']['loss']:.4f} |")
    if result["comparison"]:
        c = result["comparison"]
        lines += ["", "## 预登记配对比较", "", "比较第4/6轮控制均值，联合须提高≥2个百分点、两单项各下降≤1个百分点、硬标签损失不升。", "",
            "| 臂 | 数字 | 运动 | 联合 | 损失 |", "|---|---:|---:|---:|---:|"]
        for name, m in c["averaged"].items():
            lines.append(f"| {LABELS[name]} | {m['digit_accuracy']:.3%} | {m['motion_accuracy']:.3%} | {m['joint_accuracy']:.3%} | {m['loss']:.4f} |")
        lines += ["", f"通过：{c['passed']}；继续候选：{c['winner']}。"]
    if result["validation"]:
        lines += ["", "## 冻结后探索性验证", "", "该验证池历史曾被探索；本轮先冻结后确认，结果不参与回头调优。候选获得额外训练预算，冻结验证可能不是同轮预算比较；等预算机制比较见第4/6轮控制均值。", ""]
        for name, v in result["validation"].items():
            m = v["metrics"]
            lines.append(f"- {LABELS[name]}（冻结调优第{v['tuning_epochs_at_frozen_model']}轮）：数字{m['digit_accuracy']:.3%}、运动{m['motion_accuracy']:.3%}、联合{m['joint_accuracy']:.3%}，损失{m['loss']:.4f}。")
    online = result["post_freeze_online_fit_diagnostic"]
    if online:
        m, c = online["metrics"], online["control"]
        lines += ["", "## 冻结后当轮动态拟合诊断", "",
            "此项在整轮/冻结确认完成后单独登记，只描述冻结候选在同轮实际优化视频上的表现，不参与选择或延长训练。新增4480训练池前向，0反传/0更新/0新渲染。", "",
            f"冻结第{online['config']['epoch']}轮真实优化轨迹：数字{m['digit_accuracy']:.3%}、运动{m['motion_accuracy']:.3%}、联合{m['joint_accuracy']:.3%}，硬标签损失{m['loss']:.4f}。",
            f"同轮控制联合{c['joint_accuracy']:.3%}，动态拟合－控制联合差距{100*online['matched_epoch_gap']['joint_accuracy']:.3f}个百分点。",
            "原固定拟合率与动态拟合率分列。新轨迹上的拟合能力仍弱，有欠拟合信号；剩余低准确率不能全部归于过拟合，容量、优化和表征原因尚未区分。"]
    confidence = [e for e in result["evidence"] if e["pool"] == "control" and e["epoch"] == 6]
    if confidence:
        lines += ["", "## 第6轮误判时的自信程度", "", "对保存电位做softmax并取最大值；下表仅对误判视频求平均，是描述性诊断，不等同于已验证的概率校准。", "",
            "| 臂 | 数字误判平均自信度 | 运动误判平均自信度 |", "|---|---:|---:|"]
        for e in confidence:
            fmt = lambda task: "无误判" if e["confidence"][task]["mean_maximum_score_on_errors"] is None else f"{e['confidence'][task]['mean_maximum_score_on_errors']:.2%}"
            lines.append(f"| {LABELS[e['arm']]} | {fmt('digit')} | {fmt('motion')} |")
    cost = result["total_actual_costs"]
    lines += ["", "## 成本和复核", "",
        "阶段报告的成本读取各臂最后完整日志事件，是读取时已观察成本，不是跨文件同时刻快照。" if result["partial_report"] else "本轮各臂成本汇总与最后完整账目一致。",
        f"本轮新增优化前向/反传{cost['train_forward_videos']}/{cost['train_backward_videos']}视频次、{cost['optimizer_updates']}更新；"
        f"控制{cost['control_metric_evaluations']}次/{cost['control_forward_videos']}视频，拟合{cost['fit_metric_evaluations']}次/{cost['fit_evaluation_forward_videos']}视频，"
        f"验证{cost['validation_metric_evaluations']}次/{cost['validation_forward_videos']}视频。",
        f"共同诊断前向/反传{cost['diagnostic_forward_videos']}/{cost['diagnostic_backward_videos']}视频；增强渲染{cost['data_render_epochs']}轮/{cost['data_render_videos']}视频，"
        f"视频渲染/压缩/哈希墙钟{result['augmentation_wall_seconds']:.1f}秒（逐轮metadata写入在计时后）。OOM重试{cost['microbatch_retries']}。",
        "保存预测复算损失、三项准确率及三种混淆矩阵；Fire汇总复核吻合。该报告不运行模型、不更新参数。最终测试未读取。",
        "起点以来墙钟包含等待/中断可能性；按臂本会话墙钟、GPU/进程峰值及逐层Fire见JSON，不称真实能耗。", "",
        f"![控制曲线]({prefix.name}_curves.png)", "",
        f"完整[证据]({prefix.name}.json)；[预登记](../docs/generalization_tuning_execution.md)。",
        "所有模型/优化器/随机状态、逐帧/逐视频Fire和逐LIF统计保留在运行目录；恢复使用python -u -B -m tools.train_generalization_tuning --resume。"]
    if result["reused_origin"]:
        old = result["reused_origin"]["costs"]
        lines += ["", f"共同起点使用旧基线第20轮权重，其原累计优化前向/反传为{old['train_forward_videos']}/{old['train_backward_videos']}视频次。该历史前缀属于旧基线已计成本，本轮新增成本单列；本轮属于暖启动调优。"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=tuning.ROOT)
    parser.add_argument("--prefix", type=Path, default=PREFIX)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    result = collect(args.root, args.partial)
    args.prefix.parent.mkdir(parents=True, exist_ok=True)
    args.prefix.with_suffix(".json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    curves(result, args.prefix.with_name(args.prefix.name + "_curves.png"))
    args.prefix.with_suffix(".md").write_text(markdown(result, args.prefix), encoding="utf-8")
    print(f"仅读取保存结果，报告：{args.prefix.with_suffix('.md').resolve()}")


if __name__ == "__main__":
    main()
