"""只读保存预测与账本生成报告，不运行模型。"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--version", default="structural_v1")
    args = parser.parse_args()
    run = ROOT/"experiments"/"runs"/args.version/args.task
    summary = json.loads((run/"summary.json").read_text(encoding="utf-8"))
    reports = ROOT/"reports"
    reports.mkdir(exist_ok=True)
    stem = f"{args.version}_{args.task}_20261007"
    evidence = reports/f"{stem}_evidence"
    evidence.mkdir(exist_ok=True)
    evidence_hashes = {}
    for arm, result in summary["arms"].items():
        destination = evidence/arm
        destination.mkdir(exist_ok=True)
        for filename in ("final_model.bgs", "fit_final.npz", "confirm_final.npz",
                         f"select_{result['final_epoch']:03d}.npz"):
            target = destination/filename
            shutil.copyfile(run/arm/filename, target)
            evidence_hashes[str(target.relative_to(reports))] = hashlib.sha256(target.read_bytes()).hexdigest()
    for filename in ("registration.json", "freeze.json", "startup.json", "data_manifest.json"):
        shutil.copyfile(run/filename, evidence/filename)
    summary["evidence_sha256"] = evidence_hashes
    summary["implementation_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    (reports/f"{stem}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    total = {key: sum(arm["ledger"][key] for arm in summary["arms"].values())
             for key in ("optimization_forward_samples", "optimization_backward_samples", "attempted_updates",
                         "candidate_updates", "rolled_back_updates", "evaluation_forward_samples")}
    fc = summary["arms"]["fc"]
    lines = [f"# 首轮结构学习报告：{args.task}", "", "2026-10-07；所有成绩来自实际完成的固定预算五臂运行。", "",
             f"实现提交：`{summary['implementation_commit'][:7]}`。源哈希与配置见[完整JSON]({stem}.json)。", "",
             "## 任务与方法", "",
             "64维环面模板二分类，模板先去均值并归一化，位置独立随机，10000拟合/2000结构选择/2000冻结确认。"
             "FC为64→128→2、ReLU；CNN为8个核5环面卷积+全局均值+2输出，人工先验参照。", "",
             "模型根11、数据根20261004，5臂各15轮、157批/轮；AdamW共同日程。"
             "结构臂预热3轮，最多20个候选，每次1批适应占用原更新额度；拒绝更新状态回退但成本保留。", "",
             "prune是任务敏感删边，share是值域相邻组投影，joint交替删边与任务敏感共享。"
             "这是首版投影/候选控制器，不声称复现已有近端算法或发现卷积。", "",
             "## 实际成绩", "",
             "| 臂 | 拟合准确率 | 结构选择准确率 | 冻结确认准确率 | 选择损失 | 导出字节 | 非零多边组 | 接受/提议 |", 
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for arm, result in summary["arms"].items():
        structure = result["structure"]
        groups = structure.get("model_total", {}).get("nonzero_shared_groups", "人工共享")
        count = sum(event["accepted"] for event in result["events"])
        lines.append(f"| {arm} | {result['fit']['accuracy']:.2%} | {result['rows'][-1]['select']['accuracy']:.2%} | "
                     f"{result['confirmation']['accuracy']:.2%} | {result['rows'][-1]['select']['loss']:.6f} | "
                     f"{structure['export_bytes']} | {groups} | {count}/{len(result['events'])} |")
    lines.extend(["", "逐类混淆矩阵、每轮变化及每次拒绝原因在完整JSON，最终logits和模型文件见证据目录。"
                  "取第15轮，不按确认集选择最优轮。CNN与FC容量和文件格式不同，不据其大小判断优化器优劣。", "",
                  "## 结构与代价", "",
                  "| 臂 | 有效边 | 独立参数含偏置 | 训练分配参数槽 | 训练/搜索/选择/编码墙钟秒 |", 
                  "| --- | --- | --- | --- | --- |"])
    for arm, result in summary["arms"].items():
        stats = result["structure"].get("model_total", {})
        lines.append(f"| {arm} | {stats.get('active_edges', '卷积参照')} | "
                     f"{stats.get('independent_parameters', result['structure'].get('independent_parameters'))} | "
                     f"{stats.get('allocated_parameter_slots', result['structure'].get('independent_parameters'))} | "
                     f"{result['wall_seconds']:.3f} |")
    lines.extend(["", "共享组/掩码前向仍是稠密执行；参数槽和Adam内存未压缩。"
                  "导出FP32无损，全部映射/元数据包含在文件字节中；其压缩同时受数值和结构影响，不等于计算复用或能耗降低。", "",
                  f"设备：{summary['registration']['device']}，PyTorch {summary['registration']['torch']}；FP32，批量64，评价批量512。"
                  f"CUDA峰值分配{summary['peak_cuda_memory_bytes']}字节。完整本次会话墙钟{summary['current_session_wall_seconds']:.3f}秒；"
                  "臂间墙钟排除checkpoint/预测写入、最终拟合和确认评价，不能称完整训练端到端效率。", "",
                  f"实际优化前向/反传样本各{total['optimization_forward_samples']}/{total['optimization_backward_samples']}，"
                  f"尝试更新{total['attempted_updates']}、候选适应{total['candidate_updates']}、拒绝回退{total['rolled_back_updates']}，"
                  f"净保留更新{total['attempted_updates']-total['rolled_back_updates']}。"
                  f"评价前向样本{total['evaluation_forward_samples']}，包含每轮结构选择、候选、最终拟合与确认。", "",
                  "额外8样本GPU启动前向/反传、0更新单独记录。CPU检查5项通过，其中数据fixture路径错误只重跑该项；"
                  "没有旧模型测试或额外GPU恢复重放。", "", "## 阶段判定", ""])
    gate = summary["gate"]
    if "reuse" in summary:
        lines.extend(["本轮只新训练share/joint各15轮；FC/prune/CNN权重、预测和逐轮参照来自v1，未重训，"
                      "末态在统一v2格式重编码。复用臂成本为本轮0，其历史账单另存reuse/source_ledgers。"
                      "上表复用臂的接受次数和墙钟属于v1，不能称本轮五臂都新训练。",
                      "本轮确认集已在v1被查看，因此是探索性重复确认。复用臂墙钟来自历史运行，倍数仅为工程参照。",
                      "复用臂前14轮结构字节为v1、末态为v2，不把这些值绘成同编码曲线。"])
    lines.append(f"joint相对FC实际文件减少{gate['storage_reduction']:.2%}，计时边界内墙钟倍数{gate['runtime_ratio']:.3f}；"
                 f"阶段继续门槛为{gate['continue']}，完整分项判定见JSON。")
    if not gate["continue"]:
        lines.append("本轮未通过扩大门槛，停止运行fixed/MNIST，不追加相同条件预算、不更换种子。"
                     "失败须先解释功能、分组或存储瓶颈，后续改机制另建版本。")
    else:
        lines.append("仅支持进入下一阶段诊断；同规模随机共享对照、空间局部性、重复滤波器和跨任务机制尚未验证，不能称发现CNN。")
    lines.extend(["", "## 证据与恢复", "",
                  f"[重要证据索引]({stem}_evidence/README.md)保存所有臂最终预测与无损模型，不包含大型运行checkpoint。",
                  f"原日志/逐轮预测/完整恢复checkpoint保留在`experiments/runs/{args.version}/{args.task}/`。"
                  "运行完成后同源码使用--resume只返回已完成判定，不重训、不重复确认。代码变化会被源哈希拒绝，须登记新版本。", "",
                  "```powershell", 
                  f".venv/Scripts/python.exe -u -B -m tools.train_structure --task {args.task} --resume" if args.version == "structural_v1" else
                  ".venv/Scripts/python.exe -u -B -m tools.train_structure_v2 --resume", "```", ""])
    (reports/f"{stem}.md").write_text("\n".join(lines), encoding="utf-8")
    (evidence/"README.md").write_text("# 首轮重要证据\n\n各臂final_model.bgs及fit/select/confirm最终logits按原字节复制；"
                                    "哈希在上级完整JSON。registration记录源码/配置，freeze证明全部臂先冻结；"
                                    "startup为唯一8样本GPU启动，data_manifest只用于追溯，教师元数据不进入模型。\n",
                                    encoding="utf-8")
    print(json.dumps({"report": str(reports/f"{stem}.md"), "gate": gate, "costs": total}, ensure_ascii=False))


if __name__ == "__main__":
    main()
