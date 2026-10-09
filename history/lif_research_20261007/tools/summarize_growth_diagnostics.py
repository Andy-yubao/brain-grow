"""将实际A/B小诊断的重要汇总显式保存在Git证据目录。"""
from __future__ import annotations
import json,pickle
from pathlib import Path
from collections import Counter
import numpy as np
from tools.run_discrete_growth import write_json,save


def main():
    arun=Path("experiments/runs/discrete_growth_d1");brun=Path("experiments/runs/community_growth_d1")
    a=json.loads((arun/"summary.json").read_text(encoding="utf-8"));b=json.loads((brun/"summary.json").read_text(encoding="utf-8"))
    full=pickle.loads((arun/"full_learner.pkl").read_bytes())
    events=full.events
    totals={kind:{key:sum(e.get(key,0) for e in events if e["kind"]==kind) for key in
        ("eligible_repeated","formed","deleted","changed","mature_records")} for kind in ("input","internal","output")}
    # 只恢复隔离候选的模型学习状态；管理流与实际失败成本保持前进。
    from braingrow.network.discrete import DiscreteLearner
    ast=pickle.loads((arun/"resume.pkl").read_bytes())
    restored=DiscreteLearner(ast["initial"].clone(),ast["agendas"].copy(),ast["competitors"].copy(),full=True)
    restored.management_rng=full.management_rng
    save(arun/"restored_stage_start.pkl",{"learner":restored,"simulation_videos_cumulative":a["simulation_videos"]+1680,
        "failed_candidate_path":"full_learner.pkl","reason":"A1未通过预登记继续门槛","extra_interrupted_round_simulation_videos":1680})
    with np.load(arun/"initial_confirmation_outputs.npz") as z:baseline=z["potential"];ay=z["y"];aids=z["ids"]
    with np.load(arun/"full_confirmation_outputs.npz") as z:final=z["potential"]
    bp=np.column_stack((baseline[:,:10].argmax(axis=1),baseline[:,10:].argmax(axis=1)))
    fp=np.column_stack((final[:,:10].argmax(axis=1),final[:,10:].argmax(axis=1)))
    changed_ids={key:len(set(aids[(fp[:,col]==ay[:,col])&(bp[:,col]!=ay[:,col])].tolist())) for key,col in (("digit",0),("motion",1))}
    lineage=b["lineage"]
    actions={arm:dict(Counter(action["kind"] for e in lineage if e["arm"]==arm for action in e["actions"])) for arm in ("full","output_only")}
    births=sum("born" in action for e in lineage if e["arm"]=="full" for action in e["actions"])
    deaths=sum("deleted" in action for e in lineage if e["arm"]=="full" for action in e["actions"])
    evidence={"date":"2026-10-04","stage":"早期唯一种子11的训练子集机制诊断，非验证/最终测试",
        "model_seed":11,"data_seed":20261004,"validation_reads":0,"sealed_test_reads":0,"dependencies_added":[],
        "environment":"本地.venv，CPU，BLAS线程1","input":"8×784原灰度，4步/帧，5ms/步；视频状态清零",
        "d0":json.loads((arun/"d0.json").read_text(encoding="utf-8")),
        "A":{"initial_confirmation":a["initial_confirmation"],"full_confirmation":a["full_confirmation"],"output_only_confirmation":a["output_only_confirmation"],
            "curves":a["curves"],"interventions":a["interventions"],"event_totals":totals,"birth_batches":full.birth_batches,
            "hidden_window_count_range":[min(w.windows for w in full.hidden_windows),max(w.windows for w in full.hidden_windows)],
            "output_window_count_range":[min(w.windows for w in full.output_windows),max(w.windows for w in full.output_windows)],
            "ids_with_some_repair":changed_ids,"simulated_videos_checkpoint":a["simulation_videos"],"extra_interrupted_round_simulation_videos":1680,
            "actual_simulation_videos":a["simulation_videos"]+1680,"metric_evaluations_completed":a["metric_evaluations"],"interrupted_metric_evaluations":1,
            "wall_seconds":a["wall_seconds"],"peak_process_memory":"原进程API未正确取到，未知；不能用0替代",
            "confirmation_unique_ids":30,"development_unique_ids":30,"videos_each_pool":840,"selected_main_agenda":a["main"],
            "conclusion":"未达继续门槛；没有出生或成熟功能证据；停止A扩大，隔离失败图并恢复阶段初始完整学习状态"},
        "B":{"initial_confirmation":b["baseline_confirmation"],"full_confirmations":b["full_confirmations"],"output_only_confirmations":b["output_only_confirmations"],
            "curves":b["curves"],"lineage":lineage,"action_counts":actions,"additional_node_births_full":births,"node_deletions_full":deaths,
            "initial_birth_nodes_each_arm":48,"candidate_evaluations_each_arm":78,"actual_simulation_videos":b["simulation_videos"],
            "metric_evaluations":b["metric_evaluations"],"wall_seconds":b["wall_seconds"],"peak_working_set_bytes":b["peak_working_set_bytes"],
            "confirmation_unique_ids":20,"development_unique_ids":20,"videos_each_pool":560,"selected_main_agenda":b["agenda"],
            "graduates_full":sum(c["gate"]["graduated"] for c in b["full_confirmations"]),"graduates_output_only":sum(c["gate"]["graduated"] for c in b["output_only_confirmations"])}
    }
    evidence["A"]["training_management_array_bytes"]=sum(getattr(w,key).nbytes for w in full.hidden_windows+full.output_windows for key in ("count","source","eligible","hit","previous_sign","age"))
    evidence["A"]["mature_functional_edges_confirmed"]=0
    evidence["A"]["event_totals_counting_note"]="mature_records是每窗重复侧偏管理记录累计，含重复/缺席提名；并非成熟功能边数量"
    evidence["B"]["conclusion"]="无候选通过毕业，全局保持与跨ID任务改进未建立；停止本轮多代扩大，不进入D2"
    evidence["total_actual_simulation_videos"]=evidence["A"]["actual_simulation_videos"]+evidence["B"]["actual_simulation_videos"]
    evidence["total_neural_steps_across_videos"]=evidence["total_actual_simulation_videos"]*32
    Path("reports").mkdir(exist_ok=True);write_json("reports/growth_diagnostics_20261004.json",evidence)
    def accuracy(result):
        m=result["metrics"];return " / ".join(f"{m[k]*100:.3f}%" for k in ("digit_accuracy","motion_accuracy","joint_accuracy"))
    rows=[f"| A初图 | {accuracy(a['initial_confirmation'])} |",f"| A完整 | {accuracy(a['full_confirmation'])} |",f"| A仅输出 | {accuracy(a['output_only_confirmation'])} |",f"| B初图（不同固定子集） | {accuracy(b['baseline_confirmation'])} |"]
    for arm in ("full","output_only"):
        for c in b[arm+"_confirmations"]:rows.append(f"| B{'完整' if arm=='full' else '仅输出'}候选{c['proposal_id']} | {accuracy(c['result'])} |")
    report="""# A1/B1真实小诊断报告

2026-10-04。用户已授权实现并训练。使用根模型种子11、v5数据20261004；全部结果来自固定原始ID隔离的**训练发育/确认子集**，确认参与资格选择，不是独立泛化估计。没有读取验证或最终测试。机制实现/预算见[A1登记](../docs/discrete_growth_execution.md)、[B1登记](../docs/community_growth_execution.md)。所有精确混淆矩阵、每帧/视频Fire、各类边/密度、曲线和候选谱系保存在[重要证据](growth_diagnostics_20261004.json)。

| 确认对象 | 数字 / 运动 / 联合准确率 |
| --- | --- |
"""+"\n".join(rows)+"\n\n"
    report+=f"A完成每臂10遍840视频学习与每遍无学习发育评价，冻结后各840确认，4项整组干预；D0实际140视频次通过，正常Fire703.8/视频、零输入0。128单元中127有近阈值机会，启动静默不是主要瓶颈。完整臂输入/内部/输出实际改变次数为{totals['input']['changed']}/{totals['internal']['changed']}/{totals['output']['changed']}；主议程{a['main']}，每单元覆盖合格窗口范围{evidence['A']['hidden_window_count_range']}。无出生，因为原主混淆被另一类错误替代，不满足持续同需求出生条件；这不等于混淆需求全部解决。\n\n"
    report+="A未达到一项+5pp、联合+2pp且完整优于输出3pp的继续门槛。完整臂确认数字几乎全预测0，运动大部分预测左移；只输出臂数字也主要落在0/6。原灰度的侧偏与脉冲计数档位虽形成实际边，但很多局部相关证据被累积为全类电位偏置，没有功能组合判别。完整臂交叉熵85.09高于只输出25.42，反映输出幅度及错误自信，不把损失变化当参数方向。输入/内部改变没有独立任务收益，重复事件记录仅是巩固候选，不是成熟功能证据。四项干预的精确成绩见JSON；出生静音等于原完整图，因为确无新生节点。停止A扩大并保存失败候选；恢复阶段初始完整模型学习状态，保留管理流/失败证据/累计成本。\n\n"
    report+=f"B每臂实际评价78个完整候选图（6初生+12代×6子代），母图冻结、8随机幼年LIF起点，五类离散动作和4行为+2探索槽。完整臂动作统计{actions['full']}，只输出{actions['output_only']}；完整臂额外出生{births}、删除{deaths}，全部在候选中完成，不代表功能毕业。冻结各臂发育排序前3后确认，并对所有6个候选做新生子图静音。毕业完整/仅输出数={evidence['B']['graduates_full']}/{evidence['B']['graduates_output_only']}。\n\n"
    report+="B按预登记有限预算结束，未自动加代或扩大数据；没有候选毕业，完整子图搜索仍无足够跨ID全任务保持。此次只检验短期孤立幼年子图和固定母图接口，不排除所有多代发育机制。只输出臂读出修改比例更大、容量与随机流消费不同，评价次数配对不等于完全消除混杂。C延迟机制没有启动；当前数字也接近机会水平，暂无仅由运动顺序瓶颈解释失败的证据。下一机制需要重新设计类别间竞争/保持与可解释的任务信用，不能用更多随机出生补救本轮失败。\n\n"
    report+=f"实际仿真视频次：A检查点40,460，加第6遍浮点验算停止后已耗的1,680额外次，共42,140；B{b['simulation_videos']:,}，合计{evidence['total_actual_simulation_videos']:,}。每次32神经步，合计{evidence['total_neural_steps_across_videos']:,}视频步次；批量前向按视频数量计账。A已完成指标评价{a['metric_evaluations']}，另1次验算中断；B指标/完整图评价{b['metric_evaluations']}。A墙钟{a['wall_seconds']:.2f}秒（包含停止/恢复间隔）；B墙钟{b['wall_seconds']:.2f}秒；B原生进程峰值工作集{b['peak_working_set_bytes']/1024**2:.2f}MiB，包含数据/轨迹/候选，并非网络数组或能耗。A原进程峰值因API错误未获取，明确未知。设备CPU，BLAS线程1，本地.venv，无新增依赖、外部支出或环境更改。\n\n"
    report+="第一次A第6遍恒等复核float32累计误差3.05e-5超过2e-5容差停止；改为float64重建和绝对1e-4/相对1e-6容差，明确记录每评价最大误差，从第5遍完整检查点恢复，重复成本未隐藏。这是数值验算修复，未改机制、种子或选择成绩。两机制相关行为检查覆盖逐步复位前状态/上一Fire传导、无标签前向、输出恒等/视频清零、候选不共享图/计数/议程/RNG、三类边增删重建、节点稳定ID同步、B隐藏冻结；旧tmp_path目录需要沙箱批准运行，其通过不证明研究成功。\n\n"
    report+="恢复入口：`.venv/Scripts/python.exe -B -m tools.run_discrete_growth --resume`、`.venv/Scripts/python.exe -B -m tools.run_community_growth --resume`；已完成检查点只报告完成，不重复训练。完整学习状态分别在`experiments/runs/discrete_growth_d1/resume.pkl`与`experiments/runs/community_growth_d1/resume.pkl`；A失败完整学习器`full_learner.pkl`、止损恢复模型状态`restored_stage_start.pkl`；B所有初生/未毕业实际图、谱系、冻结候选/私有状态都在其运行目录。检查点是可信本地pickle，不加载未知外部文件；仅network.npz不包含学习管理状态。源码快照、数据哈希、开始提交与工作区差异均保存在每轮目录，生成物不默认入Git。\n"
    report+="\n交付时12项相关行为检查全部通过，文档本地链接/文件/语法检查通过；两个完成检查点的恢复命令已实际执行且未重训。B完成后以固定根11/提议ID无LIF重建全部156实际母子图，逐数组、稳定ID/动作相等并补存私有提议RNG末态，见[恢复核对](growth_recovery_audit_20261004.json)。入口完成后增加独立前向成本日志，未来模型轮次回滚时累计实际成本不回滚；本轮A中断额外成本使用真实执行记录单独补记。上述硬化不改变已运行学习规则或成绩，原运行源码快照保留。A管理数组另占"+f"{evidence['A']['training_management_array_bytes']:,}"+"字节（不含Python对象/ID集合/RNG），不能混入隐藏网络参数或当能耗。\n"
    Path("reports/growth_diagnostics_20261004.md").write_text(report,encoding="utf-8")
    print(json.dumps({"A_simulation_videos":42140,"B_simulation_videos":b["simulation_videos"],"B_graduates":[evidence["B"]["graduates_full"],evidence["B"]["graduates_output_only"]]},ensure_ascii=False))


if __name__=="__main__":main()
