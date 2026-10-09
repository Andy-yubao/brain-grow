"""只读清理预测/证书/BGM，核验真实边和逐样本输出等价，0模型前向。"""
from pathlib import Path
import hashlib
import json
import shutil
import time

import numpy as np
from tools.report_capacity_edge_curve import read_structure
from tools.report_capacity_natural import verify_scores

ROOT=Path(__file__).resolve().parents[1]


def main():
    started=time.perf_counter()
    folder=ROOT/"experiments/runs/h1_capacity_reachability_v1/mnist_static"
    summary=json.loads((folder/"summary.json").read_text(encoding="utf-8"))
    cfg=summary["registration"]["config"]
    title="h1_capacity_reachability_v1_mnist_static_20261008"
    target=ROOT/"reports"/title
    evidence=target.with_name(title+"_evidence"); evidence.mkdir(exist_ok=True)
    labels={"fixed":"固定训练","edge_task":"自然边主","edge_random":"同规模随机边","node_task":"自然缩点主","node_random":"同规模随机缩点"}
    for name,result in summary["results"].items():
        arm=folder/name
        prediction_path=arm/"validation_equivalence.npz"
        with np.load(prediction_path,allow_pickle=False) as saved:
            before,after,frozen,y=saved["before_logits"],saved["after_logits"],saved["frozen_logits"],saved["labels"]
            verify_scores(before,y,result["before_validation"])
            verify_scores(after,y,result["after_validation"])
            for key,a,b in (("before_after",before,after),("saved_before",frozen,before),("saved_after",frozen,after)):
                error=np.abs(a.astype(np.float64)-b.astype(np.float64))
                bound=cfg["logit_atol"]+cfg["logit_rtol"]*np.abs(a.astype(np.float64))
                if not (error<=bound).all() or not np.array_equal(a.argmax(1),b.argmax(1)) or not np.array_equal(error.max(1),saved[f"{key}_max_abs_per_sample"]):
                    raise ValueError("输出一致性或逐样本误差保存不符")
                if abs(float(error.max())-result["comparisons"][key]["max_absolute_error"])>1e-12:
                    raise ValueError("误差汇总不符")
        shutil.copy2(prediction_path,evidence/f"{name}_validation_equivalence.npz")
        arrays,nonzero=read_structure(arm/"model.bgm")
        if nonzero!=result["after"]["nonzero_edges"] or hashlib.sha256((arm/"model.bgm").read_bytes()).hexdigest()!=result["export"]["sha256"]:
            raise ValueError("无损导出/SHA/真实非零边不符")
        certificate=json.loads((arm/"certificate.json").read_text(encoding="utf-8"))
        for i,row in enumerate(certificate["layers"]):
            old=certificate["original_node_ids"][i]
            direct=row["direct_zero_outgoing_original_ids"]; indirect=row["indirect_unreachable_original_ids"]; alive=row["surviving_original_ids"]
            if len(direct)!=row["direct_zero_outgoing_count"] or len(indirect)!=row["indirect_unreachable_count"] or sorted(direct+indirect+alive)!=sorted(old) or len(set(direct+indirect+alive))!=len(old):
                raise ValueError("不可达/存活ID集合证书异常")
            arrays.update({f"layer_{i}_before_node_ids":np.asarray(old,dtype=np.int32),f"layer_{i}_surviving_node_ids":np.asarray(alive,dtype=np.int32),
                           f"layer_{i}_direct_removed_node_ids":np.asarray(direct,dtype=np.int32),f"layer_{i}_indirect_removed_node_ids":np.asarray(indirect,dtype=np.int32)})
        np.savez_compressed(evidence/f"{name}_structure_and_identity.npz",**arrays)
        shutil.copy2(arm/"node_ids.json",evidence/f"{name}_node_ids.json")
    cnn=cfg["cnn_edges_reference_only"]
    lines=["# 从输出反向可达性精确清理：实际报告","",f"执行前提交`{summary['registration']['execution_commit']}`；从自然五臂第16轮BGM按真实非零权重反向分析，0优化更新，不按标签、激活或边数目标选择节点。CNN433680展开有效连接排除padding/bias，仅作结果参照。", "", "## 真实结构与同权重效果", "", "| 臂 | 真实非零边：前→后 / 本清理降幅 | 清理后 / CNN | 物理宽度：前→后 | 验证准确率：前→后 | 验证CE：前→后 | BGM字节：前→后 | BGM+同格式原ID JSON：前→后 |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name,r in summary["results"].items():
        b,a=sum(r["before"]["nonzero_edges"]),sum(r["after"]["nonzero_edges"])
        lines.append(f"| {labels[name]} | {b}→{a} / {1-a/b:.3%} | {a/cnn:.3f}× | {r['before']['widths']}→{r['after']['widths']} | {r['before_validation']['accuracy']:.3%}→{r['after_validation']['accuracy']:.3%} | {r['before_validation']['cross_entropy']:.6f}→{r['after_validation']['cross_entropy']:.6f} | {r['before_model_bgm_bytes']}→{r['export']['bytes']} | {r['before_deployment_bytes_including_ids']}→{r['deployment_bytes_including_ids']} |")
    main=summary["results"]["edge_task"]
    natural_parent=25391616
    net=1-sum(main["after"]["nonzero_edges"])/natural_parent
    lines += ["",f"仅边主产生清理收益：从自然五臂清理前23818752边降到{sum(main['after']['nonzero_edges'])}，此清理删除{sum(main['before']['nonzero_edges'])-sum(main['after']['nonzero_edges'])}条此前仍存储的真实边；相对未剪FC28父25391616边累计降幅{net:.3%}（自然删边+本次清理，不能倒填旧轮结果）。其余四臂无不可达节点，0删除属于规则的自然结果。仍约{sum(main['after']['nonzero_edges'])/cnn:.3f}×CNN，远未接近其有效连接规模。", "", "| 臂 / 隐藏层 | 直接零出边 | 间接不可达 | 删除的非零bias | 逐层真实非零边：前→后 |", "| --- | --- | --- | --- | --- |"]
    for name,r in summary["results"].items():
        for i,row in enumerate(r["certificate"]):
            lines.append(f"| {labels[name]} / {row['hidden_layer']} | {row['direct_zero_outgoing_count']} | {row['indirect_unreachable_count']} | {row['discarded_nonzero_bias_count']} | {r['before']['nonzero_edges'][i]}→{r['after']['nonzero_edges'][i]} |")
    lines += ["",f"末输出层边数边主{main['before']['nonzero_edges'][-1]}→{main['after']['nonzero_edges'][-1]}，输入784及输出10全部保留。直接零出边是向整个下一层无非零边；间接不可达是仍有出边但全部指向下游不可达节点。非零bias可能令节点激活非零，但无非零输出通路时任何激活变化都不影响输出，因此可一起删；不把训练样本静默等同这种全图路径证明。证书以当前逐层Linear/ReLU结构为前提，存活权重/bias/mask位模式逐一相同，原ID以原顺序保留。", "", "## 逐样本输出一致性", "", "FP32物理维度改变可能改变乘加求和顺序；预登记门槛每logit abs(后−前)≤1e-4+1e-5·abs(前)，argmax全同，任何不通过即停且不调容差。验证标签只记录效果，不决定清理。", "", "| 臂 | 前/后最大绝对logit差 / 最大容差比例 | 前/后逐位相同 | 容差通过 / argmax同样本 | 保存末态与清理前最大差 / 清理后最大差 |", "| --- | --- | --- | --- | --- |"]
    for name,r in summary["results"].items():
        c=r["comparisons"]; ba=c["before_after"]
        lines.append(f"| {labels[name]} | {ba['max_absolute_error']:.10g} / {ba['max_bound_ratio']:.6f} | {ba['bitwise_equal']} | {ba['allclose_sample_count']}/800 / {ba['argmax_equal_sample_count']}/800 | {c['saved_before']['max_absolute_error']:.10g} / {c['saved_after']['max_absolute_error']:.10g} |")
    lines += ["", "同权重数学删除不改实数函数，实际FP32证据支持已登记容差内等价；边主并非逐位相同，其他四臂逐位相同。未以验证准确率约97%另作接受/选模，只有实现一致性 gate。", "", "## 真实成本与保存", "", "| 臂 | 装载 / 图分析slice / 前评价 / 后评价 / 导出秒 | 臂墙钟秒 | CUDA峰分配字节 | 进程工作集 / 至该臂结束累计峰字节 | 数值张量：前→后 / bool mask张量：前→后 |", "| --- | --- | --- | --- | --- | --- |"]
    for name,r in summary["results"].items():
        c=r["costs"]; m=r["process_memory_at_arm_end_cumulative"]
        times=" / ".join(f"{c[k]:.3f}" for k in ("load_seconds","graph_and_slice_seconds","before_evaluation_seconds","after_evaluation_seconds","export_seconds"))
        lines.append(f"| {labels[name]} | {times} | {r['wall_seconds']:.3f} | {r['peak_cuda_allocated_bytes']} | {m['working_set_bytes']} / {m['peak_working_set_bytes']} | {r['before']['tensor_numeric_bytes']}→{r['after']['tensor_numeric_bytes']} / {r['before']['tensor_mask_bytes']}→{r['after']['tensor_mask_bytes']} |")
    lines += ["",f"完整会话{summary['session_wall_seconds']:.3f}秒，设备{summary['registration']['device_name']}、FP32、TF32关闭、batch512、CPU线程2；评价10次/8000样本，优化/适应/探索性测试评价全部0。前/后评价计时包含模型搬移和验证整体前向，不是稳定batch延迟基准；模型数值/掩码张量字节、推理bundle文件、Windows进程工作集累计峰、Torch CUDA分配峰分别记录，均不当能耗。GPU释放，不称本轮已证明推断加速。", "", "节点ID JSON是实际映射文件字节，前/后使用同格式；BGM权重和mask编码包含全部数值/位图/bias。model_state.pt是本地备用权重编码，另列JSON中，未计入部署bundle；其中没有迁移旧Adam，不称能严格续训旧状态。既有自然五臂权重/掩码/ID/运行/报告只读保留，清理后的模型在独立新目录。", "", "直接验证缓存SHA与既有固定ID清单核对；仅反序列化已保存预测validation字段，训练字段不读取，探索性测试缓存/预测及官方测试/剩余封存未读取。没有重训或按验证标签挑清理点。唯一新增小CPU真实检查通过2.771秒，覆盖间接断路、非零bias、精确零mask槽及原ID；无重复GPU诊断。报告仅保存预测/图/真实活值/SHA核验，模型前向0。", "", "## 下一机制优先级", "", "任务敏感边主与随机同量边图在精确清理后出现不同连接规模，支持将反向可达性物理清理作为任务驱动编辑的常规闭合步骤进一步检验；该证据不是任务排序准确率优势，因为旧轮随机验证CE更低且本轮等价清理保持各自效果。下一轮须另登记有限候选/任务损伤/完整回退和匹配随机净结构收益，不设CNN边目标；当前未启动任何耦合训练。活动多基列重构也仍缺独立证据，不能从前轮静默节点删除或本轮无输出路径清理推出有效。", ""]
    lines += ["首次fixed清理前评价含CUDA/cuBLAS初始化和模型搬移，因此其15.404秒不能与后评价0.125秒直接比较推断加速；本轮未登记重复延迟基准。", ""]
    target.with_suffix(".md").write_text("\n".join(lines),encoding="utf-8")
    summary["report_verification"]={"model_forwards":0,"seconds":time.perf_counter()-started,"all_saved_predictions_counts_sha_and_identity_checked":True}
    target.with_suffix(".json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"report":str(target.with_suffix('.md')),"model_forwards":0,"seconds":time.perf_counter()-started},ensure_ascii=False))


if __name__=="__main__": main()
