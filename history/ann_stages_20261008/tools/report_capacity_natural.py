"""仅核验保存预测/无损图/事件，报告自然结构曲线，模型前向0。"""
from pathlib import Path
import hashlib
import json
import shutil
import time

import numpy as np

from tools.report_capacity_edge_curve import read_structure, verify_prediction

ROOT=Path(__file__).resolve().parents[1]


def verify_scores(logits,y,expected):
    scores=logits.astype(np.float64)
    prediction=scores.argmax(1)
    loss=np.log(np.exp(scores-scores.max(1,keepdims=True)).sum(1))+scores.max(1)-scores[np.arange(len(y)),y]
    confusion=np.bincount(y*10+prediction,minlength=100).reshape(10,10)
    if abs((prediction==y).mean()-expected["accuracy"])>1e-7 or abs(loss.mean()-expected["cross_entropy"])>1e-6 or confusion.tolist()!=expected["confusion_matrix"]:
        raise ValueError("保存预测与汇总不一致")


def main():
    started=time.perf_counter()
    folder=ROOT/"experiments/runs/h1_capacity_natural_v1/mnist_static"
    summary=json.loads((folder/"summary.json").read_text(encoding="utf-8"))
    title="h1_capacity_natural_v1_mnist_static_20261008"
    target=ROOT/"reports"/title
    evidence=target.with_name(title+"_evidence")
    evidence.mkdir(exist_ok=True)
    names={"fixed":"固定训练","edge_task":"任务边群组","edge_random":"匹配随机边","node_task":"重构物理缩点","node_random":"匹配随机缩点"}
    for name,result in summary["results"].items():
        arm=folder/name
        for row in result["rows"]:
            epoch=row["epoch"]
            saved_path=arm/f"epoch_{epoch}_predictions.npz"
            with np.load(saved_path,allow_pickle=False) as saved:
                verify_scores(saved["validation"],saved["validation_labels"],row["select"])
                verify_scores(saved["fit"],saved["fit_labels"],row["fit"])
            shutil.copy2(saved_path,evidence/f"{name}_epoch_{epoch}_predictions.npz")
            stage=arm/f"epoch_{epoch}"
            if stage.exists():
                metadata=json.loads((stage/"structure.json").read_text(encoding="utf-8"))
                arrays,nonzero=read_structure(stage/"model.bgm")
                if nonzero!=row["structure"]["nonzero_edges"]:
                    raise ValueError("保存图真实非零边与记录不符")
                arrays.update({f"node_ids_{i}":np.asarray(ids,dtype=np.int32) for i,ids in enumerate(metadata["node_ids"])})
                np.savez_compressed(evidence/f"{name}_epoch_{epoch}_structure.npz",**arrays)
        verify_prediction(arm/"control_final.npz",result["control"])
        shutil.copy2(arm/"control_final.npz",evidence/f"{name}_control_final.npz")
        arrays,nonzero=read_structure(arm/"model.bgm")
        if nonzero!=result["structure"]["nonzero_edges"] or hashlib.sha256((arm/"model.bgm").read_bytes()).hexdigest()!=result["export"]["sha256"]:
            raise ValueError("最终导出计数/SHA不符")
        for path in arm.glob("slot_*_validation_trials.npz"):
            slot=int(path.name.split("_")[1])
            epoch=summary["registration"]["config"]["edit_epochs"][slot-1]
            event=next(e for e in result["events"] if e["epoch"]==epoch)
            with np.load(path,allow_pickle=False) as saved:
                verify_scores(saved["original"],saved["labels"],event["original_select"])
                verify_scores(saved["candidate"],saved["labels"],event["candidate_select"])
            shutil.copy2(path,evidence/f"{name}_{path.name}")
        if name.endswith("random"):
            primary=summary["results"][name.replace("random","task")]
            if result["ledger"]["net_updates"]!=primary["ledger"]["net_updates"]:
                raise ValueError("随机净更新不匹配")
            for event in result["events"]:
                paired=next(e for e in primary["events"] if e["epoch"]==event["epoch"])
                equal=paired["net_deleted_per_layer"]==event["net_deleted_per_layer"]
                if equal!=event["matched_net_deletion"]:
                    raise ValueError("匹配标记与事件不符")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"]=["Microsoft YaHei","SimHei","DejaVu Sans"]
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    cnn=summary["registration"]["config"]["cnn_edges_reference_only"]
    for name,r in summary["results"].items():
        if name=="fixed": continue
        axis=axes[0] if name.startswith("edge") else axes[1]
        axis.plot([sum(row["structure"]["nonzero_edges"])/cnn for row in r["rows"]],
                  [row["select"]["accuracy"]*100 for row in r["rows"]],marker="o",markersize=3,label=names[name])
        axes[2].plot([row["epoch"] for row in r["rows"]],[sum(row["structure"]["widths"][1:-1]) for row in r["rows"]],label=names[name])
    fixed=summary["results"]["fixed"]
    for axis,title_label in zip(axes[:2],("自然边群组","自然物理缩点")):
        axis.axhline(fixed["select"]["accuracy"]*100,linestyle="--",color="gray",label="固定训练末态")
        axis.set(xlabel="真实非零边 / CNN参照有效边",ylabel="验证准确率（%）",title=title_label)
        axis.invert_xaxis(); axis.legend(fontsize=8); axis.grid(alpha=.2)
    axes[2].set(xlabel="新增轮",ylabel="物理隐藏节点总数"); axes[2].legend(fontsize=8); axes[2].grid(alpha=.2)
    fig.tight_layout(); fig.savefig(target.with_name(title+"_curves.png"),dpi=160); plt.close(fig)
    lines=["# 扩容FC自然任务驱动结构诊断实际报告","",f"训练前提交`{summary['registration']['training_commit']}`；配置/源/未剪FC28父权重SHA见JSON。所有五臂从同FC28未剪枝基线权重出发，784→6272→3136→256→10，未继承旧强制配额图。基本CNN/FC约97%已通过；此次边数只作结果，不设CNN倍数目标或必须删除比例。CNN433680有效连接排除padding/偏置，独立参数与文件不作目标。","","## 最末冻结结果","","| 臂 | 真实非零总边 / CNN倍数 | 逐层非零边 | 物理宽度 | 有效隐藏节点 | 验证准确率 / CE | 训练准确率 / CE | 探索性测试准确率 / CE | 实际BGM字节 |","| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name,r in summary["results"].items():
        s=r["structure"]; total=sum(s["nonzero_edges"])
        lines.append(f"| {names[name]} | {total} / {total/cnn:.3f} | {s['nonzero_edges']} | {s['widths']} | {s['effective_hidden_nodes']} | {r['select']['accuracy']:.3%} / {r['select']['cross_entropy']:.6f} | {r['fit']['accuracy']:.3%} / {r['fit']['cross_entropy']:.6f} | {r['control']['accuracy']:.3%} / {r['control']['cross_entropy']:.6f} | {r['export']['bytes']} |")
    silent=sum(e.get("silent_deleted",0) for e in summary["results"]["node_task"]["events"])
    lines += ["","有效隐藏节点按非零输入或bias且有非零下一层出边定义，仅图上的潜在有效性，不等于训练集激活非零。真实独立参数=非零权重+bias；BGM位图活槽可能保存精确零值，导出槽数与真实非零分别记录，均无共享。", "",f"节点主删除256个节点中{silent}个在2560拟合样本上静默，系数为零；本轮没有进入活动多基列重构检验，不能据此认定非零线性相关补偿有效。训练样本静默不代表全输入域无效。随机节点同样从低残差128候选池抽样，比较的是该受限候选域内身份，不是所有节点的均匀随机删除。","","## 主机制逐槽与接受/回退","","| 主臂 / 轮 | 试探组数量序列 | 训练内合格候选数 | 接受 | 操作后每层净删边 | 补偿新增 / 归零存活边 | 原图同4批验证 / 候选验证 |","| --- | --- | --- | --- | --- | --- | --- |"]
    for name in ("edge_task","node_task"):
        r=summary["results"][name]
        for e in r["events"]:
            trial=[f"{t['count']}:{'过' if t['pass'] else '拒'}" for t in e["trials"]]
            metric=(f"{e['original_select']['accuracy']:.3%}/{e['candidate_select']['accuracy']:.3%}" if "candidate_select" in e else "无候选，不访问额外验证")
            lines.append(f"| {names[name]} / {e['epoch']} | {', '.join(trial)} | {len(e['deleted'])} | {e['accepted']} | {e.get('net_deleted_per_layer','—')} | {e.get('compensation_new_nonzero_edges',0)} / {e.get('compensation_zeroed_survivor_edges',0)} | {metric} |")
        accepted=sum(e["accepted"] for e in r["events"])
        reduced=1-sum(r["structure"]["nonzero_edges"])/sum(fixed["structure"]["nonzero_edges"])
        lines += ["",f"{names[name]}接受{accepted}/{len(r['events'])}槽，真实边减少{reduced:.3%}；停止`{r['stop_reason']}`。没有为凑删除数补无候选槽。节点物理缩宽及补偿新增/归零边按实际记录，有限样本重构不保证全域等价。"]
    lines += ["","边主6/8槽接受的6组均达到262144边工程上限；两个输出层槽首组在训练内功能检验失败，0候选。节点主8/8均到32节点工程上限。因此边数减少是自然功能筛选/接受的结果，但本预算限制了合格群组大小，不能把末态视为自然压缩极限。随机边验证CE0.086619优于主0.116000；随机节点0.112836也低于主0.116370，不声称任务身份选择优势。", "", "边主第二隐藏层已有1055节点无任何下一层出边（有效2081/物理3136），随机同规模边仍3136有效。这是后续精确物理清理的具体入口：另登记从输出反向可达性，删除不能影响输出的隐藏节点及其入边/出边/bias，先同权重eval输出一致性核验，再登记自然编辑耦合。当前轮未实施这项清理，不能把潜在净收益计入已报告边数。", "","## 对照与成本","","五臂16轮上限、同batch64/增强/根RNG、固定lr0.0001及基线weight_decay0.0001；8槽开头全臂fresh AdamW，EMA保留，槽间保留矩。重构时优化器已统一重置，不声称等价迁移Adam。原节点ID产生canonical dropout掩码使存活单元对应，补偿不保证带dropout训练函数等价。","","主候选由训练内2560拟合/640留出真实CE+.01、准确率−.5pp、KL≤.002生成；边排序仅短名单，节点32基列/128候选与FP64窄gelsd系数为披露先验。最多群组量只是工程上限；验证局部CE+.01/准确率−.5pp、相对fixed累计CE+.05/准确率−2pp是任务保护。拒绝恢复原图同4批末完整图/权重/Adam/EMA/梯度/RNG，候选额外尝试不回退账本。随机继承实际接受时序/删量及同补偿规则，无法匹配真实逐层净删量如实标不可比，不据随机验证重选。","","| 臂 | 净 / 尝试更新 | 优化样本 | 验证及训练评价调用 / 样本 | 提议功能 / 激活样本 | 训练 / 评价 / 提议秒 | 臂墙钟秒 | CUDA峰分配字节 | 随机净删量可比 |","| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name,r in summary["results"].items():
        a=r["ledger"]
        lines.append(f"| {names[name]} | {a['net_updates']} / {a['attempted_updates']} | {a['optimization_samples']} | {a['evaluation_calls']} / {a['evaluation_samples']} | {a['proposal_function_samples']} / {a['proposal_activation_samples']} | {a.get('training_seconds',0):.3f} / {a.get('evaluation_seconds',0):.3f} / {a.get('proposal_seconds',0):.3f} | {r['wall_seconds']:.3f} | {r['peak_cuda_memory_bytes']} | {r['random_net_deletion_comparable']} |")
    total_optimization=sum(r["ledger"]["optimization_samples"] for r in summary["results"].values())
    task_ne=sum(e["accepted"] for e in summary["results"]["node_task"]["events"])
    lines += ["",f"会话{summary['session_wall_seconds']:.3f}秒，新增优化前向/反传{total_optimization}样本；初始化/候选缓存/事务/保存/导出开销包含臂墙钟，训练/评价/提议计时单列且非全部成本相加等式。设备{summary['registration']['device_name']}、FP32、CPU线程2、评价512。CUDA峰是Torch分配，不是设备全部驻留；CPU工作集峰未实测，登记缓存量级不能当实测。边支路仍dense零mask，节点支路物理缩宽dense；未实测本轮推断/能耗，不从删边或墙钟差推断加速。","","3200训练集、800验证集与旧1000探索性测试集；主每槽最多2次额外验证反馈，完整反馈实际数见账本。全臂冻结后各最末探索性测试一次，未据测试选模/调阈值；官方测试及剩余封存访问False，不扩数据/种子。旧探索性测试不是全新独立最终测试。","","## 证据与下一轮优先级","","此次有限诊断只能回答固定候选域、功能阈值和预算下自然编辑是否成立；不能把组规模上限附近停留解释为自然可压极限，也不能因零接受宣称冗余不存在。若自然删边/缩点有收益但距离CNN边数仍大，应先扩大或改进任务候选生成机制并另行登记有限预算，避免回到强制目标配额。若节点候选不足或回退，应以已保存训练内功能扰动与基组残差区分候选质量、短适应与任务损伤。", "",f"本轮节点主接受{task_ne}槽。逐轮准确率/CE/混淆、逐槽完整位图/节点ID、候选系数与接受/拒绝细节、源/配置/父SHA见JSON及evidence；大型BGM/完整最新resume本地保留。只有已有保存预测/位图核验，报告模型前向0，不重复GPU诊断。唯一小CPU机制项通过6.188秒；首次工具未保留完成输出，第二次同项保存日志，共2调用。", "",f"![自然结构与性能曲线]({title}_curves.png)",""]
    # 完整边身份无损保留为int32证据，避免百万JSON行淹没结果摘要。
    for name,result in summary["results"].items():
        for event in result["events"]:
            if event["kind"]!="edge": continue
            ids=np.asarray(event["deleted"],dtype=np.int32)
            event["deleted_count"]=len(ids)
            if not len(ids): continue
            filename=f"{name}_epoch_{event['epoch']}_deleted_edges.npz"
            path=evidence/filename
            np.savez_compressed(path,indices=ids,layer=np.int32(event["layer"]),epoch=np.int32(event["epoch"]))
            with np.load(path,allow_pickle=False) as saved:
                if not np.array_equal(saved["indices"],ids): raise ValueError("删除身份编码不一致")
            event["deleted_indices_evidence"]={"file":filename,"count":len(ids),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"dtype":"int32"}
            del event["deleted"]
    lines += ["边删除身份完整保存为evidence内int32 NPZ，JSON只引用数量/文件/SHA；这只是无损报告编码，未修改训练配置、源、运行summary或删除历史证据。", ""]
    target.with_suffix(".md").write_text("\n".join(lines),encoding="utf-8")
    summary["report_verification"]={"model_forwards":0,"seconds":time.perf_counter()-started,"all_epoch_predictions_checked":True,"all_slot_structure_counts_checked":True}
    target.with_suffix(".json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"report":str(target.with_suffix('.md')),"model_forwards":0,"seconds":time.perf_counter()-started},ensure_ascii=False))


if __name__=="__main__": main()
