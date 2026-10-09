"""有效边压缩曲线：只读保存预测、BGM位图与事件，不运行模型。"""
from pathlib import Path
import json
import shutil
import struct
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def verify_prediction(path, expected):
    with np.load(path,allow_pickle=False) as saved:
        logits,y = saved["logits"].astype(np.float64),saved["labels"]
    prediction=logits.argmax(1)
    loss=np.log(np.exp(logits-logits.max(1,keepdims=True)).sum(1))+logits.max(1)-logits[np.arange(len(y)),y]
    matrix=np.bincount(y*10+prediction,minlength=100).reshape(10,10)
    if abs((prediction==y).mean()-expected["accuracy"])>1e-7 or abs(loss.mean()-expected["cross_entropy"])>1e-6 or matrix.tolist()!=expected["confusion_matrix"]:
        raise ValueError("保存预测与实际曲线不符")


def read_structure(path):
    arrays, nonzero = {},[]
    with path.open("rb") as source:
        assert source.read(4)==b"BGM1"
        header=json.loads(source.read(struct.unpack("<I",source.read(4))[0]))
        for i,row in enumerate(header["layers"]):
            bitmap=np.frombuffer(source.read(row["bitmap_bytes"]),dtype=np.uint8).copy()
            if int(np.unpackbits(bitmap,bitorder="little")[:int(np.prod(row["shape"]))].sum())!=row["edges"]:
                raise ValueError("位图计数异常")
            values=np.frombuffer(source.read(4*row["edges"]),dtype="<f4")
            nonzero.append(int(np.count_nonzero(values)))
            arrays[f"layer_{i}"]=bitmap
            source.seek(4*row["bias_count"],1)
    arrays["header_utf8"]=np.frombuffer(json.dumps(header,ensure_ascii=False).encode(),dtype=np.uint8)
    return arrays,nonzero


def main():
    started=time.perf_counter()
    folder=ROOT/"experiments/runs/h1_capacity_edge_curve_v1/mnist_static"
    summary=json.loads((folder/"summary.json").read_text(encoding="utf-8"))
    title="h1_capacity_edge_curve_v1_mnist_static_20261008"
    target=ROOT/"reports"/title
    evidence=target.with_name(title+"_evidence")
    evidence.mkdir(exist_ok=True)
    labels={"fixed":"固定父剪枝图","task":"任务敏感身份","random":"同规模随机身份"}
    for name,result in summary["results"].items():
        for stage in result["stages"]:
            arm=folder/name/f"stage_{stage['stage']}"
            for split,file in (("select","validation.npz"),("fit","fit.npz")):
                verify_prediction(arm/file,stage[split])
                shutil.copy2(arm/file,evidence/f"{name}_stage_{stage['stage']}_{split}.npz")
            arrays,nonzero=read_structure(arm/"model.bgm")
            if nonzero!=stage["nonzero_edges"] or (name!="fixed" and sum(nonzero)>stage["target_edges"]):
                raise ValueError("真实非零边未达到目标/与曲线不符")
            if name!="fixed" and stage["structure"]["active_edges"]!=summary["registration"]["layer_quotas"][stage["stage"]-1]:
                raise ValueError("逐层配额不符")
            last_shape=[10,256]
            degree=np.unpackbits(arrays["layer_3"],bitorder="little")[:int(np.prod(last_shape))].reshape(last_shape).sum(1).tolist()
            stage["output_input_degrees"]=degree
            np.savez_compressed(evidence/f"{name}_stage_{stage['stage']}_structure.npz",**arrays)
        verify_prediction(folder/name/"control_final.npz",result["control"])
        shutil.copy2(folder/name/"control_final.npz",evidence/f"{name}_control_final.npz")
    if summary["results"]["task"]["events"]!=summary["results"]["random"]["events"]:
        raise ValueError("主/随机实际删量时序不等")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"]=["Microsoft YaHei","SimHei","DejaVu Sans"]
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    cnn=summary["registration"]["config"]["cnn_effective_edges"]
    for name in ("task","random"):
        stages=summary["results"][name]["stages"]
        x=[sum(s["nonzero_edges"])/cnn for s in stages]
        axes[0].plot(x,[s["select"]["accuracy"]*100 for s in stages],marker="o",label=labels[name])
        axes[1].plot(x,[s["select"]["cross_entropy"] for s in stages],marker="o")
        axes[2].plot(x,[s["fit"]["accuracy"]*100 for s in stages],marker="o")
    fixed=summary["results"]["fixed"]
    axes[0].axhline(fixed["select"]["accuracy"]*100,linestyle="--",color="gray",label="固定父图末态参照")
    for axis,ylabel in zip(axes,("验证准确率（%）","验证交叉熵","训练准确率（%）")):
        axis.set(xlabel="真实非零边 / CNN有效边",ylabel=ylabel,xscale="log")
        axis.grid(alpha=.2)
        axis.invert_xaxis()
    axes[0].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(target.with_name(title+"_curves.png"),dpi=160)
    plt.close(fig)
    lines=["# 扩容FC有效非零连接多档压缩实际报告","","2026-10-08；训练前实现/登记提交`"+summary["registration"]["training_commit"]+"`。用户最新主目标为有效非零边接近CNN，基本CNN/FC约97%已通过，后续每档准确率不作硬否决；完整报告任务损伤，独立参数/字节不作主目标。CNN433680是展开后的有效连接，排除padding/偏置，不是206922独立参数。","","三臂从同task16完整模型/mask/Adam/EMA/RNG开始，fixed指保持10400411边的父剪枝图，不是原25.39m完整FC。四档按初始层比例整数最大余数固定配额，保留所有bias及固定784→6272→3136→256→10宽度；不引入像素/局部/共享先验。每档首轮剪至配额，最多8轮参数适应，lr固定.0001，Adam继承不重置，同全局批次/仿射增强/dropout流。","","| 档 / 臂 | 目标边 | 逐层真实非零边 | 总非零 / 目标误差 | 验证准确率 / CE | 训练准确率 / CE | 适应轮 | 实际字节 |","| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for i in range(len(summary["results"]["task"]["stages"])):
        for name in ("task","random","fixed"):
            s=summary["results"][name]["stages"][i]
            delta=s["nonzero_edge_target_error"] if name!="fixed" else "—"
            lines.append(f"| {s['target_edges']//cnn}× / {labels[name]} | {s['target_edges'] if name!='fixed' else '固定父图'} | {' / '.join(map(str,s['nonzero_edges']))} | {sum(s['nonzero_edges'])} / {delta} | {s['select']['accuracy']:.3%} / {s['select']['cross_entropy']:.6f} | {s['fit']['accuracy']:.3%} / {s['fit']['cross_entropy']:.6f} | {s['stage_epochs_run']} | {s['structure']['bytes']} |")
    task=summary["results"]["task"]
    lines += ["",f"主已到达{len(task['stages'])}/4档；停止原因`{task['stop_reason']}`，全部预定档位到达={summary['all_planned_stages_reached']}。主mask总边各档精确等于目标、真实非零≤目标，各层配额与随机删量/时序完全匹配。若主严重崩溃提前停，后续未到达档不称完成；fixed/random只匹配实际主净轮数。","","| 最末冻结臂 | 探索性测试准确率 / CE | 净 / 尝试更新 | 优化样本 | 评价调用 / 样本 | 臂墙钟秒 | GPU峰分配字节 |","| --- | --- | --- | --- | --- | --- | --- |"]
    for name in ("task","random","fixed"):
        r=summary["results"][name]; a=r["ledger"]
        lines.append(f"| {labels[name]} | {r['control']['accuracy']:.3%} / {r['control']['cross_entropy']:.6f} | {a['net_updates']} / {a['attempted_updates']} | {a['optimization_samples']} | {a['evaluation_calls']} / {a['evaluation_samples']} | {r['wall_seconds']:.3f} | {r['peak_cuda_memory_bytes']} |")
    samples=sum(r["ledger"]["optimization_samples"] for r in summary["results"].values())
    calls=sum(r["ledger"]["evaluation_calls"] for r in summary["results"].values())
    eval_samples=sum(r["ledger"]["evaluation_samples"] for r in summary["results"].values())
    last=task["stages"][-1]
    disconnected=[i for i,n in enumerate(last["output_input_degrees"]) if not n]
    degree_note=f"末态主输出类0—9入度={last['output_input_degrees']}；无输入边类别={disconnected}。验证各类正确数={[last['select']['confusion_matrix'][i][i] for i in range(10)]}（每类80），训练各类正确数={[last['fit']['confusion_matrix'][i][i] for i in range(10)]}（每类320）。bias仍存在，无输入类输出只剩常数；1×边数达到但任务效果严重下降，不能称保持原任务效果的成功。固定比例配额的输出断连伴随类别识别丧失，是下一轮分层配额诊断的具体依据；不证明433680总边必然不足，也不证明保留输出边一定解决所有问题。此分析来自保存位图与训练/验证混淆，未根据探索性测试制定规则。"
    lines += ["",degree_note]
    lines += ["",f"新增优化前向/反传各{samples}样本，评价{calls}次/{eval_samples}样本；完整会话{summary['session_wall_seconds']:.3f}秒，三臂完全冻结后探索性测试阶段{summary['control_phase_seconds']:.3f}秒。逐项训练/排序/评价/编码/checkpoint成本见JSON，数据装载准备计入会话；首次task臂含CUDA算子初始化，不能以臂计时差称算法加速。设备{summary['registration']['device_name']}，FP32/batch64/评价512/CPU线程2，仍dense Linear+零mask。未测能耗/稀疏计算加速。","","主严重崩溃预登记为训练<80%且验证<75%连续两轮；随机崩溃只记录并在有限同轮数预算中保留对照。没有97%/损失/文件接受或回退，不挑最佳点掩盖损伤；每档完整准确率/CE/混淆矩阵和结构保存。偏置共9674，含偏置独立参数=活边+9674，仅附带信息；未物理清理静默节点，训练参数槽仍25401290。","","中间档只评价训练集/验证集，全曲线全部三臂冻结后各最末态1000探索性测试一次；该集合已历史探索，不能称全新独立最终测试。未根据测试改配额/lr/轮数，官方MNIST测试集及剩余封存ID未访问，种子11/数据20261004不变。","","唯一新增小CPU配额/真实删边行为检查通过0.038秒，复用旧导出/事务/数据证据，没有额外GPU诊断或旧训练/测试重跑。报告读取每档保存预测/位图/活值核验完整混淆、CE与真实非零边，0模型前向。所有已到达档位模型/完整状态及随机身份在本地`experiments/runs/h1_capacity_edge_curve_v1/mnist_static/`保留；报告附每档两划分预测/完整位图和最末探索性测试预测，源码/配置/父SHA在JSON，入口拒绝覆盖。", "",f"报告生成{time.perf_counter()-started:.3f}秒；单根曲线支持该机制本预算下性能/有效边权衡，不称跨任务普适或统计显著。", "",f"![真实边数与任务性能曲线]({title}_curves.png)",""]
    target.with_suffix(".md").write_text("\n".join(lines),encoding="utf-8")
    target.with_suffix(".json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"report":str(target.with_suffix('.md')),"model_forwards":0,"report_seconds":time.perf_counter()-started},ensure_ascii=False))


if __name__ == "__main__":
    main()
