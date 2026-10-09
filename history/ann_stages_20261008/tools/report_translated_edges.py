"""只读保存预测与授权验证像素活动；不加载模型、不新增模型前向。"""
from pathlib import Path
import hashlib
import json
import shutil

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def main():
    run = ROOT / "experiments/runs/h1_translated_edges_v1/mnist_circular"
    summary = json.loads((run/"summary.json").read_text(encoding="utf-8"))
    stem = "h1_translated_edges_v1_mnist_circular_20261008"
    reports = ROOT / "reports"
    evidence = reports / (stem+"_evidence")
    evidence.mkdir(parents=True, exist_ok=True)
    records, id_tables, deletion_records = {}, {}, {}
    positions = np.load(run/"position_manifest.npz", allow_pickle=False)
    validation_ids = positions["validation_original_ids"]
    freq = np.bincount((positions["validation_offsets"][...,0]*28+positions["validation_offsets"][...,1]).flatten(),minlength=784)
    data_manifest=json.loads((run/"data_manifest.json").read_text(encoding="utf-8"))
    cache=ROOT/"data/structure_mnist_v1/select.npz"
    cache_hash=hashlib.sha256(cache.read_bytes()).hexdigest()
    assert cache_hash==data_manifest["original"]["cache_sha256"]["select"]
    with np.load(cache,allow_pickle=False) as p:
        raw_images=p["x"].reshape(-1,28,28)
    transformed=np.stack([np.roll(image,tuple(shift),axis=(0,1)) for image,sh in zip(raw_images,positions["validation_offsets"]) for shift in sh])
    activity={}
    for name,images in (("original_validation",raw_images),("translated_validation",transformed)):
        mean=images.mean(0,dtype=np.float64)
        energy=np.square(images).mean(0,dtype=np.float64)
        activity[name]={"mean_pixel":mean.tolist(),"second_moment":energy.tolist(),"central14_energy_fraction":float(energy[7:21,7:21].sum()/energy.sum()),
                        "energy_spatial_cv":float(energy.std()/energy.mean()),"original_ids":800,"shifted_versions":len(images)}
    np.savez_compressed(evidence/"validation_pixel_activity.npz",**{name+"_"+key:np.asarray(value[key]) for name,value in activity.items() for key in ("mean_pixel","second_moment")})
    for source in (run/"position_manifest.npz",run/"data_manifest.json",run/"registration.json"):
        shutil.copyfile(source,evidence/source.name)
    for name, result in summary["results"].items():
        directory = run/name
        if name == "random_pair":
            candidates = [run/"node/random_pair_exploratory_test.npz"]
        else:
            candidates = sorted(directory.glob("validation*.npz")) + sorted(directory.glob("exploratory_test.npz")) + sorted(directory.glob("training_fixed_one_position.npz"))
        for source in candidates:
            target = evidence/(name+"_"+source.name)
            shutil.copyfile(source,target)
            records[target.name] = {"bytes":target.stat().st_size,"sha256":hashlib.sha256(target.read_bytes()).hexdigest()}
            if source.name.startswith("validation"):
                with np.load(source,allow_pickle=False) as p:
                    labels=p["labels"].reshape(-1,8)
                    predicted=p["logits"].argmax(1).reshape(-1,8)
                    id_tables[target.name] = [{"original_id":int(raw),"digit":int(label[0]),"correct_positions":int((guess==label).sum()),"predictions":guess.tolist()} for raw,label,guess in zip(validation_ids,labels,predicted)]
        if name in ("magnitude","node"):
            for source in sorted(directory.glob("deletion_stage_*.npz")):
                with np.load(source,allow_pickle=False) as p:
                    row_counts = {k:p[k] for k in p.files if k.startswith("row_counts_")}
                target = evidence/(name+"_"+source.stem+"_row_counts.npz")
                np.savez_compressed(target,**row_counts)
                deletion_records[name+"_"+source.name] = {"local_run_path":str(source),"bytes":source.stat().st_size,"sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"row_counts_evidence":target.name}
            for source in sorted(directory.glob("calibration_stage_*.npz")):
                shutil.copyfile(source,evidence/(name+"_"+source.name))
            for source in sorted(directory.glob("random_pair_stage_*.npz")):
                with np.load(source,allow_pickle=False) as p:
                    target=evidence/(name+"_"+source.stem+"_predictions.npz")
                    np.savez_compressed(target,logits=p["logits"],labels=p["labels"],original_ids=validation_ids)
                deletion_records[name+"_"+source.name] = {"local_run_path":str(source),"bytes":source.stat().st_size,"sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"prediction_evidence":target.name}
            source=directory/"model.bgm"
            if source.stat().st_size <= 4*1024*1024:
                shutil.copyfile(source,evidence/(name+"_model.bgm"))
    payload = {"summary":summary,"validation_shift_counts":freq.tolist(),"evidence":records,"validation_original_id_tables":id_tables,"deletion_identity_local_records":deletion_records,
               "validation_pixel_activity":activity,"report_model_forwards":0,"new_training":0,"pixels_read":"仅授权800验证ID缓存，来源SHA已核验"}
    (reports/(stem+".json")).write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    lines = ["# 位置均衡MNIST与节点贡献剪边实际报告","",f"运行状态：`{summary['gate']}`；单模型根11、数据根20261004。全体位移循环边界是诊断假设，不代表自然图像。", "",
             "## 冻结结果","","| 模型/机制 | 验证准确率 | 验证CE | 历史探索性测试准确率 | 真实展开边 | 导出字节 |","| --- | ---: | ---: | ---: | ---: | ---: |"]
    names={"baseline":"密集FC","cnn":"循环边界CNN","fixed":"固定图继续训练","magnitude":"旧全局幅度","node":"节点输入贡献","random_pair":"第5槽拒绝随机候选（非最终同规模对照）"}
    for name,r in summary["results"].items():
        v=r["validation"]
        test=r.get("exploratory_test")
        test_text=f"{test['accuracy']:.3%}" if test else "未访问"
        lines.append(f"| {names.get(name,name)} | {v['accuracy']:.3%} | {v['cross_entropy']:.6f} | {test_text} | {sum(r['structure']['nonzero_edges'])} | {r['structure']['bytes']} |")
    lines += ["", "FC物理节点始终3136/1568/128/10，未缩点/清理；CNN节点相同，循环输入展开456000边、1.2倍547200只作外评。FC执行仍为dense零mask，没有测推断加速或能耗。", "",
              "## 位置覆盖与评价","", f"验证800原ID×8固定互异位置，全784位移频数最小{freq.min()}、最大{freq.max()}；位置版本有关联，不能视为6400独立样本。训练每轮每ID随机位置，实际累计覆盖如下。", ""]
    if "random_pair" in summary["results"]:
        pair=summary["results"]["random_pair"]
        lines += [f"随机候选只跟随节点冻结所选槽stage{pair['selected_stage']}，并非与回退后的节点最终模型同规模；该槽主候选是否接受为{pair.get('forced_retained_timing',True)}。若拒绝，应比较该槽两种候选，不把随机候选与回退模型作边身份因果对照。", ""]
    if summary["gate"]=="completed":
        lines += ["当前结果：旧幅度在位置活动已均衡的任务上仍减少75%边，验证较密集FC只降约0.422pp；因此旧成功并非只能在固定中部信息条件出现。但幅度1,894,464边约CNN4.155倍，节点3,166,655边约6.944倍，均未达1.2倍外评目标。节点机制本轮最终压缩弱于旧幅度。第一槽节点93.250%对随机90.156%（+3.094pp）支持该槽边身份有用；后续槽没有稳定随机优势，不能泛称方法全面优越。", ""]
    for name,r in summary["results"].items():
        hist=np.asarray(r["ledger"].get("training_position_histogram",[]))
        if len(hist):
            lines.append(f"- {names.get(name,name)}：累计{int(hist.sum())}训练样本位移，覆盖{int((hist>0).sum())}/784，格频数{int(hist.min())}—{int(hist.max())}。")
        if "training_fixed_one_position" in r:
            lines.append(f"- {names.get(name,name)}冻结后训练ID固定1位移准确率{r['training_fixed_one_position']['accuracy']:.3%}；非全部训练位移成绩。")
    lines += ["", "按原ID的各位置预测表、逐类混淆矩阵、位置清单/来源哈希保存在机器报告和证据目录。官方测试及剩余封存数据访问0；历史探索性测试仅所有机制冻结后访问，不能称全新独立测试。", "", "## 全部剪边阶段","",
              "| 机制/阶段 | 真实边 | 验证准确率 | 验证CE | 接受 |", "| --- | ---: | ---: | ---: | --- |"]
    for name in ("fixed","magnitude","node"):
        for stage in summary["results"].get(name,{}).get("stages",[]):
            v=stage["validation"]
            lines.append(f"| {names[name]} / {stage['stage']} | {sum(stage['structure']['nonzero_edges'])} | {v['accuracy']:.3%} | {v['cross_entropy']:.6f} | {stage['accepted']} |")
    for name in ("fixed","magnitude","node"):
        r=summary["results"].get(name)
        if r:
            lines.append(f"\n{names[name]}按统一验证功能/最少边规则冻结stage{r['chosen_stage']}；40净轮末态stage5验证{r['stages'][-1]['validation']['accuracy']:.3%}。两者分开报告，不把继续训练末态换成起点。")
            for e in r["events"]:
                if not e["accepted"]:
                    v=e["candidate_validation"]
                    lines.append(f"- 拒绝候选stage{e['stage']}：{sum(e['candidate_structure']['nonzero_edges'])}边，验证{v['accuracy']:.3%}/CE{v['cross_entropy']:.6f}；已完整回退，随后普通恢复另计成本。")
    lines += ["", "完整删除身份/大模型留本地run并登记SHA；逐row删量、校准激活和完整阶段预测纳入证据，重要最终稀疏BGM仅≤4MiB时复制。", ""]
    lines += ["## 验证图像实际信息分布","", "仅读取授权800验证ID缓存和冻结位移清单，未新增模型前向；原缓存SHA一致。位移频数本身不能保证有限样本像素活动完全均匀，因此另统计像素均值/二阶活动。", ""]
    for name,a in activity.items():
        lines.append(f"- {name}：中央14×14占总像素二阶能量{a['central14_energy_fraction']:.3%}（面积25%）；全784位置二阶活动变异系数{a['energy_spatial_cv']:.5f}。")
    if "node" in summary["results"]:
        lines += ["", "## 节点贡献与同父随机候选","", "节点评分abs(w)×训练输入RMS，每节点低分前缀贡献能量预算0.05，不指定固定删率；同时保护排序尾部及最大权重边。评分忽略输入相关项，因此能量预算不保证联删无损。以下扰动用固定剪前层输入测量，最终接受由8轮恢复后的验证功能决定。", ""]
        for e in summary["results"]["node"]["events"]:
            pair=e["random_pair"]
            delta=[round(d["relative_input_rms"],4) for d in e["disturbance"]]
            lines.append(f"- 阶段{e['stage']}：逐层删{e['deleted_per_layer']}；线性输入相对RMS扰动{delta}；主候选验证{e['candidate_validation']['accuracy']:.3%}、接受{e['accepted']}；同父/逐row同删量随机验证{pair['validation']['accuracy']:.3%}。")
        lines += ["", "随机各阶段独立从节点主臂该槽剪前图、Adam/RNG开始；这是候选边身份比较，不是连续随机训练臂。全部分支保留，本报告不据随机或测试成绩挑阶段。"]
    lines += ["", "## 实际成本及边界","", f"全会话{summary['session_wall_seconds']:.3f}秒；设备{summary['registration']['device_name']}，FP32、batch64/评价512、CPU2线程、TF32关闭。训练/剪边搜索/随机分支/验证/导出/快照成本和CUDA峰见机器报告。", ""]
    lines += ["训练前冻结提交c866d83。CNN第10轮保存progress时Windows文件替换被拒绝，完整checkpoint已保存；第一次恢复因Windows默认GBK读取UTF8中文账本失败，未训练。第二次设置进程PYTHONUTF8=1后按同一锁定源/config恢复，FC200轮不重训，总预算从既有progress/failure最大耗时累计。恢复入口：PowerShell先`$env:PYTHONUTF8='1'`再`.venv/Scripts/python.exe -B -m tools.train_translated_edges --resume`。", ""]
    lines += ["CNN40轮又遇文件替换拒绝，从41轮以`-X utf8`接续；工程包装器另登记并提交8554c84，是否实际使用以engineering_io日志为准。会话Budget从各次模型会话创建后累计到结束，包含候选拒绝、随机分支及文件等待，不含解释器初始化、Budget创建前GBK启动失败或人工恢复间隔；不能称完整日历端到端成本。各臂wall_seconds只记录最后接续段，跨恢复训练/评价成本应汇总逐epoch秒数，完整总成本以累计Budget为准。", ""]
    engineering=list(run.glob("engineering_io_*.json"))
    if not engineering:
        lines += ["工程包装器本轮仅备用，实际启用0次、包装重试0次、包装等待0秒；第三次标准-X utf8恢复顺利完成。", ""]
    else:
        waits=[json.loads(p.read_text(encoding="utf-8")) for p in engineering]
        lines += [f"工程包装实际启用{len(waits)}次、重试{sum(r.get('retry_count',0) for r in waits)}次、等待{sum(r.get('waiting_seconds',0.) for r in waits):.6f}秒。", ""]
    for name,r in summary["results"].items():
        lines.append(f"- {names.get(name,name)}：主训练账本{r['ledger']['attempted_updates']}尝试更新、{r['ledger']['optimization_samples']}优化样本；CUDA峰{r.get('peak_cuda_memory_bytes','见所属节点臂')}字节。")
    if "node" in summary["results"]:
        events=summary["results"]["node"]["events"]
        random_updates=sum(e["random_pair"]["ledger"]["attempted_updates"] for e in events)
        random_samples=sum(e["random_pair"]["ledger"]["optimization_samples"] for e in events)
        main_updates=sum(r["ledger"]["attempted_updates"] for name,r in summary["results"].items() if name!="random_pair")
        lines += [f"全部5个随机分支额外{random_updates}更新/{random_samples}优化样本（汇总表随机只显示末槽，不能当全部搜索成本）；所有主臂与全部随机合计{main_updates+random_updates}尝试更新。节点每槽训练校准3200输入及联合扰动3200输入另计，共各16000输入。随机最终训练ID固定1位置评价3200样本在原随机ledger未加计，此处明确补充，其他完整账本保留原记录。", ""]
    if summary["gate"] != "completed":
        lines += ["", "本轮同分布基线未过90%功能底线，未执行剪枝，未访问历史测试。不能把当前压缩或任务泛化作为成功；失败反映此数据量/固定骨架/冻结学习规则下的困难，不能单独归因于剪枝机制。"]
    else:
        lines += ["", "单种子、重复验证反馈和历史测试范围限制结论；是否胜过旧幅度/随机、是否达到连接外评目标应根据完整表判断，不由单个准确率或边数推出。未验证跨任务、卷积发现、严格共享或实际加速。"]
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.family"]=["Microsoft YaHei","DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"]=False
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for name in ("baseline","cnn"):
            rows=summary["results"][name]["rows"]
            axes[0].plot([r["epoch"] for r in rows],[r["validation"]["accuracy"] for r in rows],label=names[name])
        axes[0].set(xlabel="训练轮数",ylabel="验证准确率",title="均匀循环平移：同分布基线")
        axes[0].legend()
        for name in ("fixed","magnitude","node"):
            rows=summary["results"].get(name,{}).get("stages",[])
            if rows:
                axes[1].plot([sum(r["structure"]["nonzero_edges"]) for r in rows],[r["validation"]["accuracy"] for r in rows],marker="o",label=names[name])
            rejected=[e for e in summary["results"].get(name,{}).get("events",[]) if not e["accepted"]]
            if rejected:
                axes[1].scatter([sum(e["candidate_structure"]["nonzero_edges"]) for e in rejected],[e["candidate_validation"]["accuracy"] for e in rejected],
                                marker="x",s=70,color="C1" if name=="magnitude" else "C2",label=("幅度" if name=="magnitude" else "节点")+"拒绝候选")
        axes[1].set(xlabel="真实非零边",ylabel="验证准确率",xscale="log",title="全部阶段：测试不参与选模")
        if "node" in summary["results"]:
            axes[1].legend()
        fig.tight_layout()
        fig.savefig(reports/(stem+"_curves.png"),dpi=160)
        plt.close(fig)
        lines += ["",f"![训练与结构曲线]({stem}_curves.png)"]
        fig,axes=plt.subplots(2,2,figsize=(8,7))
        for col,(name,a) in enumerate(activity.items()):
            for row,key in enumerate(("mean_pixel","second_moment")):
                limit=max(np.asarray(item[key]).max() for item in activity.values())
                chart=axes[row,col].imshow(a[key],cmap="magma",origin="upper",vmin=0,vmax=limit)
                axes[row,col].set_title(("原始验证图像" if col==0 else "固定均衡平移后")+"\n"+("像素均值" if row==0 else "像素二阶活动"))
                fig.colorbar(chart,ax=axes[row,col],shrink=.8)
        fig.tight_layout()
        fig.savefig(reports/(stem+"_pixel_activity.png"),dpi=160)
        plt.close(fig)
        lines += ["",f"![平移前后的像素活动]({stem}_pixel_activity.png)"]
    except ImportError:
        lines += ["", "绘图库不可用，完整数值曲线已保存。"]
    (reports/(stem+".md")).write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"report":str(reports/(stem+".md")),"model_forwards":0,"gate":summary["gate"]},ensure_ascii=False))


if __name__=="__main__":
    main()
