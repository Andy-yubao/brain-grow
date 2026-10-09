"""只读保存预测、候选身份与闭合证书；0模型前向。"""
from pathlib import Path
import hashlib
import json
import shutil
import time

import numpy as np
from tools.report_capacity_edge_curve import read_structure,verify_prediction
from tools.report_capacity_natural import verify_scores

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    started=time.perf_counter()
    folder=ROOT/'experiments/runs/h1_capacity_closed_pruning_v1/mnist_static'
    summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    registration=summary['registration']; cfg=registration['config']
    if sha(ROOT/'configs/h1_capacity_closed_pruning_v1.json')!=registration['config_sha256'] or sha(ROOT/'tools/train_capacity_closed_pruning.py')!=registration['source_sha256']:
        raise ValueError('训练后源或配置发生变化')
    parent=ROOT/'experiments/runs/h1_capacity_reachability_v1/mnist_static/edge_task'
    if sha(parent/'model.bgm')!=registration['mother_model_sha256'] or sha(parent/'node_ids.json')!=registration['mother_node_ids_sha256']:
        raise ValueError('母图证据变化')
    for path,expected in registration['helper_sha256'].items():
        if sha(ROOT/path)!=expected: raise ValueError('已运行辅助源变化')
    title='h1_capacity_closed_pruning_v1_mnist_static_20261008'
    target=ROOT/'reports'/title; evidence=target.with_name(title+'_evidence'); evidence.mkdir(exist_ok=True)
    names={'fixed':'固定母图训练','task':'任务删边+精确闭合','random':'匹配闭合前删量随机'}
    evidence_manifest={}
    def copy(path,name):
        destination=evidence/name; shutil.copy2(path,destination)
        evidence_manifest[name]={'sha256':sha(destination),'bytes':destination.stat().st_size}
    for name,r in summary['results'].items():
        arm=folder/name
        for row in r['rows']:
            epoch=row['epoch']; prediction=arm/f'epoch_{epoch}_predictions.npz'
            with np.load(prediction,allow_pickle=False) as saved:
                verify_scores(saved['validation'],saved['validation_labels'],row['select'])
                verify_scores(saved['fit'],saved['fit_labels'],row['fit'])
            copy(prediction,f'{name}_{prediction.name}')
            stage=arm/f'epoch_{epoch}'
            if stage.exists():
                metadata=json.loads((stage/'structure.json').read_text(encoding='utf-8'))
                arrays,nonzero=read_structure(stage/'model.bgm')
                if nonzero!=row['structure']['nonzero_edges'] or sha(stage/'model.bgm')!=metadata['export']['sha256']:
                    raise ValueError('逐槽实际图计数或SHA不符')
                arrays.update({f'node_ids_{i}':np.asarray(ids,dtype=np.int32) for i,ids in enumerate(metadata['node_ids'])})
                path=evidence/f'{name}_epoch_{epoch}_structure.npz'; np.savez_compressed(path,**arrays)
                evidence_manifest[path.name]={'sha256':sha(path),'bytes':path.stat().st_size}
        verify_prediction(arm/'control_final.npz',r['control']); copy(arm/'control_final.npz',f'{name}_control_final.npz')
        _,actual=read_structure(arm/'model.bgm')
        if actual!=r['structure']['nonzero_edges'] or sha(arm/'model.bgm')!=r['export']['sha256']:
            raise ValueError('末态导出异常')
        copy(arm/'node_ids.json',f'{name}_node_ids.json')
        for e in r['events']:
            slot=cfg['edit_epochs'].index(e['additional_epoch'])+1
            if 'deleted_indices_evidence' in e:
                identity=e['deleted_indices_evidence']; path=arm/identity['file']
                if sha(path)!=identity['sha256']: raise ValueError('删除身份SHA异常')
                with np.load(path,allow_pickle=False) as saved:
                    ids=saved['indices']
                    if ids.dtype!=np.int32 or len(ids)!=e['deleted_count'] or len(np.unique(ids))!=len(ids) or int(saved['layer'])!=e['layer']:
                        raise ValueError('删除身份数量/唯一性/层不符')
                copy(path,f'{name}_{path.name}')
            if 'candidate_select' in e:
                path=arm/f'slot_{slot}_validation_trials.npz'
                with np.load(path,allow_pickle=False) as saved:
                    verify_scores(saved['original'],saved['labels'],e['original_select'])
                    verify_scores(saved['candidate'],saved['labels'],e['candidate_select'])
                copy(path,f'{name}_{path.name}')
            if 'closure' in e:
                c=e['closure']; path=arm/f'slot_{slot}_closure_equivalence.npz'
                with np.load(path,allow_pickle=False) as saved:
                    before=saved['before_logits'].astype(np.float64); after=saved['after_logits'].astype(np.float64)
                    delta=np.abs(after-before); bound=cfg['logit_atol']+cfg['logit_rtol']*np.abs(before)
                    if not np.isfinite(before).all() or not np.isfinite(after).all() or not (delta<=bound).all() or not np.array_equal(before.argmax(1),after.argmax(1)) or not np.array_equal(delta.max(1),saved['max_abs_per_sample']):
                        raise ValueError('逐样本闭合输出一致性不符')
                    if abs(float(delta.max())-c['comparison']['max_absolute_error'])>1e-12: raise ValueError('闭合误差汇总不符')
                if c['closed_deleted_per_layer']!=[a-b for a,b in zip(c['before']['nonzero_edges'],c['after']['nonzero_edges'])]:
                    raise ValueError('闭合实际边收益不符')
                for row in c['certificate']:
                    if len(row['direct_zero_outgoing_original_ids'])!=row['direct_zero_outgoing_count'] or len(row['indirect_unreachable_original_ids'])!=row['indirect_unreachable_count'] or row['before_width']-row['after_width']!=row['direct_zero_outgoing_count']+row['indirect_unreachable_count']:
                        raise ValueError('直接/间接不可达证书不符')
                copy(path,f'{name}_{path.name}'); copy(path.with_suffix('.json'),f'{name}_{path.with_suffix(".json").name}')
    task=summary['results']['task']; random=summary['results']['random']; fixed=summary['results']['fixed']
    if random['ledger']['net_updates']!=task['ledger']['net_updates']: raise ValueError('随机净更新未匹配')
    for e in random['events']:
        paired=next(p for p in task['events'] if p['epoch']==e['epoch'])
        if not paired['accepted'] or e['preclosure_deleted_per_layer']!=paired['preclosure_deleted_per_layer'] or not e['matched_preclosure_counts']:
            raise ValueError('随机接受时序/闭合前逐层删量不符')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for name,r in summary['results'].items():
        x=[row['additional_epoch'] for row in r['rows']]
        axes[0].plot(x,[sum(row['structure']['nonzero_edges'])/1e6 for row in r['rows']],label=names[name])
        axes[1].plot(x,[row['select']['accuracy']*100 for row in r['rows']],label=names[name])
        axes[2].plot(x,[row['select']['cross_entropy'] for row in r['rows']],label=names[name])
    for axis,ylabel in zip(axes,('真实非零边（百万）','验证准确率（%）','验证交叉熵')):
        axis.set(xlabel='新增轮',ylabel=ylabel); axis.grid(alpha=.2); axis.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(target.with_name(title+'_curves.png'),dpi=160); plt.close(fig)
    mother=sum(registration['mother_structure']['nonzero_edges']); cnn=cfg['cnn_edges_reference_only']
    lines=['# 自然任务剪边与精确闭合耦合：实际报告','',f"训练前提交`{registration['training_commit']}`。三臂同一清理后母图{mother}真实非零边，宽{registration['mother_structure']['widths']}；均新增16轮上限，不设目标边数/必删比例。CNN{cnn}是排除padding及bias的展开有效连接，仅作报告参照。基础CNN/FC约97%已达，不逐候选强守97%；性能完整记录。",'','## 全臂冻结末态','','| 臂 | 真非零边 / 母图下降 / CNN倍数 | 逐层边 | 物理宽度 / 有效隐藏节点 | 训练准确率 / CE | 验证准确率 / CE | 探索性测试准确率 / CE | BGM / BGM+原ID字节 | 独立非零参数 |','| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for name,r in summary['results'].items():
        s=r['structure']; n=sum(s['nonzero_edges'])
        lines.append(f"| {names[name]} | {n} / {1-n/mother:.3%} / {n/cnn:.3f}× | {s['nonzero_edges']} | {s['widths']} / {s['effective_hidden_nodes']} | {r['fit']['accuracy']:.3%} / {r['fit']['cross_entropy']:.6f} | {r['select']['accuracy']:.3%} / {r['select']['cross_entropy']:.6f} | {r['control']['accuracy']:.3%} / {r['control']['cross_entropy']:.6f} | {r['export']['bytes']} / {r['deployment_bytes_including_ids']} | {s['independent_parameters']} |")
    lines+=['','有效节点只表示图上的非零输入或bias且有出边，不是活动重构证据。无共享映射，独立参数、字节均非成功门槛；精确零权重与位图活槽分别记录。探索性测试为已历史使用的1000样本，不是全新独立最终测试。','', '## 主候选与闭合实际贡献','','| 轮 / 层 | 训练内尝试数 / 合格最大组 | 接受 | 闭合前逐层删除 | 闭合额外逐层删除 | 候选最终逐层净删除 | 原图4批验证准确率 / CE → 候选 |','| --- | --- | --- | --- | --- | --- | --- |']
    for e in task['events']:
        scores='无候选，0额外验证' if 'candidate_select' not in e else f"{e['original_select']['accuracy']:.3%} / {e['original_select']['cross_entropy']:.6f} → {e['candidate_select']['accuracy']:.3%} / {e['candidate_select']['cross_entropy']:.6f}"
        lines.append(f"| {e['epoch']} / {e['layer']+1} | {len(e['trials'])} / {e['deleted_count']} | {e['accepted']} | {e.get('preclosure_deleted_per_layer','—')} | {e.get('closure',{}).get('closed_deleted_per_layer','—')} | {e.get('net_deleted_per_layer','—')} | {scores} |")
    lines+=['','只把接受事务计入最终收益；拒绝恢复原图同4批完整状态，候选尝试与闭合成本保留。候选起组128、上限1048576、14级仅工程域，首次训练内损伤失败停止扩组；0合格/0接受允许发生。到工程上限不说明自然极限。训练内640功能检验并非参数学习独立留出，因为参数/EMA使用所有3200训练样本。','', '## 清理证书与随机比较','','| 臂 / 轮 | 直接 / 间接不可达节点（逐隐藏层） | 删除非零bias（逐层） | 闭合前后最大logit差 / 容差比例 | 全640同argmax | 闭合额外真边 |','| --- | --- | --- | --- | --- | --- |']
    for name in ('task','random'):
        for e in summary['results'][name]['events']:
            if 'closure' not in e: continue
            c=e['closure']; cert=c['certificate']; comparison=c['comparison']
            lines.append(f"| {names[name]} / {e['epoch']} | {[p['direct_zero_outgoing_count'] for p in cert]} / {[p['indirect_unreachable_count'] for p in cert]} | {[p['discarded_nonzero_bias_count'] for p in cert]} | {comparison['max_absolute_error']:.9g} / {comparison['max_bound_ratio']:.6f} | {comparison['argmax_equal_sample_count']}/640 | {sum(c['closed_deleted_per_layer'])} |")
    accepted=[e for e in task['events'] if e['accepted']]
    primary_closed=sum(sum(e['closure']['closed_deleted_per_layer']) for e in accepted)
    random_closed=sum(sum(e['closure']['closed_deleted_per_layer']) for e in random['events'] if e['accepted'])
    direct=sum(sum(e['preclosure_deleted_per_layer']) for e in accepted)
    lines += ['',f"主接受{len(accepted)}/{len(task['events'])}槽。接受的直接删边合计{direct}，闭合额外删边{primary_closed}；随机匹配同槽/同闭合前逐层删量，随后独立闭合额外{random_closed}边。最终边数允许不同，不称同最终边数随机。净更新主/随机{task['ledger']['net_updates']}/{random['ledger']['net_updates']}；随机可比标记{random['random_preclosure_comparable']}。", '', '直接不可达是没有向下一层任何非零出边；间接不可达仍有出边但全部通向下游不可达节点。非零bias也无法经不存在的输出路径影响logits，可同时删除。存活原ID/权重/掩码保持，每次闭合用同权重训练内640逐样本核验：abs差≤1e-4+1e-5·abs(前)，argmax全同；切维可能改变FP32求和，容差只为实现核验，未据标签选节点。闭合后短适应属于新参数变化，不冒称整个训练输出严格不变。', '', '## 对照、成本与有限结论', '', '全臂同新增批次/增强/原ID dropout，batch64、AdamW lr0.0001、weight_decay0.0001；8编辑槽全臂统一fresh Adam，EMA保留，切维后的fresh Adam与槽开头空状态一致。固定臂同头4批提供累计参照，主候选另4批产生额外成本。局部验证CE+.01/准确率−.5pp及相对固定CE+.05/准确率−2pp为任务保护；没有边数目标。完整回退含图/ID/权重/梯度/Adam/EMA/RNG/mode。', '', '| 臂 | 净 / 尝试更新 | 优化样本 | 常规评价调用 / 样本 | 功能提议样本 | 闭合评价调用 / 样本 | 回退 | 训练 / 常规评价 / 提议 / 闭合秒 | 臂墙钟秒 | CUDA峰字节 | 工作集 / 进程累计峰字节 |', '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for name,r in summary['results'].items():
        a=r['ledger']; m=r['process_memory_at_arm_end_cumulative']
        proposal=sum(e['proposal_seconds'] for e in r['events']); closure=sum(e.get('closure',{}).get('seconds',0) for e in r['events'])
        lines.append(f"| {names[name]} | {a['net_updates']} / {a['attempted_updates']} | {a['optimization_samples']} | {a['evaluation_calls']} / {a['evaluation_samples']} | {a['proposal_function_samples']} | {a['closure_evaluation_calls']} / {a['closure_evaluation_samples']} | {a['rollbacks']} | {a.get('training_seconds',0):.3f} / {a.get('evaluation_seconds',0):.3f} / {proposal:.3f} / {closure:.3f} | {r['wall_seconds']:.3f} | {r['peak_cuda_memory_bytes']} | {m['working_set_bytes']} / {m['peak_working_set_bytes']} |")
    lines += ['',f"会话{summary['session_wall_seconds']:.3f}秒，设备{registration['device_name']}、FP32、TF32关闭、CPU线程2、评价batch512。执行仍dense零mask，但不可达节点被物理slice。BGM编码字节、进程工作集累计峰与CUDA分配峰区分报告；候选/闭合/保存均纳入实际墙钟，计时栏目非严格可加成本分解。未登记延迟/能耗测量，不能由边数或墙钟推断推断加速。", '', '3200训练/800验证，额外验证每有候选主槽2次（原图4批与候选4批），固定每槽1次参照；常规逐轮训练/验证反馈另如实记录。三臂全部冻结后最末探索性测试各一次，未据其追调；官方及剩余封存访问False，不扩数据/种子、不据验证选末态。保存逐轮混淆/CE、每槽图位图及原ID、删除身份和闭合前后logits，完整最新resume及BGM本地保留。唯一新增CPU事务/闭合检查一次通过7.884秒；报告仅已存证据核验，0新模型前向。', '', f"本轮主相对母图真实边下降{1-sum(task['structure']['nonzero_edges'])/mother:.3%}，仍{sum(task['structure']['nonzero_edges'])/cnn:.3f}倍CNN参照。四个前两层候选均到1048576工程上限，两个第三层与两个输出层候选在更大组训练功能检验失败，不把末态称自然极限。正式主8/8接受、0验证回退；小CPU拒绝恢复已验证，不冒称正式发生拒绝。三臂停止均finite_budget，无严重崩溃/闭合不等价/不可比触发。任务验证CE低于随机，随机探索性测试准确率/CE更好，不宣称任务排序有全面性能优势；最终规模差反映剪边身份与图闭合共同作用，不等同同规模任务优势。", '', '下一独立优先检验活动多节点重构，明确排除静默目标及基列；不继续扩大同一候选上限以追CNN数量。旧自然节点主所删256目标均静默，不能作为活动重构成功证据；只读旧随机证据有24列非零多基补偿但缺完整激活残差。当前没有混入此机制，下一须另预登记训练内激活范数/多系数/重构残差和功能筛选、物理净边收益、完整回退及随机补偿对照，不设边数目标。', '',f'![自然耦合结构与验证曲线]({title}_curves.png)','']
    target.with_suffix('.md').write_text('\n'.join(lines),encoding='utf-8')
    summary['report_verification']={'model_forwards':0,'seconds':time.perf_counter()-started,'source_config_parent_sha_unchanged':True,'all_predictions_counts_id_closure_pairs_checked':True,'evidence_manifest':evidence_manifest}
    target.with_suffix('.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'report':str(target.with_suffix('.md')),'seconds':time.perf_counter()-started,'evidence_files':len(evidence_manifest),'model_forwards':0},ensure_ascii=False))


if __name__=='__main__': main()
