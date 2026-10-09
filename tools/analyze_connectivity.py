"""只读BGM/冻结CNN权重的离线连接分析；不导入训练入口或读取数据。"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import struct
import time
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'experiments/runs/h1_aligned_edges_v1/mnist_static'
OUT = ROOT / 'docs/reports/2026-10-08/h1_connectivity_analysis_v1'
QUANTILES = [0, .1, .25, .5, .75, .9, 1]
COORD = np.array([(i // 28, i % 28) for i in range(784)])

def dist(values):
    values = np.asarray(values)
    return {'count': int(values.size), 'zero': int((values == 0).sum()),
            'quantiles_0_10_25_50_75_90_100': np.quantile(values, QUANTILES).tolist() if values.size else [],
            'mean': float(values.mean()) if values.size else None}

def decode(path):
    raw = path.read_bytes()
    if raw[:4] != b'BGM1':
        raise ValueError('BGM magic错误')
    length = struct.unpack('<I', raw[4:8])[0]
    header = json.loads(raw[8:8+length])
    pos, matrices, biases = 8+length, [], []
    for row in header['layers']:
        mask = np.unpackbits(np.frombuffer(raw[pos:pos+row['bitmap_bytes']], np.uint8), bitorder='little')[:int(np.prod(row['shape']))].astype(bool)
        pos += row['bitmap_bytes']
        values = np.frombuffer(raw[pos:pos+4*row['edges']], '<f4')
        pos += 4*row['edges']
        weights = np.zeros(mask.size, np.float32)
        weights[mask] = values
        matrices.append(weights.reshape(row['shape']))
        biases.append(np.frombuffer(raw[pos:pos+4*row['bias_count']], '<f4').copy())
        pos += 4*row['bias_count']
    if pos != len(raw):
        raise ValueError('BGM尾部错误')
    return matrices, biases, {'path': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw), 'header': header}

def graph(matrices, biases):
    masks = [w != 0 for w in matrices]
    widths = [masks[0].shape[1]] + [a.shape[0] for a in masks]
    backward = [None] * len(widths)
    backward[-1] = np.ones(widths[-1], bool)
    for i in range(len(masks)-1, -1, -1):
        backward[i] = masks[i][backward[i+1]].any(axis=0)
    forward = [np.ones(widths[0], bool)]
    for m, b in zip(masks, biases):
        forward.append(m[:, forward[-1]].any(axis=1) | (b != 0))
    node_rows = []
    for i, n in enumerate(widths):
        indeg = masks[i-1].sum(axis=1) if i else np.zeros(n, int)
        outdeg = masks[i].sum(axis=0) if i < len(masks) else np.zeros(n, int)
        node_rows.append({'layer': i, 'physical_nodes': n, 'incoming_degree': dist(indeg) if i else None,
            'outgoing_degree': dist(outdeg) if i < len(masks) else None,
            'backward_reachable_nodes': int(backward[i].sum()),
            'potential_forward_functional_nodes': int(forward[i].sum()),
            'potential_forward_and_backward_nodes': int((forward[i] & backward[i]).sum()),
            'nonzero_in_or_bias_and_direct_out_nodes': int(((indeg > 0) | (biases[i-1] != 0)) .__and__(outdeg > 0).sum()) if 0 < i < len(masks) else None})
    edge_rows = [{'layer': i+1, 'nonzero_edges': int(m.sum()), 'to_output_unreachable_edges': int(m[~backward[i+1]].sum()),
                  'from_potentially_nonfunctional_edges': int(m[:, ~forward[i]].sum()),
                  'on_potential_input_or_bias_to_output_paths': int(m[backward[i+1]][:,forward[i]].sum())} for i,m in enumerate(masks)]
    return {'widths': widths, 'nodes': node_rows, 'edges': edge_rows, 'nonzero_edges': sum(r['nonzero_edges'] for r in edge_rows)}

def locality(mask, weights=None):
    rows = np.flatnonzero(mask.any(axis=1))
    rad, area, fill, tight, centers, weighted = [], [], [], [], [], []
    for i in rows:
        points = COORD[mask[i]]
        center = points.mean(axis=0)
        sq = ((points-center)**2).sum(axis=1)
        box = int(np.prod(points.max(axis=0)-points.min(axis=0)+1))
        rad.append(np.sqrt(sq.mean())); area.append(box); fill.append(len(points)/box)
        tight.append((sq <= 9).mean()); centers.append(center)
        if weights is not None:
            a = np.abs(weights[i, mask[i]]).astype(float)
            weighted.append(a.sum()**2 / (a*a).sum())
    col = mask.sum(axis=0).astype(float)
    probs = col / col.sum()
    center_mask = ((COORD[:,0]>=7)&(COORD[:,0]<21)&(COORD[:,1]>=7)&(COORD[:,1]<21))
    result = {'nonempty_nodes': len(rows), 'rms_radius_pixels': dist(rad), 'bbox_area_pixels': dist(area),
              'bbox_density': dist(fill), 'fraction_edges_within_radius3_centroid': dist(tight),
              'pixel_degree': dist(col), 'pixel_degree_entropy_normalized': float(-(probs[probs>0]*np.log(probs[probs>0])).sum()/np.log(784)),
              'top10percent_pixel_edge_share': float(np.sort(col)[-79:].sum()/col.sum()),
              'center14x14_pixel_edge_share': float(col[center_mask].sum()/col.sum())}
    if weights is not None: result['absolute_weight_effective_support'] = dist(weighted)
    return result, {'rows': rows, 'radius': np.array(rad), 'area': np.array(area), 'centers': np.array(centers)}

def template(mask, ids):
    codes = np.zeros((len(ids),55*55), np.float32)
    for k,i in enumerate(ids):
        p = COORD[mask[i]]
        center = np.rint(p.mean(axis=0)).astype(int)
        z = p-center+27
        codes[k,z[:,0]*55+z[:,1]] = 1
    normalized = codes / np.sqrt(codes.sum(axis=1,keepdims=True))
    sim = normalized @ normalized.T
    pair = sim[np.triu_indices(len(ids),1)]
    np.fill_diagonal(sim,-1)
    nearest = sim.max(axis=1)
    packed = np.packbits(codes.astype(bool),axis=1)
    _, counts = np.unique(packed,axis=0,return_counts=True)
    return {'nodes': len(ids), 'mean_pair_cosine': float(pair.mean()), 'median_nearest_cosine': float(np.median(nearest)),
            'fraction_nodes_in_exact_duplicate_centered_templates': float(counts[counts>1].sum()/len(ids))}

def weighted_random(mask, rng):
    p = mask.sum(axis=0).astype(float)
    p /= p.sum()
    out = np.zeros_like(mask)
    for i,d in enumerate(mask.sum(axis=1)):
        if d: out[i,rng.choice(mask.shape[1],int(d),replace=False,p=p)] = True
    return out

def swaps(mask, rng, attempts):
    out = mask.copy()
    rows, cols = np.nonzero(out)
    first = rng.integers(len(rows),size=attempts)
    second = rng.integers(len(rows),size=attempts)
    accepted = 0
    for a,b in zip(first,second):
        ra,rb,ca,cb = rows[a],rows[b],cols[a],cols[b]
        if ra == rb or ca == cb or out[ra,cb] or out[rb,ca]: continue
        out[ra,ca] = out[rb,cb] = False
        out[ra,cb] = out[rb,ca] = True
        cols[a],cols[b] = cb,ca
        accepted += 1
    if not np.array_equal(out.sum(axis=1),mask.sum(axis=1)) or not np.array_equal(out.sum(axis=0),mask.sum(axis=0)):
        raise AssertionError('交换度数未保存')
    return out, accepted

def cnn_graph():
    import torch
    torch.set_num_threads(2)
    path = ROOT/'experiments/runs/capacity_baseline_v1/mnist_static/cnn/best.pt'
    saved = torch.load(path,map_location='cpu',weights_only=False)
    state = saved['state']
    matrices, biases, shared = [], [], []
    for name, channels, size in [('conv1',1,28),('conv2',16,14)]:
        w = state[name+'.weight'].numpy()
        b = state[name+'.bias'].numpy()
        outsize = size//2
        expanded = np.zeros((w.shape[0]*outsize*outsize,channels*size*size),np.float32)
        for oc in range(w.shape[0]):
            for oy in range(outsize):
                for ox in range(outsize):
                    row = (oc*outsize+oy)*outsize+ox
                    for ic in range(channels):
                        for ky in range(3):
                            for kx in range(3):
                                iy,ix = oy*2-1+ky,ox*2-1+kx
                                if 0<=iy<size and 0<=ix<size:
                                    expanded[row,(ic*size+iy)*size+ix] = w[oc,ic,ky,kx]
        matrices.append(expanded); biases.append(np.repeat(b,outsize*outsize))
        shared.append({'layer':name,'independent_kernel_weights':int(w.size),'nonzero_kernel_weights':int((w!=0).sum()),'expanded_nonzero_edges':int((expanded!=0).sum())})
    for name in ['fc1','fc2']:
        matrices.append(state[name+'.weight'].numpy().copy()); biases.append(state[name+'.bias'].numpy().copy())
    return matrices,biases,{'path':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'shared_convolution':shared,
                            'independent_weights':sum(int(v.numel()) for k,v in state.items() if k.endswith('.weight')),
                            'independent_biases':sum(int(v.numel()) for k,v in state.items() if k.endswith('.bias'))}

def draw_receptive_fields(base,random_first,cnn_mask,chosen):
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']
    plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(3,3,figsize=(10,10))
    # 卷积输出(7,7)对应原输入质心(14,14)，展示真实内部窗口。
    cnn_internal=cnn_mask[7*14+7].reshape(28,28)
    for row,node in enumerate(chosen):
        for col,arr,title in [(0,base[node].reshape(28,28),f'FC半径{[10,50,90][row]}%：节点{node}，度{base[node].sum()}'),(1,random_first[node].reshape(28,28),f'随机同节点，度{random_first[node].sum()}'),(2,cnn_internal,'CNN内部3×3，中心(14,14)')]:
            axes[row,col].imshow(arr,cmap='Greys',vmin=0,vmax=1);axes[row,col].set_title(title);axes[row,col].set_xticks([0,13,27]);axes[row,col].set_yticks([0,13,27])
    fig.suptitle('直接输入邻接；黑色为保留连接；固定半径分位示例',fontsize=14);fig.tight_layout();fig.savefig(str(OUT)+'_receptive_fields.png',dpi=160);plt.close(fig)

def main():
    started = time.perf_counter()
    result = {'registration':'docs/h1_connectivity_analysis_v1.md','random_seed':20261004,'model_forward_calls':0,'training_updates':0,'data_access':False,'models':{},'random_controls':[]}
    masks = {}
    weights5 = None
    for name,path in [('FC密集',RUN/'baseline/model.bgm'),('FC阶段4',RUN/'magnitude/stage_4.bgm'),('FC阶段5',RUN/'magnitude/model.bgm')]:
        w,b,source = decode(path)
        report = graph(w,b)
        report['source'] = source
        report['first_layer_locality'], aux = locality(w[0]!=0,w[0])
        report['strict_shared_weight_binding'] = False
        result['models'][name] = report
        masks[name] = w[0]!=0
        if name == 'FC阶段5': weights5, final_mask, aux5 = w[0], w[-1]!=0, aux
    w,b,source = cnn_graph()
    report = graph(w,b); report['source'] = source
    report['first_layer_locality'], cnn_aux = locality(w[0]!=0,w[0])
    report['strict_shared_weight_binding'] = True
    result['models']['CNN'] = report
    masks['CNN'] = w[0]!=0
    base = masks['FC阶段5']
    active = np.flatnonzero(base.any(axis=1))
    ordered = active[np.lexsort((active,base[active].sum(axis=1)))]
    ids = ordered[np.rint(np.linspace(0,len(ordered)-1,256)).astype(int)]
    result['template_node_ids'] = ids.tolist()
    result['template_sample_degree1_nodes'] = int((base[ids].sum(axis=1)==1).sum())
    result['observed_template'] = template(base,ids)
    result['cnn_template'] = template(masks['CNN'],np.arange(256))
    rng = np.random.default_rng(20261004)
    random_first = None
    chain = base.copy()
    for family in ['行度与像素偏好近似','行列度精确交换']:
        for i in range(4):
            if time.perf_counter()-started > 300:
                raise TimeoutError('达到登记CPU预算')
            if family == '行列度精确交换':
                chain, accepted = swaps(chain,rng,int(base.sum())*10)
                control, attempts = chain, int(base.sum())*10
            else:
                control,accepted,attempts = weighted_random(base,rng),None,None
            if random_first is None: random_first=control.copy()
            loc,_ = locality(control)
            result['random_controls'].append({'family':family,'replicate':i,'locality':loc,'template':template(control,ids),
                'swap_attempts':attempts,'swap_accepts':accepted,'dependent_chain':family=='行列度精确交换',
                'maximum_column_degree_difference':int(np.abs(control.sum(axis=0)-base.sum(axis=0)).max())})
    inter = final_mask.astype(int) @ final_mask.astype(int).T
    union = final_mask.sum(axis=1)[:,None]+final_mask.sum(axis=1)[None,:]-inter
    jaccard=inter/union
    result['output'] = {'class_input_degree':final_mask.sum(axis=1).tolist(),'hidden_output_degree':dist(final_mask.sum(axis=0)),
                       'class_jaccard':jaccard.tolist(),'last_hidden_nodes_used_by_outputs':int(final_mask.any(axis=0).sum()),
                       'nodes_used_by_multiple_classes':int((final_mask.sum(axis=0)>1).sum())}
    # 中文字体来自Windows既有字体，不安装环境依赖。
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']
    plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(2,3,figsize=(15,8.5))
    names=list(result['models'])
    for name in names:
        axes[0,0].plot(range(1,5),[r['nonzero_edges'] for r in result['models'][name]['edges']],marker='o',label=name)
    axes[0,0].set_yscale('log'); axes[0,0].set_xticks(range(1,5));axes[0,0].set_title('逐层展开非零边（对数轴）');axes[0,0].set_xlabel('连接层');axes[0,0].legend()
    for name in ['FC阶段5','CNN']:
        _,aux=locality(masks[name]);r=np.sort(aux['radius']);axes[0,1].plot(r,np.arange(1,len(r)+1)/len(r),label=name)
    _,aux=locality(random_first);r=np.sort(aux['radius']);axes[0,1].plot(r,np.arange(1,len(r)+1)/len(r),label='行度/像素偏好随机')
    axes[0,1].set_title('首层直接感受野RMS半径');axes[0,1].set_xlabel('像素');axes[0,1].set_ylabel('累计比例');axes[0,1].legend()
    axes[0,2].imshow(base.sum(axis=0).reshape(28,28),cmap='viridis');axes[0,2].set_title('FC阶段5：每像素连接数');axes[0,2].set_xlabel('输入像素列');axes[0,2].set_ylabel('输入像素行')
    backward=[result['models']['FC阶段5']['nodes'][i]['backward_reachable_nodes'] for i in range(1,5)]
    axes[1,0].bar(np.arange(4)-.2,[3136,1568,128,10],.4,label='物理节点');axes[1,0].bar(np.arange(4)+.2,backward,.4,label='输出反向可达');axes[1,0].set_xticks(np.arange(4),['隐藏1','隐藏2','隐藏3','输出']);axes[1,0].set_title('固定节点与输出通路');axes[1,0].legend()
    labels=['观测','加权随机','度精确随机'];tvalues=[result['observed_template']['median_nearest_cosine']]+[np.mean([r['template']['median_nearest_cosine'] for r in result['random_controls'] if r['family']==fam]) for fam in ['行度与像素偏好近似','行列度精确交换']]
    axes[1,1].bar(labels,tvalues);axes[1,1].set_ylim(0,1);axes[1,1].set_title('固定256节点：最近质心对齐模板cosine')
    axes[1,2].imshow(jaccard,vmin=0,vmax=1,cmap='Blues');axes[1,2].set_xticks(range(10));axes[1,2].set_yticks(range(10));axes[1,2].set_title('输出类别邻接Jaccard');axes[1,2].set_xlabel('数字类别');axes[1,2].set_ylabel('数字类别')
    fig.tight_layout();fig.savefig(str(OUT)+'_structure.png',dpi=160);plt.close(fig)
    order=np.lexsort((aux5['rows'],aux5['radius']))
    chosen=[int(aux5['rows'][order[int(round(q*(len(order)-1)))]] ) for q in [.1,.5,.9]]
    result['illustrated_node_ids']=chosen
    draw_receptive_fields(base,random_first,masks['CNN'],chosen)
    result['cpu_wall_seconds']=time.perf_counter()-started
    result['execution']={'device':'CPU','threads':2,'precision':'读取FP32权重，统计numpy float64（模板cosine为FP32）','model_instantiated':False,'data_access':False,'gpu_access':False}
    result['analysis_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    OUT.with_suffix('.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    write_report(result)
    print(json.dumps({'output':str(OUT),'wall_seconds':result['cpu_wall_seconds'],'observed_locality':result['models']['FC阶段5']['first_layer_locality'],'template':result['observed_template']},ensure_ascii=False))

def write_report(result):
    lines=['# FC剪枝与CNN连接结构离线比较','', '当前剪后FC较匹配行列度随机图有适度局部和二值模板偏向，但首层直接连接仍覆盖广、没有跨位置严格参数绑定；目前不能称发现卷积。', '', '2026-10-08。本报告只读取保存模型，不训练、不实例化模型、不前向、不读取任何数据集。分析前已冻结[指标及对照](../../../../history/ann_stages_20261008/docs/h1_connectivity_analysis_v1.md)。', '',
    '## 连接规模与节点', '', '|模型|第一层|第二层|第三层|输出层|合计|', '|---|---:|---:|---:|---:|---:|']
    for name,m in result['models'].items(): lines.append('|'+name+'|'+'|'.join(str(r['nonzero_edges']) for r in m['edges'])+'|'+str(m['nonzero_edges'])+'|')
    lines += ['', 'FC第一、第二隐藏层没有预先位置坐标；各层节点数与CNN相同是人为容量先验，不代表节点能一一对应CNN的通道/位置。FC首层95602边仍是CNN首层26896边的3.55倍；总边更少主要来自第三层和末层的强稀疏化，不能只据总边数称连接方式接近卷积。','', '|FC阶段5节点层|物理节点|入度均值/中位数|出度均值/中位数|输出反向可达|潜在功能且反向可达|', '|---|---:|---|---|---:|---:|']
    m=result['models']['FC阶段5']
    for r in m['nodes']:
        def form(d): return '—' if d is None else f"{d['mean']:.3f}/{d['quantiles_0_10_25_50_75_90_100'][3]:.1f}"
        lines.append(f"|{r['layer']}|{r['physical_nodes']}|{form(r['incoming_degree'])}|{form(r['outgoing_degree'])}|{r['backward_reachable_nodes']}|{r['potential_forward_and_backward_nodes']}|")
    waste=[r['to_output_unreachable_edges'] for r in m['edges']]
    lines += ['', f'指向输出反向不可达节点的非零边逐层为{waste}，合计{sum(waste)}（{sum(waste)/m["nonzero_edges"]:.2%}）。这里只给图统计，全部4842个非输入物理节点保持。潜在功能从输入符号或非零bias前向传播，不等于验证样本实际激活；反向可达也不保证节点对分类有必要。', '', '## 首层直接像素感受野', '', '|图|RMS均值/中位数（像素）|包围盒中位面积|包围盒密度中位数|质心半径3内边比例均值|中心14×14边占比|', '|---|---:|---:|---:|---:|---:|']
    for name,m in result['models'].items():
        l=m['first_layer_locality'];lines.append(f"|{name}|{l['rms_radius_pixels']['mean']:.3f}/{l['rms_radius_pixels']['quantiles_0_10_25_50_75_90_100'][3]:.3f}|{l['bbox_area_pixels']['quantiles_0_10_25_50_75_90_100'][3]:.0f}|{l['bbox_density']['quantiles_0_10_25_50_75_90_100'][3]:.3f}|{l['fraction_edges_within_radius3_centroid']['mean']:.3f}|{l['center14x14_pixel_edge_share']:.3f}|")
    for fam in ['行度与像素偏好近似','行列度精确交换']:
        controls=[r for r in result['random_controls'] if r['family']==fam]
        for key in ['rms_radius_pixels','bbox_area_pixels','fraction_edges_within_radius3_centroid']:
            nums=[r['locality'][key]['mean'] for r in controls]
            lines.append(f'\n{fam}4图的{key}均值范围：{min(nums):.4f}–{max(nums):.4f}。')
    l=result['models']['FC阶段5']['first_layer_locality']
    lines += ['', f"FC阶段5全局像素列度归一化熵{l['pixel_degree_entropy_normalized']:.4f}，连接最多的79像素占{l['top10percent_pixel_edge_share']:.2%}。中心偏好来自已训练模型的连接统计；本分析没有读取像素内容。仅比均匀随机更集中可能是全局中心偏好，故主要对照保留全局列度。", '',
    'CNN首层内部核最多9个输入，padding边界4或6个；核内局部邻接是人工架构。感受野图固定展示中心(14,14)附近一个真实3×3内部窗口；相同二值窗口遍及各空间位置，步长2移动，并在边界裁掉padding。FC的包围盒、RMS均在真实28×28输入位置计算，首层直接感受野与经多层路径形成的几何感受野不同。本轮不将FC第二层序号重塑成空间，也不报告第二层像素距离。', '', '## 有限重复模板诊断', '', '|图|平均两两cosine|最近模板cosine中位数|对齐完全重复节点比例|', '|---|---:|---:|---:|']
    for name,t in [('观测FC',result['observed_template']),('CNN前256节点（含边界，仅描述）',result['cnn_template'])]: lines.append(f"|{name}|{t['mean_pair_cosine']:.4f}|{t['median_nearest_cosine']:.4f}|{t['fraction_nodes_in_exact_duplicate_centered_templates']:.4f}|")
    for fam in ['行度与像素偏好近似','行列度精确交换']:
        rows=[r for r in result['random_controls'] if r['family']==fam]
        vals={k:[r['template'][k] for r in rows] for k in ['mean_pair_cosine','median_nearest_cosine','fraction_nodes_in_exact_duplicate_centered_templates']}
        lines.append('|'+fam+'4图范围|'+'|'.join(f'{min(v):.4f}–{max(v):.4f}' for v in vals.values())+'|')
    exact=[r for r in result['random_controls'] if r['swap_attempts']]
    lines += ['', '256节点按入度固定等分位抽样，不按模板相似度挑节点；随机图使用同节点/同入度。对齐只平移四舍五入质心，没有转动、缩放、非线性匹配。相似度只看二值邻接，不能证明功能相同、带符号权重核相同或卷积计算复用。CNN内部节点的二值模板全是3×3，即使不同通道的权重核数值不同。', '',
    f"固定样本含{result.get('template_sample_degree1_nodes',3)}个仅1输入节点，它们质心对齐后必然全等；观测完全重复节点比例3/256=1.171875%恰由这些退化单点模板构成，没有观察到额外非单点完全重复模板。因此这里的NN相似度差是有限部分重叠信号，完全重复比例不是卷积模板发现证据。", '',
    f"精确行列度交换每图{exact[0]['swap_attempts']}次尝试，接受率分别为{[round(r['swap_accepts']/r['swap_attempts'],4) for r in exact]}，所有行列度逐项一致。4图是同一交换链的连续检查点，未证明充分混合，结果为描述性范围，不作p值或独立重复结论。加权随机只近似保持列度，其最大列度差见JSON。", '',
    '## 共享与输出连接', '',
    'FC保存格式sharing=none，每条存活权重独立训练；没有跨节点/位置的参数绑定。CNN conv1的144个、conv2的4608个核权重在多个位置严格复用，分别展开26896/204800条非零边。首层局部性、邻接模板相似或浮点值碰巧相等，均不构成严格共享证据。FC仍dense零mask执行，CNN核复用由Conv2d执行；本轮没有测推断时延、内存或能耗。', '',
    f"十输出类别入度为{result['output']['class_input_degree']}，共同使用{result['output']['last_hidden_nodes_used_by_outputs']}个末隐藏节点，其中{result['output']['nodes_used_by_multiple_classes']}个连向多个类别。完整类别Jaccard和逐层分位数保存在JSON。仅拓扑重叠不代表类别特征相同。", '',
    '## 结论边界与证据', '',
    '剪后FC的首层相对精确保行列度随机图，平均半径约短9.4%，质心半径3内连接均值11.16%对随机5.40%–5.65%；这支持当前图有一定局部组织偏向，且不能完全由全局像素偏好解释。但中位包围盒594/784像素、RMS9.50像素仍呈广域稀疏连接，与CNN固定紧密3×3不同。stage4→stage5半径和包围盒继续降低是两个保存阶段的描述，未据它选择新剪边强度。', '',
    '固定模板最近相似度0.259对随机0.182，提示部分重复邻接偏向；其绝对相似度较低，没有非单点完全重复模板证据，更没有跨位置严格绑定权重。当前更合适的描述是“保持任务效果的广域稀疏FC，带有一定局部/部分模式偏向”，不称已经恢复卷积。', '',
    '已完成真实稀疏性、图可达性、首层直接局部性与有限二值模板诊断。结构差别必须分别看：局部邻接、重复模板、严格参数绑定、计算复用；本轮未评价平移等变、深层功能、跨种子/任务稳健性。验证97.125%/历史探索性测试97.3%来自既有训练报告，未重评；历史探索性测试不能称独立最终测试。', '',
    f"实际CPU分析墙钟{result['cpu_wall_seconds']:.2f}秒；读取FP32权重，numpy统计为float64、模板cosine为float32，线程2、0模型前向/反传/更新、0数据读取、0GPU使用。该时间是分析成本，不是推断速度。分析源码与源模型SHA保存在JSON。", '',
    '[结构总图](h1_connectivity_analysis_v1_20261008_structure.png) · [固定分位感受野](h1_connectivity_analysis_v1_20261008_receptive_fields.png) · [完整JSON](h1_connectivity_analysis_v1_20261008.json) · [分析源码](../../../../tools/analyze_connectivity.py)', '',
    '![结构统计](h1_connectivity_analysis_v1_20261008_structure.png)', '', '![固定半径分位感受野](h1_connectivity_analysis_v1_20261008_receptive_fields.png)', '']
    OUT.with_suffix('.md').write_text('\n'.join(lines),encoding='utf-8')

if __name__ == '__main__':
    import sys
    if '--report-only' in sys.argv:
        write_report(json.loads(OUT.with_suffix('.json').read_text(encoding='utf-8')))
    elif '--receptive-fields-only' in sys.argv:
        saved=json.loads(OUT.with_suffix('.json').read_text(encoding='utf-8'))
        w,_,_=decode(RUN/'magnitude/model.bgm')
        base=w[0]!=0
        first=weighted_random(base,np.random.default_rng(20261004))
        cw,_,_=cnn_graph()
        draw_receptive_fields(base,first,cw[0]!=0,saved['illustrated_node_ids'])
    else:
        main()
