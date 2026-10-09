"""冻结BGM前两层的只读结构分析：无模型前向、无数据访问。"""
from __future__ import annotations

import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tools.analyze_connectivity import decode, graph

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/reports/2026-10-08/h1_front_layers_patterns_v1'
SEED = 20261004
SOURCES = [
    ('静态幅度', 'experiments/runs/h1_aligned_edges_v1/mnist_static/magnitude/model.bgm', '6e69ffb1e1860eaf2c9d146501e3a0bbf215c958d1ccd546d9cbbed4c943b857', False),
    ('循环幅度', 'history/ann_stages_20261008/experiments/runs/h1_translated_edges_v1/mnist_circular/magnitude/model.bgm', '74202b856d95f29ed31128b159c33362ed5f875dfedd400d0b9166a6488f0180', True),
    ('循环节点贡献', 'history/ann_stages_20261008/experiments/runs/h1_translated_edges_v1/mnist_circular/node/model.bgm', 'bc800fb08a2096d8d57f89eaae3eb89231d711db5ac7f6bc4cc24274e9dc610b', True),
]
Q = [0, .1, .25, .5, .75, .9, 1]


def stats(a):
    a = np.asarray(a, dtype=float)
    if not a.size:
        return {'count': 0, 'mean': None, 'quantiles': [], 'cv': None, 'zero': 0}
    total = a.sum()
    return {'count': int(a.size), 'mean': float(a.mean()), 'quantiles': np.quantile(a, Q).tolist(),
            'cv': float(a.std() / a.mean()) if a.mean() else None, 'zero': int((a == 0).sum()),
            'top10percent_share': float(np.sort(a)[-max(1, int(np.ceil(a.size * .1))):].sum() / total) if total else None}


def budget(started):
    # Windows工作集；运行无需psutil依赖。
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [(n, ctypes.c_size_t) for n in ['PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage']]
    c = Counters(); c.cb = ctypes.sizeof(c)
    get_info = ctypes.windll.psapi.GetProcessMemoryInfo
    get_info.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
    if not get_info(wintypes.HANDLE(-1), ctypes.byref(c), c.cb):
        raise OSError('无法读取进程工作集')
    if time.perf_counter() - started > 600 or c.WorkingSetSize > 2 * 1024**3:
        raise RuntimeError('达到预登记墙钟或内存停止信号')
    return int(c.PeakWorkingSetSize)


def curveball(a, rng, trades=None):
    out = a.copy()
    changed = 0
    for _ in range(trades or 8 * len(a)):
        i, j = rng.choice(len(a), 2, replace=False)
        different = np.flatnonzero(out[i] ^ out[j])
        if not different.size:
            continue
        count_i = int(out[i, different].sum())
        rng.shuffle(different)
        out[i, different] = False; out[j, different] = False
        out[i, different[:count_i]] = True; out[j, different[count_i:]] = True
        changed += 1
    if not np.array_equal(out.sum(0), a.sum(0)) or not np.array_equal(out.sum(1), a.sum(1)):
        raise AssertionError('curveball未保存行列度')
    return out, {'trades': trades or 8 * len(a), 'nontrivial_trades': changed,
                 'changed_edge_fraction': float((a & ~out).sum() / a.sum()), 'row_degree_max_error': 0, 'column_degree_max_error': 0}


def remap_weights(w, a, rng):
    z = np.zeros_like(w)
    for i in range(len(w)):
        values = w[i, w[i] != 0].copy()
        rng.shuffle(values)
        z[i, a[i]] = values
    return z


def axis_radius(mass, circular):
    total = mass.sum(1)
    p = np.arange(28, dtype=float)
    base = mass @ p
    if not circular:
        centers = base / total
        return centers, ((p[None, :] - centers[:, None])**2 * mass).sum(1) / total
    cumulative = np.cumsum(mass, 1)
    best = np.full(len(mass), np.inf); centers = np.zeros(len(mass))
    for cut in range(28):
        c = (base + 28 * (cumulative[:, cut - 1] if cut else 0)) / total % 28
        d = (p[None, :] - c[:, None] + 14) % 28 - 14
        sq = (d * d * mass).sum(1) / total
        improve = sq < best
        best[improve] = sq[improve]; centers[improve] = c[improve]
    return centers, best


def locality(a, w, circular):
    ids = np.flatnonzero(a.any(1))
    fields = a[ids].reshape(-1, 28, 28).astype(float)
    energy = w[ids].reshape(-1, 28, 28).astype(float)**2
    result, aux = {}, {'ids': ids}
    for kind, z in [('binary', fields), ('energy', energy)]:
        cy, sy = axis_radius(z.sum(2), circular)
        cx, sx = axis_radius(z.sum(1), circular)
        radius = np.sqrt(sx + sy)
        local = ndimage.uniform_filter(z, size=(1, 3, 3), mode='wrap' if circular else 'constant') * 9
        concentration = local.max((1, 2)) / z.sum((1, 2))
        result[kind + '_rms_radius'] = stats(radius)
        result[kind + '_best3x3_fraction'] = stats(concentration)
        aux[kind + '_centers'] = np.column_stack([cy, cx])
        aux[kind + '_radius'] = radius
    result['geometry'] = '环面连续中心' if circular else '平面连续中心'
    signed = w[ids].reshape(-1, 28, 28).astype(float)
    signed -= signed.mean((1, 2), keepdims=True)
    power = np.abs(np.fft.fft2(signed, axes=(1, 2)))**2
    freq = np.fft.fftfreq(28) * 28
    low = (np.abs(freq[:, None]) <= 3) & (np.abs(freq[None, :]) <= 3)
    low[0, 0] = False
    power[:, 0, 0] = 0
    result['signed_weight_non_dc_low_frequency_fraction'] = stats(power[:, low].sum(1) / power.sum((1, 2)))
    return result, aux


def select_ids(a):
    degrees = a.sum(1)
    ids = np.flatnonzero((degrees >= 9) & (degrees <= a.shape[1] - 9))
    ids = ids[np.lexsort((ids, degrees[ids]))]
    return ids[np.rint(np.linspace(0, len(ids) - 1, min(128, len(ids)))).astype(int)] if len(ids) else ids


def pair_stats(z, subtract_mean=False):
    z = z.astype(float)
    if subtract_mean:
        z -= z.mean(1, keepdims=True)
    norm = np.linalg.norm(z, axis=1, keepdims=True)
    z = np.divide(z, norm, out=np.zeros_like(z), where=norm > 0)
    sim = z @ z.T
    pair = sim[np.triu_indices(len(sim), 1)]
    np.fill_diagonal(sim, -np.inf)
    return {'mean_pair_cosine': float(pair.mean()), 'median_nearest_cosine': float(np.median(sim.max(1)))}


def templates(a, w, ids, aux=None, circular=False):
    binary = a[ids].astype(float); signed = w[ids].astype(float)
    if aux is not None:
        lookup = {int(i): k for k, i in enumerate(aux['ids'])}
        def center(z, kind):
            fields = np.zeros((len(ids), 28 if circular else 55, 28 if circular else 55))
            for k, i in enumerate(ids):
                cy, cx = np.rint(aux[kind + '_centers'][lookup[int(i)]]).astype(int)
                image = z[k].reshape(28, 28)
                if circular:
                    fields[k] = np.roll(image, (14 - cy, 14 - cx), axis=(0, 1))
                else:
                    y, x = np.nonzero(image)
                    fields[k, y - cy + 27, x - cx + 27] = image[y, x]
            return fields.reshape(len(ids), -1)
        binary = center(binary, 'binary'); signed = center(signed, 'energy')
    packed = np.packbits(binary.astype(bool), axis=1)
    _, count = np.unique(packed, axis=0, return_counts=True)
    return {'nodes': len(ids), 'node_ids': ids.tolist(), 'minimum_degree': int(a[ids].sum(1).min()),
            'binary_centered_residual_cosine': pair_stats(binary, True), 'signed_weight_cosine': pair_stats(signed),
            'exact_binary_duplicate_node_fraction': float(count[count > 1].sum() / len(ids)),
            'spatially_aligned': aux is not None}


def kmeans(x, clusters=8):
    x = np.divide(x, np.linalg.norm(x, axis=1, keepdims=True), out=np.zeros_like(x), where=np.linalg.norm(x, axis=1, keepdims=True) > 0)
    rng = np.random.default_rng(SEED)
    chosen = [int(rng.integers(len(x)))]
    distances = np.full(len(x), np.inf)
    for _ in range(1, clusters):
        distances = np.minimum(distances, ((x - x[chosen[-1]])**2).sum(1))
        chosen.append(int(np.argmax(distances)))
    centers = x[chosen].copy()
    labels = np.zeros(len(x), int)
    for _ in range(20):
        labels = ((x[:, None, :] - centers[None, :, :])**2).sum(2).argmin(1)
        for i in range(clusters):
            if (labels == i).any(): centers[i] = x[labels == i].mean(0)
    return labels


def organization(a):
    dr = a.sum(1).astype(float); dc = a.sum(0).astype(float); edges = float(dr.sum())
    r = np.flatnonzero(dr); c = np.flatnonzero(dc)
    ar = a[np.ix_(r, c)].astype(float)
    row = dr[r]; col = dc[c]
    b = ar / np.sqrt(row[:, None] * col[None, :]) - np.sqrt(row[:, None] * col[None, :]) / edges
    frob2 = float(np.sum(b * b))
    rng = np.random.default_rng(SEED)
    q, _ = np.linalg.qr(b @ rng.standard_normal((len(c), 16)), mode='reduced')
    for _ in range(2):
        z, _ = np.linalg.qr(b.T @ q, mode='reduced')
        q, _ = np.linalg.qr(b @ z, mode='reduced')
    small_u, singular, vt = np.linalg.svd(q.T @ b, full_matrices=False)
    u = q @ small_u[:, :8]; v = vt[:8].T
    rl = kmeans(u); cl = kmeans(v)
    block = np.zeros((8, 8))
    for i in range(8):
        sums = ar[rl == i].sum(0)
        block[i] = np.bincount(cl, weights=sums, minlength=8)
    rd = np.bincount(rl, weights=row, minlength=8); cd = np.bincount(cl, weights=col, minlength=8)
    expected = rd[:, None] * cd[None, :] / edges
    ratio = np.divide(block, expected, out=np.ones_like(block), where=expected > 0)
    chi = np.divide((block - expected)**2, expected, out=np.zeros_like(block), where=expected > 0)
    return {'singular_values_first8': singular[:8].tolist(), 'residual_frobenius_squared': frob2,
            'first_residual_energy_fraction': float(singular[0]**2 / frob2),
            'top8_residual_energy_fraction': float((singular[:8]**2).sum() / frob2),
            'block_chi_squared_per_edge': float(chi.sum() / edges), 'block_observed_edges': block.tolist(),
            'block_expected_edges': expected.tolist(), 'block_degree_corrected_enrichment': ratio.tolist(),
            'block_row_nodes': np.bincount(rl, minlength=8).tolist(), 'block_column_nodes': np.bincount(cl, minlength=8).tolist(),
            'large_block_max_enrichment': float(ratio[expected >= .005 * edges].max()),
            'active_row_nodes': len(r), 'active_column_nodes': len(c)}


def layer_analysis(a, w, ids, circular, first):
    degree = a.sum(1); energy = (w.astype(float)**2).sum(1); fourth = (w.astype(float)**4).sum(1)
    effective = np.divide(energy**2, fourth, out=np.zeros_like(energy), where=fourth > 0)
    active = degree > 0
    report = {'edges': int(a.sum()), 'density': float(a.mean()), 'row_degree': stats(degree), 'column_degree': stats(a.sum(0)),
              'energy_effective_support': stats(effective[active]), 'effective_support_over_degree': stats(effective[active] / degree[active])}
    aux = None
    if first:
        report['locality'], aux = locality(a, w, circular)
    report['templates'] = templates(a, w, ids, aux, circular)
    report['organization'] = organization(a)
    return report, aux


def self_check():
    rng = np.random.default_rng(SEED)
    a = rng.random((20, 30)) < .3
    b, _ = curveball(a, rng)
    assert np.array_equal(a.sum(0), b.sum(0)) and np.array_equal(a.sum(1), b.sum(1))
    mass = np.zeros((1, 28)); mass[0, [0, 27]] = 1
    _, circular = axis_radius(mass, True); _, planar = axis_radius(mass, False)
    assert np.allclose(circular, .25) and np.allclose(planar, 182.25)
    block = np.kron(np.eye(8, dtype=bool), np.ones((3, 4), bool))
    org = organization(block)
    assert np.allclose(org['singular_values_first8'][:7], 1)
    assert org['block_chi_squared_per_edge'] > 5.9
    print('最小检查通过：curveball行列度、环面翻边、平面距离、已知8块图谱。')


def make_figures(result, examples):
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    names = list(result['models'])
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for j, name in enumerate(names):
        m = result['models'][name]
        for layer in range(2):
            actual = m['layers'][layer]; controls = m['controls'][layer]
            points = [actual['organization']['singular_values_first8']] + [z['analysis']['organization']['singular_values_first8'] for z in controls]
            axes[layer, j].plot(range(1, 9), points[0], 'o-', label='观测')
            for z in points[1:]: axes[layer, j].plot(range(1, 9), z, '--', color='gray', alpha=.7)
            axes[layer, j].set_title(f'{name}：第{layer + 1}层度校正谱')
            axes[layer, j].set_xlabel('奇异值序号'); axes[layer, j].set_ylabel('奇异值'); axes[layer, j].legend()
    fig.tight_layout(); fig.savefig(OUT / 'degree_corrected_spectrum.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for j, name in enumerate(names):
        arr = np.array(result['models'][name]['layers'][1]['organization']['block_degree_corrected_enrichment'])
        im = axes[j].imshow(arr, vmin=0, vmax=2, cmap='RdBu_r'); axes[j].set_title(name + '：第二层群组富集')
        axes[j].set_xlabel('前层节点群组'); axes[j].set_ylabel('后层节点群组')
    fig.colorbar(im, ax=axes.ravel().tolist(), label='实际边/按度预期边；1为基线', shrink=.85)
    fig.savefig(OUT / 'second_layer_groups.png', dpi=160, bbox_inches='tight'); plt.close(fig)
    fig, axes = plt.subplots(3, 6, figsize=(17, 9))
    for row, (name, fields) in enumerate(examples.items()):
        for k, item in enumerate(fields):
            for offset, key in enumerate(['observed', 'null']):
                arr = item[key]
                vmax = item['color_limit']
                axes[row, 2*k+offset].imshow(arr, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
                axes[row, 2*k+offset].set_title(f'{name}\n半径{item["quantile"]}% 节点{item["id"]}\n' + ('观测' if offset == 0 else '同度同权重随机'), fontsize=9)
                axes[row, 2*k+offset].set_xticks([0, 27]); axes[row, 2*k+offset].set_yticks([0, 27])
    fig.suptitle('首层带符号权重：固定能量半径10/50/90分位；每对共用色标，显示原像素坐标')
    fig.tight_layout(); fig.savefig(OUT / 'first_layer_examples.png', dpi=150); plt.close(fig)


def write_report(result):
    lines = ['# 前两层是否仍有规律：冻结权重与连接的只读分析', '',
             '2026-10-08。按[预登记](../../../h1_front_layers_patterns_v1.md)读取三份冻结BGM；不读取数据、不训练、不前向。所有随机图精确保留每个节点的入出度，每行权重多重集保持。2个独立启发随机参照不提供显著性检验，群组结果为探索性描述。', '',
             '## 首层空间组织', '', '|模型|平均入度 / 中位数|二值RMS：观测 / null范围|能量RMS：观测 / null范围|最佳3×3能量：观测 / null范围|能量有效支撑中位数|去DC低频能量：观测 / null范围|', '|---|---:|---:|---:|---:|---:|---:|---:|']
    def rngtext(v): return f'{min(v):.4f}—{max(v):.4f}'
    for name, model in result['models'].items():
        obs = model['layers'][0]; ctrl = model['controls'][0]
        def vals(key): return [z['analysis']['locality'][key]['mean'] for z in ctrl]
        loc = obs['locality']
        lines.append(f'|{name}|{obs["row_degree"]["mean"]:.2f} / {obs["row_degree"]["quantiles"][3]:.0f}|{loc["binary_rms_radius"]["mean"]:.4f} / {rngtext(vals("binary_rms_radius"))}|{loc["energy_rms_radius"]["mean"]:.4f} / {rngtext(vals("energy_rms_radius"))}|{loc["energy_best3x3_fraction"]["mean"]:.4f} / {rngtext(vals("energy_best3x3_fraction"))}|{obs["energy_effective_support"]["quantiles"][3]:.2f}|{loc["signed_weight_non_dc_low_frequency_fraction"]["mean"]:.4f} / {rngtext(vals("signed_weight_non_dc_low_frequency_fraction"))}|')
    lines += ['', '静态使用平面距离，循环任务使用环面最短距离，跨模型的半径不可直接当作相同任务成绩。半径只对非空首层节点平均；最佳3×3为逐节点搜索所有窗口后的最大值，对随机图同样搜索。完整3×3均匀卷积窗口的二值RMS为1.155像素；宽广且较随机更集中的连接仍不能叫3×3卷积。W²衡量权重能量分布，不等于实际任务贡献。', '',
              '傅里叶指标对完整28×28带符号权重场（包括零边）逐节点减均值去DC，低频固定为|kx|≤3且|ky|≤3、排除DC，共48/783个频率。它描述广域平滑/纹理的空间相关；静态图的FFT也使用周期展开，因此只与同任务同处理null比较。该分析阈值是外部评价先验，没有送入优化器。', '',
              '## 重复与第二层群组', '', '|模型/层|二值最近残差cos：观测 / null|带符号最近cos：观测 / null|第一度校正奇异值：观测 / null|8簇块偏差χ²/E：观测 / null|', '|---|---:|---:|---:|---:|']
    for name, model in result['models'].items():
        for layer in range(2):
            obs = model['layers'][layer]; ctrl = [z['analysis'] for z in model['controls'][layer]]
            def val(x, kind):
                if kind in ['binary_centered_residual_cosine', 'signed_weight_cosine']: return x['templates'][kind]['median_nearest_cosine']
                if kind == 'sigma': return x['organization']['singular_values_first8'][0]
                return x['organization']['block_chi_squared_per_edge']
            items = [f'{val(obs, k):.4f} / {rngtext([val(z, k) for z in ctrl])}' for k in ['binary_centered_residual_cosine', 'signed_weight_cosine', 'sigma', 'chi']]
            lines.append(f'|{name}/{layer+1}|' + '|'.join(items) + '|')
    lines += ['', '首层模板分别按二值中心与能量中心平移对齐；二值先去掉行均值，排除高密度的共同背景。第二层保留原节点身份，不作空间对齐。抽样固定最多128个、入度至少9且至少9条空缺；最近cosine是同一个样本集合中找最相似邻居。观测与null均重新对齐及寻找最近邻，因此数值较高本身不证明重复模板。没有精确跨位置权重绑定，signed cosine也不等于参数共享。', '',
              '第二层8个群组来自度校正后的连接谱，群组编号任意，不代表空间位置、数字类别或生物功能。块偏差是在同一图上拟合和计量，因此只用同流程null排除拟合噪声；未证明唯一模块划分。度校正奇异值与群组富集仅证明连接的相关组织，不能说明其分类作用。', '',
              '## 容量分布及输出通路', '', '|模型/层|边数|入度10/50/90分位|入度CV|出度CV|入度最高10%边占比|无入边节点|无出边前层节点|指向输出不可达边|', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, model in result['models'].items():
        for i, obs in enumerate(model['layers']):
            d = obs['row_degree']; c = obs['column_degree']; q = d['quantiles']
            lines.append(f'|{name}/{i+1}|{obs["edges"]}|{q[1]:.0f}/{q[3]:.0f}/{q[5]:.0f}|{d["cv"]:.3f}|{c["cv"]:.3f}|{d["top10percent_share"]:.2%}|{d["zero"]}|{c["zero"]}|{model["graph"]["edges"][i]["to_output_unreachable_edges"]}|')
    lines += ['', '这些度差异说明容量分配不均；随机对照保留它们，所以不能把该项本身当作额外模板或群组。所有物理节点保留；反向可达仅表示存在路径，未使用激活验证必要性。完整逐层图、bias潜在前向通路及不可达节点统计在JSON。', '',
              '## 来源、成本与限制', '', f'正式CPU分析墙钟{result["wall_seconds"]:.3f}秒，2线程；峰值进程工作集{result["peak_working_set_bytes"] / 1024**2:.1f}MiB。读取FP32权重，统计/谱计算float64；网络batch不适用，执行为离线NumPy/SciPy图分析。0模型前向、0训练更新、0GPU、0数据或测试集访问；未测模型延迟、能耗或准确率。源模型及分析源SHA在[机器报告](analysis.json)，固定样本和图度校验也在其中。', '',
              '每条curveball链8×行数交易，从观测图独立开始，未证明充分混合；两个参照只能显示本轮差异的量级。模型只有一个既有种子，分析超参数和外部像素几何均在登记中披露。谱是固定随机低秩近似，不能作为精确谱定理。没有观察到所测偏差不能推出完全无规律；观察到偏差也不能推出卷积结构、语义模块、严格共享、因果重要性或跨任务能力。', '',
              '![两层度校正谱](degree_corrected_spectrum.png)', '', '![第二层群组富集](second_layer_groups.png)', '', '![固定分位首层权重示例](first_layer_examples.png)', '']
    (OUT / 'report.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--self-check', action='store_true'); args = parser.parse_args()
    if args.self_check:
        self_check(); return
    started = time.perf_counter(); OUT.mkdir(parents=True, exist_ok=True)
    result = {'registration': 'docs/h1_front_layers_patterns_v1.md', 'analysis_random_root': SEED,
              'analysis_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'decoder_source_sha256': hashlib.sha256((ROOT / 'tools/analyze_connectivity.py').read_bytes()).hexdigest(),
              'model_forward_calls': 0, 'training_updates': 0, 'data_access': False, 'gpu_access': False,
              'threads': 2, 'quantile_levels': Q, 'models': {}}
    examples = {}; peak = 0
    try:
        for mi, (name, relative, expected_sha, circular) in enumerate(SOURCES):
            weights, biases, source = decode(ROOT / relative)
            if source['sha256'] != expected_sha: raise ValueError('冻结BGM来源SHA不符')
            model = {'source': source, 'graph': graph(weights, biases), 'layers': [], 'controls': [], 'strict_weight_binding': False}
            result['models'][name] = model
            for layer in range(2):
                w = weights[layer]; a = w != 0; ids = select_ids(a)
                observed, aux = layer_analysis(a, w, ids, circular, layer == 0)
                model['layers'].append(observed); model['controls'].append([])
                chosen = []
                if layer == 0:
                    order = np.lexsort((aux['ids'], aux['energy_radius']))
                    for q in [.1, .5, .9]:
                        idx = int(aux['ids'][order[round(q * (len(order) - 1))]])
                        chosen.append({'id': idx, 'quantile': round(q * 100), 'observed': w[idx].reshape(28, 28).copy()})
                peak = max(peak, budget(started))
                for replicate in range(2):
                    rng = np.random.default_rng(np.random.SeedSequence([SEED, mi, layer, replicate]))
                    null_a, details = curveball(a, rng)
                    null_w = remap_weights(w, null_a, rng)
                    null_report, _ = layer_analysis(null_a, null_w, ids, circular, layer == 0)
                    details['observed_edge_overlap_fraction'] = float((a & null_a).sum() / a.sum())
                    if replicate == 0:
                        first_null = null_a.copy()
                    else:
                        model['null_pair_edge_overlap_fraction'] = model.get('null_pair_edge_overlap_fraction', [])
                        model['null_pair_edge_overlap_fraction'].append({'layer': layer + 1, 'fraction': float((first_null & null_a).sum() / a.sum())})
                        del first_null
                    model['controls'][layer].append({'replicate': replicate, 'random_stream': [SEED, mi, layer, replicate], 'trade_verification': details, 'analysis': null_report})
                    if layer == 0 and replicate == 0:
                        for item in chosen:
                            item['null'] = null_w[item['id']].reshape(28, 28).copy()
                            item['color_limit'] = float(max(np.abs(item['observed']).max(), np.abs(item['null']).max()))
                        examples[name] = chosen
                        model['illustrated_node_ids'] = [z['id'] for z in chosen]
                    del null_a, null_w
                    peak = max(peak, budget(started))
                print(f'{name} 第{layer+1}层完成；累计{time.perf_counter()-started:.1f}秒', flush=True)
            del weights, biases
        make_figures(result, examples)
        peak = max(peak, budget(started))
        result['wall_seconds'] = time.perf_counter() - started; result['peak_working_set_bytes'] = peak
        result['status'] = '已运行完成'
        (OUT / 'analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        write_report(result)
        print(json.dumps({'status': result['status'], 'wall_seconds': result['wall_seconds'], 'output': str(OUT)}, ensure_ascii=False))
    except Exception as exc:
        result['status'] = '停止或失败'; result['failure'] = repr(exc); result['wall_seconds'] = time.perf_counter() - started
        (OUT / 'failure.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
