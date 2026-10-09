"""输出反向可达子图的事后只读补充，原主分析实现不修改。"""
from __future__ import annotations
import hashlib
import json
import time
from pathlib import Path
from tools.analyze_front_layers import ROOT, OUT, SOURCES, SEED, budget, curveball, remap_weights, organization, stats
from tools.analyze_connectivity import decode
import numpy as np


def low_frequency(w):
    active = np.any(w != 0, axis=1)
    z = w[active].reshape(-1, 28, 28).astype(float)
    z -= z.mean((1, 2), keepdims=True)
    power = np.abs(np.fft.fft2(z, axes=(1, 2)))**2
    frequencies = np.fft.fftfreq(28) * 28
    low = (np.abs(frequencies[:, None]) <= 3) & (np.abs(frequencies[None, :]) <= 3)
    low[0, 0] = False; power[:, 0, 0] = 0
    return stats(power[:, low].sum(1) / power.sum((1, 2)))


def backward_reachable(weights):
    back = [None] * (len(weights) + 1)
    back[-1] = np.ones(weights[-1].shape[0], dtype=bool)
    for i in range(len(weights) - 1, -1, -1):
        back[i] = (weights[i][back[i + 1]] != 0).any(0)
    return back


def main():
    started = time.perf_counter()
    primary_path = OUT / 'analysis.json'
    primary = json.loads(primary_path.read_text(encoding='utf-8'))
    source_sha = hashlib.sha256((ROOT / 'tools/analyze_front_layers.py').read_bytes()).hexdigest()
    if primary['analysis_source_sha256'] != source_sha:
        raise ValueError('主分析实现SHA与已保存证据不符')
    result = {'status': '开始', 'post_hoc': True, 'reason': '主分析发现输出不可达支路，需要解释混杂复核',
              'primary_json_sha256': hashlib.sha256(primary_path.read_bytes()).hexdigest(),
              'primary_analysis_source_sha256': source_sha,
              'addendum_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'model_forward_calls': 0, 'training_updates': 0, 'data_access': False, 'gpu_access': False, 'threads': 2,
              'models': {}}
    peak = 0
    try:
        for mi, (name, relative, expected_sha, _) in enumerate(SOURCES[1:], 1):
            weights, _, source = decode(ROOT / relative)
            if source['sha256'] != expected_sha: raise ValueError('源BGM SHA不符')
            back = backward_reachable(weights)
            assert back[0].all(), '首层输入列需保持28×28完整784列'
            model = {'source': source, 'node_selection': [], 'edge_selection': [], 'layers': []}
            result['models'][name] = model
            for i, selected in enumerate(back):
                model['node_selection'].append({'node_layer': i, 'original_nodes': len(selected), 'retained_nodes': int(selected.sum()),
                                                'removed_nodes': int((~selected).sum()), 'retained_node_ids': np.flatnonzero(selected).tolist(),
                                                'removed_node_ids': np.flatnonzero(~selected).tolist()})
            for layer, w in enumerate(weights):
                original_edges = int((w != 0).sum())
                sub = w[np.ix_(back[layer + 1], back[layer])]
                kept = int((sub != 0).sum())
                model['edge_selection'].append({'layer': layer + 1, 'original_edges': original_edges, 'retained_edges': kept, 'removed_edges': original_edges - kept})
            for layer in range(2):
                w = weights[layer][np.ix_(back[layer + 1], back[layer])]
                a = w != 0
                layer_report = {'layer': layer + 1, 'shape': list(a.shape), 'edges': int(a.sum()),
                                'observed': low_frequency(w) if layer == 0 else organization(a), 'controls': []}
                model['layers'].append(layer_report)
                for replicate in range(2):
                    # 新事后流命名区分主分析，仍沿用分析根。
                    stream = [SEED, 100, mi, layer, replicate]
                    rng = np.random.default_rng(np.random.SeedSequence(stream))
                    null_a, verification = curveball(a, rng)
                    if layer == 0:
                        null_w = remap_weights(w, null_a, rng)
                        analysis = low_frequency(null_w)
                        del null_w
                    else:
                        analysis = organization(null_a)
                    verification['observed_edge_overlap_fraction'] = float((a & null_a).sum() / a.sum())
                    if replicate == 0: first_null = null_a.copy()
                    else:
                        layer_report['null_pair_edge_overlap_fraction'] = float((first_null & null_a).sum() / a.sum())
                        del first_null
                    layer_report['controls'].append({'replicate': replicate, 'stream': stream, 'trade_verification': verification, 'analysis': analysis})
                    peak = max(peak, budget(started))
                    if time.perf_counter() - started > 120 or time.perf_counter() - started + primary['wall_seconds'] > 600:
                        raise RuntimeError('达到补充120秒或合并600秒预算')
                print(f'{name} 可达子图第{layer+1}层完成，累计{time.perf_counter()-started:.1f}秒', flush=True)
            del weights
        result['status'] = '已运行完成'; result['wall_seconds'] = time.perf_counter() - started
        result['combined_primary_addendum_wall_seconds'] = result['wall_seconds'] + primary['wall_seconds']
        result['peak_working_set_bytes'] = peak
        (OUT / 'reachable_addendum.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        print(json.dumps({'status': result['status'], 'wall_seconds': result['wall_seconds'], 'combined_seconds': result['combined_primary_addendum_wall_seconds']}, ensure_ascii=False))
    except Exception as exc:
        result['status'] = '失败或停止'; result['error'] = repr(exc); result['wall_seconds'] = time.perf_counter() - started
        (OUT / 'reachable_addendum_failure.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
