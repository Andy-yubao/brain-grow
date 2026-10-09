"""固定节点，另存冻结FC的输出不可达边清理副本。"""
from pathlib import Path
import copy
import hashlib
import json
import time
import torch
from tools.train_capacity_pruning import MaskedFC, load_export, export_model

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/runs/h1_dead_edges_cleanup_v1'
REPORT = ROOT / 'docs/reports/2026-10-08/h1_dead_edges_cleanup_v1'
SOURCES = [
    ('静态幅度', 'experiments/runs/h1_aligned_edges_v1/mnist_static/magnitude/model.bgm', '6e69ffb1e1860eaf2c9d146501e3a0bbf215c958d1ccd546d9cbbed4c943b857'),
    ('循环幅度', 'history/ann_stages_20261008/experiments/runs/h1_translated_edges_v1/mnist_circular/magnitude/model.bgm', '74202b856d95f29ed31128b159c33362ed5f875dfedd400d0b9166a6488f0180'),
    ('循环节点贡献', 'history/ann_stages_20261008/experiments/runs/h1_translated_edges_v1/mnist_circular/node/model.bgm', 'bc800fb08a2096d8d57f89eaae3eb89231d711db5ac7f6bc4cc24274e9dc610b'),
]

def reach(model):
    back = [None] * (len(model.layers) + 1)
    back[-1] = torch.ones(model.layers[-1].out_features, dtype=torch.bool)
    for i in range(len(model.layers)-1, -1, -1):
        back[i] = (model.layers[i].weight[back[i+1]] != 0).any(0)
    return back

def clean(model):
    back = reach(model)
    with torch.no_grad():
        for i, (layer, mask) in enumerate(zip(model.layers, model.masks)):
            mask[~back[i+1]] = False
        model.enforce()
    return back

def counts(model):
    return [int(torch.count_nonzero(layer.weight)) for layer in model.layers]

def main():
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.manual_seed(20261004)
    toy = MaskedFC([2, 3, 2, 1], 0).eval()
    with torch.no_grad():
        for l in toy.layers:
            l.weight.fill_(1); l.bias.fill_(1)
        toy.layers[-1].weight[0, 1] = 0
        toy.layers[1].weight[0, 1:] = 0
    original = copy.deepcopy(toy)
    clean(toy)
    assert counts(toy) == [2, 1, 1]
    assert torch.equal(original(torch.ones(4, 2)), toy(torch.ones(4, 2)))
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    result = {'training_updates': 0, 'dataset_access': False, 'gpu_access': False,
              'device': 'CPU', 'threads': 2, 'precision': 'FP32', 'execution': 'dense_linear_zero_mask_eval',
              'synthetic_batch': 128, 'toy_check': True,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'models': []}
    g = torch.Generator().manual_seed(20261004)
    x = torch.cat([torch.zeros(16, 784), torch.ones(16, 784),
                   torch.rand(48, 784, generator=g), torch.randn(48, 784, generator=g)])
    with torch.no_grad():
        for index, (name, relative, expected) in enumerate(SOURCES):
            path = ROOT / relative
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
            original = load_export(path).eval()
            assert all(torch.isfinite(p).all() for p in original.parameters())
            model = copy.deepcopy(original)
            before = counts(model)
            back = clean(model)
            after = counts(model)
            for i, (a, b) in enumerate(zip(original.layers, model.layers)):
                assert torch.equal(a.bias, b.bias)
                assert torch.equal(a.weight[back[i+1]], b.weight[back[i+1]])
                assert not torch.count_nonzero(b.weight[~back[i+1]])
            remaining = reach(model)
            assert all(not torch.count_nonzero(l.weight[~remaining[i+1]]) for i, l in enumerate(model.layers))
            target = OUT / f'model_{index}.bgm'
            exported = export_model(model, target)
            loaded = load_export(target).eval()
            assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), loaded.state_dict().values()))
            y0, y1 = original(x), loaded(x)
            error = float((y0-y1).abs().max())
            same = bool(torch.equal(y0.argmax(1), y1.argmax(1)))
            assert error <= 1e-6 and same
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
            result['models'].append({'name': name, 'source': relative, 'source_sha256': expected,
                'before_edges': before, 'removed_edges': [a-b for a,b in zip(before, after)], 'after_edges': after,
                'removed_fraction': 1-sum(after)/sum(before), 'source_bytes': path.stat().st_size,
                'export': exported, 'output': str(target.relative_to(ROOT)),
                'max_absolute_logit_difference': error, 'same_synthetic_predictions': same,
                'remaining_dead_edges': 0, 'physical_nodes_changed': False, 'bias_changed': False})
            if time.perf_counter()-started > 120: raise TimeoutError('预算到达')
    result['wall_seconds'] = time.perf_counter()-started
    (REPORT/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))

if __name__ == '__main__':
    main()
