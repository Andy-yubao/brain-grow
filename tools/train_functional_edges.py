"""功能补偿、任务门控与结构替换；固定节点，训练内闭合，可恢复。"""
from __future__ import annotations

import argparse
import copy
import heapq
import json
import math
from pathlib import Path
import random
import subprocess
import tempfile
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from tools.train_capacity_baseline import ROOT, digest, evaluate, save_json
from tools.train_capacity_pruning import MaskedFC, capture, restore, cpu_tree, export_model, sync
from tools.train_aligned_edges import seed, edge_counts, empty_ledger
from tools.train_adaptive_edges import shifts, closed_edges, learning_rate, train_epoch, adaptation_status, tree_equal
from braingrow.datasets import load_bundle


def qualifies(value, cfg):
    return value['accuracy'] + 1e-8 >= cfg['accuracy_floor'] and value['cross_entropy'] <= cfg['cross_entropy_ceiling']


def reset_parameters(optimizer, model):
    for layer in model.layers:
        optimizer.state.pop(layer.weight, None)
        optimizer.state.pop(layer.bias, None)


def new_plain(cfg, device, state=None):
    seed(cfg['model_seed'])
    model = MaskedFC(cfg['fc_widths'], cfg['dropout']).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    if state is not None:
        restore(model, opt, [], state)
    return model, opt


def atomic(path, value):
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    for attempt in range(21):
        try:
            temporary.replace(path)
            return
        except OSError as exc:
            if getattr(exc, 'winerror', None) not in (5, 32) or attempt == 20:
                raise
            time.sleep(.05)


class Gates(nn.Module):
    """只对支持集建参数；确定性系数为拉伸 sigmoid 后截断。"""
    def __init__(self, plain, cfg):
        super().__init__()
        self.plain, self.cfg = plain, cfg
        self.log_alpha = nn.ParameterList()
        shift = cfg['gate_temperature'] * math.log(-cfg['gate_low']/cfg['gate_high'])
        initial = math.log(cfg['gate_initial_probability']/(1-cfg['gate_initial_probability'])) + shift
        for i, mask in enumerate(plain.masks):
            indices = mask.flatten().nonzero().flatten()
            self.register_buffer(f'indices_{i}', indices)
            self.log_alpha.append(nn.Parameter(torch.full((len(indices),), initial, device=mask.device)))

    def probability(self, i):
        c = self.cfg
        return torch.sigmoid(self.log_alpha[i] - c['gate_temperature']*math.log(-c['gate_low']/c['gate_high']))

    def coefficient(self, i, stochastic):
        c = self.cfg
        a = self.log_alpha[i]
        if stochastic:
            u = torch.rand_like(a).clamp(1e-6, 1-1e-6)
            z = torch.sigmoid((torch.log(u)-torch.log1p(-u)+a)/c['gate_temperature'])
        else:
            z = torch.sigmoid(a)
        return (z*(c['gate_high']-c['gate_low'])+c['gate_low']).clamp(0, 1)

    def forward(self, x):
        for i, layer in enumerate(self.plain.layers):
            indices = getattr(self, f'indices_{i}')
            factor = torch.zeros_like(layer.weight).flatten().scatter(0, indices, self.coefficient(i, self.training)).reshape_as(layer.weight)
            x = F.linear(x, layer.weight*factor, layer.bias)
            if i < len(self.plain.layers)-1:
                x = F.dropout(F.relu(x), self.plain.dropout, self.training)
        return x

    def complexity(self):
        return sum(self.probability(i).sum() for i in range(len(self.log_alpha)))

    def harden(self):
        result = MaskedFC(self.cfg['fc_widths'], self.cfg['dropout']).to(next(self.parameters()).device)
        result.load_state_dict(self.plain.state_dict())
        closed = []
        with torch.no_grad():
            for i, (layer, mask) in enumerate(zip(result.layers, result.masks)):
                indices = getattr(self, f'indices_{i}')
                keep = self.probability(i) >= self.cfg['gate_threshold']
                factors = self.coefficient(i, False)
                layer.weight.flatten()[indices] *= factors
                mask.flatten()[indices[~keep]] = False
                closed.append(int((~keep).sum()))
            result.enforce()
        return result, closed


def gate_capture(model, weight_opt, gate_opt):
    state = capture(model, weight_opt, [])
    state['gate_optimizer'] = cpu_tree(gate_opt.state_dict())
    state['schema'] = 'functional_gates_v1'
    return state


def gate_restore(model, weight_opt, gate_opt, state):
    restore(model, weight_opt, [], state)
    gate_opt.load_state_dict(state['gate_optimizer'])
    gate_opt.zero_grad(set_to_none=True)


class Runner:
    def __init__(self, cfg, folder, tensors, device, start):
        self.cfg, self.folder, self.device = cfg, folder, device
        self.x, self.y, self.vx, self.vy, self.cx, self.fx, self.fy, self.hx, self.hy = tensors
        self.start = start
        self.ledger = empty_ledger()
        self.costs = {k: 0. for k in ('solve_seconds', 'training_seconds', 'validation_seconds', 'save_seconds', 'closure_seconds', 'failure_search_seconds', 'data_setup_seconds')}
        self.deadline = start + cfg['wall_seconds_per_method']

    def budget(self):
        if time.perf_counter() >= self.deadline:
            raise TimeoutError('本轮端到端两小时预算耗尽；未完成状态已保存')
        if self.device.type == 'cuda' and torch.cuda.max_memory_allocated() > self.cfg['peak_memory_limit_bytes']:
            raise MemoryError('本轮峰值显存超过8GiB')

    def save(self, path, value):
        t = time.perf_counter()
        atomic(path, value)
        self.costs['save_seconds'] += time.perf_counter()-t

    def measure(self, model):
        t = time.perf_counter()
        val, logits = evaluate(model, self.vx, self.vy, self.cfg['evaluation_batch_size'])
        self.costs['validation_seconds'] += time.perf_counter()-t
        self.ledger['evaluation_calls'] += 1
        self.ledger['evaluation_samples'] += len(self.vx)
        return val, logits

    def close(self, model, optimizer):
        t = time.perf_counter()
        value = closed_edges(model, optimizer)
        self.costs['closure_seconds'] += time.perf_counter()-t
        return value

    def adapt(self, model, optimizer, path, sequence, force48=False):
        """独立分支逐轮保存；拒绝时调用者恢复父状态。"""
        cfg = self.cfg
        branch_started, prior_wall = time.perf_counter(), 0.
        epoch, rows, points, calibration = 0, [], [], []
        if path.exists():
            saved = torch.load(path, map_location='cpu', weights_only=False)
            restore(model, optimizer, [], saved['state'])
            epoch, rows, points, calibration = [saved[k] for k in ('epoch', 'rows', 'points', 'calibration')]
            prior_wall = saved.get('branch_wall_seconds', 0.)
            self.ledger, self.costs = saved.get('ledger',self.ledger), saved.get('costs',self.costs)
            if 'method_wall_seconds' in saved:
                self.start = time.perf_counter()-saved['method_wall_seconds']
                self.deadline = self.start+cfg['wall_seconds_per_method']
            if saved.get('completed'):
                return saved['result']
        while epoch < cfg['maximum_epochs']:
            self.budget()
            epoch += 1
            t = time.perf_counter()
            row = train_epoch(model, optimizer, self.x, self.y, cfg, sequence+epoch,
                              learning_rate(epoch, True, cfg), self.ledger)
            sync(self.device)
            self.costs['training_seconds'] += time.perf_counter()-t
            rows.append({'epoch': epoch, **row})
            if epoch % 4 == 0:
                t = time.perf_counter()
                fit, _ = evaluate(model, self.cx, self.y, cfg['evaluation_batch_size'])
                self.costs['validation_seconds'] += time.perf_counter()-t
                self.ledger['evaluation_calls'] += 1
                self.ledger['evaluation_samples'] += len(self.cx)
                calibration.append({'step': epoch, **fit})
            status = 'extend'
            if epoch in cfg['validation_epochs']:
                val, _ = self.measure(model)
                training_status = adaptation_status(calibration, cfg)
                improving = bool(points and (points[-1]['validation']['cross_entropy']-val['cross_entropy'] > cfg['validation_improvement_ce'] or val['accuracy']-points[-1]['validation']['accuracy'] > cfg['validation_improvement_accuracy']))
                if qualifies(val, cfg) and not force48:
                    status = 'accept'
                elif not force48 and training_status == 'plateau' and not improving:
                    status = 'completed_failure'
                elif epoch == cfg['maximum_epochs']:
                    status = 'control_complete' if force48 else 'optimization_incomplete'
                points.append({'epoch': epoch, 'validation': val, 'training_status': training_status, 'validation_improving': improving, 'decision': status})
                print(json.dumps({'branch': path.stem, 'epoch': epoch, 'accuracy': val['accuracy'], 'ce': val['cross_entropy'], 'decision': status}), flush=True)
            completed = status != 'extend'
            wall = prior_wall+time.perf_counter()-branch_started
            result = {'status': status, 'epochs': epoch, 'rows': rows, 'validation_points': points, 'validation': points[-1]['validation'] if points else None, 'wall_seconds':wall}
            self.save(path, {'state': capture(model, optimizer, []), 'epoch': epoch, 'rows': rows, 'points': points, 'calibration': calibration, 'completed': completed, 'result': result, 'branch_wall_seconds':wall, 'ledger':self.ledger, 'costs':self.costs, 'method_wall_seconds':time.perf_counter()-self.start})
            if completed:
                return result

    def finish(self, method, model, optimizer, events, rows, stop, selected):
        arm = self.folder/method
        val, logits = self.measure(model)
        t = time.perf_counter()
        fit, _ = evaluate(model, self.cx, self.y, self.cfg['evaluation_batch_size'])
        self.costs['validation_seconds'] += time.perf_counter()-t
        self.ledger['evaluation_calls'] += 1
        self.ledger['evaluation_samples'] += len(self.cx)
        closure = self.close(model, optimizer)
        t = time.perf_counter()
        structure = {**export_model(model, arm/'model.bgm'), **edge_counts(model)}
        indegrees = []
        for i, mask in enumerate(model.masks):
            v = mask.sum(1).cpu().numpy()
            indegrees.append({'layer': i+1, 'min': int(v.min()), 'max': int(v.max()), 'mean': float(v.mean()), 'quantiles': np.quantile(v, [0, .1, .25, .5, .75, .9, 1]).tolist(), 'zero_input_nodes': int((v==0).sum())})
        np.savez_compressed(arm/'validation_final.npz', logits=logits, labels=self.vy.cpu().numpy())
        self.costs['save_seconds'] += time.perf_counter()-t
        self.save(arm/'model_full.pt', {'state': capture(model, optimizer, []), 'info': {'method': method, 'validation': val, 'selected_full_checkpoint': selected, 'events': events}})
        result = {'method': method, 'validation': val, 'training_fixed_one_position':fit, 'structure': structure, 'indegrees': indegrees, 'events': events, 'rows': rows, 'costs': copy.deepcopy(self.costs), 'ledger': copy.deepcopy(self.ledger),
                  'wall_seconds': time.perf_counter()-self.start, 'selected_full_checkpoint': selected, 'stop': stop, 'final_closure': closure,
                  'peak_cuda_memory_bytes': torch.cuda.max_memory_allocated() if self.device.type=='cuda' else None,
                  'full_checkpoint_sha256': digest(arm/'model_full.pt'), 'official_test_accessed': False, 'historical_test_accessed': False, 'sealed_remaining_ids_accessed': False}
        save_json(arm/'result.json', result)
        return result


def activations(model, x, y, batch):
    inputs = [[] for _ in model.layers]
    sensitivity = [torch.zeros(l.out_features, dtype=torch.float64, device=x.device) for l in model.layers]
    energy = [torch.zeros_like(v) for v in sensitivity]
    model.eval()
    for bx, by in zip(x.split(batch), y.split(batch)):
        h, zs = bx, []
        for i, layer in enumerate(model.layers):
            inputs[i].append(h.detach())
            z = layer(h)
            zs.append(z)
            h = F.relu(z) if i < len(model.layers)-1 else z
        grads = torch.autograd.grad(F.cross_entropy(h, by, reduction='sum'), zs)
        for i, (z, grad) in enumerate(zip(zs, grads)):
            sensitivity[i] += grad.detach().double().square().sum(0)
            energy[i] += z.detach().double().square().sum(0)
    return [torch.cat(v) for v in inputs], [v/len(x) for v in sensitivity], [v/len(x) for v in energy]


def stabilized_inverse(matrix, cfg):
    diagonal = matrix.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(0)
    damping = (diagonal*cfg['damping_relative']).clamp_min(cfg['damping_minimum'])
    eye = torch.eye(matrix.shape[-1], dtype=matrix.dtype, device=matrix.device)
    repairs = 0
    for attempt in range(4):
        h = matrix + damping[..., None, None]*eye
        chol, info = torch.linalg.cholesky_ex(h)
        if not info.any():
            return torch.cholesky_inverse(chol), damping, repairs
        if attempt == 3:
            raise FloatingPointError('受限支持集FP64 Cholesky四次均失败')
        damping = torch.where(info!=0, damping*10, damping)
        repairs += int((info!=0).sum())


def compensation_proposal(model, fx, fy, hx, cfg, strength, runner):
    """各行支持集逆＋OBS序列；最终联合重解，始终以父响应为锚点。"""
    t = time.perf_counter()
    with torch.no_grad():
        for layer, mask in zip(model.layers, model.masks):
            mask.logical_and_(layer.weight != 0)
    xs, sensitivity, energy = activations(model, fx, fy, cfg['evaluation_batch_size'])
    held, _, _ = activations(model, hx, torch.zeros(len(hx), dtype=torch.long, device=hx.device), cfg['evaluation_batch_size'])
    sequences, covariances, means, dampings = [], [], [], []
    repairs, next_print = 0, time.perf_counter()+30
    original_weights = [l.weight.detach().clone() for l in model.layers]
    original_bias = [l.bias.detach().clone() for l in model.layers]
    for li, (layer, mask, xx, sens) in enumerate(zip(model.layers, model.masks, xs, sensitivity)):
        mu = xx.double().mean(0)
        center = xx.double()-mu
        cov = center.T@center/len(xx)
        covariances.append(cov)
        means.append(mu)
        layer_seq, layer_damp = [None]*layer.out_features, torch.zeros(layer.out_features, dtype=torch.float64, device=fx.device)
        supports = [row.nonzero().flatten() for row in mask]
        buckets = {}
        for row, support in enumerate(supports):
            if len(support):
                buckets.setdefault(int(len(support)), []).append(row)
            else:
                layer_seq[row] = (np.empty(0, np.int32), np.empty(0))
        for n, ids_all in sorted(buckets.items()):
            block = max(1, min(len(ids_all), cfg['matrix_block_bytes']//max(1, n*n*8*5)))
            for offset in range(0, len(ids_all), block):
                runner.budget()
                ids = ids_all[offset:offset+block]
                indices = torch.stack([supports[row] for row in ids])
                mats = cov[indices[:, :, None], indices[:, None, :]]
                q, damping, repaired = stabilized_inverse(mats, cfg)
                repairs += repaired
                layer_damp[ids] = damping
                w = layer.weight[ids].gather(1, indices).detach().double()
                alive = torch.ones_like(w, dtype=torch.bool)
                order, damage = [], []
                batches = torch.arange(len(ids), device=fx.device)
                for step in range(n):
                    diag = q.diagonal(dim1=-2, dim2=-1).clamp_min(1e-30)
                    score = (w.square()/diag).masked_fill(~alive, float('inf'))
                    j = score.argmin(1)
                    column = q[batches, :, j].clone()
                    pivot = q[batches, j, j].clone()
                    cost = score[batches, j]*sens[ids]
                    order.append(indices[batches, j])
                    damage.append(cost)
                    w -= column*(w[batches, j]/pivot)[:, None]
                    w[batches, j] = 0
                    q -= column[:, :, None]*column[:, None, :]/pivot[:, None, None]
                    q[batches, j, :] = 0
                    q[batches, :, j] = 0
                    q[batches, j, j] = 1
                    alive[batches, j] = False
                orders, damages = torch.stack(order, 1).cpu().numpy(), torch.stack(damage, 1).cpu().numpy()
                for row, oo, dd in zip(ids, orders, damages):
                    layer_seq[row] = (oo.astype(np.int32), dd)
                if time.perf_counter() >= next_print:
                    print(json.dumps({'proposal': '支持集贪心序列', 'layer': li+1, 'support_size': n, 'processed_bucket_rows': offset+len(ids)}), flush=True)
                    next_print = time.perf_counter()+30
        sequences.append(layer_seq)
        dampings.append(layer_damp)
        print(json.dumps({'proposal_layer_complete': li+1, 'rows': layer.out_features}), flush=True)
    budget = float(sum((s*e).sum() for s, e in zip(sensitivity, energy)))*strength
    cap = math.floor(sum(int(m.sum()) for m in model.masks)*cfg['maximum_delete_fraction'])
    heap, deleted = [], [[[] for _ in layer] for layer in sequences]
    for li, layer in enumerate(sequences):
        for row, (_, damage) in enumerate(layer):
            if len(damage):
                heapq.heappush(heap, (float(damage[0]), li, row, 0))
    total_damage, count = 0., 0
    while heap and count < cap:
        cost, li, row, step = heapq.heappop(heap)
        if not math.isfinite(cost) or total_damage+cost > budget:
            break
        total_damage += cost
        deleted[li][row].append(int(sequences[li][row][0][step]))
        count += 1
        if step+1 < len(sequences[li][row][0]):
            heapq.heappush(heap, (float(sequences[li][row][1][step+1]), li, row, step+1))
    metrics, deleted_quantiles, flats = [], [], []
    with torch.no_grad():
        for li, (layer, mask, cov, mu, damping) in enumerate(zip(model.layers, model.masks, covariances, means, dampings)):
            previous = original_weights[li].double()
            direct = previous.clone()
            removed_weights, indices_flat = [], []
            for row, values in enumerate(deleted[li]):
                if not values:
                    continue
                runner.budget()
                d = torch.tensor(values, device=fx.device)
                removed_weights.append(previous[row, d].abs())
                indices_flat.extend(row*layer.in_features+value for value in values)
                mask[row, d] = False
                direct[row, d] = 0
                r = mask[row].nonzero().flatten()
                layer.weight[row, d] = 0
                if len(r):
                    hrr = cov[r[:, None], r[None, :]]+damping[row]*torch.eye(len(r), dtype=torch.float64, device=fx.device)
                    delta = torch.linalg.solve(hrr, cov[r[:, None], d[None, :]]@previous[row, d])
                    layer.weight[row, r] = (previous[row, r]+delta).float()
                layer.bias[row] = original_bias[li][row] + (mu@(previous[row]-layer.weight[row].double())).float()
            current = layer.weight.detach().double()
            row_metrics = {}
            for name, inputs in [('fit', xs[li]), ('diagnostic', held[li])]:
                before = inputs.double()@previous.T+original_bias[li].double()
                naive = inputs.double()@direct.T+original_bias[li].double()
                after = inputs.double()@current.T+layer.bias.double()
                row_metrics[name] = {'uncompensated_mse': float((naive-before).square().mean()), 'compensated_mse': float((after-before).square().mean()), 'parent_response_energy': float(before.square().mean())}
            metrics.append(row_metrics)
            flat = torch.cat(removed_weights).cpu().numpy() if removed_weights else np.empty(0)
            deleted_quantiles.append(np.quantile(flat, [0, .25, .5, .75, 1]).tolist() if len(flat) else [])
            flats.append(np.asarray(indices_flat, dtype=np.int32))
        model.enforce()
    runner.costs['solve_seconds'] += time.perf_counter()-t
    return {'strength': strength, 'damage_budget': budget, 'predicted_damage': total_damage, 'edge_operation_cap': cap, 'direct_deleted_per_layer': [len(v) for v in flats], 'reconstruction': metrics, 'deleted_abs_weight_quantiles': deleted_quantiles, 'numeric_repairs': repairs, 'proposal_wall_seconds':time.perf_counter()-t}, flats


def run_compensation(runner, source):
    cfg, arm = runner.cfg, runner.folder/'compensation'
    arm.mkdir(exist_ok=True)
    model, optimizer = new_plain(cfg, runner.device, source)
    progress = arm/'resume.pt'
    c = {'slot': 1, 'attempt': 1, 'strength': cfg['damage_fraction'], 'mode': 'proposal', 'events': [], 'rows': [], 'best': source, 'best_key': sum(edge_counts(model)['nonzero_edges']), 'parent': source, 'selected': str(cfg['source_checkpoint']), 'controls_done': False}
    if progress.exists():
        saved = torch.load(progress, map_location='cpu', weights_only=False)
        restore(model, optimizer, [], saved['state'])
        c = saved['controller']
        runner.ledger, runner.costs = saved['ledger'], saved['costs']
        runner.start = time.perf_counter()-saved['wall_seconds']
        runner.deadline = runner.start+cfg['wall_seconds_per_method']
    def save():
        runner.save(progress, {'state': capture(model, optimizer, []), 'controller': c, 'ledger': runner.ledger, 'costs': runner.costs, 'wall_seconds': time.perf_counter()-runner.start})
        save_json(arm/'progress.json', {k: v for k, v in c.items() if k not in ('best', 'parent', 'candidate')})
    stop = 'parent_slots_completed'
    try:
        while c['slot'] <= cfg['parent_slots']:
            runner.budget()
            if c['mode'] == 'proposal':
                c['parent'] = capture(model, optimizer, [])
                event, flats = compensation_proposal(model, runner.fx, runner.fy, runner.hx, cfg, c['strength'], runner)
                reset_parameters(optimizer, model)
                event['closure'] = runner.close(model, optimizer)
                event.update({'slot': c['slot'], 'attempt': c['attempt'], 'before_edges': sum(int(torch.count_nonzero(v)) for k,v in c['parent']['model'].items() if k.endswith('.weight')), 'candidate_structure': edge_counts(model)})
                np.savez_compressed(arm/f"slot_{c['slot']}_attempt_{c['attempt']}_deleted.npz", **{f'layer_{i}': v for i,v in enumerate(flats)})
                c.update({'event': event, 'candidate': capture(model, optimizer, []), 'mode': 'controls' if not c['controls_done'] else 'adaptation'})
                save()
            if c['mode'] == 'controls':
                for name in ('uncompensated', 'continue_training'):
                    control, opt = new_plain(cfg, runner.device, c['parent'])
                    reset_parameters(opt, control)
                    if name == 'uncompensated':
                        for i, mask in enumerate(control.masks):
                            mask.copy_(c['candidate']['model'][f'mask_{i}'].to(mask.device))
                        control.enforce(opt)
                        runner.close(control, opt)
                    result = runner.adapt(control, opt, arm/f'{name}_resume.pt', 10000, True)
                    save_json(arm/f'{name}_result.json', result)
                restore(model, optimizer, [], c['candidate'])
                c.update({'controls_done': True, 'mode': 'adaptation'})
                save()
            path = arm/f"slot_{c['slot']}_attempt_{c['attempt']}_resume.pt"
            outcome = runner.adapt(model, optimizer, path, 10000+(c['slot']-1)*100)
            c['event'].update(outcome)
            net = sum(edge_counts(model)['nonzero_edges']) < c['event']['before_edges']
            accepted = qualifies(outcome['validation'], cfg) and net
            c['event']['accepted'] = accepted
            c['events'].append(c['event'])
            c['rows'].extend(outcome['rows'])
            if accepted:
                key = sum(edge_counts(model)['nonzero_edges'])
                if key < c['best_key']:
                    c['best'], c['best_key'], c['selected'] = capture(model, optimizer, []), key, str(path.relative_to(ROOT))
                c.update({'slot': c['slot']+1, 'attempt': 1, 'mode': 'proposal'})
            else:
                # 失败搜索是已计solve/train/save的子集，不能再累加作总成本。
                runner.costs['failure_search_seconds'] += outcome['wall_seconds']+c['event']['proposal_wall_seconds']
                restore(model, optimizer, [], c['parent'])
                runner.ledger['rollbacks'] += 1
                if outcome['status'] == 'completed_failure' and c['attempt'] < cfg['maximum_attempts_per_parent']:
                    c.update({'attempt': c['attempt']+1, 'strength': c['strength']*.5, 'mode': 'proposal'})
                else:
                    stop = outcome['status']
                    break
            save()
    except TimeoutError:
        stop = 'method_budget_optimization_incomplete'
        save()
    restore(model, optimizer, [], c['best'])
    return runner.finish('compensation', model, optimizer, c['events'], c['rows'], stop, c['selected'])


def gate_epoch(gated, wo, go, teacher, runner, epoch, pressure, initial):
    cfg = runner.cfg
    torch.manual_seed(cfg['model_seed']+4000+epoch)
    stream = torch.Generator(device=runner.device).manual_seed(cfg['data_seed']+5000+epoch)
    order = torch.randperm(len(runner.x), generator=stream, device=runner.device)
    offsets = torch.randint(28, (len(runner.x), 2), generator=stream, device=runner.device)
    histogram = np.bincount((offsets[:,0]*28+offsets[:,1]).cpu().numpy(), minlength=784)
    runner.ledger['training_position_histogram'] = (np.asarray(runner.ledger.get('training_position_histogram',[0]*784))+histogram).tolist()
    cosine = .01+.99*.5*(1+math.cos(math.pi*(epoch-1)/cfg['gate_epochs']))
    wo.param_groups[0]['lr'] = cfg['learning_rate']*cosine
    go.param_groups[0]['lr'] = cfg['gate_learning_rate']*cosine
    penalty = pressure*max(0, min(1, (epoch-8)/48))
    kdweight = cfg['distillation_weight']*max(0, 1-epoch/56)
    losses = np.zeros(3)
    gated.train()
    teacher.eval()
    for ids in order.split(cfg['batch_size']):
        inputs = shifts(runner.x[ids], offsets[ids])
        wo.zero_grad(set_to_none=True)
        go.zero_grad(set_to_none=True)
        output = gated(inputs)
        ce = F.cross_entropy(output, runner.y[ids])
        with torch.no_grad():
            target = teacher(inputs)
        temp = cfg['distillation_temperature']
        kd = F.kl_div(F.log_softmax(output/temp, 1), F.softmax(target/temp, 1), reduction='batchmean')*temp*temp
        complexity = gated.complexity()/initial
        loss = ce+kdweight*kd+penalty*complexity
        if not torch.isfinite(loss):
            raise FloatingPointError('连接门控训练非有限')
        loss.backward()
        for layer, mask in zip(gated.plain.layers, gated.plain.masks):
            layer.weight.grad.mul_(mask)
        wo.step()
        go.step()
        gated.plain.enforce(wo)
        losses += np.asarray([float(ce.detach()), float(kd.detach()), float(complexity.detach())])*len(ids)
        runner.ledger['attempted_updates'] += 1
        runner.ledger['optimization_samples'] += len(ids)
    wo.zero_grad(set_to_none=True)
    go.zero_grad(set_to_none=True)
    return {'epoch': epoch, 'ce': losses[0]/len(runner.x), 'kd': losses[1]/len(runner.x), 'expected_nonzero_fraction': losses[2]/len(runner.x), 'quantity_weight': penalty, 'distillation_weight': kdweight, 'weight_learning_rate': wo.param_groups[0]['lr'], 'gate_learning_rate': go.param_groups[0]['lr']}


def run_gate_arm(runner, source, name, pressure, best_source=None):
    arm_started = time.perf_counter()
    prior_arm_wall = 0.
    if runner.device.type=='cuda':
        torch.cuda.reset_peak_memory_stats()
    cfg, folder = runner.cfg, runner.folder/name
    folder.mkdir(exist_ok=True)
    teacher, _ = new_plain(cfg, runner.device, torch.load(ROOT/cfg['source_checkpoint'], map_location='cpu', weights_only=False)['state'])
    progress = folder/'resume.pt'
    loaded = torch.load(progress, map_location='cpu', weights_only=False) if progress.exists() else None
    plain, wo = new_plain(cfg, runner.device, source)
    if loaded:
        plain.load_state_dict({k.removeprefix('plain.'):v for k,v in loaded['state']['model'].items() if k.startswith('plain.')})
    gated = Gates(plain, cfg)
    go = torch.optim.Adam(gated.log_alpha.parameters(), lr=cfg['gate_learning_rate'])
    c = {'epoch': 0, 'mode': 'joint', 'events': [], 'rows': [], 'best': best_source or source, 'best_key': sum(int(v.sum()) for k,v in (best_source or source)['model'].items() if k.startswith('mask_')), 'selected': cfg['source_checkpoint'], 'gate_start_count':sum(len(a) for a in gated.log_alpha), 'online_replaced':False, 'costs_baseline':copy.deepcopy(runner.costs), 'ledger_baseline':copy.deepcopy(runner.ledger)}
    if loaded:
        saved = loaded
        gate_restore(gated, wo, go, saved['state'])
        c = saved['controller']
        runner.ledger, runner.costs = saved['ledger'], saved['costs']
        runner.start = time.perf_counter()-saved['wall_seconds']
        runner.deadline = runner.start+cfg['wall_seconds_per_method']
        prior_arm_wall = saved.get('arm_wall_seconds', 0.)
    def save():
        runner.save(progress, {'state': gate_capture(gated, wo, go), 'controller': c, 'wall_seconds': time.perf_counter()-runner.start, 'arm_wall_seconds':prior_arm_wall+time.perf_counter()-arm_started, 'ledger': runner.ledger, 'costs': runner.costs})
        save_json(folder/'progress.json', {k:v for k,v in c.items() if k not in ('best',)})
    stop = 'gate_epochs_completed'
    try:
        while c['epoch'] < cfg['gate_epochs'] or c['mode'] == 'hard_branch':
            runner.budget()
            if c['mode'] == 'joint':
                c['epoch'] += 1
                t = time.perf_counter()
                row = gate_epoch(gated, wo, go, teacher, runner, c['epoch'], pressure, c['gate_start_count'])
                sync(runner.device)
                runner.costs['training_seconds'] += time.perf_counter()-t
                c['rows'].append(row)
                if name.startswith('relearn/') and c['epoch']==40 and not c['online_replaced']:
                    # 训练中根据任务反馈改变可用支持集；新的门继续共同优化。
                    hard, removed = gated.harden()
                    closed = runner.close(hard, None)
                    replacement = grow_edges(hard, runner)
                    wo = torch.optim.AdamW(hard.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
                    gated = Gates(hard, cfg)
                    go = torch.optim.Adam(gated.log_alpha.parameters(), lr=cfg['gate_learning_rate'])
                    c['events'].append({'joint_epoch':40, 'operation':'online_task_driven_support_replacement', 'threshold_deleted_per_layer':removed, 'closure':closed, **replacement, 'optimizer_reset':'all_weight_bias_and_gate_parameters'})
                    c['online_replaced'] = True
                if c['epoch'] in cfg['gate_checkpoints']:
                    c['mode'] = 'hard_branch'
                save()
                if c['epoch'] % 8 == 0:
                    print(json.dumps({'arm': name, **row}), flush=True)
            if c['mode'] == 'hard_branch':
                joint = gate_capture(gated, wo, go)
                hard, threshold_deleted = gated.harden()
                deleted_quantiles = []
                for i, layer in enumerate(gated.plain.layers):
                    indices = getattr(gated, f'indices_{i}')
                    removed = layer.weight.detach().flatten()[indices[gated.probability(i)<cfg['gate_threshold']]].abs().cpu().numpy()
                    deleted_quantiles.append(np.quantile(removed,[0,.25,.5,.75,1]).tolist() if len(removed) else [])
                opt = torch.optim.AdamW(hard.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
                closure = runner.close(hard, opt)
                path = folder/f"hard_{c['epoch']}_resume.pt"
                outcome = runner.adapt(hard, opt, path, 20000+c['epoch']*100)
                val = outcome['validation']
                counts = edge_counts(hard)
                event = {'joint_epoch': c['epoch'], 'threshold_deleted_per_layer': threshold_deleted, 'deleted_abs_weight_quantiles':deleted_quantiles, 'expected_nonzero_edges':float(gated.complexity().detach()), 'closure': closure, 'structure': counts, **outcome}
                accepted = qualifies(val, cfg) and sum(counts['nonzero_edges']) < c['best_key']
                event['accepted'] = accepted
                if accepted:
                    c['best'], c['best_key'], c['selected'] = capture(hard, opt, []), sum(counts['nonzero_edges']), str(path.relative_to(ROOT))
                c['events'].append(event)
                gate_restore(gated, wo, go, joint)
                c['mode'] = 'joint'
                save()
    except TimeoutError:
        stop = 'method_budget_optimization_incomplete'
        save()
    model, opt = new_plain(cfg, runner.device, c['best'])
    result = runner.finish(name, model, opt, c['events'], c['rows'], stop, c['selected'])
    result['method_cumulative_wall_seconds'] = result['wall_seconds']
    result['wall_seconds'] = prior_arm_wall+time.perf_counter()-arm_started
    result['method_cumulative_costs'] = copy.deepcopy(result['costs'])
    result['costs'] = {k:v-c['costs_baseline'].get(k,0.) for k,v in runner.costs.items()}
    result['method_cumulative_ledger'] = copy.deepcopy(result['ledger'])
    result['ledger'] = {k:(v-c['ledger_baseline'].get(k,0) if isinstance(v,(int,float)) else (np.asarray(v)-np.asarray(c['ledger_baseline'].get(k,[0]*len(v)))).tolist()) for k,v in runner.ledger.items()}
    save_json(folder/'result.json',result)
    return result


def grow_edges(model, runner):
    """未掩码任务梯度outer product与对角平方代理；不按幅度排序。"""
    cfg = runner.cfg
    sums, squares = [torch.zeros_like(l.weight) for l in model.layers], [torch.zeros_like(l.weight) for l in model.layers]
    model.eval()
    for x, y in zip(runner.fx.split(cfg['batch_size']), runner.fy.split(cfg['batch_size'])):
        runner.budget()
        h, inputs, zs = x, [], []
        for i, layer in enumerate(model.layers):
            inputs.append(h.detach())
            z = layer(h)
            zs.append(z)
            h = F.relu(z) if i<len(model.layers)-1 else z
        grads = torch.autograd.grad(F.cross_entropy(h, y, reduction='sum'), zs)
        for i, (g, a) in enumerate(zip(grads, inputs)):
            sums[i] += g.T@a
            squares[i] += g.square().T@a.square()
    scores = [(g/len(runner.fx)).square()/(v/len(runner.fx)+1e-12) for g,v in zip(sums, squares)]
    flat = torch.cat([s.masked_fill(m, -float('inf')).flatten() for s,m in zip(scores, model.masks)])
    count = math.floor(sum(int(m.sum()) for m in model.masks)*cfg['rewire_fraction'])
    chosen = flat.topk(count, sorted=False).indices
    start, grown = 0, []
    with torch.no_grad():
        for layer, mask, g in zip(model.layers, model.masks, sums):
            ids = chosen[(chosen>=start)&(chosen<start+mask.numel())]-start
            rms = layer.weight[mask].square().mean().sqrt() if mask.any() else torch.tensor(1e-3, device=runner.device)
            layer.weight.flatten()[ids] = -g.flatten()[ids].sign()*rms*cfg['rewire_initial_rms_fraction']
            mask.flatten()[ids] = True
            grown.append(int(len(ids)))
            start += mask.numel()
    return {'grown_per_layer': grown, 'score': 'task_gradient_squared_over_diagonal_gradient_squared', 'initialization': 'negative_gradient_sign_times_layer_rms_0.1'}


def run_upgrade(runner, source, method):
    cfg, folder = runner.cfg, runner.folder/method
    folder.mkdir(exist_ok=True)
    model, opt = new_plain(cfg, runner.device, source)
    events, current, best = [], source, source
    best_edges = sum(edge_counts(model)['nonzero_edges'])
    slots = cfg['parent_slots'] if method=='rewire' else 1
    for slot in range(1, slots+1):
        marker = folder/f'slot_{slot}_result.json'
        completed = folder/f'slot_{slot}'/'model_full.pt'
        if marker.exists():
            event = json.loads(marker.read_text(encoding='utf-8'))
            events.append(event)
            if event['accepted']:
                current = torch.load(completed, map_location='cpu', weights_only=False)['state']
                best = current
                best_edges = sum(int(v.sum()) for k,v in best['model'].items() if k.startswith('mask_'))
            continue
        model, opt = new_plain(cfg, runner.device, current)
        event = {'slot': slot, 'before_edges': best_edges}
        if method=='relearn':
            seed(cfg['model_seed'])
            for layer in model.layers:
                layer.reset_parameters()
            model.enforce()
            reset_parameters(opt, model)
            event['initialization'] = 'same_mask_all_weights_biases_reinitialized_seed11'
        else:
            try:
                event.update(grow_edges(model, runner))
            except TimeoutError:
                save_json(folder/'incomplete.json', {'slot':slot, 'status':'growth_budget_optimization_incomplete'})
                break
            reset_parameters(opt, model)
            path = folder/f'slot_{slot}_warmup.pt'
            warmcfg = {**cfg, 'maximum_epochs': 8, 'validation_epochs': [8]}
            warmrunner = copy.copy(runner)
            warmrunner.cfg = warmcfg
            try:
                warmrunner.adapt(model, opt, path, 30000+slot*100, True)
            except TimeoutError:
                save_json(folder/'incomplete.json', {'slot':slot, 'status':'warmup_budget_optimization_incomplete', 'resume':f'python -X utf8 -B -m tools.train_functional_edges --resume --method {method}'})
                break
            runner.ledger, runner.costs, runner.start, runner.deadline = warmrunner.ledger, warmrunner.costs, warmrunner.start, warmrunner.deadline
        start_state = capture(model, opt, [])
        nestedname = f'{method}/slot_{slot}'
        result = run_gate_arm(runner, start_state, nestedname, 1, current)
        candidate = torch.load(completed, map_location='cpu', weights_only=False)['state']
        event.update({'validation': result['validation'], 'structure': result['structure'], 'gate_events': result['events'], 'accepted': qualifies(result['validation'], cfg) and sum(result['structure']['nonzero_edges'])<best_edges})
        if event['accepted']:
            current, best, best_edges = candidate, candidate, sum(result['structure']['nonzero_edges'])
        events.append(event)
        save_json(marker, event)
        if time.perf_counter()>=runner.deadline:
            break
    model, opt = new_plain(cfg, runner.device, best)
    return runner.finish(method, model, opt, events, [], 'structure_slots_completed' if len(events)==slots else 'method_budget_optimization_incomplete', str((folder/'model_full.pt').relative_to(ROOT)))


def check(cfg):
    torch.set_num_threads(2)
    tiny = {**cfg, 'fc_widths': [3, 2, 2, 10], 'dropout': 0.}
    plain, wo = new_plain(tiny, torch.device('cpu'))
    gates = Gates(plain, tiny)
    go = torch.optim.Adam(gates.log_alpha.parameters(), lr=.01)
    x = torch.randn(6, 3)
    gates.train()
    gates(x).square().mean().backward()
    wo.step(); go.step(); wo.zero_grad(set_to_none=True); go.zero_grad(set_to_none=True)
    state = gate_capture(gates, wo, go)
    with torch.no_grad():
        gates.log_alpha[0][0] = -20
        plain.layers[0].weight.add_(2)
    gate_restore(gates, wo, go, state)
    assert tree_equal(state, gate_capture(gates, wo, go))
    assert torch.allclose(gates.probability(1), torch.full_like(gates.probability(1), .999), atol=.001)
    with torch.no_grad():
        gates.log_alpha[0][0] = -20
    hard, deleted = gates.harden()
    assert deleted[0] >= 1 and not hard.masks[0].flatten()[0]
    opt = torch.optim.AdamW(hard.parameters())
    with torch.no_grad():
        hard.masks[-1][:,1] = False
    hard.enforce()
    expected = hard.eval()(x).detach().clone()
    closure = closed_edges(hard, opt)
    assert sum(closure['removed_nonzero_per_layer'])>0
    torch.testing.assert_close(hard(x), expected, rtol=0, atol=0)
    assert torch.count_nonzero(hard.layers[0].weight[~hard.masks[0]])==0
    reset_parameters(wo, plain)
    assert not wo.state
    # 大权重可被相关输入承接；受限逆必须重新求，不是全逆切片。
    a = torch.linspace(-1, 1, 100, dtype=torch.float64)
    xx = torch.stack([a, a, torch.randn(100, dtype=torch.float64)], 1)
    center = xx-xx.mean(0)
    cov = center.T@center/len(xx)
    inv, damping, _ = stabilized_inverse(cov[:2,:2][None], cfg)
    w = torch.tensor([2., 3.], dtype=torch.float64)
    delta = torch.linalg.solve(cov[1:2,1:2]+damping[0]*torch.eye(1), cov[1:2,:1]@w[:1])
    naive = xx[:,1]*w[1]
    compensated = xx[:,1]*(w[1]+delta[0])
    parent = xx[:,:2]@w
    assert (compensated-parent).square().mean() < (naive-parent).square().mean()*1e-6
    fullinv, _, _ = stabilized_inverse(cov[None], cfg)
    assert not torch.allclose(inv, fullinv[:,:2,:2])
    (ROOT/'tmp').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as directory:
        path = Path(directory)/'resume.pt'
        atomic(path, {'state': state, 'controller': {'epoch': 32, 'mode': 'hard_branch'}})
        loaded = torch.load(path, weights_only=False)
        gate_restore(gates, wo, go, loaded['state'])
        assert tree_equal(state, gate_capture(gates, wo, go))
        assert loaded['controller']['mode']=='hard_branch'
    report = {'ok': True, 'device': 'CPU', 'dataset_access': False, 'new_gpu_check': False, 'large_weight_substitution': True, 'restricted_inverse_recomputed': True, 'gate_probability_and_hardening': True, 'dual_optimizer_rng_restore_exact': True, 'optimizer_full_parameter_reset': True, 'training_closure': True, 'resume_roundtrip': True, 'config_sha256': digest(ROOT/'configs/h1_functional_edges_v1.json'), 'source_sha256': digest(Path(__file__))}
    target = ROOT/'experiments/runs'/cfg['version']
    target.mkdir(parents=True, exist_ok=True)
    save_json(target/'check.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--method', choices=['compensation', 'gates', 'rewire', 'relearn', 'all'], default='all')
    args = parser.parse_args()
    cfgpath = ROOT/'configs/h1_functional_edges_v1.json'
    cfg = json.loads(cfgpath.read_text(encoding='utf-8'))
    if args.check:
        check(cfg)
        return
    assert cfg['model_seed']==11 and cfg['data_seed']==20261004 and cfg['fc_widths']==[784,3136,1568,128,10]
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda')
    if not torch.cuda.is_available():
        raise RuntimeError('完整实验使用既有GPU，不静默改CPU')
    torch.cuda.reset_peak_memory_stats()
    folder = ROOT/'experiments/runs'/cfg['version']/'mnist_circular'
    folder.mkdir(parents=True, exist_ok=True)
    if any(folder.iterdir()) and not args.resume:
        raise FileExistsError('已有运行证据，接续请--resume')
    reg = {'config': cfg, 'config_sha256': digest(cfgpath), 'source_sha256': digest(Path(__file__)), 'parent_sha256': digest(ROOT/cfg['source_checkpoint']), 'positions_sha256': digest(ROOT/cfg['source_positions']),
           'training_commit': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(), 'device_name': torch.cuda.get_device_name(), 'torch': torch.__version__, 'precision': 'FP32_weights_FP64_reconstruction', 'execution': 'dense_linear_zero_mask', 'cpu_threads': 2, 'official_test_accessed': False, 'historical_test_accessed': False}
    reg['reused_sha256'] = {p:digest(ROOT/p) for p in ('tools/train_adaptive_edges.py','tools/train_capacity_pruning.py','tools/train_capacity_baseline.py','tools/train_aligned_edges.py','src/braingrow/datasets.py')}
    if (folder/'registration.json').exists():
        old = json.loads((folder/'registration.json').read_text(encoding='utf-8'))
        for k in ('config_sha256','source_sha256','parent_sha256','positions_sha256','reused_sha256'):
            assert old[k]==reg[k], f'{k}变化，拒绝混源恢复'
    else:
        save_json(folder/'registration.json', reg)
    data = load_bundle('mnist', include_confirm=False)
    assert data.fit_x.shape==(3200,784) and data.select_x.shape==(800,784) and not len(data.confirm_x)
    oldpositions = np.load(ROOT/cfg['source_positions'])
    assert np.array_equal(oldpositions['train_original_ids'], data.manifest['splits']['fit'])
    assert np.array_equal(oldpositions['validation_original_ids'], data.manifest['splits']['select'])
    x,y = torch.from_numpy(data.fit_x).to(device), torch.from_numpy(data.fit_y).to(device)
    vx = shifts(torch.from_numpy(data.select_x).to(device).repeat_interleave(8,0), torch.from_numpy(oldpositions['validation_offsets'].reshape(-1,2)))
    vy = torch.from_numpy(data.select_y).to(device).repeat_interleave(8)
    cx = shifts(x, torch.from_numpy(oldpositions['calibration_offsets'][:,0]))
    rng = np.random.default_rng(cfg['data_seed']+6100)
    split = rng.permutation(3200)
    offsets = rng.integers(28, size=(3200,4,2), dtype=np.int64)
    n = cfg['reconstruction_fit_ids']
    fitids, heldids = split[:n], split[n:]
    fx = shifts(x[fitids].repeat_interleave(4,0), torch.from_numpy(offsets[fitids].reshape(-1,2)))
    hx = shifts(x[heldids].repeat_interleave(4,0), torch.from_numpy(offsets[heldids].reshape(-1,2)))
    fy,hy = y[fitids].repeat_interleave(4), y[heldids].repeat_interleave(4)
    if not (folder/'reconstruction_manifest.npz').exists():
        np.savez_compressed(folder/'reconstruction_manifest.npz', fit_original_ids=np.asarray(data.manifest['splits']['fit'])[fitids], diagnostic_original_ids=np.asarray(data.manifest['splits']['fit'])[heldids], offsets=offsets, partition_indices=split)
    save_json(folder/'data_manifest.json', {'original': data.manifest, 'reconstruction_manifest_sha256': digest(folder/'reconstruction_manifest.npz'), 'fit_original_ids':2560, 'diagnostic_original_ids':640, 'positions_per_id':4, 'validation_original_ids':800, 'validation_positions_per_id':8, 'test_accessed':False})
    source = torch.load(ROOT/cfg['source_checkpoint'], map_location='cpu', weights_only=False)['state']
    tensors = x,y,vx,vy,cx,fx,fy,hx,hy
    results = {}
    methods = ['compensation','gates','rewire','relearn'] if args.method=='all' else [args.method]
    for method in methods:
        if (folder/method/'result.json').exists() and method!='gates':
            results[method] = json.loads((folder/method/'result.json').read_text(encoding='utf-8'))
            continue
        if method in ('rewire','relearn') and args.method=='all' and any(r['structure']['nonzero_edges'][0]<=cfg['evaluation_first_layer_limit'] and qualifies(r['validation'],cfg) for r in results.values()):
            break
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        methodstart = started if not results else time.perf_counter()
        runner = Runner(cfg,folder,tensors,device,methodstart)
        if not results:
            runner.costs['data_setup_seconds'] = time.perf_counter()-started
        if method=='compensation':
            result = run_compensation(runner,source)
        elif method=='gates':
            if (folder/'gates'/'result.json').exists():
                result = json.loads((folder/'gates'/'result.json').read_text(encoding='utf-8'))
                runner.costs, runner.ledger = copy.deepcopy(result.get('method_cumulative_costs',result['costs'])), copy.deepcopy(result['ledger'])
                runner.start = time.perf_counter()-result.get('method_cumulative_wall_seconds',result['wall_seconds'])
                runner.deadline = runner.start+cfg['wall_seconds_per_method']
            else:
                result = run_gate_arm(runner,source,'gates',1)
            if not (folder/'gates_control'/'result.json').exists():
                control = run_gate_arm(runner,source,'gates_control',0)
            else:
                control = json.loads((folder/'gates_control'/'result.json').read_text(encoding='utf-8'))
            result['control'] = {k:control[k] for k in ('validation','structure','wall_seconds','events','rows')}
            result['paired_wall_seconds'] = max(result.get('method_cumulative_wall_seconds',result['wall_seconds']),control.get('method_cumulative_wall_seconds',control['wall_seconds']))
            result['paired_costs'] = copy.deepcopy(control.get('method_cumulative_costs',control['costs']))
            save_json(folder/'gates'/'result.json',result)
        else:
            eligible = [r for r in results.values() if qualifies(r['validation'],cfg)]
            if not eligible:
                eligible = [json.loads(p.read_text(encoding='utf-8')) for p in folder.glob('*/result.json') if p.parent.name!='gates_control']
            parentmethod = min(eligible,key=lambda r:sum(r['structure']['nonzero_edges']))['method'] if eligible else None
            parent = torch.load(folder/parentmethod/'model_full.pt', map_location='cpu', weights_only=False)['state'] if parentmethod else source
            result = run_upgrade(runner,parent,method)
        results[method] = result
        save_json(folder/'summary.json', {'results':results, 'end_to_end_wall_seconds':time.perf_counter()-started, 'status':'completed_registered_rounds', 'official_test_accessed':False, 'historical_test_accessed':False})
        print(json.dumps({'method_complete':method,'edges':result['structure']['nonzero_edges'],'accuracy':result['validation']['accuracy'],'ce':result['validation']['cross_entropy'],'stop':result['stop']}),flush=True)


if __name__=='__main__':
    main()
