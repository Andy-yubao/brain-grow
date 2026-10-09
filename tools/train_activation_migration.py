"""固定物理节点的跨激活输出边迁移；独立入口，禁止读取CNN配额生成候选。"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np
import torch
from torch.nn import functional as F

from braingrow.datasets import load_bundle
from tools.train_capacity_baseline import ROOT, digest, evaluate, save_json
from tools.train_capacity_pruning import MaskedFC, capture, restore as _restore, export_model, sync
from tools.train_aligned_edges import seed, edge_counts
from tools.train_adaptive_edges import shifts, closed_edges, train_epoch, learning_rate, adaptation_status, tree_equal
from tools.train_functional_edges import Runner, atomic, new_plain as _new_plain, qualifies, reset_parameters, stabilized_inverse


def restore(model, optimizer, ema, saved):
    # optimizer.load_state_dict可复用CPU moment/step张量，必须隔离冻结父状态。
    _restore(model,optimizer,ema,copy.deepcopy(saved))


def new_plain(cfg, device, state=None):
    return _new_plain(cfg,device,None if state is None else copy.deepcopy(state))


class MigrationRunner(Runner):
    """分支每轮保存；搜索/对照成本不回退，只有接受主分支计入retained。"""
    def __init__(self, *args):
        super().__init__(*args)
        self.ledger.update(main_attempted_updates=0, control_attempted_updates=0,
                           retained_optimization_samples=0, proposal_forward_samples=0,
                           proposal_backward_samples=0, diagnostic_samples=0,
                           rejected_candidate_updates=0, teacher_samples=0)

    def budget(self):
        if time.perf_counter() >= self.deadline-10:
            raise TimeoutError('两小时预算剩余10秒供最终接受父图保存与评价')
        if self.device.type=='cuda' and torch.cuda.max_memory_allocated()>self.cfg['peak_memory_limit_bytes']:
            raise MemoryError('峰值显存超过8GiB')

    def adapt(self, model, optimizer, path, sequence, force48=False,
              control=False, stop_after_epochs=None, allowed_support=None):
        cfg = self.cfg
        epoch, rows, points, calibration = 0, [], [], []
        branch_started,prior_branch_wall=time.perf_counter(),0.
        if allowed_support is None:
            allowed_support = [m.detach().clone() for m in model.masks]
        if path.exists():
            saved = torch.load(path, map_location='cpu', weights_only=False)
            restore(model, optimizer, [], saved['state'])
            epoch, rows, points, calibration = [saved[k] for k in ('epoch', 'rows', 'points', 'calibration')]
            self.ledger, self.costs = saved['ledger'], saved['costs']
            self.start = time.perf_counter()-saved['method_wall_seconds']
            self.deadline = self.start+cfg['wall_seconds_per_method']
            prior_branch_wall=saved.get('branch_wall_seconds',0.)
            allowed_support = [m.to(self.device) for m in saved['allowed_support']]
            if saved['completed']:
                return saved['result']
        while epoch < cfg['maximum_epochs']:
            self.budget()
            epoch += 1
            t = time.perf_counter()
            closure_before=self.costs['closure_seconds']
            before_updates = self.ledger['attempted_updates']
            hook = getattr(self, 'epoch_hook', self.train_supported_epoch)
            self.allowed_support = allowed_support
            row = hook(model, optimizer, self.x, self.y, cfg, sequence+epoch,
                       learning_rate(epoch, True, cfg), self.ledger)
            sync(self.device)
            self.costs['training_seconds'] += time.perf_counter()-t-(self.costs['closure_seconds']-closure_before)
            key = 'control_attempted_updates' if control else 'main_attempted_updates'
            self.ledger[key] += self.ledger['attempted_updates']-before_updates
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
                improving = bool(points and (points[-1]['validation']['cross_entropy']-val['cross_entropy'] > cfg['validation_improvement_ce']
                    or val['accuracy']-points[-1]['validation']['accuracy'] > cfg['validation_improvement_accuracy']))
                if qualifies(val, cfg) and not force48:
                    status = 'accept'
                elif not force48 and training_status == 'plateau' and not improving:
                    status = 'completed_failure'
                elif epoch == cfg['maximum_epochs']:
                    status = 'control_complete' if force48 else 'optimization_incomplete'
                points.append({'epoch': epoch, 'validation': val, 'training_status': training_status,
                               'validation_improving': improving, 'decision': status})
                print(json.dumps({'branch':path.stem, 'epoch':epoch, 'accuracy':val['accuracy'],
                                  'ce':val['cross_entropy'], 'decision':status}), flush=True)
            result = {'status':status, 'epochs':epoch, 'rows':rows, 'validation_points':points,
                      'validation':points[-1]['validation'] if points else None,
                      'attempted_updates':epoch*math.ceil(len(self.x)/cfg['batch_size']),
                      'optimization_samples':epoch*len(self.x), 'control':control,
                      'wall_seconds':prior_branch_wall+time.perf_counter()-branch_started}
            self.save(path, {'state':capture(model, optimizer, []), 'epoch':epoch, 'rows':rows,
                'points':points, 'calibration':calibration, 'completed':status!='extend',
                'result':result, 'ledger':copy.deepcopy(self.ledger), 'costs':copy.deepcopy(self.costs),
                'allowed_support':[m.cpu().clone() for m in allowed_support],
                'branch_wall_seconds':result['wall_seconds'],
                'method_wall_seconds':time.perf_counter()-self.start})
            if stop_after_epochs is not None and epoch >= stop_after_epochs:
                raise InterruptedError('仅CPU检查使用的逐轮中断')
            if status != 'extend':
                return result

    def train_supported_epoch(self, model, optimizer, x, y, cfg, seed_epoch, lr, ledger):
        """reserve不计实际边；每步支持梯度→真实mask→闭合/Adam同步。"""
        torch.manual_seed(cfg['model_seed']+2000+seed_epoch)
        stream = torch.Generator(device=x.device).manual_seed(cfg['data_seed']+3000+seed_epoch)
        order = torch.randperm(len(x),device=x.device,generator=stream)
        offsets = torch.randint(28,(len(x),2),device=x.device,generator=stream)
        histogram=np.bincount((offsets[:,0]*28+offsets[:,1]).cpu().numpy(),minlength=784)
        ledger['training_position_histogram']=(np.asarray(ledger.get('training_position_histogram',[0]*784))+histogram).tolist()
        optimizer.param_groups[0]['lr']=lr
        model.train()
        total=0.
        for ids in order.split(cfg['batch_size']):
            optimizer.zero_grad(set_to_none=True)
            loss=F.cross_entropy(model(shifts(x[ids],offsets[ids])),y[ids])
            if not torch.isfinite(loss):
                raise FloatingPointError('训练损失非有限')
            loss.backward()
            for layer,allowed in zip(model.layers,self.allowed_support):
                layer.weight.grad.mul_(allowed)
            optimizer.step()
            with torch.no_grad():
                for layer,mask,allowed in zip(model.layers,model.masks,self.allowed_support):
                    layer.weight.mul_(allowed)
                    mask.copy_(allowed&(layer.weight!=0))
            self.close(model,optimizer)
            ledger['attempted_updates']+=1
            ledger['optimization_samples']+=len(ids)
            total+=float(loss.detach())*len(ids)
        optimizer.zero_grad(set_to_none=True)
        return {'hard_label_ce':total/len(x),'learning_rate':lr}

    def finish(self, method, model, optimizer, events, rows, stop, selected):
        closure=self.close(model,optimizer)
        val,logits=self.measure(model)
        t=time.perf_counter()
        fit,_=evaluate(model,self.cx,self.y,self.cfg['evaluation_batch_size'])
        self.costs['validation_seconds']+=time.perf_counter()-t
        self.ledger['evaluation_calls']+=1
        self.ledger['evaluation_samples']+=len(self.cx)
        t=time.perf_counter()
        structure={**export_model(model,self.folder/'model.bgm'),**edge_counts(model)}
        indegrees=[]
        for i,(layer,mask) in enumerate(zip(model.layers,model.masks)):
            degrees=((layer.weight!=0)&mask).sum(1).cpu().numpy()
            indegrees.append({'layer':i+1,'min':int(degrees.min()),'max':int(degrees.max()),
                'mean':float(degrees.mean()),'quantiles':np.quantile(degrees,[0,.1,.25,.5,.75,.9,1]).tolist(),
                'zero_input_nodes':int((degrees==0).sum())})
        np.savez_compressed(self.folder/'validation_final.npz',logits=logits,labels=self.vy.cpu().numpy())
        self.costs['save_seconds']+=time.perf_counter()-t
        self.save(self.folder/'model_full.pt',{'state':capture(model,optimizer,[]),'info':{
            'method':method,'validation':val,'selected_full_checkpoint':selected,'events':events}})
        return {'method':method,'validation':val,'training_fixed_one_position':fit,'structure':structure,
            'indegrees':indegrees,'events':events,'rows':rows,'costs':copy.deepcopy(self.costs),
            'ledger':copy.deepcopy(self.ledger),'wall_seconds':time.perf_counter()-self.start,
            'selected_full_checkpoint':selected,'stop':stop,'final_closure':closure,
            'full_checkpoint_sha256':digest(self.folder/'model_full.pt'),
            'peak_cuda_memory_bytes':torch.cuda.max_memory_allocated() if self.device.type=='cuda' else None,
            'official_test_accessed':False,'historical_test_accessed':False,'sealed_remaining_ids_accessed':False}


def context(model, x, y, runner, gradients=True):
    """post-ReLU激活、线性响应、真实支持输出可达；无dropout校准。"""
    model.eval()
    hs, zs = [[] for _ in model.layers[:-1]], [[] for _ in model.layers]
    sens = [torch.zeros(l.out_features, dtype=torch.float64, device=x.device) for l in model.layers]
    for bx, by in zip(x.split(runner.cfg['evaluation_batch_size']), y.split(runner.cfg['evaluation_batch_size'])):
        runner.budget()
        with torch.set_grad_enabled(gradients):
            h, batch_z = bx, []
            for i, layer in enumerate(model.layers):
                z = layer(h)
                batch_z.append(z)
                zs[i].append(z.detach())
                if i < len(model.layers)-1:
                    h = F.relu(z)
                    hs[i].append(h.detach())
                else:
                    h = z
            if gradients:
                grads = torch.autograd.grad(F.cross_entropy(h, by, reduction='sum'), batch_z)
                for i, g in enumerate(grads):
                    sens[i] += g.detach().double().square().sum(0)
        runner.ledger['proposal_forward_samples'] += len(bx)
        if gradients:
            runner.ledger['proposal_backward_samples'] += len(bx)
    response = [torch.cat(v) for v in zs]
    hidden = [torch.cat(v) for v in hs]
    reachable, outgoing = [], torch.ones(model.layers[-1].out_features, dtype=torch.bool, device=x.device)
    for layer, mask in reversed(list(zip(model.layers, model.masks))):
        incoming = ((layer.weight != 0) & mask & outgoing[:, None]).any(0)
        reachable.append(incoming)
        outgoing = incoming
    reachable.reverse()
    means, cov, active = [], [], []
    for i, h in enumerate(hidden):
        mu = h.double().mean(0)
        centered = h-mu.to(h.dtype)
        # 大激活协方差FP32；小组求解和迁移系数为FP64。
        cc = (centered.T@centered/len(h)).double()
        means.append(mu)
        cov.append(cc)
        active.append(((h>0).float().mean(0)>=runner.cfg['active_fraction_floor'])
                      & (cc.diagonal()>0) & reachable[i+1])
    return {'hidden':hidden, 'response':response, 'sensitivity':[v/len(x) for v in sens],
            'means':means, 'covariances':cov, 'active':active}


def groups_from_context(ctx, cfg):
    """同层相关激活组，层间轮转；不按节点幅度或输入RMS排列。"""
    available = [a.nonzero().flatten().tolist() for a in ctx['active']]
    groups = []
    while any(len(v)>1 for v in available) and len(groups)<cfg['proposal_group_cap']:
        for li, ids in enumerate(available):
            if len(ids)<2 or len(groups)>=cfg['proposal_group_cap']:
                continue
            anchor = ids[0]
            covariance = ctx['covariances'][li]
            options = torch.tensor(ids, device=covariance.device)
            denominator = (covariance[anchor,anchor]*covariance.diagonal()[options]).sqrt().clamp_min(1e-30)
            similarity = (covariance[anchor,options]/denominator).abs()
            order = similarity.argsort(descending=True, stable=True)
            chosen = options[order[:cfg['maximum_group_size']]].tolist()
            # 包括anchor；舍去已分组节点，保持组不重叠。
            if anchor not in chosen:
                chosen[-1] = anchor
            groups.append({'layer':li, 'nodes':chosen})
            selected = set(chosen)
            available[li] = [v for v in ids if v not in selected]
    return groups


def representatives(covariance, nodes, limit):
    """对标准化post-ReLU功能协方差做贪心独立列选择。"""
    indices = torch.tensor(nodes, device=covariance.device)
    matrix = covariance[indices[:, None], indices[None, :]]
    norms = matrix.diagonal().sqrt().clamp_min(1e-30)
    residual = matrix/(norms[:,None]*norms[None,:])
    chosen = []
    for _ in range(min(limit, len(nodes)-1)):
        diagonal = residual.diagonal().clone()
        if chosen:
            diagonal[chosen] = -float('inf')
        j = int(diagonal.argmax())
        if float(diagonal[j]) <= 1e-8:
            break
        chosen.append(j)
        column = residual[:,j].clone()
        residual = residual-column[:,None]*column[None,:]/diagonal[j]
        residual = (residual+residual.T)*.5
    return [nodes[j] for j in chosen]


def fit_migration(model, ctx, group, cfg):
    li, nodes = group['layer'], group['nodes']
    if not all(bool(ctx['active'][li][j]) for j in nodes):
        return None
    covariance, mu = ctx['covariances'][li], ctx['means'][li]
    keep = representatives(covariance, nodes, cfg['maximum_representatives'])
    removed = [j for j in nodes if j not in keep]
    if not keep or not removed:
        return None
    r = torch.tensor(keep, device=mu.device)
    d = torch.tensor(removed, device=mu.device)
    layer = model.layers[li+1]
    old = layer.weight.detach().double()
    contribution = old[:,d]
    rr = covariance[r[:,None],r[None,:]]
    rd = covariance[r[:,None],d[None,:]]
    dd = covariance[d[:,None],d[None,:]]
    inverse, damping, repairs = stabilized_inverse(rr[None], cfg)
    delta = (inverse[0]@rd@contribution.T).T
    bias_delta = contribution@mu[d]-delta@mu[r]
    residual = ((contribution@dd)*contribution).sum(1)+((delta@rr)*delta).sum(1)-2*((delta@rd)*contribution).sum(1)
    damage = float((residual.clamp_min(0)*ctx['sensitivity'][li+1]).sum())
    direct = int(((layer.weight[:,d]!=0)&model.masks[li+1][:,d]).sum())
    incoming = int(((model.layers[li].weight[d]!=0)&model.masks[li][d]).sum())
    prospective = (old[:,r]+delta).to(layer.weight.dtype)
    new = int(((prospective!=0)&~((layer.weight[:,r]!=0)&model.masks[li+1][:,r])).sum())
    predicted_net = direct+incoming-new
    return {**group, 'representatives':keep, 'removed':removed, 'delta':delta,
            'bias_delta':bias_delta, 'predicted_net':predicted_net, 'predicted_new':new,
            'local_weighted_damage':damage, 'score':damage/max(predicted_net,1),
            'damping':float(damping[0]), 'numerical_repairs':repairs}


def apply_migration(model, optimizer, fit, runner=None):
    before = edge_counts(model)
    zero_before=[int((~((l.weight!=0)&m).any(1)).sum()) for l,m in zip(model.layers,model.masks)]
    layer_index = fit['layer']+1
    layer, mask = model.layers[layer_index], model.masks[layer_index]
    r = torch.tensor(fit['representatives'], device=layer.weight.device)
    d = torch.tensor(fit['removed'], device=layer.weight.device)
    old_nonzero = (layer.weight!=0)&mask
    with torch.no_grad():
        layer.weight[:,d] = 0
        mask[:,d] = False
        layer.weight[:,r] += fit['delta'].to(layer.weight.dtype)
        layer.bias += fit['bias_delta'].to(layer.bias.dtype)
        mask[:,r] = layer.weight[:,r]!=0
    new_nonzero = (layer.weight!=0)&mask
    new = int((new_nonzero&~old_nonzero).sum())
    direct = int((old_nonzero&~new_nonzero).sum())
    closure = closed_edges(model,optimizer) if runner is None else runner.close(model,optimizer)
    after = edge_counts(model)
    closure_removed = sum(closure['removed_nonzero_per_layer'])
    net = sum(before['nonzero_edges'])-sum(after['nonzero_edges'])
    assert net == direct+closure_removed-new
    return {'before':before, 'after':after, 'direct_output_deleted':direct,
            'new_nonzero_output_edges':new, 'closure_deleted':closure_removed,
            'net_removed':net, 'closure':closure, 'physical_nodes_changed':False,
            'new_output_disconnected_nodes':len(fit['removed']),
            'new_zero_input_nodes_per_layer':[int((~((l.weight!=0)&m).any(1)).sum())-a
                for l,m,a in zip(model.layers,model.masks,zero_before)]}


def no_compensation_control(parent, candidate, cfg, device):
    """允许support独立保存；真实mask/闭合从结构操作起始终准确。"""
    model, optimizer = new_plain(cfg, device, parent)
    with torch.no_grad():
        for i, (layer, mask) in enumerate(zip(model.layers, model.masks)):
            allowed = candidate['model'][f'mask_{i}'].to(device)
            layer.weight.mul_(allowed)
            mask.copy_(allowed&(layer.weight!=0))
    reset_parameters(optimizer, model)
    closed_edges(model,optimizer)
    return model, optimizer


def actual_damage(ctx, reference):
    return sum(float(((z.double()-p.double()).square().mean(0)*a).sum())
               for z,p,a in zip(ctx['response'],reference['response'],reference['sensitivity']))


def source_context(state, cfg, runner, x=None, y=None, gradients=True):
    current = capture(runner.model, runner.optimizer, [])
    parent, opt = new_plain(cfg, runner.device, state)
    value = context(parent,runner.fx if x is None else x,runner.fy if y is None else y,runner,gradients)
    restore(runner.model, runner.optimizer, [], current)
    del parent, opt
    return value


def summarize_fit(fit):
    return {k:v for k,v in fit.items() if k not in ('delta','bias_delta')}


def proposal(runner, c, save):
    cfg, model, optimizer = runner.cfg, runner.model, runner.optimizer
    t = time.perf_counter()
    closure_start,save_start=runner.costs['closure_seconds'],runner.costs['save_seconds']
    reference = source_context(c['before'], cfg, runner)
    energy = sum(float((z.double().square().mean(0)*a).sum())
                 for z,a in zip(reference['response'],reference['sensitivity']))
    limit = cfg['damage_fraction']*energy
    guard_n = min(len(runner.fx), cfg['guard_original_ids']*cfg['reconstruction_positions'])
    if c['proposal'] is None:
        parent_ce = float(F.cross_entropy(reference['response'][-1][:guard_n], runner.fy[:guard_n]))
        c['proposal'] = {'pending':groups_from_context(reference,cfg), 'events':[], 'accepted_groups':0,
                         'parent_fit_ce':parent_ce, 'parent_response_energy':energy,
                         'damage_limit':limit, 'guard_samples':guard_n}
        save()
    p = c['proposal']
    while p['pending']:
        runner.budget()
        ctx = context(model,runner.fx,runner.fy,runner)
        fits = [fit_migration(model,ctx,g,cfg) for g in p['pending']]
        options = [(f['score'],i,f) for i,f in enumerate(fits) if f is not None and f['predicted_net']>0]
        if not options:
            p['events'].append({'status':'no_positive_net_group', 'remaining_groups':len(p['pending'])})
            p['pending'] = []
            break
        _, index, fit = min(options,key=lambda v:(v[0],v[1]))
        del p['pending'][index]
        before = capture(model,optimizer,[])
        group_started=time.perf_counter()
        event = summarize_fit(fit)
        event.update(apply_migration(model,optimizer,fit,runner))
        if event['net_removed']>0:
            changed = context(model,runner.fx,runner.fy,runner,gradients=False)
            damage = actual_damage(changed,reference)
            ce = float(F.cross_entropy(changed['response'][-1][:guard_n],runner.fy[:guard_n]))
            event.update(actual_parent_weighted_damage=damage,fit_guard_ce=ce)
            accepted = damage<=limit and ce<=p['parent_fit_ce']+cfg['fit_ce_increase']
        else:
            accepted = False
        event['accepted'] = accepted
        if accepted:
            p['accepted_groups'] += 1
        else:
            restore(model,optimizer,[],before)
            runner.ledger['rollbacks'] += 1
            runner.costs['failure_search_seconds']+=time.perf_counter()-group_started
        p['events'].append(event)
        runner.costs['solve_seconds'] += time.perf_counter()-t-(runner.costs['closure_seconds']-closure_start)-(runner.costs['save_seconds']-save_start)
        save()
        t = time.perf_counter()
        closure_start,save_start=runner.costs['closure_seconds'],runner.costs['save_seconds']
        print(json.dumps({'proposal_slot':c['slot'],'groups_checked':len(p['events']),
                          'accepted_groups':p['accepted_groups'],'net_edges':event['net_removed'],
                          'accepted':accepted}),flush=True)
    reset_parameters(optimizer,model)
    runner.close(model,optimizer)
    # 诊断仅报告，不参与排序、守门或适应的验证选择。
    diagnostic = context(model,runner.hx,runner.hy,runner,gradients=False)
    parent_diagnostic=source_context(c['before'],cfg,runner,runner.hx,runner.hy,gradients=False)
    p['diagnostic_ce'] = float(F.cross_entropy(diagnostic['response'][-1],runner.hy))
    p['parent_diagnostic_ce']=float(F.cross_entropy(parent_diagnostic['response'][-1],runner.hy))
    p['diagnostic_linear_response_mse']=[float((z.double()-old.double()).square().mean())
        for z,old in zip(diagnostic['response'],parent_diagnostic['response'])]
    runner.ledger['diagnostic_samples'] += len(runner.hx)
    p['candidate_structure'] = edge_counts(model)
    runner.costs['solve_seconds'] += time.perf_counter()-t-(runner.costs['closure_seconds']-closure_start)-(runner.costs['save_seconds']-save_start)
    return p['accepted_groups']>0


def execute(runner, source, resume=False):
    cfg, folder = runner.cfg, runner.folder
    progress = folder/'resume.pt'
    runner.model, runner.optimizer = new_plain(cfg,runner.device,source)
    c = {'slot':1,'mode':'new','before':None,'proposal':None,'candidate':None,'outcome':None,
         'best':source,'best_path':'--parent','events':[], 'controls':{}, 'first_pair':True,'stop':None}
    if progress.exists():
        if not resume:
            raise FileExistsError('已有证据，必须--resume')
        saved = torch.load(progress,map_location='cpu',weights_only=False)
        restore(runner.model,runner.optimizer,[],saved['state'])
        c, runner.ledger, runner.costs = saved['controller'],saved['ledger'],saved['costs']
        runner.start = time.perf_counter()-saved['method_wall_seconds']
        runner.deadline = runner.start+cfg['wall_seconds_per_method']

    def save():
        runner.save(progress,{'state':capture(runner.model,runner.optimizer,[]),
            'controller':copy.deepcopy(c),'ledger':copy.deepcopy(runner.ledger),
            'costs':copy.deepcopy(runner.costs),'method_wall_seconds':time.perf_counter()-runner.start})
        save_json(folder/'progress.json',{'slot':c['slot'],'mode':c['mode'],'stop':c['stop'],
            'events':c['events'],'controls':c['controls'],'ledger':runner.ledger,'costs':runner.costs,
            'wall_seconds':time.perf_counter()-runner.start})

    try:
        while c['slot']<=cfg['parent_slots'] and c['stop'] is None:
            runner.budget()
            if c['mode']=='new':
                runner.close(runner.model,runner.optimizer)
                c['before'] = capture(runner.model,runner.optimizer,[])
                c['proposal'] = None
                c['mode'] = 'proposal'
                save()
            if c['mode']=='proposal':
                if not proposal(runner,c,save):
                    restore(runner.model,runner.optimizer,[],c['before'])
                    runner.ledger['rollbacks'] += 1
                    c['stop'] = 'no_function_preserving_net_migration'
                    save()
                    break
                c['candidate'] = capture(runner.model,runner.optimizer,[])
                c['mode'] = 'adapt'
                save()
            if c['mode']=='adapt':
                path = folder/f"slot_{c['slot']}_adapt.pt"
                c['outcome'] = runner.adapt(runner.model,runner.optimizer,path,50000+c['slot']*100)
                c['candidate_trained'] = capture(runner.model,runner.optimizer,[])
                c['mode'] = 'control_no_compensation' if c['first_pair'] else 'decide'
                save()
            if c['mode']=='control_no_compensation':
                model, opt = no_compensation_control(c['before'],c['candidate'],cfg,runner.device)
                allowed_counts = edge_counts(model)
                allowed=[c['candidate']['model'][f'mask_{i}'].to(runner.device).clone() for i in range(len(model.layers))]
                outcome = runner.adapt(model,opt,folder/'control_no_compensation.pt',
                                      50000+c['slot']*100,True,control=True,allowed_support=allowed)
                closure = runner.close(model,opt)
                export_model(model,folder/'control_no_compensation.bgm')
                c['controls']['no_compensation'] = {**outcome,'initial_allowed_structure':allowed_counts,
                    'final_structure':edge_counts(model),'final_closure':closure,
                    'allowed_support_edges':[int(v.sum()) for v in allowed],
                    'independent_allowed_support_gradient_and_per_update_actual_closure':True}
                restore(runner.model,runner.optimizer,[],c['candidate_trained'])
                c['mode'] = 'control_no_delete'
                save()
            if c['mode']=='control_no_delete':
                model, opt = new_plain(cfg,runner.device,c['before'])
                reset_parameters(opt,model)
                outcome = runner.adapt(model,opt,folder/'control_no_delete.pt',
                                      50000+c['slot']*100,True,control=True)
                closure = runner.close(model,opt)
                export_model(model,folder/'control_no_delete.bgm')
                c['controls']['no_delete'] = {**outcome,'final_structure':edge_counts(model),'final_closure':closure}
                restore(runner.model,runner.optimizer,[],c['candidate_trained'])
                c['first_pair'] = False
                c['mode'] = 'decide'
                save()
            if c['mode']=='decide':
                closure = runner.close(runner.model,runner.optimizer)
                counts = edge_counts(runner.model)
                before_edges = sum(int(torch.count_nonzero(v)) for k,v in c['before']['model'].items() if k.endswith('.weight'))
                accepted = qualifies(c['outcome']['validation'],cfg) and sum(counts['nonzero_edges'])<before_edges
                event = {'slot':c['slot'],'proposal':c['proposal'],'adaptation':c['outcome'],
                         'structure':counts,'final_closure':closure,'accepted':accepted}
                c['events'].append(event)
                if accepted:
                    runner.ledger['net_updates'] += c['outcome']['attempted_updates']
                    runner.ledger['retained_optimization_samples'] += c['outcome']['optimization_samples']
                    c['best'] = capture(runner.model,runner.optimizer,[])
                    c['best_path'] = f"slot_{c['slot']}_accepted.pt"
                    runner.save(folder/c['best_path'],{'state':c['best'],'event':event})
                    c['slot'] += 1
                    c.update(mode='new',before=None,proposal=None,candidate=None,outcome=None)
                    c.pop('candidate_trained',None)
                else:
                    runner.ledger['rejected_candidate_updates'] += c['outcome']['attempted_updates']
                    runner.costs['failure_search_seconds']+=c['outcome']['wall_seconds']
                    restore(runner.model,runner.optimizer,[],c['before'])
                    runner.ledger['rollbacks'] += 1
                    c['stop'] = c['outcome']['status']
                save()
    except TimeoutError:
        c['stop'] = 'method_budget_optimization_incomplete'
        save()
    saved_stop = c['stop'] or 'parent_slots_completed'
    restore(runner.model,runner.optimizer,[],c['best'])
    # 验证及导出预留在两小时预算内；预算结束仅输出最后接受父图。
    result = runner.finish('activation_migration',runner.model,runner.optimizer,c['events'],[],saved_stop,c['best_path'])
    result['controls'] = c['controls']
    result['ledger_semantics'] = {'attempted_updates':'主候选和对照全部已执行更新',
        'net_updates':'最终接受主路径保留的更新，不含对照或拒绝候选',
        'teacher_samples':0,'prototype_comparison':'只比较主臂和对照共同已保存轮次'}
    result['cost_semantics']={'closure_seconds':'结构及每步实际图闭合独立计时，从training/solve中剔除',
        'failure_search_seconds':'重叠子集：被拒组应用/实际守门/回退及被拒主分支适应；不重复求和',
        'solve_seconds':'功能缓存/排序/拟合/守门/诊断，不含已单列闭合与保存',
        'wall_seconds':'唯一端到端总量；其余分项未覆盖主控制流快照等全部开销'}
    result['evaluation_first_layer_limit'] = cfg['evaluation_first_layer_limit']
    result['evaluation_first_layer_reached'] = qualifies(result['validation'],cfg) and result['structure']['nonzero_edges'][0]<=cfg['evaluation_first_layer_limit']
    save_json(folder/'result.json',result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent',type=Path)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--check',action='store_true')
    args = parser.parse_args()
    cfgpath = ROOT/'configs/h1_activation_migration_v1.json'
    cfg = json.loads(cfgpath.read_text(encoding='utf-8'))
    if args.check:
        check(cfg)
        return
    if args.parent is None:
        parser.error('正式实验必须--parent指定最小功能合格完整检查点')
    parent_path = args.parent.resolve()
    if not parent_path.is_absolute() or not parent_path.is_file():
        raise FileNotFoundError(parent_path)
    assert cfg['fc_widths']==[784,3136,1568,128,10] and cfg['model_seed']==11 and cfg['data_seed']==20261004
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('正式实验使用既有GPU，不静默改为CPU')
    device = torch.device('cuda')
    folder = ROOT/'experiments/runs'/cfg['version']/'mnist_circular'
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'result.json').exists():
        if not args.resume:
            raise FileExistsError('已有完成证据，禁止覆盖')
        print((folder/'result.json').read_text(encoding='utf-8'),flush=True)
        return
    if any(folder.iterdir()) and not args.resume:
        raise FileExistsError('已有证据，必须--resume')
    reused = ['tools/train_functional_edges.py','tools/train_adaptive_edges.py','tools/train_capacity_pruning.py',
              'tools/train_capacity_baseline.py','tools/train_aligned_edges.py','src/braingrow/datasets.py']
    manifest_path=ROOT/'experiments/runs/h1_functional_edges_v1/mnist_circular/reconstruction_manifest.npz'
    registration = {'config':cfg,'config_sha256':digest(cfgpath),'source_sha256':digest(Path(__file__)),
        'parent_path':str(parent_path),'parent_sha256':digest(parent_path),
        'position_sha256':digest(ROOT/cfg['source_positions']),
        'reconstruction_manifest_sha256':digest(manifest_path),
        'reused_sha256':{p:digest(ROOT/p) for p in reused},
        'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'device_name':torch.cuda.get_device_name(),'torch':torch.__version__,
        'precision':'FP32_tasks_and_large_activation_covariance_FP64_group_solve',
        'execution':'dense_linear_zero_mask','official_test_accessed':False,'historical_test_accessed':False}
    regpath = folder/'registration.json'
    if regpath.exists():
        old = json.loads(regpath.read_text(encoding='utf-8'))
        for key in ('config_sha256','source_sha256','parent_sha256','position_sha256','reconstruction_manifest_sha256','reused_sha256'):
            assert old[key]==registration[key], f'{key}改变，禁止混源恢复'
    else:
        save_json(regpath,registration)
    source = torch.load(parent_path,map_location='cpu',weights_only=False)['state']
    data = load_bundle('mnist',include_confirm=False)
    assert data.fit_x.shape==(3200,784) and data.select_x.shape==(800,784) and not len(data.confirm_x)
    positions = np.load(ROOT/cfg['source_positions'])
    assert np.array_equal(positions['train_original_ids'],data.manifest['splits']['fit'])
    assert np.array_equal(positions['validation_original_ids'],data.manifest['splits']['select'])
    x,y = torch.from_numpy(data.fit_x).to(device),torch.from_numpy(data.fit_y).to(device)
    vx = shifts(torch.from_numpy(data.select_x).to(device).repeat_interleave(8,0),torch.from_numpy(positions['validation_offsets'].reshape(-1,2)))
    vy = torch.from_numpy(data.select_y).to(device).repeat_interleave(8)
    cx = shifts(x,torch.from_numpy(positions['calibration_offsets'][:,0]))
    old_manifest=np.load(manifest_path)
    partition=old_manifest['partition_indices']
    offsets=old_manifest['offsets']
    fit, held = partition[:cfg['reconstruction_fit_ids']],partition[cfg['reconstruction_fit_ids']:]
    assert np.array_equal(old_manifest['fit_original_ids'],np.asarray(data.manifest['splits']['fit'])[fit])
    assert np.array_equal(old_manifest['diagnostic_original_ids'],np.asarray(data.manifest['splits']['fit'])[held])
    fx = shifts(x[fit].repeat_interleave(4,0),torch.from_numpy(offsets[fit].reshape(-1,2)))
    hx = shifts(x[held].repeat_interleave(4,0),torch.from_numpy(offsets[held].reshape(-1,2)))
    fy,hy = y[fit].repeat_interleave(4),y[held].repeat_interleave(4)
    np.savez_compressed(folder/'reconstruction_manifest.npz',partition_indices=partition,offsets=offsets,
        fit_original_ids=np.asarray(data.manifest['splits']['fit'])[fit],
        diagnostic_original_ids=np.asarray(data.manifest['splits']['fit'])[held])
    save_json(folder/'data_manifest.json',{'original':data.manifest,'fit_original_ids':len(fit),
        'diagnostic_original_ids':len(held),'validation_original_ids':800,'validation_positions_per_id':8,
        'source_reconstruction_manifest_sha256':digest(manifest_path),
        'historical_test_accessed':False,'official_test_accessed':False,'sealed_remaining_ids_accessed':False})
    torch.cuda.reset_peak_memory_stats()
    runner = MigrationRunner(cfg,folder,(x,y,vx,vy,cx,fx,fy,hx,hy),device,started)
    runner.costs['data_setup_seconds'] = time.perf_counter()-started
    model,opt = new_plain(cfg,device,source)
    val,_ = runner.measure(model)
    assert qualifies(val,cfg), '指定父模型不满足固定功能门槛'
    result = execute(runner,source,args.resume)
    print(json.dumps({'method_complete':'activation_migration','structure':result['structure'],
                      'validation':result['validation'],'stop':result['stop']}),flush=True)


def check(cfg):
    """一次无数据、无GPU的机制与逐轮恢复检查。"""
    torch.set_num_threads(2)
    device = torch.device('cpu')
    tiny = {**cfg,'fc_widths':[3,4,3,2,2],'dropout':0.,'batch_size':4,
            'maximum_epochs':2,'validation_epochs':[1,2],
            'accuracy_floor':0.,'cross_entropy_ceiling':100.,'maximum_representatives':1}
    model,opt = new_plain(tiny,device)
    with torch.no_grad():
        for layer,mask in zip(model.layers,model.masks):
            layer.weight.fill_(.1)
            layer.bias.fill_(.2)
            mask.fill_(True)
        model.layers[0].weight.zero_()
        model.layers[0].weight[0,0]=1
        model.layers[0].weight[1,0]=2
        model.layers[0].weight[2,0]=3
        model.layers[0].weight[3,1]=1
        model.layers[0].bias.copy_(torch.tensor([.1,.2,.3,.1]))
        model.layers[1].weight[:,2]=0
        model.layers[1].weight[0,2]=2
        model.layers[1].weight[0,0]=0
        model.layers[-1].weight[1].mul_(2)
        for layer,mask in zip(model.layers,model.masks):
            mask.copy_(layer.weight!=0)
    x = torch.linspace(.1,1.,24).reshape(8,3)
    y = torch.tensor([0,1]*4)
    model(x).square().mean().backward()
    opt.step()
    # 恢复明确的代数相关，保留非空Adam以检查完整回退。
    with torch.no_grad():
        model.layers[0].weight[1].copy_(2*model.layers[0].weight[0])
        model.layers[0].weight[2].copy_(3*model.layers[0].weight[0])
        model.layers[0].bias[1]=2*model.layers[0].bias[0]
        model.layers[0].bias[2]=3*model.layers[0].bias[0]
        model.layers[1].weight[0,0]=0
    model.enforce(opt)
    opt.zero_grad(set_to_none=True)
    parent = capture(model,opt,[])
    with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as directory:
        folder = Path(directory)
        tensors = (x,y,x,y,x,x,y,x,y)
        runner = MigrationRunner(tiny,folder,tensors,device,time.perf_counter())
        ctx = context(model,x,y,runner)
        assert bool(ctx['active'][0][2])
        fit = fit_migration(model,ctx,{'layer':0,'nodes':[0,2]},tiny)
        expected = model.eval()(x).detach().clone()
        nodes = edge_counts(model)['nodes']
        event = apply_migration(model,opt,fit)
        assert event['new_nonzero_output_edges']>0
        assert event['net_removed']==event['direct_output_deleted']+event['closure_deleted']-event['new_nonzero_output_edges']
        assert edge_counts(model)['nodes']==nodes
        assert not model.masks[0][fit['removed']].any()
        torch.testing.assert_close(model(x),expected,rtol=5e-4,atol=5e-4)
        candidate = capture(model,opt,[])
        control,copt = no_compensation_control(parent,candidate,tiny,device)
        allowed=[v.clone() for v in model.masks]
        added = candidate['model']['mask_1']&~parent['model']['mask_1']
        assert added.any() and not control.layers[1].weight[added].any()
        F.cross_entropy(control(x),y).backward()
        assert control.layers[1].weight.grad[added].abs().sum()>0
        copt.step()
        with torch.no_grad():
            for layer,mask,reserve in zip(control.layers,control.masks,allowed):
                layer.weight.mul_(reserve)
                mask.copy_(reserve&(layer.weight!=0))
        closed_edges(control,copt)
        assert control.layers[1].weight[added].abs().sum()>0
        restore(model,opt,[],parent)
        assert tree_equal(capture(model,opt,[]),parent)

        def toy_epoch(m,o,xx,yy,cc,epoch,lr,ledger):
            torch.manual_seed(cc['model_seed']+epoch)
            order = torch.randperm(len(xx))
            o.param_groups[0]['lr']=lr
            total=0.
            for ids in order.split(cc['batch_size']):
                o.zero_grad(set_to_none=True)
                loss=F.cross_entropy(m(xx[ids]),yy[ids])
                loss.backward()
                for layer,mask in zip(m.layers,m.masks):
                    layer.weight.grad.mul_(mask)
                o.step()
                m.enforce(o)
                ledger['attempted_updates']+=1
                ledger['optimization_samples']+=len(ids)
                total+=float(loss.detach())*len(ids)
            o.zero_grad(set_to_none=True)
            return {'hard_label_ce':total/len(xx),'learning_rate':lr}

        full_runner = MigrationRunner(tiny,folder,tensors,device,time.perf_counter())
        full_runner.epoch_hook=toy_epoch
        fm,fo=new_plain(tiny,device,parent)
        full_runner.adapt(fm,fo,folder/'full.pt',123,True,control=False)
        expected_state=capture(fm,fo,[])
        interrupted = MigrationRunner(tiny,folder,tensors,device,time.perf_counter())
        interrupted.epoch_hook=toy_epoch
        im,io=new_plain(tiny,device,parent)
        try:
            interrupted.adapt(im,io,folder/'resumed.pt',123,True,stop_after_epochs=1)
        except InterruptedError:
            pass
        interrupted.adapt(im,io,folder/'resumed.pt',123,True)
        assert tree_equal(capture(im,io,[]),expected_state)
        assert interrupted.ledger==full_runner.ledger
        assert interrupted.ledger['attempted_updates']==4
        assert interrupted.ledger['main_attempted_updates']==4
        assert interrupted.ledger['net_updates']==0 and interrupted.ledger['teacher_samples']==0
    report={'ok':True,'device':'CPU','dataset_access':False,'new_gpu_check':False,
        'post_relu_joint_output_compensation':True,'new_edge_net_accounting':True,
        'physical_nodes_fixed_and_training_closure':True,'same_allowed_support_zero_new_control_gradient':True,
        'reserve_independent_actual_mask_and_first_update_closure':True,
        'full_rollback_exact':True,'per_epoch_resume_exact_and_ledger_equal':True,
        'config_sha256':digest(ROOT/'configs/h1_activation_migration_v1.json'),
        'source_sha256':digest(Path(__file__))}
    target=ROOT/'experiments/runs'/cfg['version']
    target.mkdir(parents=True,exist_ok=True)
    save_json(target/'check.json',report)
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
