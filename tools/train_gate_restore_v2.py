"""仅修复冻结检查点恢复别名；独立重跑原B及零数量代价对照。"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from tools import train_functional_edges as m

LEGACY_RESTORE = m.restore
LEGACY_GATE_RESTORE = m.gate_restore


def isolated_restore(model, optimizer, ema, saved):
    # Adam的CPU step和CPU训练的moment不得复用冻结父材料。
    LEGACY_RESTORE(model, optimizer, ema, copy.deepcopy(saved))


def isolated_gate_restore(model, weight_optimizer, gate_optimizer, saved):
    LEGACY_GATE_RESTORE(model, weight_optimizer, gate_optimizer, copy.deepcopy(saved))


def install():
    m.restore = isolated_restore
    m.gate_restore = isolated_gate_restore


def update_once(model, optimizer):
    optimizer.zero_grad(set_to_none=True)
    model(torch.randn(5, model.layers[0].in_features)).square().mean().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def check(cfg):
    torch.set_num_threads(2)
    tiny = {**cfg, 'fc_widths':[3,2,2,10], 'dropout':0.}
    model,opt = m.new_plain(tiny,torch.device('cpu'))
    update_once(model,opt)
    frozen = m.capture(model,opt,[])
    original = copy.deepcopy(frozen)
    legacy_model,legacy_opt = m.new_plain(tiny,torch.device('cpu'))
    LEGACY_RESTORE(legacy_model,legacy_opt,[],frozen)
    aliased_step = any(legacy_opt.state[p]['step'].data_ptr()==frozen['optimizer']['state'][i]['step'].data_ptr() for i,p in enumerate(legacy_model.parameters()))
    update_once(legacy_model,legacy_opt)
    legacy_mutates_parent = not m.tree_equal(frozen,original)
    assert aliased_step and legacy_mutates_parent
    install()
    saved = copy.deepcopy(original)
    first,first_opt = m.new_plain(tiny,torch.device('cpu'),saved)
    before_first = m.capture(first,first_opt,[])
    update_once(first,first_opt)
    assert m.tree_equal(saved,original)
    second,second_opt = m.new_plain(tiny,torch.device('cpu'),saved)
    assert m.tree_equal(before_first,m.capture(second,second_opt,[]))
    update_once(second,second_opt)
    assert m.tree_equal(saved,original)
    gated = m.Gates(second,tiny)
    gate_opt = torch.optim.Adam(gated.log_alpha.parameters(),lr=.01)
    second_opt.zero_grad(set_to_none=True)
    gated.train()(torch.randn(5,3)).square().mean().backward()
    second_opt.step();gate_opt.step()
    second_opt.zero_grad(set_to_none=True);gate_opt.zero_grad(set_to_none=True)
    joint = m.gate_capture(gated,second_opt,gate_opt)
    immutable_joint = copy.deepcopy(joint)
    isolated_gate_restore(gated,second_opt,gate_opt,joint)
    assert m.tree_equal(m.gate_capture(gated,second_opt,gate_opt),joint)
    gated(torch.randn(5,3)).square().mean().backward()
    second_opt.step();gate_opt.step()
    second_opt.zero_grad(set_to_none=True);gate_opt.zero_grad(set_to_none=True)
    assert m.tree_equal(joint,immutable_joint)
    isolated_gate_restore(gated,second_opt,gate_opt,joint)
    assert m.tree_equal(m.gate_capture(gated,second_opt,gate_opt),joint)
    report = {'ok':True,'device':'CPU','new_gpu_check':False,'dataset_access':False,
              'legacy_step_alias_reproduced':aliased_step,'legacy_parent_mutation_reproduced':legacy_mutates_parent,
              'isolated_parent_optimizer_and_rng_immutable':True,'paired_initial_state_exact':True,
              'dual_optimizer_frozen_checkpoint_immutable_after_updates':True,'joint_rng_restore_exact':True,
              'source_sha256':m.digest(Path(__file__)),'config_sha256':m.digest(m.ROOT/'configs/h1_gate_restore_v2.json')}
    folder = m.ROOT/'experiments/runs'/cfg['version']
    folder.mkdir(parents=True,exist_ok=True)
    m.save_json(folder/'check.json',report)
    print(json.dumps(report,ensure_ascii=False),flush=True)


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    cfgpath = m.ROOT/'configs/h1_gate_restore_v2.json'
    cfg = json.loads(cfgpath.read_text(encoding='utf-8'))
    original_cfg = json.loads((m.ROOT/'configs/h1_functional_edges_v1.json').read_text(encoding='utf-8'))
    assert {k:v for k,v in cfg.items() if k!='version'}=={k:v for k,v in original_cfg.items() if k!='version'},'本修复不得改变协议数值'
    if args.check:
        check(cfg)
        return
    install()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available():
        raise RuntimeError('正式B配对实验使用原GPU')
    device = torch.device('cuda')
    folder = m.ROOT/'experiments/runs'/cfg['version']/'mnist_circular'
    folder.mkdir(parents=True,exist_ok=True)
    if any(folder.iterdir()) and not args.resume:
        raise FileExistsError('已有证据，恢复请--resume')
    registration = {'config':cfg,'config_sha256':m.digest(cfgpath),'source_sha256':m.digest(Path(__file__)),
                    'reused_sha256':m.digest(Path(m.__file__)),'parent_sha256':m.digest(m.ROOT/cfg['source_checkpoint']),
                    'positions_sha256':m.digest(m.ROOT/cfg['source_positions']),
                    'training_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=m.ROOT,text=True).strip(),
                    'device_name':torch.cuda.get_device_name(),'torch':torch.__version__,'precision':'FP32',
                    'execution':'dense_linear_zero_mask','change':'deepcopy_optimizer_checkpoint_material_before_load_only',
                    'official_test_accessed':False,'historical_test_accessed':False,'sealed_remaining_ids_accessed':False}
    if (folder/'registration.json').exists():
        previous = json.loads((folder/'registration.json').read_text(encoding='utf-8'))
        for key in ('config_sha256','source_sha256','reused_sha256','parent_sha256','positions_sha256'):
            assert previous[key]==registration[key],'禁止混源恢复'
    else:
        m.save_json(folder/'registration.json',registration)
    data = m.load_bundle('mnist',include_confirm=False)
    assert data.fit_x.shape==(3200,784) and data.select_x.shape==(800,784) and not len(data.confirm_x)
    pos = np.load(m.ROOT/cfg['source_positions'])
    assert np.array_equal(pos['train_original_ids'],data.manifest['splits']['fit'])
    assert np.array_equal(pos['validation_original_ids'],data.manifest['splits']['select'])
    x,y = torch.from_numpy(data.fit_x).to(device),torch.from_numpy(data.fit_y).to(device)
    vx = m.shifts(torch.from_numpy(data.select_x).to(device).repeat_interleave(8,0),torch.from_numpy(pos['validation_offsets'].reshape(-1,2)))
    vy = torch.from_numpy(data.select_y).to(device).repeat_interleave(8)
    cx = m.shifts(x,torch.from_numpy(pos['calibration_offsets'][:,0]))
    source = torch.load(m.ROOT/cfg['source_checkpoint'],map_location='cpu',weights_only=False)['state']
    frozen_source = copy.deepcopy(source)
    runner = m.Runner(cfg,folder,(x,y,vx,vy,cx,x,y,x,y),device,started)
    runner.costs['data_setup_seconds']=time.perf_counter()-started
    results = {}
    for name,pressure in [('gates',1),('gates_control',0)]:
        if (folder/name/'result.json').exists():
            result = json.loads((folder/name/'result.json').read_text(encoding='utf-8'))
            runner.ledger = copy.deepcopy(result.get('method_cumulative_ledger',result['ledger']))
            runner.costs = copy.deepcopy(result.get('method_cumulative_costs',result['costs']))
            runner.start=time.perf_counter()-result.get('method_cumulative_wall_seconds',result['wall_seconds'])
            runner.deadline=runner.start+cfg['wall_seconds_per_method']
        else:
            probe,opt = m.new_plain(cfg,device,source)
            assert m.tree_equal(m.cpu_tree(opt.state_dict()),source['optimizer'])
            del probe,opt
            result = m.run_gate_arm(runner,source,name,pressure)
        assert m.tree_equal(source,frozen_source),'训练不得污染冻结父材料'
        result['frozen_parent_material_immutable_after_training']=True
        results[name]=result
        m.save_json(folder/name/'result.json',result)
        print(json.dumps({'arm_complete':name,'actual_edges':result['structure']['nonzero_edges'],'source_material_immutable':True}),flush=True)
    summary = {'results':results,'paired_wall_seconds':time.perf_counter()-runner.start,
               'end_to_end_wall_seconds':time.perf_counter()-started,'paired_costs':runner.costs,
               'optimizer_restore_isolated':True,'frozen_parent_material_immutable_after_both_arms':True,
               'all_protocol_values_unchanged':True,'official_test_accessed':False,'historical_test_accessed':False}
    m.save_json(folder/'summary.json',summary)
    print(json.dumps({'gate':'completed','paired_wall_seconds':summary['paired_wall_seconds']}),flush=True)


if __name__=='__main__':
    main()
