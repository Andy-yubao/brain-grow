"""与真实A机制相关的状态、离散事件和逐步接口检查。"""
import pickle
import numpy as np
from braingrow.network.discrete import initial_network, DiscreteLearner, Window, sync_masks


def learner(n=8):
    net=initial_network(n)
    return DiscreteLearner(net,[(0,0,1)]*n,[1]*10+[1]*7,stable_ids=list(range(n)),next_id=n)


def test_step_trace_is_pre_reset_and_only_fire_transmits():
    net=initial_network(8);net.win[:]=0;net.rec[:]=0;net.out[:]=0
    net.bias[:]=0;net.alpha[:]=0;net.threshold[:]=1
    net.win[0,0]=.9;net.rec[0,1]=100;net.out[1,0]=2
    sync_masks(net);x=np.zeros((2,8,784),np.float32);x[:,:,0]=1
    p,a,t=net.forward(x,record_steps=True)
    assert t["membrane_before_reset"].shape==(2,32,8)
    assert np.all(t["membrane_before_reset"][:,:,0]==np.float32(.9))
    assert not t["spikes"].any() and not p.any()
    net.win[0,0]=2;sync_masks(net)
    p,a,t=net.forward(x,record_steps=True)
    assert t["membrane_before_reset"][0,0,0]==2
    assert t["spikes"][0,0,1]==0 and t["spikes"][0,1,1]==1
    np.testing.assert_allclose(p,t["spikes"].sum(axis=1)@net.out,atol=1e-5)
    np.testing.assert_array_equal(p,net.forward(x)[0])
    np.testing.assert_array_equal(p[:1],net.forward(x[:1])[0])


def test_window_other_and_id_coverage_and_restore():
    a=learner();w=Window(4);bits=np.ones(4,bool)
    a._add(w,-1,1,bits,bits,True);assert w.count.sum()==0
    for side in (0,1):
        for j in range(20):a._add(w,side,1,bits,bits,True)
    assert w.count.tolist()==[16,16] and not a._ready(w)
    w.ids=[{1,2,3},{4,5,6}];assert a._ready(w)
    before=pickle.dumps(a)
    rejected=a.clone();rejected.net.win[:]=0;rejected.hidden_windows[0].ids[0].add(99)
    rejected.rng.random();rejected.agendas[0]=(1,1,2);rejected.events.append({"rejected":True})
    assert pickle.dumps(a)==before
    restored=pickle.loads(before)
    assert restored.rng.random()==a.rng.random()


def test_all_edge_classes_form_delete_rebuild_and_birth_state():
    a=learner();rng=np.random.default_rng(11)
    # 使用真实事件表的合法反复证据，反复随机抽样直到得到状态迁移；不检查模型成绩。
    for kind,weak,strong in (("input",.5,1.),("internal",2.,4.),("output",.03,.10)):
        w=Window(200);weights=np.zeros(200,np.float32)
        w.count[:]=16;w.ids=[{1,2,3},{4,5,6}];w.source[0]=16;w.eligible[0]=16;w.previous_sign[:]=1
        a._transition(weights,w,weak,strong,rng,kind,0)
        assert (weights==weak).any()
        w.count[:]=16;w.ids=[{1,2,3},{4,5,6}];w.source[1]=16;w.eligible[1]=16;w.previous_sign[:]=-1
        a._transition(weights,w,weak,strong,rng,kind,0)
        assert (weights==-weak).any()
    n=a.net.n;a.birth((0,0,1),2)
    assert a.net.n==n+2 and a.net.valid()
    assert len(a.hidden_windows)==a.net.n and len(a.stable_ids)==a.net.n
    assert all(w.width==784+a.net.n for w in a.hidden_windows)
    assert all(w.width==a.net.n*5 for w in a.output_windows)
    assert a.hidden_windows[-1].windows==0
    # 三类实际边的删除/重建、既有节点更换来源。
    for weights,mask,index in ((a.net.win,a.net.input_mask,(0,0)),(a.net.rec,a.net.rec_mask,(1,0)),(a.net.out,a.net.output_mask,(0,0))):
        weights[index]=0;sync_masks(a.net);assert not mask[index]
        weights[index]=.5;sync_masks(a.net);assert mask[index]
