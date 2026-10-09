"""B机制：独立完整图的有限多代离散发育、行为多样性保留。"""
from __future__ import annotations
import copy
from dataclasses import dataclass, field
import numpy as np
from .discrete import sync_masks


@dataclass(eq=False)
class CommunityCandidate:
    net: object
    proposal_id: int
    parent_id: int | None = None
    stable_ids: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    prediction: object = None
    score: dict = field(default_factory=dict)
    age: int = 0
    proposal_rng_state: dict | None = None

    def clone(self): return copy.deepcopy(self)


def add_random_node(candidate,rng,stable_id):
    net=candidate.net;n=net.n
    wi=np.zeros(784,np.float32);inc=np.zeros(n,np.float32);outg=np.zeros(n,np.float32);out=np.zeros(17,np.float32)
    pixels=rng.choice(784,int(rng.integers(16,65)),False);wi[pixels]=rng.choice([-.5,.5,1.],len(pixels))
    sources=rng.choice(n,6,False);inc[sources]=rng.choice([-4.,-2.,2.,4.],6)
    # 不回接母图，幼年子图内部可任意重连；只传递真正Fire。
    if n>128:
        targets=rng.choice(np.arange(128,n),min(3,n-128),False);outg[targets]=rng.choice([-2.,2.],len(targets))
    targets=rng.choice(17,5,False);out[targets]=rng.choice([-.03,.03],5)
    net.add_node(wi,inc,outg,out,bias=0.,alpha=float(np.exp(-5/rng.choice([10.,20.,40.,80.]))),threshold=float(rng.choice([.5,1.,2.])))
    candidate.stable_ids.append(int(stable_id))


def born_candidate(initial,proposal_id):
    rng=np.random.default_rng(np.random.SeedSequence([11,4,proposal_id]))
    c=CommunityCandidate(initial.clone(),proposal_id,stable_ids=list(range(128)))
    for j in range(8): add_random_node(c,rng,128+proposal_id*100+j)
    c.proposal_rng_state=copy.deepcopy(rng.bit_generator.state)
    assert c.net.valid()
    return c


def offspring(parent,proposal_id,full=True):
    c=parent.clone();c.parent_id=parent.proposal_id;c.proposal_id=proposal_id;c.age=parent.age+1
    c.prediction=None;c.score={};c.actions=[]
    rng=np.random.default_rng(np.random.SeedSequence([11,5,proposal_id]))
    for action_index in range(int(rng.integers(1,4))):
        kind=int(rng.integers(5));net=c.net
        # 仅输出臂每个提议也进行1–3个动作，全部转为实际输出编辑，预算更偏向输出。
        if not full: kind=3
        index=int(rng.integers(128,net.n));stable=c.stable_ids[index]
        action={"kind":("input","internal","dynamics","output","node")[kind],"node_id":stable}
        if kind==0:
            sources=rng.choice(784,int(rng.integers(4,17)),False)
            net.win[sources,index]=rng.choice([0.,-.5,.5,-1.,1.],len(sources))
            action["source_pixels"]=sources.tolist()
        elif kind==1:
            sources=rng.choice(np.delete(np.arange(net.n),index),min(8,net.n-1),False)
            net.rec[sources,index]=rng.choice([0.,-2.,2.,-4.,4.],len(sources))
            # 子图到子图已有节点可换源，不形成外部计算。
            action["source_ids"]=[c.stable_ids[j] for j in sources]
        elif kind==2:
            net.alpha[index]=np.exp(-5/rng.choice([10.,20.,40.,80.]))
            net.threshold[index]=rng.choice([.25,.5,1.,2.]);net.bias[index]=0.
        elif kind==3:
            targets=rng.choice(17,int(rng.integers(2,7)),False)
            net.out[index,targets]=rng.choice([0.,-.03,.03,-.10,.10],len(targets))
            action["targets"]=targets.tolist()
        else:
            if net.n>132 and rng.random()<.5:
                net.delete_node(index);c.stable_ids.pop(index);action["deleted"]=stable
            else:
                new_id=100000+proposal_id*10+action_index
                add_random_node(c,rng,new_id);action["born"]=new_id
        c.actions.append(action)
    sync_masks(c.net)
    c.proposal_rng_state=copy.deepcopy(rng.bit_generator.state)
    assert c.net.valid() and len(c.stable_ids)==c.net.n
    return c


def behavior_score(candidate,baseline_prediction,y,ids,agenda):
    p=candidate.prediction;g,target,r=agenda
    target_error=(y[:,g]==target)&(baseline_prediction[:,g]==r)
    repaired=target_error&(p[:,g]==target)
    harmed=(baseline_prediction==y)&(p!=y)
    fixed=(baseline_prediction!=y)&(p==y)
    candidate.score={"target_repairs":int(repaired.sum()),"damage":int(harmed.sum()),
        "total_repairs":int(fixed.sum()),"repair_ids":sorted(set(ids[repaired].tolist())),
        "signature":sorted(set(np.flatnonzero(repaired).tolist()))}
    return candidate.score


def select_candidates(candidates,rng):
    # 4个任务/保持非支配或行为多样性槽，2个无资源奖励探索槽。
    nondominated=[]
    for c in candidates:
        s=c.score
        dominated=any(o.score["target_repairs"]>=s["target_repairs"] and o.score["damage"]<=s["damage"] and
            (o.score["target_repairs"]>s["target_repairs"] or o.score["damage"]<s["damage"]) for o in candidates if o is not c)
        if not dominated:nondominated.append(c)
    ordered=sorted(nondominated,key=lambda c:(-c.score["target_repairs"],c.score["damage"],-c.score["total_repairs"],c.proposal_id))
    remainder=sorted([c for c in candidates if c not in nondominated],key=lambda c:(-c.score["target_repairs"],c.score["damage"],c.proposal_id))
    selected=[];covered=set()
    pool=ordered+remainder
    while pool and len(selected)<4:
        best=min(pool,key=lambda c:(-len(set(c.score["signature"])-covered),-c.score["target_repairs"],c.score["damage"],c.proposal_id))
        selected.append(best);covered.update(best.score["signature"]);pool=[c for c in pool if c is not best]
    if pool:
        for j in rng.choice(len(pool),min(2,len(pool)),False):selected.append(pool[int(j)])
    return selected
