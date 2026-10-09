"""A机制：标签仅在真实前向之后控制离散事件迁移，无参数方向估计。"""
from __future__ import annotations
import copy
from dataclasses import dataclass, field
import numpy as np
from .growing import GrowingNetwork


def initial_network(n=128):
    rng = np.random.default_rng(11)
    wi = np.zeros((784, n), np.float32)
    rec = np.zeros((n, n), np.float32)
    out = np.zeros((n, 17), np.float32)
    for i in range(n):
        pixels = rng.choice(784, int(rng.integers(16, 65)), replace=False)
        wi[pixels, i] = rng.choice([-.5, .5, 1.], len(pixels), p=[.2, .4, .4])
        sources = rng.choice(np.delete(np.arange(n), i), 6, replace=False)
        rec[sources, i] = rng.choice([-2., 2.], 6)
        targets = rng.choice(17, 5, replace=False)
        out[i, targets] = rng.choice([-.03, .03], 5)
    return GrowingNetwork(wi, rec, out, wi != 0, rec != 0, out != 0,
        np.zeros(n, np.float32), np.exp(-5/rng.choice([10.,20.,40.,80.], n)).astype(np.float32),
        np.full(n, .5, np.float32))


def sync_masks(net):
    net.input_mask[:] = net.win != 0
    net.rec_mask[:] = net.rec != 0
    np.fill_diagonal(net.rec, 0); np.fill_diagonal(net.rec_mask, False)
    net.output_mask[:] = net.out != 0


@dataclass
class Window:
    width: int
    count: np.ndarray = field(default_factory=lambda: np.zeros(2, int))
    ids: list = field(default_factory=lambda: [set(), set()])
    source: np.ndarray | None = None
    eligible: np.ndarray | None = None
    hit: np.ndarray = field(default_factory=lambda: np.zeros(2, int))
    previous_sign: np.ndarray | None = None
    windows: int = 0
    age: np.ndarray | None = None

    def __post_init__(self):
        self.source = np.zeros((2, self.width), np.uint16)
        self.eligible = np.zeros((2, self.width), np.uint16)
        self.previous_sign = np.zeros(self.width, np.int8)
        self.age = np.zeros(self.width, np.uint16)

    def clear(self):
        self.count[:] = 0; self.ids = [set(), set()]
        self.source[:] = 0; self.eligible[:] = 0; self.hit[:] = 0


@dataclass
class DiscreteLearner:
    net: GrowingNetwork
    agendas: list
    output_competitors: list
    full: bool = True
    rng: object = field(default_factory=lambda: np.random.default_rng(np.random.SeedSequence([11,1])))
    output_rng: object = field(default_factory=lambda: np.random.default_rng(np.random.SeedSequence([11,2])))
    management_rng: object = field(default_factory=lambda: np.random.default_rng(np.random.SeedSequence([11,3])))
    next_id: int = 128
    stable_ids: list = field(default_factory=lambda: list(range(128)))
    hidden_windows: list = field(default_factory=list)
    output_windows: list = field(default_factory=list)
    events: list = field(default_factory=list)
    birth_batches: int = 0
    exposures: int = 0

    def __post_init__(self):
        if not self.hidden_windows:
            self.hidden_windows = [Window(784+self.net.n) for _ in range(self.net.n)]
        if not self.output_windows:
            self.output_windows = [Window(self.net.n*5) for _ in range(17)]

    def clone(self):
        return copy.deepcopy(self)

    @staticmethod
    def _side(label, agenda):
        g,c,r = agenda
        return 0 if label[g] == c else 1 if label[g] == r else -1

    def _add(self, window, side, raw_id, source, eligible, fired):
        if side < 0 or window.count[side] >= 16:
            return
        window.count[side] += 1; window.ids[side].add(int(raw_id))
        window.source[side] += source.astype(np.uint16)
        window.eligible[side] += eligible.astype(np.uint16)
        window.hit[side] += int(fired)

    @staticmethod
    def _ready(w):
        return np.all(w.count == 16) and all(len(ids) >= 3 for ids in w.ids)

    def _transition(self, weights, w, weak, strong, rng, kind, identifier, output=False):
        rates = w.source/16
        if output:
            difference = rates[0]-rates[1]
            sign = np.where(difference >= .25, 1, np.where(difference <= -.25, -1, 0)).astype(np.int8)
        else:
            sign = np.where((rates[0]>=.5)&(rates[1]<=.25), 1,
                np.where((rates[1]>=.5)&(rates[0]<=.25), -1, 0)).astype(np.int8)
            # 只有本单元近阈值/真实Fire资格可提名，源膜状态从未记录。
            sign *= (w.eligible.sum(axis=0)>0)
        repeated = (sign != 0)&(sign == w.previous_sign)
        if output:
            repeated = repeated.reshape(self.net.n, 5)
            signs = sign.reshape(self.net.n, 5)
            choices = np.argmax(repeated, axis=1)
            edge_sign = signs[np.arange(self.net.n), choices]
            qualified = repeated.any(axis=1)
        else:
            edge_sign, qualified = sign, repeated
        selected = qualified & (rng.random(len(weights)) < .1)
        old = weights.copy()
        opposite = selected & (np.sign(weights) != edge_sign) & (weights != 0)
        weights[opposite] = 0
        absent = selected & (old == 0)
        weights[absent] = weak*edge_sign[absent]
        reinforce = selected & (np.sign(old)==edge_sign) & (old != 0)
        weights[reinforce] = strong*edge_sign[reinforce]
        # 覆盖充分且连续两窗无侧偏的幼年连接：强退弱、弱退出；成熟需干预。
        if output:
            uninformative = (sign.reshape(self.net.n,5)==0).all(axis=1)&(w.previous_sign.reshape(self.net.n,5)==0).all(axis=1)
            age = w.age.reshape(self.net.n,5).max(axis=1)
        else:
            uninformative = (sign==0)&(w.previous_sign==0)
            age = w.age
        retire = uninformative & (age<2)&(rng.random(len(weights))<.01)&(old!=0)
        weights[retire] = np.where(np.abs(old[retire])>weak, weak*np.sign(old[retire]), 0)
        w.age[repeated.reshape(-1)] += 1
        w.previous_sign = sign.copy(); w.windows += 1
        self.events.append({"kind":kind,"id":identifier,"window":w.windows,
            "positive_ids":sorted(w.ids[0]),"negative_ids":sorted(w.ids[1]),
            "eligible_repeated":int(qualified.sum()),"formed":int(((old==0)&(weights!=0)).sum()),
            "deleted":int(((old!=0)&(weights==0)).sum()),"changed":int((old!=weights).sum()),
            "mature_records":int((w.age>=2).sum()),"positive_hits":int(w.hit[0]),"negative_hits":int(w.hit[1])})
        w.clear()

    def learn_batch(self, x, y, ids):
        # 不接受标签的前向先完整结束，标签只在下方管理记录中使用。
        p, activity, trace = self.net.forward(x, record_steps=True)
        spikes = trace["spikes"] > 0
        counts = spikes.sum(axis=1)
        near = (trace["membrane_before_reset"] >= .5*self.net.threshold)
        input_presence = (x > .1).any(axis=1)
        # 最近4步真实脉冲（严格过去，跨帧，视频开头补零）。
        arrival = np.zeros_like(spikes)
        for lag in range(1,5):
            arrival[:,lag:] |= spikes[:,:-lag]
        for row in range(len(x)):
            if self.full:
                for i,w in enumerate(self.hidden_windows):
                    side = self._side(y[row], self.agendas[i])
                    if side < 0 or w.count[side]>=16: continue
                    input_elig = ((x[row]>.1)&near[row].reshape(8,4,self.net.n).any(axis=1)[:,i,None]).any(axis=0)
                    rec_elig = (arrival[row]&near[row,:,i,None]).any(axis=0)
                    source = np.concatenate((input_presence[row], spikes[row].any(axis=0)))
                    eligible = np.concatenate((input_elig,rec_elig))
                    self._add(w,side,ids[row],source,eligible,counts[row,i]>0)
            for target,w in enumerate(self.output_windows):
                g,c = (0,target) if target<10 else (1,target-10)
                side = self._side(y[row],(g,c,self.output_competitors[target]))
                source = (counts[row,:,None]>=np.array([1,2,5,9,17])).reshape(-1)
                self._add(w,side,ids[row],source,source,True)
        if self.full:
            for i,w in enumerate(self.hidden_windows):
                if self._ready(w):
                    allweights = np.concatenate((self.net.win[:,i],self.net.rec[:,i]))
                    # 不同量纲分开调用，窗口计数仅在最后清空。
                    sub = copy.deepcopy(w); sub.width=784
                    for name in ("source","eligible"): setattr(sub,name,getattr(w,name)[:,:784].copy())
                    for name in ("previous_sign","age"): setattr(sub,name,getattr(w,name)[:784].copy())
                    self._transition(allweights[:784],sub,.5,1.,self.rng,"input",self.stable_ids[i])
                    w.previous_sign[:784]=sub.previous_sign; w.age[:784]=sub.age
                    sub = copy.deepcopy(w); sub.width=self.net.n
                    for name in ("source","eligible"): setattr(sub,name,getattr(w,name)[:,784:].copy())
                    for name in ("previous_sign","age"): setattr(sub,name,getattr(w,name)[784:].copy())
                    self._transition(allweights[784:],sub,2.,4.,self.rng,"internal",self.stable_ids[i])
                    w.previous_sign[784:]=sub.previous_sign; w.age[784:]=sub.age
                    self.net.win[:,i]=allweights[:784]; self.net.rec[:,i]=allweights[784:]
                    # 无标签兴奋性适应只依赖本单元活动；每个覆盖窗口一档。
                    rate = counts[:,i].mean()/32
                    if rate<.005: self.net.threshold[i]=max(.25,self.net.threshold[i]/2)
                    if rate>.6: self.net.threshold[i]=min(2.,self.net.threshold[i]*2)
                    # 无偏小比例形成或换源，所有现有单元可改变支撑集。
                    if self.rng.random()<.1:
                        source=int(self.rng.integers(784+self.net.n))
                        if source<784: self.net.win[source,i]=self.rng.choice([0.,-.5,.5])
                        elif source-784!=i: self.net.rec[source-784,i]=self.rng.choice([0.,-2.,2.])
                    w.windows+=1; w.clear()
        for target,w in enumerate(self.output_windows):
            if self._ready(w):
                self._transition(self.net.out[:,target],w,.03,.10,self.output_rng,"output",target,True)
        sync_masks(self.net); self.exposures+=len(x)
        return p,activity,trace

    def birth(self, agenda, number=4):
        if not self.full or self.birth_batches>=2: return
        # 新增全部可塑随机汇聚单元；不注入目标类别电流，不模板化空间结构。
        for _ in range(number):
            n=self.net.n; wi=np.zeros(784,np.float32); inc=np.zeros(n,np.float32); outgoing=np.zeros(n,np.float32)
            pixels=self.management_rng.choice(784,48,replace=False)
            wi[pixels]=self.management_rng.choice([-.5,.5,1.],48)
            sources=self.management_rng.choice(n,6,replace=False)
            inc[sources]=self.management_rng.choice([-2.,2.],6)
            targets=self.management_rng.choice(n,3,replace=False)
            outgoing[targets]=self.management_rng.choice([-2.,2.],3)
            output=np.zeros(17,np.float32)
            ot=self.management_rng.choice(17,5,replace=False); output[ot]=self.management_rng.choice([-.03,.03],5)
            self.net.add_node(wi,inc,outgoing,output,bias=0.,alpha=float(self.management_rng.choice(self.net.alpha)),threshold=.5)
            for w in self.hidden_windows:
                w.width+=1
                for name in ("source","eligible"): setattr(w,name,np.pad(getattr(w,name),((0,0),(0,1))))
                for name in ("previous_sign","age"): setattr(w,name,np.pad(getattr(w,name),(0,1)))
            self.hidden_windows.append(Window(784+self.net.n))
            for w in self.output_windows:
                w.width+=5
                for name in ("source","eligible"): setattr(w,name,np.pad(getattr(w,name),((0,0),(0,5))))
                for name in ("previous_sign","age"): setattr(w,name,np.pad(getattr(w,name),(0,5)))
            self.agendas.append(tuple(agenda)); self.stable_ids.append(self.next_id); self.next_id+=1
        self.birth_batches+=1
        self.events.append({"kind":"birth","batch":self.birth_batches,"number":number,"agenda":agenda,"exposures":self.exposures})
