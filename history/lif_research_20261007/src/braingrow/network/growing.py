"""纯LIF隐藏图、仅脉冲通信、无泄漏/无发放的固定电位累积输出。"""
from __future__ import annotations
import copy
from dataclasses import dataclass
import numpy as np


@dataclass
class GrowingNetwork:
    win: np.ndarray
    rec: np.ndarray
    out: np.ndarray
    input_mask: np.ndarray
    rec_mask: np.ndarray
    output_mask: np.ndarray
    bias: np.ndarray
    alpha: np.ndarray
    threshold: np.ndarray
    steps_per_frame: int = 4
    max_nodes: int | None = None
    max_edges: int | None = None
    # 工程内存保护，不是研究规模预算；超保护直接报错而非缩网。
    allocation_guard_bytes: int = 512 * 1024 * 1024

    @classmethod
    def create(cls, seed=11, n=24):
        if seed != 11:
            raise ValueError("早期模型种子固定11")
        rng = np.random.default_rng(seed)
        im = rng.random((784, n)) < .08
        rm = rng.random((n, n)) < .10
        np.fill_diagonal(rm, False)
        om = rng.random((n, 17)) < .30
        return cls((rng.normal(.12, .9, im.shape)*im).astype(np.float32),
                   (rng.normal(0, .8, rm.shape)*rm).astype(np.float32),
                   (rng.normal(0, .12, om.shape)*om).astype(np.float32), im, rm, om,
                   rng.uniform(.25, .7, n).astype(np.float32),
                   np.exp(-5/rng.choice([10., 20., 40., 80.], n)).astype(np.float32),
                   np.ones(n, np.float32))

    @property
    def n(self):
        return len(self.bias)

    def clone(self):
        return copy.deepcopy(self)

    def resources(self):
        inputs, internal, outputs = [int(m.sum()) for m in (self.input_mask, self.rec_mask, self.output_mask)]
        maximum = self.n*(self.n-1)
        allocated = sum(a.nbytes for a in (self.win, self.rec, self.out, self.input_mask, self.rec_mask,
                                           self.output_mask, self.bias, self.alpha, self.threshold))
        return {"hidden_lif": self.n, "output_lif": 0, "total_lif": self.n,
                "output_accumulators": 17, "total_units": self.n+17, "input_dimensions": 784,
                "input_edges": inputs, "hidden_graph_edges": internal,
                "hidden_graph_density": internal/maximum if maximum else 0.,
                "hidden_graph_max_edges": maximum, "output_edges": outputs,
                "output_density": outputs/(17*self.n), "input_density": inputs/(784*self.n),
                "effective_weight_slots": inputs+internal+outputs, "external_classifier_parameters": 0,
                "allocated_array_bytes": allocated, "model_max_nodes": self.max_nodes,
                "model_max_edges": self.max_edges, "allocation_guard_bytes": self.allocation_guard_bytes}

    def valid(self):
        r = self.resources()
        if r["allocated_array_bytes"] > self.allocation_guard_bytes:
            raise MemoryError("达到明确的工程数组内存保护")
        return (self.n >= 1 and self.win.shape == self.input_mask.shape == (784, self.n)
                and self.rec.shape == self.rec_mask.shape == (self.n, self.n)
                and self.out.shape == self.output_mask.shape == (self.n, 17)
                and self.alpha.shape == self.threshold.shape == (self.n,)
                and 0 < self.steps_per_frame and np.all((self.alpha >= 0) & (self.alpha < 1))
                and np.all(self.threshold > 0) and not np.diag(self.rec_mask).any()
                and all(np.isfinite(a).all() for a in (self.win, self.rec, self.out, self.bias, self.alpha, self.threshold))
                and all(not a[~m].any() for a,m in ((self.win,self.input_mask),(self.rec,self.rec_mask),(self.out,self.output_mask)))
                and (self.max_nodes is None or self.n <= self.max_nodes)
                and (self.max_edges is None or r["effective_weight_slots"] <= self.max_edges))

    def forward(self, x, trace=False, record_steps=False):
        if x.ndim != 3 or x.shape[-1] != 784 or not np.isfinite(x).all():
            raise ValueError("输入必须是有限的batch×frames×784原像素")
        # 原始灰度直接作为输入电流；没有卷积/空间聚合/运动特征。
        current = x @ self.win + self.bias
        v = np.zeros((len(x), self.n), np.float32)
        previous = np.zeros_like(v)
        potential = np.zeros((len(x), 17), np.float32)
        frame_spikes = np.zeros((len(x), x.shape[1], self.n), np.float32)
        step_records = []
        membrane_records = []
        input_scale = float(np.abs(current).mean())
        recurrent_scale = 0.
        for frame in range(x.shape[1]):
            for _ in range(self.steps_per_frame):
                recurrent = previous @ self.rec
                recurrent_scale += float(np.abs(recurrent).mean())
                v = self.alpha*v + (1-self.alpha)*(current[:, frame] + recurrent)
                spikes = (v >= self.threshold).astype(np.float32)
                if record_steps:
                    step_records.append(spikes.copy())
                    membrane_records.append(v.copy())
                v = np.where(spikes > 0, 0., v).astype(np.float32)
                # 输出只积分实际隐藏脉冲，没有阈值、复位、泄漏或bonus。
                potential += spikes @ self.out
                frame_spikes[:, frame] += spikes
                previous = spikes
        total = float(frame_spikes.sum())
        count = len(x)
        activity = {"hidden_spikes": int(total), "output_spikes": 0, "total_spikes": int(total),
                    "mean_spikes_per_image": total/(count*x.shape[1]) if count else 0.,
                    "mean_spikes_per_video": total/count if count else 0.,
                    "spikes_per_lif_step": total/(count*x.shape[1]*self.steps_per_frame*self.n) if count else 0.,
                    "silent_hidden_units": int((frame_spikes.sum(axis=(0,1)) == 0).sum()),
                    "frames": x.shape[1], "steps_per_frame": self.steps_per_frame,
                    "step_ms": 5, "state_boundary": "all_states_zero_at_each_video",
                    "counted_units": "hidden_LIF_only_output_accumulators_do_not_fire"}
        if record_steps:
            return potential, activity, {"frame_spikes": frame_spikes,
                "spikes": np.stack(step_records, axis=1),
                "membrane_before_reset": np.stack(membrane_records, axis=1),
                "mean_absolute_input_current": input_scale,
                "mean_absolute_recurrent_current": recurrent_scale/(x.shape[1]*self.steps_per_frame)}
        return potential, activity, frame_spikes if trace else None

    def add_node(self, win, incoming, outgoing, output, bias=.5, alpha=.8, threshold=1., masks=None):
        n = self.n
        self.win = np.column_stack((self.win, win)).astype(np.float32)
        new_rec = np.zeros((n+1, n+1), np.float32)
        new_rec[:n, :n] = self.rec
        new_rec[:n, n], new_rec[n, :n] = incoming, outgoing
        self.rec = new_rec
        self.out = np.vstack((self.out, output)).astype(np.float32)
        im, inc, outg, om = masks if masks is not None else (np.asarray(win)!=0, np.asarray(incoming)!=0,
                                                          np.asarray(outgoing)!=0, np.asarray(output)!=0)
        self.input_mask = np.column_stack((self.input_mask, im))
        rm = np.zeros((n+1, n+1), bool)
        rm[:n,:n] = self.rec_mask
        rm[:n,n], rm[n,:n] = inc, outg
        self.rec_mask = rm
        self.output_mask = np.vstack((self.output_mask, om))
        self.bias = np.append(self.bias, bias).astype(np.float32)
        self.alpha = np.append(self.alpha, alpha).astype(np.float32)
        self.threshold = np.append(self.threshold, threshold).astype(np.float32)

    def split_node(self, index):
        """复制动力学并平分所有出边，未分化前保持逐步计算（浮点误差除外）。"""
        outgoing, output = self.rec[index].copy()/2, self.out[index].copy()/2
        incoming = self.rec[:,index].copy()
        masks = (self.input_mask[:,index].copy(), self.rec_mask[:,index].copy(),
                 self.rec_mask[index].copy(), self.output_mask[index].copy())
        self.rec[index] = outgoing
        self.out[index] = output
        self.add_node(self.win[:,index].copy(), incoming, outgoing, output,
                      self.bias[index], self.alpha[index], self.threshold[index], masks)

    def delete_node(self, index):
        if not 0 <= index < self.n or self.n <= 1:
            raise ValueError("不能删除固定类别输出或最后一个隐藏节点")
        for name in ("win", "input_mask"):
            setattr(self, name, np.delete(getattr(self,name), index, axis=1))
        for name in ("rec", "rec_mask"):
            setattr(self, name, np.delete(np.delete(getattr(self,name),index,axis=0),index,axis=1))
        for name in ("out", "output_mask", "bias", "alpha", "threshold"):
            setattr(self, name, np.delete(getattr(self,name),index,axis=0))

    def save(self, path):
        scalars = {"steps_per_frame": self.steps_per_frame, "max_nodes": -1 if self.max_nodes is None else self.max_nodes,
                   "max_edges": -1 if self.max_edges is None else self.max_edges,
                   "allocation_guard_bytes": self.allocation_guard_bytes}
        np.savez_compressed(path, **scalars, **{k:getattr(self,k) for k in
            ("win","rec","out","input_mask","rec_mask","output_mask","bias","alpha","threshold")})

    @classmethod
    def load(cls, path):
        with np.load(path) as z:
            args = {k:z[k].copy() for k in z.files}
        for k in ("steps_per_frame","max_nodes","max_edges","allocation_guard_bytes"):
            args[k] = int(args[k])
        for k in ("max_nodes","max_edges"):
            if args[k] == -1:
                args[k] = None
        net = cls(**args)
        if not net.valid():
            raise ValueError("检查点不合法")
        return net
