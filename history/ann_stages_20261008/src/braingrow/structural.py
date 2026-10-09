"""低频任务敏感结构编辑；所有候选均使用全局结构选择反馈。"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Proposal:
    kind: str
    edits: list[tuple[int, np.ndarray]]
    count: int


class StructureController:
    def __init__(self, model, arm, fraction=0.05, decay=0.9):
        self.arm = arm
        self.fraction = fraction
        self.decay = decay
        self.proposals = 0
        self.accepted = 0
        self.moments = [{"mean": torch.zeros_like(layer.theta),
                         "square": torch.zeros_like(layer.theta)}
                        for layer in model.shared_layers()]

    @torch.no_grad()
    def observe_gradients(self, model):
        for layer, moment in zip(model.shared_layers(), self.moments):
            grad = layer.last_edge_grad
            if grad is None:
                continue
            moment["mean"].mul_(self.decay).add_(grad, alpha=1-self.decay)
            moment["square"].mul_(self.decay).addcmul_(grad, grad, value=1-self.decay)

    def state_dict(self):
        return {"arm": self.arm, "fraction": self.fraction, "decay": self.decay,
                "proposals": self.proposals, "accepted": self.accepted,
                "moments": self.moments}

    def load_state_dict(self, state):
        self.arm, self.fraction, self.decay = state["arm"], state["fraction"], state["decay"]
        self.proposals, self.accepted = state["proposals"], state["accepted"]
        self.moments = copy.deepcopy(state["moments"])

    def propose(self, model):
        kind = "prune" if self.arm == "prune" else "share"
        if self.arm == "joint":
            kind = "prune" if self.proposals % 2 == 0 else "share"
        edits, total = [], 0
        for li, (layer, moment) in enumerate(zip(model.shared_layers(), self.moments)):
            ids = layer.group_index.detach().cpu().numpy()
            weights = layer.theta.detach().cpu().numpy()
            mean = moment["mean"].detach().cpu().numpy()
            square = moment["square"].detach().cpu().numpy()
            live = np.flatnonzero(ids >= 0)
            if len(live) < 2:
                continue
            if kind == "prune":
                edge_w = weights[ids[live]]
                score = np.abs(edge_w * mean[live]) + 0.5*edge_w**2*square[live]
                n = max(1, int(len(live)*self.fraction))
                chosen = live[np.argsort(score, kind="stable")[:n]]
            else:
                counts = np.bincount(ids[live], minlength=len(weights))
                groups = np.flatnonzero(counts)
                if len(groups) < 2:
                    continue
                gm = np.bincount(ids[live], weights=mean[live], minlength=len(weights))
                gq = np.bincount(ids[live], weights=square[live], minlength=len(weights))
                gm[groups] /= counts[groups]
                gq[groups] /= counts[groups]
                order = groups[np.lexsort((groups, gm[groups], weights[groups]))]
                pairs = order[:len(order)//2*2].reshape(-1, 2)
                a, b = pairs.T
                delta = np.abs(weights[a]-weights[b])
                score = delta
                if self.arm == "joint":
                    score = delta*np.abs(gm[a]-gm[b]) + delta**2*(gq[a]+gq[b])
                n = min(len(pairs), max(1, int(len(groups)*self.fraction)))
                chosen = pairs[np.argsort(score, kind="stable")[:n]]
            edits.append((li, chosen.copy()))
            total += len(chosen)
        # 游标在事务之外推进，拒绝后不会重复无限重放同一候选。
        self.proposals += 1
        return Proposal(kind, edits, total)

    @torch.no_grad()
    def apply(self, model, optimizer, proposal):
        for li, indices in proposal.edits:
            layer = model.shared_layers()[li]
            index = torch.as_tensor(indices, dtype=torch.long, device=layer.theta.device)
            affected = layer.prune_edges(index) if proposal.kind == "prune" else layer.tie_pairs(index)
            state = optimizer.state.get(layer.theta, {})
            for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                if name in state:
                    state[name][affected] = 0


class CandidateTransaction:
    """保存编辑相关状态；账本与候选预算由外部管理，永不回退。"""
    def __init__(self, model, optimizer, controller):
        self.model, self.optimizer, self.controller = model, optimizer, controller
        self.snapshot = {
            "model": copy.deepcopy(model.state_dict()),
            "training": model.training,
            "optimizer": copy.deepcopy(optimizer.state_dict()),
            "controller": copy.deepcopy(controller.state_dict()),
            "edge_grad": [None if layer.last_edge_grad is None else layer.last_edge_grad.clone()
                          for layer in model.shared_layers()],
            "parameter_grad": [None if p.grad is None else p.grad.clone() for p in model.parameters()],
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else None,
        }
        self.committed = False

    def __enter__(self):
        return self

    def commit(self):
        self.committed = True

    def __exit__(self, exc_type, exc, tb):
        if self.committed and exc_type is None:
            return False
        self.model.load_state_dict(self.snapshot["model"])
        self.model.train(self.snapshot["training"])
        self.optimizer.load_state_dict(self.snapshot["optimizer"])
        self.controller.load_state_dict(self.snapshot["controller"])
        for layer, grad in zip(self.model.shared_layers(), self.snapshot["edge_grad"]):
            layer.last_edge_grad = grad
        for parameter, grad in zip(self.model.parameters(), self.snapshot["parameter_grad"]):
            parameter.grad = grad
        random.setstate(self.snapshot["python_rng"])
        np.random.set_state(self.snapshot["numpy_rng"])
        torch.set_rng_state(self.snapshot["torch_rng"])
        if self.snapshot["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(self.snapshot["cuda_rng"])
        return False
