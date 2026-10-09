"""从普通拟合梯度恢复 H1 identity-slot 图中的连接，不读取空间或共享先验。"""

from __future__ import annotations

import copy
from collections.abc import Sequence

import numpy as np
import torch

from .structural import Proposal


class EdgeRecoveryController:
    """恢复分数是展开权重梯度的 EMA(g²)，含当前已删除连接。

    runner 仅在普通拟合反传后调用 observe_gradients；候选适应不更新评分。
    propose 在事务外推进游标；接受计数由 runner 在提交事务前递增。
    """

    def __init__(self, model, decay: float = 0.9):
        if not 0 <= decay < 1:
            raise ValueError("EMA decay 必须位于 [0,1)")
        self._validate_graph(model)
        self.decay = float(decay)
        self.proposals = 0
        self.accepted = 0
        self.observations = 0
        self.moments = [{"mean": torch.zeros_like(layer.theta),
                         "square": torch.zeros_like(layer.theta)}
                        for layer in model.shared_layers()]

    @staticmethod
    def _validate_graph(model) -> None:
        for layer in model.shared_layers():
            groups = layer.group_index
            identity = torch.arange(layer.theta.numel(), device=groups.device)
            if groups.shape != identity.shape or not torch.all((groups == -1) | (groups == identity)):
                raise ValueError("边恢复仅允许 H1 原始 identity-slot 剪枝图，不支持共享映射")

    @torch.no_grad()
    def observe_gradients(self, model) -> None:
        self._validate_graph(model)
        layers = model.shared_layers()
        if len(layers) != len(self.moments):
            raise ValueError("控制器与模型层数不一致")
        gradients = [layer.last_edge_grad for layer in layers]
        if any(grad is None for grad in gradients):
            raise ValueError("普通拟合反传后才能记录展开连接梯度")
        for grad, moment in zip(gradients, self.moments):
            if grad.shape != moment["mean"].shape or not torch.isfinite(grad).all():
                raise ValueError("拟合展开梯度维度不一致或非有限")
        for grad, moment in zip(gradients, self.moments):
            moment["mean"].mul_(self.decay).add_(grad, alpha=1-self.decay)
            moment["square"].mul_(self.decay).addcmul_(grad, grad, value=1-self.decay)
        self.observations += 1

    def state_dict(self) -> dict:
        return {"version": 1, "decay": self.decay, "proposals": self.proposals,
                "accepted": self.accepted, "observations": self.observations,
                "moments": self.moments}

    def load_state_dict(self, state: dict) -> None:
        if state.get("version") != 1:
            raise ValueError("不支持的边恢复状态版本")
        self.decay = state["decay"]
        self.proposals, self.accepted = state["proposals"], state["accepted"]
        self.observations = state["observations"]
        self.moments = copy.deepcopy(state["moments"])

    @torch.no_grad()
    def propose(self, model, fraction: float = .005,
                random_match: Proposal | Sequence[int] | None = None,
                seed_parts: Sequence[int] | None = None) -> Proposal:
        """按原始层边数 fraction 的预算恢复；随机模式只匹配已接受提议的数量。

        random_match 是主已接受 Proposal 或逐层整数数量；调用方负责只传已接受槽。
        随机根为 seed_parts=[11, epoch, source_proposal]，每层追加 layer_index。
        主模式以分数降序、原始边 ID 升序确定并列次序。
        """
        self._validate_graph(model)
        if not 0 < fraction <= 1:
            raise ValueError("fraction 必须位于 (0,1]")
        layers = model.shared_layers()
        if len(layers) != len(self.moments):
            raise ValueError("控制器与模型层数不一致")
        counts = None
        if random_match is not None:
            if seed_parts is None or not len(seed_parts) or seed_parts[0] != 11:
                raise ValueError("匹配随机恢复必须提供从唯一根 11 派生的 seed_parts")
            if any(type(part) is not int or part < 0 for part in seed_parts):
                raise ValueError("随机子流标识必须为非负整数")
            if isinstance(random_match, Proposal):
                if random_match.kind != "recover":
                    raise ValueError("随机恢复只能匹配 recover 提议")
                counts = [0] * len(layers)
                for layer_index, edges in random_match.edits:
                    if not 0 <= layer_index < len(layers) or counts[layer_index]:
                        raise ValueError("匹配提议层编号非法或重复")
                    counts[layer_index] = len(edges)
                if sum(counts) != random_match.count:
                    raise ValueError("匹配提议总数不一致")
            else:
                counts = list(random_match)
            if len(counts) != len(layers) or any(type(count) is not int or count < 0 for count in counts):
                raise ValueError("随机匹配须给每层非负整数恢复数量")
        elif not self.observations:
            raise ValueError("任务梯度恢复必须先记录普通拟合梯度")
        edits = []
        for layer_index, (layer, moment) in enumerate(zip(layers, self.moments)):
            inactive = np.flatnonzero(layer.group_index.detach().cpu().numpy() == -1)
            budget = min(len(inactive), int(layer.theta.numel() * fraction))
            count = budget if counts is None else counts[layer_index]
            if count > budget:
                raise ValueError("随机恢复匹配数超过当前未激活边或原始边数比例预算")
            if not count:
                continue
            if counts is None:
                score = moment["square"].detach().cpu().numpy()[inactive]
                if not np.isfinite(score).all():
                    raise ValueError("恢复分数非有限")
                chosen = inactive[np.argsort(-score, kind="stable")[:count]]
            else:
                rng = np.random.default_rng(np.random.SeedSequence([*seed_parts, layer_index]))
                chosen = rng.choice(inactive, size=count, replace=False)
            edits.append((layer_index, chosen.copy()))
        self.proposals += 1
        return Proposal("recover", edits, sum(len(edges) for _, edges in edits))

    @torch.no_grad()
    def apply(self, model, optimizer, proposal: Proposal) -> None:
        """恢复权重为零，初始输出不变；沿用 Adam 全局 step，仅清除相关槽矩。"""
        self._validate_graph(model)
        if proposal.kind != "recover":
            raise ValueError("恢复控制器只接受 recover 提议")
        layers, checked, seen = model.shared_layers(), [], set()
        for layer_index, edges in proposal.edits:
            if not 0 <= layer_index < len(layers) or layer_index in seen:
                raise ValueError("恢复提议层编号非法或重复")
            seen.add(layer_index)
            layer = layers[layer_index]
            array = np.asarray(edges)
            if array.ndim != 1 or array.dtype.kind not in "iu":
                raise ValueError("恢复边必须是一维整数 ID")
            index = torch.as_tensor(array, device=layer.theta.device, dtype=torch.long)
            if (index.unique().numel() != index.numel() or
                    (index.numel() and ((index < 0).any() or (index >= layer.theta.numel()).any()))):
                raise ValueError("恢复边重复或超出原始槽范围")
            if not torch.all(layer.group_index[index] == -1):
                raise ValueError("只能恢复当前 inactive 边")
            checked.append((layer, index))
        if sum(index.numel() for _, index in checked) != proposal.count:
            raise ValueError("恢复提议总数不一致")
        # 所有编辑先验证；调用方仍须用 CandidateTransaction 包裹适应与选择评价。
        for layer, index in checked:
            layer.group_index[index] = index
            layer.theta[index] = 0
            state = optimizer.state.get(layer.theta, {})
            for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                if name in state:
                    state[name][index] = 0
