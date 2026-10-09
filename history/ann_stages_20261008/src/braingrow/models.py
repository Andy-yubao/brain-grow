"""严格标量共享图；当前前向仍为稠密矩阵执行，不宣称计算复用。"""

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class StrictSharedLinear(nn.Module):
    """每个活跃连接引用一个固定 theta slot；-1 表示删除。

    展平顺序为输出单元优先。初始化与同形状 nn.Linear 相同，外部负责
    设置随机根。共享梯度由 gather 的反传自然相加，不按组大小平均。
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("层维度必须为正整数")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        edges = self.in_features * self.out_features
        self.theta = nn.Parameter(torch.empty(edges))
        self.bias = nn.Parameter(torch.empty(self.out_features))
        self.last_edge_grad: torch.Tensor | None = None
        self.register_buffer("group_index", torch.arange(edges, dtype=torch.long))
        nn.init.kaiming_uniform_(self.theta.view(self.out_features, self.in_features), a=math.sqrt(5))
        bound = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.bias, -bound, bound)

    def expanded_weight(self) -> torch.Tensor:
        weight = self.theta[self.group_index.clamp_min(0)]
        return weight.masked_fill(self.group_index < 0, 0).view(self.out_features, self.in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.expanded_weight()
        if weight.requires_grad:
            weight.register_hook(self._capture_edge_grad)
        return F.linear(x, weight, self.bias)

    def _capture_edge_grad(self, grad: torch.Tensor) -> None:
        # 训练派生状态，不入state_dict；候选拒绝由控制器另外恢复。
        # 这里是展开矩阵导数，控制器仅用group_index>=0的有效边。
        self.last_edge_grad = grad.detach().reshape(-1).clone()

    def _indices(self, values) -> torch.Tensor:
        result = torch.as_tensor(values, dtype=torch.long, device=self.theta.device).flatten().unique()
        if result.numel() and ((result < 0).any() or (result >= self.theta.numel()).any()):
            raise ValueError("连接或组 ID 超出 theta slot 范围")
        return result

    @torch.no_grad()
    def prune_edges(self, indices) -> torch.Tensor:
        """删除指定连接，返回相关组 slot，供控制器重置 Adam 矩。

        删除代表所在连接不会迁移该 slot；其他成员仍可严格引用它。
        """
        indices = self._indices(indices)
        affected = self.group_index[indices]
        affected = affected[affected >= 0].unique()
        self.group_index[indices] = -1
        return affected

    @torch.no_grad()
    def tie_groups(self, group_ids) -> torch.Tensor:
        """把一组活跃组投影到按实际连接数加权的均值，代表取最小 ID。"""
        ids = self._indices(group_ids)
        if ids.numel() < 2:
            return ids[:0]
        active = self.group_index[self.group_index >= 0]
        counts = torch.bincount(active, minlength=self.theta.numel())
        if (counts[ids] == 0).any():
            raise ValueError("只能合并活跃组")
        sizes = counts[ids].to(self.theta.dtype)
        projected = (self.theta[ids] * sizes).sum() / sizes.sum()
        lookup = torch.arange(self.theta.numel(), device=self.theta.device)
        lookup[ids] = ids[0]
        mask = self.group_index >= 0
        self.group_index[mask] = lookup[self.group_index[mask]]
        self.theta[ids[0]] = projected
        return ids

    @torch.no_grad()
    def tie_pairs(self, pairs: torch.Tensor) -> torch.Tensor:
        """一次合并多对互不重叠的活跃组，避免逐对遍历整层连接。"""
        pairs = torch.as_tensor(pairs, dtype=torch.long, device=self.theta.device)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pairs 必须为 N×2 的组 ID")
        if not pairs.numel():
            return pairs.flatten()
        ids = self._indices(pairs)
        if ids.numel() != pairs.numel():
            raise ValueError("成对合并要求组 ID 互不重叠")
        counts = torch.bincount(self.group_index[self.group_index >= 0], minlength=self.theta.numel())
        if (counts[ids] == 0).any():
            raise ValueError("只能合并活跃组")
        canonical = pairs.min(dim=1).values
        removed = pairs.max(dim=1).values
        sizes = counts[pairs].to(self.theta.dtype)
        projected = (self.theta[pairs] * sizes).sum(dim=1) / sizes.sum(dim=1)
        lookup = torch.arange(self.theta.numel(), device=self.theta.device)
        lookup[removed] = canonical
        mask = self.group_index >= 0
        self.group_index[mask] = lookup[self.group_index[mask]]
        self.theta[canonical] = projected
        return ids

    @torch.no_grad()
    def stats(self) -> dict:
        ids, counts = torch.unique(self.group_index[self.group_index >= 0], return_counts=True)
        nonzero = self.theta[ids] != 0
        multi = counts > 1
        active = int(counts.sum().item())
        groups = ids.numel()
        biases = self.bias.numel()
        return {
            "in_features": self.in_features, "out_features": self.out_features,
            "total_edges": self.theta.numel(), "active_edges": active,
            "nonzero_edges": int(counts[nonzero].sum().item()),
            "independent_weight_parameters": groups, "bias_parameters": biases,
            "independent_parameters": groups + biases,
            "allocated_parameter_slots": self.theta.numel() + biases,
            "shared_groups": int(multi.sum().item()),
            "nonzero_shared_groups": int((multi & nonzero).sum().item()),
            "zero_groups": int((~nonzero).sum().item()),
            "shared_edges": int(counts[multi].sum().item()),
            "nonzero_shared_edges": int(counts[multi & nonzero].sum().item()),
            "max_group_size": int(counts.max().item()) if counts.numel() else 0,
        }


class StructuredMLP(nn.Module):
    """人为固定层数、宽度和 ReLU 骨架，连接及标量共享关系可编辑。"""

    def __init__(self, widths: Sequence[int]):
        super().__init__()
        self.widths = tuple(int(value) for value in widths)
        if len(self.widths) < 2 or any(value <= 0 for value in self.widths):
            raise ValueError("widths 必须包含至少两个正维度")
        self.layers = nn.ModuleList(StrictSharedLinear(a, b) for a, b in zip(self.widths, self.widths[1:]))

    def shared_layers(self) -> list[StrictSharedLinear]:
        return list(self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index + 1 < len(self.layers):
                x = F.relu(x)
        return x


class ConvReference1D(nn.Module):
    """具有预置局部卷积及平移共享先验的长度 64 参照。"""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1, 8, 5, padding=2, padding_mode="circular")
        self.head = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        return self.head(F.relu(self.conv(x)).mean(dim=-1))


class ConvReference2D(nn.Module):
    """具有预置局部卷积及平移共享先验的 28×28 MNIST 参照。"""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 8, 5)
        self.head = nn.Linear(4608, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.reshape(-1, 1, 28, 28)
        elif x.ndim == 3:
            x = x.unsqueeze(1)
        return self.head(F.relu(self.conv(x)).flatten(1))
