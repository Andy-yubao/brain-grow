"""Louizos等既有hard-concrete机制的逐边版本；部署冻结不保留门参数。

原论文式10–13：https://arxiv.org/abs/1712.01312 。作者L0Dense门控输入
单元，本模块每条边独立；稠密训练不代表稀疏执行或新机制首创。
"""
from __future__ import annotations

import copy
import hashlib
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .models import StructuredMLP

BETA, GAMMA, ZETA = 2. / 3., -.1, 1.1
UNIFORM_EPS = 1e-6
LOG_ALPHA_BOUND = math.log(100.)


def deterministic_gate(log_alpha):
    """论文式13，不除温度；与随机门的开概率不是同一量。"""
    return (torch.sigmoid(log_alpha) * (ZETA - GAMMA) + GAMMA).clamp(0, 1)


def open_probability(log_alpha):
    # 理想Uniform(0,1)的解析开概率；端点保护后的实际采样仅近似该期望。
    return torch.sigmoid(log_alpha - BETA * math.log(-GAMMA / ZETA))


def sample_gate(log_alpha, generator):
    u = torch.rand(log_alpha.shape, dtype=log_alpha.dtype, device=log_alpha.device,
                   generator=generator) * (1 - 2 * UNIFORM_EPS) + UNIFORM_EPS
    s = torch.sigmoid((torch.log(u) - torch.log1p(-u) + log_alpha) / BETA)
    return (s * (ZETA - GAMMA) + GAMMA).clamp(0, 1)


class L0GatedMLP(nn.Module):
    """所有层逐边门；bias独立，批内共享采样门，不接受空间先验。"""

    def __init__(self, base: StructuredMLP, initial_log_alpha=2.5):
        super().__init__()
        self.base = copy.deepcopy(base)
        self.widths = self.base.widths
        for layer in self.base.layers:
            identity = torch.arange(layer.theta.numel(), device=layer.theta.device)
            if not torch.equal(layer.group_index, identity):
                raise ValueError("本轮门控只接受父未剪枝identity图，不复用共享或稀疏起点")
        self.log_alpha = nn.ParameterList([
            nn.Parameter(torch.full_like(layer.theta, initial_log_alpha)) for layer in self.base.layers])
        self.gate_generator = None

    def forward(self, x):
        if self.training and self.gate_generator is None:
            raise RuntimeError("随机门训练必须设置独立且可保存的generator")
        for index, (layer, a) in enumerate(zip(self.base.layers, self.log_alpha)):
            z = sample_gate(a, self.gate_generator) if self.training else deterministic_gate(a)
            # theta与门同为FP32，freeze采用完全相同的乘法顺序。
            weight = (layer.theta * z).view(layer.out_features, layer.in_features)
            x = F.linear(x, weight, layer.bias)
            if index + 1 < len(self.base.layers):
                x = F.relu(x)
        return x

    def expected_active_edges(self):
        return sum(open_probability(a).sum() for a in self.log_alpha)

    @torch.no_grad()
    def constrain(self):
        for a in self.log_alpha:
            a.clamp_(-LOG_ALPHA_BOUND, LOG_ALPHA_BOUND)

    @torch.no_grad()
    def gate_stats(self):
        rows = []
        for index, a in enumerate(self.log_alpha):
            z, p = deterministic_gate(a), open_probability(a)
            rows.append(dict(layer_index=index, gate_parameters=a.numel(),
                expected_active_edges=float(p.sum()), deterministic_zero_gates=int((z == 0).sum()),
                deterministic_one_gates=int((z == 1).sum()), deterministic_fractional_gates=int(((z > 0) & (z < 1)).sum()),
                log_alpha_min=float(a.min()), log_alpha_max=float(a.max()),
                log_alpha_mean=float(a.mean()), mean_open_probability=float(p.mean()),
                mean_probability_derivative=float((p * (1 - p)).mean()),
                gates_at_lower_bound=int((a <= -LOG_ALPHA_BOUND).sum()),
                gates_at_upper_bound=int((a >= LOG_ALPHA_BOUND).sum())))
        return dict(layers=rows, gate_parameters=sum(r["gate_parameters"] for r in rows),
                    expected_active_edges=sum(r["expected_active_edges"] for r in rows),
                    deterministic_zero_gates=sum(r["deterministic_zero_gates"] for r in rows))


@torch.no_grad()
def freeze_gated(model: L0GatedMLP, *, permutation_seed=None):
    """独立返回普通MLP与证据，源参数/状态/RNG不变；不执行模型前向。

    随机分支逐层置乱完整门值（含零）。z>0的有效位置可有精确零theta；
    活跃边与FP32非零边分别报告。此操作仅折入门，不做额外排序硬剪。
    """
    with torch.random.fork_rng(devices=[]):
        result = StructuredMLP(model.widths)
    result = result.to(next(model.parameters()).device)
    arrays, layers = {}, []
    for index, (source, target, a) in enumerate(zip(model.base.layers, result.layers, model.log_alpha)):
        original = deterministic_gate(a)
        z = original
        if permutation_seed is not None:
            permutation = np.random.default_rng(np.random.SeedSequence([permutation_seed, 811, index])).permutation(a.numel())
            arrays[f"permutation_{index}"] = permutation.astype(np.int32)
            z = original[torch.as_tensor(permutation, device=a.device)]
        target.theta.copy_(source.theta * z)
        target.bias.copy_(source.bias)
        target.group_index.copy_(torch.where(z > 0, torch.arange(z.numel(), device=z.device), -1))
        values = z.cpu().numpy().copy()
        arrays[f"gate_{index}"] = values
        layers.append(dict(layer_index=index, active_gate_positions=int((z > 0).sum()),
            zero_gates=int((z == 0).sum()), fractional_gates=int(((z > 0) & (z < 1)).sum()),
            one_gates=int((z == 1).sum()), nonzero_effective_weights=int((target.theta != 0).sum()),
            gate_values_sha256=hashlib.sha256(values.tobytes()).hexdigest(),
            same_gate_multiset=bool(np.array_equal(np.sort(original.cpu().numpy()), np.sort(values)))))
    result.train(model.training)
    return result, dict(rule="theta*z_det" if permutation_seed is None else "theta*per_layer_permuted_full_z_det",
                        layers=layers, permutation_seed=permutation_seed, source_unchanged=True,
                        additional_model_forward_samples=0), arrays
