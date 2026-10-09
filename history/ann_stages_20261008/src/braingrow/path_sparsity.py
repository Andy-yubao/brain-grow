"""偏置包含在完整路径中的节点组正则与无补偿物理剪宽。"""
from __future__ import annotations

import torch

from .models import StructuredMLP


def zero_safe_sqrt(value: torch.Tensor) -> torch.Tensor:
    """正值用sqrt，零值选择一阶零次梯度；先替换分母域避免0*inf产生NaN。"""
    positive = value > 0
    rooted = torch.sqrt(torch.where(positive, value, torch.ones_like(value)))
    return torch.where(positive, rooted, torch.zeros_like(rooted))


def path_node_scores(model: StructuredMLP) -> list[torch.Tensor]:
    """每节点全部输入/偏置→输出路径权重积平方和的平方根。

    输入质量1，f_l=(W_l²)f_(l-1)+b_l²，r_l=(W_(l+1)²)^T r_(l+1)，
    输出反向质量1，score_l=sqrt(f_l*r_l)。偏置是固定常数输入的边。
    对任意隐藏节点正缩放入边/偏置、反缩放出边，所有路径积保持不变。
    """
    weights = [layer.expanded_weight() for layer in model.layers]
    forward = torch.ones(model.widths[0], device=weights[0].device, dtype=weights[0].dtype)
    masses = []
    for layer, weight in zip(model.layers[:-1], weights[:-1]):
        forward = weight.square() @ forward + layer.bias.square()
        masses.append(forward)
    backward = torch.ones(model.widths[-1], device=weights[0].device, dtype=weights[0].dtype)
    scores = [None] * len(masses)
    for index in range(len(weights) - 1, 0, -1):
        backward = weights[index].square().T @ backward
        scores[index-1] = zero_safe_sqrt(masses[index-1] * backward)
    return scores


def path_group_penalty(model: StructuredMLP, initial_widths, initial_scale: float) -> torch.Tensor:
    """固定原始隐藏宽度分母和根11初始R0；剪宽后不重算归一化常数。"""
    if initial_scale <= 0:
        raise ValueError("路径正则归一化常数必须为正")
    scores = path_node_scores(model)
    return sum(score.sum() / width for score, width in zip(scores, initial_widths[1:-1])) / initial_scale


@torch.no_grad()
def delete_hidden_nodes(model: StructuredMLP, layer_index: int, dropped_indices) -> StructuredMLP:
    """物理删除隐藏输入行/偏置及下一层列，不恢复、不共享、不补偿新边。"""
    if not 0 <= layer_index < len(model.layers) - 1:
        raise ValueError("只允许删除隐藏层节点")
    width = model.widths[layer_index+1]
    dropped = sorted(set(int(index) for index in dropped_indices))
    if not dropped or len(dropped) >= width or dropped[0] < 0 or dropped[-1] >= width:
        raise ValueError("删除索引必须合法，且保留至少一个隐藏节点")
    dropped_set = set(dropped)
    kept = [index for index in range(width) if index not in dropped_set]
    matrices = [layer.expanded_weight().clone() for layer in model.layers]
    biases = [layer.bias.detach().clone() for layer in model.layers]
    matrices[layer_index] = matrices[layer_index][kept]
    biases[layer_index] = biases[layer_index][kept]
    matrices[layer_index+1] = matrices[layer_index+1][:, kept]
    widths = list(model.widths)
    widths[layer_index+1] = len(kept)
    with torch.random.fork_rng(devices=[]):
        compact = StructuredMLP(widths)
    compact = compact.to(model.layers[0].theta.device)
    for layer, matrix, bias in zip(compact.layers, matrices, biases):
        layer.theta.copy_(matrix.flatten())
        layer.bias.copy_(bias)
        # 本方案只物理删节点，零值仍是可训练独立连接，不新增稀疏掩码操作。
        layer.group_index.copy_(torch.arange(layer.theta.numel(), device=layer.theta.device))
    compact.train(model.training)
    return compact
