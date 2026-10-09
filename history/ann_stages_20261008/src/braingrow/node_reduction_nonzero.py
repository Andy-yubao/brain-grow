"""非零拟合激活的多节点线性依赖提议与隐藏宽度物理缩减。

只从当前StructuredMLP的训练拟合集输入提取ReLU激活，不接收标签、
选择数据、坐标或外部分类器。CPU float64列残差贪心确定基列，随后
无截距最小二乘估计H_dropped≈H_kept@C。有限拟合样本上的小残差
不保证全输入域等价，任务损失/准确率和泛化须由调用方另行评价。
候选必须训练拟合激活L2范数严格大于0，且原float64系数中至少2个
绝对值大于固定1e-8；不清理零节点、不把单节点复制计为多节点重构。

出边补偿W_next[:,kept]+=W_next[:,dropped]@C.T可能使存活出边稠密化。
返回模型的层宽真实变小；所有有效FP32矩阵按非零identity/零-1重建，
不保留ghost节点或共享关系。未迁移Adam、调度器或其他训练状态；
调用方必须登记新建Adam，并提供同等状态重置的对照。
"""

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .models import StructuredMLP

MIN_COEFFICIENT_ABS = 1e-8
MIN_RECONSTRUCTION_COEFFICIENTS = 2


@dataclass(frozen=True)
class NodeReductionProposal:
    layer_index: int
    widths_before: tuple[int, ...]
    kept_indices: tuple[int, ...]
    dropped_indices: tuple[int, ...]
    basis_indices: tuple[int, ...]
    coefficients: torch.Tensor  # CPU float64，len(kept)×len(dropped)。
    relative_residuals: tuple[float, ...]
    absolute_residuals: tuple[float, ...]
    fit_samples: int
    max_relative_residual: float
    max_basis: int
    coefficient_source: str = "当前网络训练拟合输入的ReLU激活；CPU float64无截距lstsq(driver=gelsd)，无标签"
    candidate_filter_enforced: bool = True
    fit_activation_norms: tuple[float, ...] = ()

    def to_dict(self) -> dict:
        return {"layer_index": self.layer_index, "widths_before": list(self.widths_before),
                "kept_indices": list(self.kept_indices), "dropped_indices": list(self.dropped_indices),
                "basis_indices": list(self.basis_indices), "coefficients": self.coefficients.tolist(),
                "coefficient_shape": list(self.coefficients.shape),
                "relative_residuals": list(self.relative_residuals),
                "absolute_residuals": list(self.absolute_residuals), "fit_samples": self.fit_samples,
                "max_relative_residual": self.max_relative_residual, "max_basis": self.max_basis,
                "coefficient_source": self.coefficient_source,
                "equation": "H_dropped ≈ H_kept @ C；无截距，不改下游偏置",
                "scope": "有限拟合样本依赖，不保证全域等价",
                "candidate_filter_enforced": self.candidate_filter_enforced,
                "fit_activation_L2_norms_of_dropped": list(self.fit_activation_norms),
                "nonzero_fit_activation_rule": "训练拟合激活L2范数严格大于0",
                "coefficient_eligibility_rule": "原float64 solution中至少2个abs>1e-8；固定阈值不扫描",
                "coefficient_support_counts": (self.coefficients.abs() > MIN_COEFFICIENT_ABS).sum(dim=0).tolist()}


def _check_hidden_layer(model: StructuredMLP, layer_index: int) -> None:
    if not isinstance(model, StructuredMLP):
        raise TypeError("节点缩减只支持StructuredMLP")
    if not isinstance(layer_index, int) or not 0 <= layer_index < len(model.layers) - 1:
        raise ValueError("layer_index须指向有下一层的隐藏线性层")


@torch.no_grad()
def _fit_activations(model: StructuredMLP, layer_index: int, fit_x: torch.Tensor) -> torch.Tensor:
    inputs = torch.as_tensor(fit_x)
    if inputs.ndim != 2 or inputs.shape[1] != model.widths[0] or inputs.shape[0] == 0:
        raise ValueError("fit_x须为非空N×输入维度矩阵，且只能来自训练拟合集")
    if not torch.isfinite(inputs).all():
        raise ValueError("fit_x含非有限值")
    device, dtype = model.layers[0].theta.device, model.layers[0].theta.dtype
    rows = []
    # StructuredMLP无随机/批归一化模块，保留调用方training标志和RNG。
    for batch in inputs.split(256):
        hidden = batch.to(device=device, dtype=dtype)
        for index in range(layer_index + 1):
            hidden = F.relu(model.layers[index](hidden))
        rows.append(hidden.to(device="cpu", dtype=torch.float64))
    result = torch.cat(rows, dim=0)
    if not torch.isfinite(result).all():
        raise ValueError("拟合激活含非有限值")
    return result


def _relative_norm(norms: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
    # 全零激活的相对残差定义为0；不以近零阈值删除小幅度的非零激活。
    return torch.where(baseline > 0, norms / baseline, torch.zeros_like(norms))


@torch.no_grad()
def propose_node_reduction(model: StructuredMLP, layer_index: int, fit_x: torch.Tensor,
                           max_remove: int = 8, max_relative_residual: float = 0.05,
                           max_basis: int = 64) -> NodeReductionProposal | None:
    """返回一次最多max_remove节点的联合候选；没有合格依赖则返回None。

    基列按相对列残差最大值贪心选择，并列取最小节点ID，最多max_basis。
    非基列按最小二乘相对L2残差/节点ID排序，选满足阈值、拟合激活非零
    且float64基组系数至少2个abs>1e-8的删除节点；无合格候选返回None。
    未选中删除的其他节点也保留，其C行填0；不是仅做节点配对。
    """
    _check_hidden_layer(model, layer_index)
    if not isinstance(max_remove, int) or max_remove < 0:
        raise ValueError("max_remove须为非负整数")
    if not isinstance(max_basis, int) or max_basis <= 0:
        raise ValueError("max_basis须为正整数")
    if not 0 <= max_relative_residual < 1:
        raise ValueError("max_relative_residual须在[0,1)内")
    width = model.widths[layer_index + 1]
    if max_remove == 0 or width < 2:
        return None
    hidden = _fit_activations(model, layer_index, fit_x)
    norms = torch.linalg.vector_norm(hidden, dim=0)
    residual = hidden.clone()
    basis = []
    for _ in range(min(max_basis, width - 1)):
        residual_norms = torch.linalg.vector_norm(residual, dim=0)
        scores = _relative_norm(residual_norms, norms)
        if basis:
            scores[basis] = -1
            if float(scores.max()) <= max_relative_residual:
                break
        chosen = int(scores.argmax())
        basis.append(chosen)
        size = residual_norms[chosen]
        if float(size) == 0:
            break  # 全零层仍保留一个节点，避免空宽度。
        q = residual[:, chosen] / size
        # 两次投影降低正交化舍入误差；不引入位置、标签或人为共享组。
        for _ in range(2):
            residual -= q[:, None] * (q @ residual)[None, :]
    basis_hidden = hidden[:, basis]
    solution = torch.linalg.lstsq(basis_hidden, hidden, driver="gelsd").solution
    error_norms = torch.linalg.vector_norm(hidden - basis_hidden @ solution, dim=0)
    relative = _relative_norm(error_norms, norms)
    coefficient_support = (solution.abs() > MIN_COEFFICIENT_ABS).sum(dim=0)
    basis_set = set(basis)
    eligible = [index for index in range(width)
                if (index not in basis_set and float(norms[index]) > 0
                    and int(coefficient_support[index]) >= MIN_RECONSTRUCTION_COEFFICIENTS
                    and float(relative[index]) <= max_relative_residual)]
    eligible.sort(key=lambda index: (float(relative[index]), index))
    dropped = tuple(sorted(eligible[:min(max_remove, width - 1)]))
    if not dropped:
        return None
    dropped_set = set(dropped)
    kept = tuple(index for index in range(width) if index not in dropped_set)
    kept_row = {index: row for row, index in enumerate(kept)}
    coefficients = torch.zeros((len(kept), len(dropped)), dtype=torch.float64)
    for basis_row, index in enumerate(basis):
        coefficients[kept_row[index]] = solution[basis_row, list(dropped)]
    if not torch.isfinite(coefficients).all():
        raise ValueError("最小二乘系数含非有限值")
    return NodeReductionProposal(layer_index, tuple(model.widths), kept, dropped, tuple(basis),
                                 coefficients, tuple(float(relative[index]) for index in dropped),
                                 tuple(float(error_norms[index]) for index in dropped), hidden.shape[0],
                                 float(max_relative_residual), max_basis,
                                 fit_activation_norms=tuple(float(norms[index]) for index in dropped))


@torch.no_grad()
def apply_node_reduction(model: StructuredMLP, proposal: NodeReductionProposal) -> StructuredMLP:
    """物理缩小隐藏层和下一层输入，返回独立FP32模型，不修改原模型。"""
    _check_hidden_layer(model, proposal.layer_index)
    if tuple(model.widths) != proposal.widths_before:
        raise ValueError("候选宽度与当前模型不符；修改模型后应重新提议")
    width = model.widths[proposal.layer_index + 1]
    kept, dropped = proposal.kept_indices, proposal.dropped_indices
    if (not kept or not dropped or sorted(kept + dropped) != list(range(width))
            or len(set(kept + dropped)) != width):
        raise ValueError("保留/删除节点必须构成原节点的互斥完整划分")
    if proposal.coefficients.shape != (len(kept), len(dropped)) or not torch.isfinite(proposal.coefficients).all():
        raise ValueError("补偿系数形状不符或非有限")
    if any(layer.theta.dtype != torch.float32 or layer.bias.dtype != torch.float32 for layer in model.layers):
        raise ValueError("首版节点缩减只支持FP32")
    device = model.layers[0].theta.device
    if any(layer.theta.device != device for layer in model.layers):
        raise ValueError("模型各层须在同一设备")
    matrices = [layer.expanded_weight().clone() for layer in model.layers]
    biases = [layer.bias.detach().clone() for layer in model.layers]
    index = proposal.layer_index
    before_outgoing = matrices[index + 1][:, list(kept)].clone()
    compensation = matrices[index + 1][:, list(dropped)] @ proposal.coefficients.to(device=device, dtype=torch.float32).T
    compensated = before_outgoing + compensation
    if not torch.isfinite(compensated).all():
        raise ValueError("补偿出边含非有限值")
    matrices[index] = matrices[index][list(kept)]
    biases[index] = biases[index][list(kept)]
    matrices[index + 1] = compensated
    widths = list(model.widths)
    widths[index + 1] = len(kept)
    with torch.random.fork_rng(devices=[]):
        compact = StructuredMLP(widths)
    compact = compact.to(device)
    for layer, matrix, bias in zip(compact.shared_layers(), matrices, biases):
        layer.theta.copy_(matrix.flatten())
        layer.bias.copy_(bias)
        group_index = torch.arange(layer.theta.numel(), dtype=torch.long, device=device)
        group_index[matrix.flatten() == 0] = -1
        layer.group_index.copy_(group_index)
    compact.train(model.training)
    compact.node_reduction_metadata = {
        "layer_index": index, "widths_before": list(model.widths), "widths_after": widths,
        "removed_nodes": len(dropped), "coefficient_source": proposal.coefficient_source,
        "compensation_new_nonzero_edges": int(((before_outgoing == 0) & (compensated != 0)).sum()),
        "compensation_zeroed_edges": int(((before_outgoing != 0) & (compensated == 0)).sum()),
        "retained_outgoing_nonzero_before": int((before_outgoing != 0).sum()),
        "retained_outgoing_nonzero_after": int((compensated != 0).sum()),
        "mapping_policy": "所有层有效非零权重identity，精确零映射-1；不保留共享",
        "optimizer_policy": "不迁移Adam；调用方新建Adam并登记同等重置对照",
        "guarantee": "有限拟合激活残差，不保证全域等价；接受须另验任务效果和完整导出字节",
    }
    return compact
