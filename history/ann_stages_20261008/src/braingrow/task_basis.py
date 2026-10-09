"""固定FC快照上的功能群组选择；不改模型、Adam或母态梯度。

所有候选严格复用同一快照的归一化激活基列和原单位C32库。
J是逐样本CE的一阶敏感度代理，Q是按固定拟合通道尺度归一化的
联合下一层扰动；两者均不是任务损失变化上界或全域等价证明。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from .node_reduction import NodeReductionProposal


def _cpu_bytes(records):
    return sum(value.numel()*value.element_size() for record in records
               for value in record.values() if isinstance(value, torch.Tensor))


def extract_snapshot(model, x, y, batch_size=64, cpu_limit=512*1024**2, cuda_limit=1024**3):
    """一次带梯度提取两隐藏层H、下一层Z和逐样本CE对Z的G。

sum CE求导，无参数backward/优化更新；显式F.linear避免旧模型权重
hook改写last_edge_grad。批图及时释放，参数grad/Adam/RNG不改变。
"""
    if len(model.layers) != 3 or len(x) != len(y):
        raise ValueError("本轮仅两个隐藏层的同一FC快照")
    widths = model.widths
    estimated = len(x)*8*(widths[1]+widths[2]+2*(widths[2]+widths[3]))
    if 2*estimated > cpu_limit:
        raise MemoryError("快照CPU数组/拼接预算不足")
    values = [dict(hidden=[], z=[], gradient=[]) for _ in range(2)]
    before = model.training
    model.eval()
    try:
        for start in range(0, len(x), batch_size):
            hidden, pre, activations = x[start:start+batch_size], [], []
            for index, layer in enumerate(model.layers):
                z = F.linear(hidden, layer.expanded_weight(), layer.bias)
                pre.append(z)
                if index < 2:
                    hidden = F.relu(z)
                    activations.append(hidden)
                else:
                    logits = z
            loss = F.cross_entropy(logits, y[start:start+batch_size], reduction="sum")
            gradients = torch.autograd.grad(loss, pre[1:], create_graph=False)
            for index in range(2):
                for key, value in (("hidden", activations[index]), ("z", pre[index+1]),
                                   ("gradient", gradients[index])):
                    if not torch.isfinite(value).all():
                        raise FloatingPointError("快照激活/梯度非有限")
                    values[index][key].append(value.detach().to(device="cpu", dtype=torch.float64))
            if logits.device.type == "cuda" and torch.cuda.memory_allocated(logits.device) > cuda_limit:
                raise MemoryError("快照CUDA allocated上限用完")
            del hidden, pre, activations, logits, gradients, loss, z
        records = [{key: torch.cat(parts) for key, parts in record.items()} for record in values]
        for index, record in enumerate(records):
            record["outgoing"] = model.layers[index+1].expanded_weight().detach().to("cpu", torch.float64)
        if _cpu_bytes(records) > cpu_limit:
            raise MemoryError("快照CPU数组上限用完")
        return records
    finally:
        model.train(before)


@dataclass
class BasisLibrary:
    layer_index: int
    widths: tuple[int, ...]
    basis: tuple[int, ...]
    pool: tuple[int, ...]
    coefficients: torch.Tensor  # 基列×原列，CPU FP32，已还原原激活单位。
    fit_indices: torch.Tensor
    holdout_indices: torch.Tensor
    relative_residuals: torch.Tensor
    support_counts: torch.Tensor
    sigma: torch.Tensor


def build_library(record, layer_index, widths, fit_indices, holdout_indices, max_basis,
                  minimum_support=2):
    """列L2归一化后贪心基组；无标签float64无截距lstsq，再固定C32。"""
    fit = torch.as_tensor(fit_indices, dtype=torch.long)
    holdout = torch.as_tensor(holdout_indices, dtype=torch.long)
    hidden = record["hidden"][fit]
    norms = torch.linalg.vector_norm(hidden, dim=0)
    nonzero = norms > 0
    normalized = hidden/torch.where(nonzero, norms, torch.ones_like(norms))
    residual, basis = normalized.clone(), []
    for _ in range(min(max_basis, hidden.shape[1]-1)):
        scores = torch.linalg.vector_norm(residual, dim=0)
        scores[~nonzero] = -1
        if basis:
            scores[basis] = -1
        chosen = int(scores.argmax())
        if float(scores[chosen]) <= 0:
            break
        basis.append(chosen)
        q = residual[:, chosen]/scores[chosen]
        for _ in range(2):
            residual -= q[:, None]*(q@residual)[None, :]
    if not basis:
        coefficients = torch.empty((0, hidden.shape[1]), dtype=torch.float32)
    else:
        a = torch.linalg.lstsq(normalized[:, basis], normalized, driver="gelsd").solution
        coefficients = (a*norms[None, :]/norms[basis, None]).to(torch.float32)
    if not torch.isfinite(coefficients).all():
        raise FloatingPointError("C32库非有限")
    error = hidden-hidden[:, basis]@coefficients.double()
    absolute = torch.linalg.vector_norm(error, dim=0)
    relative = absolute/torch.where(nonzero, norms, torch.ones_like(norms))
    support = (coefficients != 0).sum(dim=0)
    pool = tuple(index for index in range(hidden.shape[1])
                 if index not in basis and bool(nonzero[index]) and int(support[index]) >= minimum_support)
    sigma = record["z"][fit].square().mean(dim=0).sqrt()
    # 只缓存N×节点的激活残差，不缓存N×候选×下游通道。
    record["errors"] = record["hidden"]-record["hidden"][:, basis]@coefficients.double()
    return BasisLibrary(layer_index, tuple(widths), tuple(basis), pool, coefficients, fit, holdout,
                        relative, support, sigma)


def group_delta(record, library, dropped, indices):
    error = record["errors"][indices][:, list(dropped)]
    return error@record["outgoing"][:, list(dropped)].T


def function_scores(record, library, dropped, indices, batch_size=64):
    """仅保留一个群组的样本分块，不保存N×候选×通道张量。"""
    j_sum, q_sum, finite, count = 0., 0., True, 0
    for block in torch.as_tensor(indices, dtype=torch.long).split(batch_size):
        delta = group_delta(record, library, dropped, block)
        j_sum += float((record["gradient"][block]*delta).abs().sum())
        sigma = library.sigma
        safe = torch.where(sigma > 0, sigma, torch.ones_like(sigma))
        normalized = delta/safe
        invalid_zero = (sigma == 0)[None, :] & (delta != 0)
        finite = finite and not bool(invalid_zero.any())
        q_sum += float(normalized.square().sum())
        count += len(block)
    j, q = j_sum/count, (q_sum/(count*len(library.sigma)))**.5 if finite else float("inf")
    return j, q


def score_record(record, library, dropped, j_limit=.01, q_limit=.10):
    fit_j, fit_q = function_scores(record, library, dropped, library.fit_indices)
    hold_j, hold_q = function_scores(record, library, dropped, library.holdout_indices)
    def clean(value):
        return value if np.isfinite(value) else "inf"
    support = library.support_counts[list(dropped)].tolist()
    return dict(fit_j=clean(fit_j), fit_q=clean(fit_q), holdout_j=clean(hold_j), holdout_q=clean(hold_q),
        fit_j_scaled=clean(fit_j/j_limit), fit_q_scaled=clean(fit_q/q_limit),
        fit_dominant="J" if fit_j/j_limit > fit_q/q_limit else "Q" if fit_j/j_limit < fit_q/q_limit else "tie",
        protection_passed=all(np.isfinite(v) and v <= limit for v, limit in
            ((fit_j, j_limit), (hold_j, j_limit), (fit_q, q_limit), (hold_q, q_limit))),
        support_counts=support, zero_support_columns=sum(value == 0 for value in support),
        single_support_columns=sum(value == 1 for value in support), multi_support_columns=sum(value >= 2 for value in support),
        coefficients_absolute_max=float(library.coefficients[:, list(dropped)].abs().max()),
        coefficients_frobenius_norm=float(torch.linalg.vector_norm(library.coefficients[:, list(dropped)])),
        fit_relative_residuals=library.relative_residuals[list(dropped)].tolist())


def select_group(record, library, method, count=8, j_limit=.01, q_limit=.10):
    if len(library.pool) < count:
        return None, []
    if method == "residual":
        chosen = sorted(library.pool, key=lambda index: (float(library.relative_residuals[index]), index))[:count]
        return tuple(sorted(chosen)), []
    if method == "random":
        rng = np.random.default_rng(np.random.SeedSequence([11, 25, library.layer_index, 0]))
        return tuple(sorted(rng.choice(library.pool, size=count, replace=False).tolist())), []
    if method != "task":
        raise ValueError("群组方法仅residual/task/random")
    chosen, steps = [], []
    # 逐身份、逐64样本分块；不同时构建全部候选误差。
    for step in range(count):
        candidates = []
        for index in library.pool:
            if index in chosen:
                continue
            j, q = function_scores(record, library, chosen+[index], library.fit_indices)
            candidates.append((max(j/j_limit, q/q_limit), index, j, q))
        value, index, j, q = min(candidates, key=lambda item: (item[0], item[1]))
        chosen.append(index)
        steps.append(dict(step=step+1, added_index=index, fit_j=j, fit_q=q if np.isfinite(q) else "inf",
                          j_scaled=j/j_limit, q_scaled=q/q_limit if np.isfinite(q) else "inf",
                          score=value if np.isfinite(value) else "inf",
                          dominant="J" if j/j_limit > q/q_limit else "Q" if j/j_limit < q/q_limit else "tie"))
    return tuple(sorted(chosen)), steps


def reduction_proposal(library, dropped):
    kept = tuple(index for index in range(library.widths[library.layer_index+1]) if index not in dropped)
    coefficients = torch.zeros((len(kept), len(dropped)), dtype=torch.float64)
    rows = {index: row for row, index in enumerate(kept)}
    for row, basis in enumerate(library.basis):
        coefficients[rows[basis]] = library.coefficients[row, list(dropped)].double()
    return NodeReductionProposal(library.layer_index, library.widths, kept, tuple(dropped), library.basis,
        coefficients, tuple(library.relative_residuals[list(dropped)].tolist()), (), len(library.fit_indices),
        1., len(library.basis), coefficient_source="严格共享母FC快照的归一化float64激活lstsq，原单位固定C32；标签仅用于J")
