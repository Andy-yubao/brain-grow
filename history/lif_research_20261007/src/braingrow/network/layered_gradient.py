"""两层纯LIF前馈图：前向只传播Fire，固定动力学，仅边权可训练。"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


class _Fire(torch.autograd.Function):
    @staticmethod
    def forward(ctx, membrane_minus_threshold: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(membrane_minus_threshold)
        return (membrane_minus_threshold >= 0).to(membrane_minus_threshold.dtype)

    @staticmethod
    def backward(ctx, upstream: torch.Tensor) -> torch.Tensor:
        (distance,) = ctx.saved_tensors
        # 只在反传中使用平滑导数；前向和推断始终是真实的二值Fire。
        return upstream / (1.0 + 5.0 * distance.abs()).square()


@dataclass(frozen=True)
class Dynamics:
    alpha: float = math.exp(-5.0 / 20.0)
    threshold: float = 0.5
    steps_per_frame: int = 4
    step_ms: int = 5


class LayeredGradientSNN(nn.Module):
    def __init__(self, widths: tuple[int, int] = (512, 512), seed: int = 11,
                 dynamics: Dynamics = Dynamics(), device: str | torch.device = "cuda") -> None:
        super().__init__()
        if seed != 11:
            raise ValueError("早期模型唯一根种子为11")
        if len(widths) != 2 or min(widths) < 1:
            raise ValueError("首轮仅实现两个非空隐藏LIF层")
        self.widths = tuple(int(n) for n in widths)
        self.dynamics = dynamics
        target = torch.device(device)
        generator = torch.Generator(device=target).manual_seed(seed)
        n1, n2 = self.widths
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(784, n1, generator=generator, device=target) * 0.15),
            nn.Parameter(torch.randn(n1, n2, generator=generator, device=target) * 0.08),
            nn.Parameter(torch.randn(n2, 17, generator=generator, device=target) * 0.02),
        ])
        self.register_buffer("mask_input", torch.ones(784, n1, dtype=torch.bool, device=target))
        self.register_buffer("mask_internal", torch.ones(n1, n2, dtype=torch.bool, device=target))
        self.register_buffer("mask_output", torch.ones(n2, 17, dtype=torch.bool, device=target))

    def forward(self, x: torch.Tensor, trace: bool = False):
        if x.ndim != 3 or x.shape[1:] != (8, 784):
            raise ValueError("输入必须是batch×8帧×784原灰度")
        batch = x.shape[0]
        n1, n2 = self.widths
        alpha = self.dynamics.alpha
        threshold = self.dynamics.threshold
        w0, w1, w2 = (w * mask for w, mask in zip(self.weights,
                              (self.mask_input, self.mask_internal, self.mask_output)))
        voltage1 = x.new_zeros((batch, n1))
        voltage2 = x.new_zeros((batch, n2))
        prior_fire1 = x.new_zeros((batch, n1))
        output_potential = x.new_zeros((batch, 17))
        if trace:
            per_video_fire = x.new_zeros((batch, 2))
            per_unit_fire = [x.new_zeros(n1), x.new_zeros(n2)]
            frame_fire = []
            last_layer_counts = x.new_zeros((batch, n2))
        for frame in range(8):
            input_current = x[:, frame] @ w0
            if trace:
                this_frame = x.new_zeros((batch, 2))
            for _ in range(self.dynamics.steps_per_frame):
                voltage1 = alpha * voltage1 + (1.0 - alpha) * input_current
                fire1 = _Fire.apply(voltage1 - threshold)
                voltage1 = voltage1 * (1.0 - fire1.detach())
                # 下一隐藏层仅收到上一个神经步真实Fire；不传膜值或额外特征。
                voltage2 = alpha * voltage2 + (1.0 - alpha) * (prior_fire1 @ w1)
                fire2 = _Fire.apply(voltage2 - threshold)
                voltage2 = voltage2 * (1.0 - fire2.detach())
                output_potential = output_potential + fire2 @ w2
                prior_fire1 = fire1
                if trace:
                    counts = torch.stack((fire1.sum(1), fire2.sum(1)), dim=1)
                    this_frame += counts
                    per_video_fire += counts
                    per_unit_fire[0] += fire1.sum(0)
                    per_unit_fire[1] += fire2.sum(0)
                    last_layer_counts += fire2
            if trace:
                frame_fire.append(this_frame)
        if not trace:
            return output_potential
        return output_potential, {
            "frame_fire": torch.stack(frame_fire, dim=1),
            "per_video_fire": per_video_fire,
            "per_unit_fire": per_unit_fire,
            "last_layer_counts": last_layer_counts,
        }

    @torch.no_grad()
    def grow(self, layer: int, count: int, proposal_id: int) -> list[tuple[nn.Parameter, nn.Parameter]]:
        """新节点接满相邻层，出边从零开始，出生当下保持现有输出。"""
        if layer not in (0, 1) or count < 1:
            raise ValueError("首轮只允许向现有两个隐藏层加正数个节点")
        n1, n2 = self.widths
        device = self.weights[0].device
        generator = torch.Generator(device=device).manual_seed(11 + 1000003 * proposal_id + layer)
        before = list(self.weights)
        if layer == 0:
            incoming = torch.randn(784, count, generator=generator, device=device) * 0.15
            outgoing = torch.zeros(count, n2, device=device)
            tensors = [torch.cat((before[0], incoming), dim=1),
                       torch.cat((before[1], outgoing), dim=0), before[2].detach().clone()]
            self.mask_input = torch.cat((self.mask_input, torch.ones_like(incoming, dtype=torch.bool)), dim=1)
            self.mask_internal = torch.cat((self.mask_internal, torch.ones_like(outgoing, dtype=torch.bool)), dim=0)
            self.widths = (n1 + count, n2)
        else:
            incoming = torch.randn(n1, count, generator=generator, device=device) * 0.08
            outgoing = torch.zeros(count, 17, device=device)
            tensors = [before[0].detach().clone(), torch.cat((before[1], incoming), dim=1),
                       torch.cat((before[2], outgoing), dim=0)]
            self.mask_internal = torch.cat((self.mask_internal, torch.ones_like(incoming, dtype=torch.bool)), dim=1)
            self.mask_output = torch.cat((self.mask_output, torch.ones_like(outgoing, dtype=torch.bool)), dim=0)
            self.widths = (n1, n2 + count)
        self.weights = nn.ParameterList(nn.Parameter(t) for t in tensors)
        return list(zip(before, self.weights))

    @torch.no_grad()
    def split_active(self, layer: int, parents: torch.Tensor, proposal_id: int
                     ) -> list[tuple[nn.Parameter, nn.Parameter]]:
        """复制活跃LIF，守恒分配出边；反对称小扰动使训练中可以分化。"""
        if layer not in (0, 1) or parents.ndim != 1 or parents.numel() < 1:
            raise ValueError("拆分需指定现有层中的非空父节点集合")
        n1, n2 = self.widths
        width = self.widths[layer]
        if parents.min() < 0 or parents.max() >= width or parents.unique().numel() != parents.numel():
            raise ValueError("拆分父节点必须是不同的现有LIF")
        if not all(bool(mask.all()) for mask in
                   (self.mask_input, self.mask_internal, self.mask_output)):
            raise ValueError("本轮拆分仅适用于尚未剪边的全连接图")
        device = self.weights[0].device
        parents = parents.to(device=device, dtype=torch.long)
        generator = torch.Generator(device=device).manual_seed(11 + 1000003 * proposal_id + layer)
        before = list(self.weights)
        count = parents.numel()
        if layer == 0:
            incoming = before[0][:, parents].detach().clone()
            original_outgoing = before[1][parents].detach().clone()
            noise_scale = original_outgoing.std(dim=1, keepdim=True).clamp_min(0.001) * 0.02
            noise = torch.randn(original_outgoing.shape, device=device, generator=generator) * noise_scale
            updated = before[1].detach().clone()
            updated[parents] = original_outgoing / 2 + noise
            new_outgoing = original_outgoing / 2 - noise
            tensors = [torch.cat((before[0], incoming), dim=1),
                       torch.cat((updated, new_outgoing), dim=0), before[2].detach().clone()]
            self.mask_input = torch.cat((self.mask_input, torch.ones_like(incoming, dtype=torch.bool)), dim=1)
            self.mask_internal = torch.cat((self.mask_internal, torch.ones_like(new_outgoing, dtype=torch.bool)), dim=0)
            self.widths = (n1 + count, n2)
        else:
            incoming = before[1][:, parents].detach().clone()
            original_outgoing = before[2][parents].detach().clone()
            noise_scale = original_outgoing.std(dim=1, keepdim=True).clamp_min(0.001) * 0.02
            noise = torch.randn(original_outgoing.shape, device=device, generator=generator) * noise_scale
            updated = before[2].detach().clone()
            updated[parents] = original_outgoing / 2 + noise
            new_outgoing = original_outgoing / 2 - noise
            tensors = [before[0].detach().clone(), torch.cat((before[1], incoming), dim=1),
                       torch.cat((updated, new_outgoing), dim=0)]
            self.mask_internal = torch.cat((self.mask_internal, torch.ones_like(incoming, dtype=torch.bool)), dim=1)
            self.mask_output = torch.cat((self.mask_output, torch.ones_like(new_outgoing, dtype=torch.bool)), dim=0)
            self.widths = (n1, n2 + count)
        self.weights = nn.ParameterList(nn.Parameter(t) for t in tensors)
        return list(zip(before, self.weights))

    def resources(self) -> dict:
        n1, n2 = self.widths
        a, b, c = (int(mask.sum().item()) for mask in
                   (self.mask_input, self.mask_internal, self.mask_output))
        return {"hidden_lif": n1 + n2, "hidden_layers": [n1, n2],
                "output_accumulators": 17, "output_lif": 0, "total_lif": n1 + n2,
                "total_units": n1 + n2 + 17,
                "input_edges": a, "internal_edges": b, "output_edges": c,
                "input_density": a / (784 * n1), "internal_density": b / (n1 * n2),
                "output_density": c / (n2 * 17), "effective_weight_slots": a + b + c,
                "external_classifier_parameters": 0}
