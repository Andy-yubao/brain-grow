"""任意层数的固定纯LIF图：只传播真实Fire，只训练相邻层边权。"""
from __future__ import annotations

import math

import torch
from torch import nn

from braingrow.network.layered_gradient import Dynamics, _Fire


class FixedLayeredSNN(nn.Module):
    def __init__(self, widths=(5000, 3000, 2000), *, seed=11,
                 dynamics=Dynamics(), device="cuda"):
        super().__init__()
        if seed != 11:
            raise ValueError("早期模型唯一根种子为11")
        if not widths or any(int(n) != n or n < 1 for n in widths):
            raise ValueError("隐藏层必须是非空的正整数宽度序列")
        self.widths = tuple(int(n) for n in widths)
        self.dynamics = dynamics
        dimensions = (784, *self.widths, 17)
        generator = torch.Generator(device=device).manual_seed(seed)
        # 隐藏入边按fan-in缩放；输出尺度保持初始读出方差与旧512宽度接近。
        self.initial_scales = [0.15 * math.sqrt(784 / n) for n in dimensions[:-2]]
        self.initial_scales.append(0.02 * math.sqrt(512 / self.widths[-1]))
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(a, b, generator=generator, device=device) * scale)
            for a, b, scale in zip(dimensions[:-1], dimensions[1:], self.initial_scales)
        ])

    def forward(self, x, *, trace=False, voltage_diagnostics=False):
        if x.ndim != 3 or tuple(x.shape[1:]) != (8, 784):
            raise ValueError("输入必须是batch×8帧×784原灰度")
        batch, layers = len(x), len(self.widths)
        voltages = [x.new_zeros(batch, n) for n in self.widths]
        previous_fire = [x.new_zeros(batch, n) for n in self.widths]
        potential = x.new_zeros(batch, 17)
        if trace:
            unit_fire = [x.new_zeros(n) for n in self.widths]
            video_fire = x.new_zeros(batch, layers)
            frame_fire = []
            last_counts = x.new_zeros(batch, self.widths[-1])
        if voltage_diagnostics:
            square_sums = x.new_zeros(layers)
            peaks = x.new_zeros(layers)
        for frame in range(8):
            raw_current = x[:, frame] @ self.weights[0]
            if trace:
                frame_counts = x.new_zeros(batch, layers)
            for _ in range(self.dynamics.steps_per_frame):
                current_fire = []
                for layer in range(layers):
                    incoming = raw_current if layer == 0 else previous_fire[layer - 1] @ self.weights[layer]
                    v = self.dynamics.alpha * voltages[layer] + (1 - self.dynamics.alpha) * incoming
                    if voltage_diagnostics:
                        positive = v.detach().clamp_min(0)
                        square_sums[layer] += positive.square().sum()
                        peaks[layer] = torch.maximum(peaks[layer], positive.max())
                    fire = _Fire.apply(v - self.dynamics.threshold)
                    voltages[layer] = v * (1 - fire.detach())
                    current_fire.append(fire)
                    if trace:
                        real_fire = fire.detach()
                        unit_fire[layer] += real_fire.sum(0)
                        frame_counts[:, layer] += real_fire.sum(1)
                # 所有层均使用前一神经步的脉冲，不能在同一步穿透多层。
                previous_fire = current_fire
                potential = potential + current_fire[-1] @ self.weights[-1]
                if trace:
                    last_counts += current_fire[-1].detach()
            if trace:
                video_fire += frame_counts
                frame_fire.append(frame_counts)
        if not trace and not voltage_diagnostics:
            return potential
        activity = {}
        if trace:
            activity.update(per_unit_fire=unit_fire, per_video_fire=video_fire,
                            frame_fire=torch.stack(frame_fire, dim=1), last_layer_counts=last_counts)
        if voltage_diagnostics:
            counts = x.new_tensor(self.widths) * batch * 8 * self.dynamics.steps_per_frame
            activity.update(positive_voltage_rms=(square_sums / counts).sqrt(),
                            positive_voltage_max=peaks)
        return potential, activity

    def resources(self):
        dimensions = (784, *self.widths, 17)
        edges = [a * b for a, b in zip(dimensions[:-1], dimensions[1:])]
        return dict(hidden_lif=sum(self.widths), hidden_layers=list(self.widths),
                    output_accumulators=17, output_lif=0, total_lif=sum(self.widths),
                    total_units=sum(self.widths) + 17, input_edges=edges[0],
                    internal_edges=sum(edges[1:-1]), internal_edges_by_layer=edges[1:-1],
                    output_edges=edges[-1], input_density=1.0, internal_density=1.0,
                    output_density=1.0, effective_weight_slots=sum(edges),
                    external_classifier_parameters=0)
