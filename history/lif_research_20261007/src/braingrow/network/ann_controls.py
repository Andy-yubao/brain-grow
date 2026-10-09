"""用户授权的非SNN对照；只读取同一v5原始像素视频。"""
from __future__ import annotations

import torch
from torch import nn


ARMS = ("video_mlp", "stacked_cnn", "cnn_gru")
DESCRIPTIONS = {
    "video_mlp": "按时间顺序展平8×784像素；6272→2500→2000→2000→17，隐藏ReLU",
    "stacked_cnn": "8帧作为8通道；Conv3×3 8→32→64→128，前两层MaxPool2，6272→256→17",
    "cnn_gru": "共享逐帧Conv3×3 1→32→64→128，前两层MaxPool2，6272→128；单层GRU128→256，最后时刻→17",
}


def convolutional_encoder(channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(channels, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.Flatten(),
    )


class VideoMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Flatten(), nn.Linear(6272, 2500), nn.ReLU(),
            nn.Linear(2500, 2000), nn.ReLU(), nn.Linear(2000, 2000), nn.ReLU(),
            nn.Linear(2000, 17))

    def forward(self, x):
        return self.layers(x)


class StackedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = convolutional_encoder(8)
        self.head = nn.Sequential(nn.Linear(6272, 256), nn.ReLU(), nn.Linear(256, 17))

    def forward(self, x):
        return self.head(self.encoder(x.reshape(-1, 8, 28, 28)))


class CNNGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = convolutional_encoder(1)
        self.projection = nn.Sequential(nn.Linear(6272, 128), nn.ReLU())
        self.temporal = nn.GRU(128, 256, batch_first=True)
        self.head = nn.Linear(256, 17)

    def forward(self, x):
        features = self.projection(self.encoder(x.reshape(-1, 1, 28, 28)))
        # 未传入隐藏状态：每个视频从零状态开始；完整8帧按时间顺序读取。
        sequence, _ = self.temporal(features.reshape(len(x), 8, 128))
        return self.head(sequence[:, -1])


def make_model(arm: str) -> nn.Module:
    constructors = dict(video_mlp=VideoMLP, stacked_cnn=StackedCNN, cnn_gru=CNNGRU)
    return constructors[arm]()


def resources(model: nn.Module, arm: str) -> dict:
    """参数与激活位置分开记录；卷积共享参数不冒充稠密边数。"""
    groups = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.GRU)):
            weight = sum(p.numel() for key, p in module.named_parameters(recurse=False)
                         if "weight" in key)
            bias = sum(p.numel() for key, p in module.named_parameters(recurse=False)
                       if "bias" in key)
            groups.append(dict(name=name, kind=type(module).__name__, weights=weight,
                               biases=bias, parameter_density=1.0))
    # 每视频逻辑激活位置，GRU为8时刻的256状态；不计池化/门内中间量。
    hidden_positions = {
        "video_mlp": [2500, 2000, 2000],
        "stacked_cnn": [32 * 28 * 28, 64 * 14 * 14, 128 * 7 * 7, 256],
        "cnn_gru": [8 * 32 * 28 * 28, 8 * 64 * 14 * 14,
                    8 * 128 * 7 * 7, 8 * 128, 8 * 256],
    }[arm]
    return dict(architecture=DESCRIPTIONS[arm],
        trainable_parameters=sum(p.numel() for p in model.parameters()),
        parameter_groups=groups, hidden_activation_positions_per_video=hidden_positions,
        hidden_activation_positions_total=sum(hidden_positions), output_logits=17,
        total_activation_positions=sum(hidden_positions) + 17,
        hidden_lif=0, output_lif=0, fire_statistics=None, growth_curve=None,
        note="非LIF网络，无Fire/神经仿真步；激活位置不是SNN物理单元数，参数量/密度不是能耗。")
