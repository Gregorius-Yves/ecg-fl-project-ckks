"""
ResNet1D Architecture for 12-Lead ECG Multi-Label Classification
Tailored for Federated Learning with Local Batch Normalization (FedBN)
Input dimension: (Batch_Size, 12, 1000) -> 12 leads, 10s @ 100 Hz
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, padding=3):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ResidualBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=7, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1
        out_channels = channels * 2 if downsample else channels

        self.conv1 = nn.Conv1d(channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if downsample:
            self.shortcut = nn.Sequential(
                nn.Conv1d(channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return self.relu(out)


class ResNet1D(nn.Module):
    def __init__(self, in_channels=12, num_classes=27, base_filters=64):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Initial Feature Extraction
        self.initial_conv = ConvBlock1D(in_channels, base_filters, kernel_size=15, stride=2, padding=7)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Residual Stages
        self.stage1 = ResidualBlock1D(base_filters, downsample=False)
        self.stage2 = ResidualBlock1D(base_filters, downsample=True)       # filters -> 128
        self.stage3 = ResidualBlock1D(base_filters * 2, downsample=True)   # filters -> 256
        self.stage4 = ResidualBlock1D(base_filters * 4, downsample=True)   # filters -> 512

        # Global Average Pooling & Multilabel Classifier
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(base_filters * 8, num_classes)

    def forward(self, x):
        # Expected input shape: (B, 12, 5000)
        x = self.initial_conv(x)
        x = self.maxpool(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.gap(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits

    def predict_proba(self, x):
        logits = self.forward(x)
        return torch.sigmoid(logits)


def is_batchnorm_param(name: str) -> bool:
    """Helper untuk mendeteksi apakah parameter termasuk layer BatchNorm1d (Krusial untuk FedBN)"""
    bn_indicators = ["bn", "running_mean", "running_var", "num_batches_tracked"]
    if any(ind in name.lower() for ind in bn_indicators):
        return True
    # Di ResidualBlock1D, shortcut downsample berupa nn.Sequential(Conv1d, BatchNorm1d)
    # sehingga layer indeks 1 adalah BatchNorm1d (shortcut.1.weight, shortcut.1.bias)
    if "shortcut.1." in name:
        return True
    return False


def get_model_parameters(model: nn.Module, exclude_bn: bool = False):
    """Mengembalikan parameter state dict, dengan opsi mengecualikan BatchNorm untuk FedBN"""
    state_dict = model.state_dict()
    if not exclude_bn:
        return state_dict
    return {k: v for k, v in state_dict.items() if not is_batchnorm_param(k)}
