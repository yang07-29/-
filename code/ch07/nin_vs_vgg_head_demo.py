"""用可运行实验比较 VGG 式全连接分类头与 NiN 式 GAP 分类头。

本程序只隔离比较分类头，不声称复刻完整 VGG 或完整 NiN。
重点观察参数量、输入输出 Shape，以及输入分辨率改变后的行为。
"""

import torch
from torch import nn


class VGGStyleHead(nn.Module):
    """一个缩小的 VGG 式分类头：展平后接两层全连接。"""

    def __init__(self, channels: int = 64, height: int = 7, width: int = 7) -> None:
        super().__init__()
        # 全连接层必须提前知道展平长度，因此这里固定为 64×7×7。
        flattened_features = channels * height * width
        self.classifier = nn.Sequential(
            # 把每个样本的通道、高、宽全部展平成一条向量。
            nn.Flatten(),
            # 用一个小型隐藏层代替原始 VGG 更大的全连接层，便于快速演示。
            nn.Linear(flattened_features, 128),
            # ReLU 让两层 Linear 不能直接合并为一层线性变换。
            nn.ReLU(),
            # 最后一层把 128 维隐藏表示映射为 10 个类别 logits。
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入预期为 (N,64,7,7)，输出为 (N,10)。
        return self.classifier(x)


class NiNStyleHead(nn.Module):
    """NiN 式分类头：先产生类别通道，再做全局平均池化。"""

    def __init__(self, channels: int = 64, classes: int = 10) -> None:
        super().__init__()
        # 1×1 卷积在每个空间位置把 64 个通道组合成 10 个类别通道。
        self.to_class_maps = nn.Conv2d(channels, classes, kernel_size=1)
        # 不管输入高宽是多少，都把每个类别通道平均成一个 1×1 数值。
        self.global_average = nn.AdaptiveAvgPool2d((1, 1))
        # 最后只删除两个大小为 1 的空间维，保留 batch 和类别维。
        self.flatten = nn.Flatten(start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # class_maps 的 Shape 为 (N,10,H,W)，每张图表示一个类别证据图。
        class_maps = self.to_class_maps(x)
        # pooled 的 Shape 为 (N,10,1,1)，GAP 本身没有可学习参数。
        pooled = self.global_average(class_maps)
        # logits 的 Shape 为 (N,10)。
        logits = self.flatten(pooled)
        return logits


def count_parameters(module: nn.Module) -> int:
    """统计一个模块中全部可学习参数的元素个数。"""
    return sum(parameter.numel() for parameter in module.parameters())


def main() -> None:
    # 固定随机种子只是为了让每次演示的随机输入一致。
    torch.manual_seed(7)

    # 两个样本、64 个通道、空间高宽为 7×7。
    features_7 = torch.randn(2, 64, 7, 7)

    # 创建两种分类头；二者最终都输出 10 个类别 logits。
    vgg_head = VGGStyleHead(channels=64, height=7, width=7)
    nin_head = NiNStyleHead(channels=64, classes=10)

    # 执行前向，确认两种头在 7×7 输入上输出 Shape 相同。
    vgg_logits = vgg_head(features_7)
    nin_logits_7 = nin_head(features_7)

    assert tuple(vgg_logits.shape) == (2, 10)
    assert tuple(nin_logits_7.shape) == (2, 10)

    # 手算值：3136×128+128+128×10+10=402826。
    vgg_parameters = count_parameters(vgg_head)
    assert vgg_parameters == 402_826

    # 手算值：1×1 卷积有 64×10 个权重和 10 个偏置，共 650。
    nin_parameters = count_parameters(nin_head)
    assert nin_parameters == 650

    print("=== 相同 7×7 特征输入 ===")
    print("输入 Shape:", tuple(features_7.shape))
    print("VGG 式头输出 Shape:", tuple(vgg_logits.shape))
    print("NiN 式头输出 Shape:", tuple(nin_logits_7.shape))
    print("VGG 式头参数量:", vgg_parameters)
    print("NiN 式头参数量:", nin_parameters)

    # 把空间高宽改为 9×9；通道数仍是 64。
    features_9 = torch.randn(2, 64, 9, 9)

    # GAP 会自动对整个 9×9 求平均，因此仍输出 10 个类别值。
    nin_logits_9 = nin_head(features_9)
    assert tuple(nin_logits_9.shape) == (2, 10)

    # VGG 式头第一层固定需要 64×7×7=3136 个输入特征。
    expected_flattened = vgg_head.classifier[1].in_features
    actual_flattened = features_9[0].numel()
    assert expected_flattened == 3_136
    assert actual_flattened == 5_184

    print("\n=== 输入高宽改成 9×9 ===")
    print("NiN 式头仍可输出:", tuple(nin_logits_9.shape))
    print("VGG 式 Linear 期望展平长度:", expected_flattened)
    print("当前实际展平长度:", actual_flattened)
    print("两者不相等，所以固定 Linear 头不能直接接收该输入。")


if __name__ == "__main__":
    main()
