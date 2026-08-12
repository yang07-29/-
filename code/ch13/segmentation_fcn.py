"""第 13 章：合成像素标签上的转置卷积与微型 FCN。

运行：python code/ch13/segmentation_fcn.py --quick
"""

from __future__ import annotations

import argparse

import torch
from torch import nn
from torch.nn import functional as F


def make_segmentation_batch(
    batch: int, size: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """生成背景/矩形/圆形三类的离线分割数据。"""

    # 输入图像为 RGB，Shape (B,3,H,W)。
    images = torch.rand(batch, 3, size, size, generator=generator) * 0.08
    # 标签是类别索引而非 RGB，Shape (B,H,W)，dtype 必须为 long。
    masks = torch.zeros(batch, size, size, dtype=torch.long)
    # 构造一次坐标网格，圆形掩码可广播复用。
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    for index in range(batch):
        if index % 2 == 0:
            # 偶数样本绘制类别 1 的矩形。
            top = int(torch.randint(3, size // 2, (), generator=generator))
            left = int(torch.randint(3, size // 2, (), generator=generator))
            masks[index, top : top + size // 3, left : left + size // 3] = 1
            # 目标区域使用偏红颜色，给模型提供可学习线索。
            images[index, 0, masks[index] == 1] = 0.95
        else:
            # 奇数样本绘制类别 2 的圆。
            center_y = int(torch.randint(size // 3, 2 * size // 3, (), generator=generator))
            center_x = int(torch.randint(size // 3, 2 * size // 3, (), generator=generator))
            radius = size // 6
            circle = (yy - center_y).square() + (xx - center_x).square() <= radius**2
            masks[index, circle] = 2
            # 圆形区域使用偏绿颜色。
            images[index, 1, circle] = 0.95
    return images, masks


def bilinear_kernel(channels: int, kernel_size: int) -> torch.Tensor:
    """生成各通道独立的二维双线性上采样核。"""

    # 偶数核与奇数核的中心定义略有不同。
    factor = (kernel_size + 1) // 2
    center = factor - 0.5 if kernel_size % 2 == 0 else factor - 1
    positions = torch.arange(kernel_size, dtype=torch.float32)
    # 一维帐篷函数在中心为 1，向两边线性衰减。
    one_dimensional = 1 - (positions - center).abs() / factor
    # 外积得到二维可分离双线性核。
    kernel_2d = one_dimensional[:, None] * one_dimensional[None, :]
    # ConvTranspose2d 权重 Shape 为 (C_in,C_out/groups,K,K)。
    weights = torch.zeros(channels, channels, kernel_size, kernel_size)
    for channel in range(channels):
        # 只初始化同一类别通道，避免类别间混合。
        weights[channel, channel] = kernel_2d
    return weights


class TinyFCN(nn.Module):
    """下采样提取语义，再用转置卷积恢复逐像素预测。"""

    def __init__(self, classes: int = 3) -> None:
        super().__init__()
        # 两次 stride=2 把 H,W 各缩小 4 倍。
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        # 1x1 卷积逐位置把 32 维特征变成 classes 个类别分数。
        self.classifier = nn.Conv2d(32, classes, kernel_size=1)
        # k=4,s=4 恢复四倍空间尺寸，不改变类别通道数。
        self.upsample = nn.ConvTranspose2d(classes, classes, kernel_size=4, stride=4, bias=False)
        # 双线性初始化比随机棋盘格更适合作为上采样起点。
        with torch.no_grad():
            self.upsample.weight.copy_(bilinear_kernel(classes, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B,3,H,W) -> (B,32,H/4,W/4)。
        encoded = self.encoder(x)
        # 低分辨率类别 logits 为 (B,C,H/4,W/4)。
        low_resolution_logits = self.classifier(encoded)
        # 转置卷积返回 (B,C,H,W)，仍是 logits，不做 softmax。
        return self.upsample(low_resolution_logits)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="减少训练步数")
    args = parser.parse_args()
    torch.manual_seed(13)
    generator = torch.Generator().manual_seed(130)

    # 32 能被 4 整除，编码/解码 Shape 恰好对齐。
    images, masks = make_segmentation_batch(16, 32, generator)
    model = TinyFCN(classes=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    steps = 3 if args.quick else 25

    for step in range(steps):
        # 清空上一轮累积梯度。
        optimizer.zero_grad(set_to_none=True)
        # 输出 logits Shape (16,3,32,32)。
        logits = model(images)
        # 交叉熵逐像素比较类别维 C 与标签 (B,H,W)。
        loss = F.cross_entropy(logits, masks)
        # 反向同时训练 encoder、1x1 分类器和上采样核。
        loss.backward()
        optimizer.step()
        if step in {0, steps - 1}:
            prediction = logits.argmax(dim=1)
            pixel_accuracy = (prediction == masks).float().mean().item()
            print(f"step {step + 1}: loss={loss.item():.4f}, pixel_acc={pixel_accuracy:.3f}")

    # 单独验证转置卷积的输出公式：(H-1)*s-2p+k。
    probe = torch.randn(1, 3, 8, 8)
    upsampled = model.upsample(probe)
    print(f"转置卷积 Shape: {tuple(probe.shape)} -> {tuple(upsampled.shape)}")
    print(f"FCN Shape: {tuple(images.shape)} -> {tuple(model(images).shape)}，标签 {tuple(masks.shape)}")


if __name__ == "__main__":
    main()
