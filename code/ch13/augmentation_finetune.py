"""第 13 章：合成图像上的增广与微调，全程离线、无需下载数据。

运行：python code/ch13/augmentation_finetune.py --quick
"""

from __future__ import annotations

import argparse
import copy

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


def make_stripe_images(count: int, size: int, shifted: bool, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """生成横条/竖条二分类图像；shifted=True 模拟目标域变化。"""

    # 使用独立生成器，保证不同调用可复现且不污染全局随机状态。
    generator = torch.Generator().manual_seed(seed)
    # 标签 0 表示竖条，1 表示横条，Shape 为 (N,)。
    labels = torch.randint(0, 2, (count,), generator=generator)
    # 图像按 NCHW 保存，初始为弱噪声背景。
    images = torch.rand(count, 1, size, size, generator=generator) * 0.12
    for index, label in enumerate(labels.tolist()):
        # 目标域把条纹中心向右/下移动，制造轻微分布偏移。
        offset = size // 3 if shifted else size // 2
        # 每张图再加一个小随机偏移，防止死记像素坐标。
        jitter = int(torch.randint(-2, 3, (), generator=generator))
        center = max(2, min(size - 3, offset + jitter))
        if label == 0:
            # 竖条贯穿高度，宽度为 3 个像素。
            images[index, 0, :, center - 1 : center + 2] += 0.85
        else:
            # 横条贯穿宽度，高度为 3 个像素。
            images[index, 0, center - 1 : center + 2, :] += 0.85
    # 截断到合法灰度范围 [0,1]。
    return images.clamp(0, 1), labels


def augment_batch(images: torch.Tensor) -> torch.Tensor:
    """在张量 batch 上做标签保持的随机平移、翻转和亮度变化。"""

    batch, channels, height, width = images.shape
    # 反射填充 2 像素，为随机裁剪制造平移空间。
    padded = F.pad(images, (2, 2, 2, 2), mode="reflect")
    # 为每张图独立抽取裁剪起点，范围 0..4。
    top = torch.randint(0, 5, (batch,), device=images.device)
    left = torch.randint(0, 5, (batch,), device=images.device)
    # 逐样本裁剪；输出仍是 (B,C,H,W)。
    cropped = torch.stack(
        [padded[i, :, top[i] : top[i] + height, left[i] : left[i] + width] for i in range(batch)]
    )
    # 横竖条标签在水平翻转后不变，因此这种增广是安全的。
    flip_mask = torch.rand(batch, device=images.device) < 0.5
    cropped[flip_mask] = torch.flip(cropped[flip_mask], dims=[3])
    # 每张图使用一个亮度比例，Shape (B,1,1,1) 依靠广播作用于所有像素。
    brightness = 0.8 + 0.4 * torch.rand(batch, 1, 1, 1, device=images.device)
    # 增广不含可学习参数，但保持张量计算图属性。
    return (cropped * brightness).clamp(0, 1)


class TinyVisionNet(nn.Module):
    """把特征提取器与任务分类头明确分开，方便微调。"""

    def __init__(self) -> None:
        super().__init__()
        # features 学习边缘/方向等可迁移视觉特征。
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        # classifier 是目标任务专用的随机初始化分类头。
        self.classifier = nn.Linear(16, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B,1,H,W) -> (B,16,1,1)。
        features = self.features(x)
        # flatten(1) 保留 batch 维，得到 (B,16)。
        features = features.flatten(1)
        # 返回未做 softmax 的 (B,2) logits。
        return self.classifier(features)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    use_augmentation: bool,
) -> float:
    """训练一个 epoch，并返回样本加权平均损失。"""

    model.train()
    total_loss, total_examples = 0.0, 0
    for images, labels in loader:
        # 增广只用于训练；验证数据必须保持固定。
        if use_augmentation:
            images = augment_batch(images)
        # 每批更新前清空累积梯度。
        optimizer.zero_grad(set_to_none=True)
        # 前向输出 (B,2)，建立计算图。
        logits = model(images)
        # CrossEntropyLoss 内部已包含 log_softmax。
        loss = F.cross_entropy(logits, labels)
        # 反向为所有 requires_grad=True 的参数求梯度。
        loss.backward()
        # 优化器按各参数组自己的学习率更新。
        optimizer.step()
        total_loss += loss.item() * len(images)
        total_examples += len(images)
    return total_loss / total_examples


@torch.inference_mode()
def accuracy(model: nn.Module, images: torch.Tensor, labels: torch.Tensor) -> float:
    """固定验证集上的分类准确率。"""

    model.eval()
    # argmax 沿类别维选择预测类别，结果 Shape 为 (N,)。
    predictions = model(images).argmax(dim=1)
    # 比较后取平均得到 [0,1] 内准确率。
    return float((predictions == labels).float().mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="缩小数据和训练轮数")
    args = parser.parse_args()
    torch.manual_seed(13)

    # 源域数据多，用来学习通用条纹方向特征。
    source_count = 96 if args.quick else 320
    source_x, source_y = make_stripe_images(source_count, 20, shifted=False, seed=1)
    source_loader = DataLoader(TensorDataset(source_x, source_y), batch_size=32, shuffle=True)

    # 预训练阶段同时更新特征层和旧分类头。
    pretrained = TinyVisionNet()
    pretrain_optimizer = torch.optim.Adam(pretrained.parameters(), lr=0.01)
    for epoch in range(1 if args.quick else 4):
        loss = train_epoch(pretrained, source_loader, pretrain_optimizer, use_augmentation=True)
        print(f"预训练 epoch {epoch + 1}: loss={loss:.4f}")

    # 目标域训练样本很少，验证样本固定且不增广。
    target_train_x, target_train_y = make_stripe_images(48, 20, shifted=True, seed=2)
    target_val_x, target_val_y = make_stripe_images(96, 20, shifted=True, seed=3)
    target_loader = DataLoader(
        TensorDataset(target_train_x, target_train_y), batch_size=16, shuffle=True
    )

    # 深复制保留预训练特征；接着替换目标任务分类头。
    finetuned = copy.deepcopy(pretrained)
    finetuned.classifier = nn.Linear(16, 2)
    # 预训练层使用小学习率，随机头使用十倍学习率。
    optimizer = torch.optim.SGD(
        [
            {"params": finetuned.features.parameters(), "lr": 0.01},
            {"params": finetuned.classifier.parameters(), "lr": 0.1},
        ],
        momentum=0.9,
    )
    for epoch in range(2 if args.quick else 6):
        loss = train_epoch(finetuned, target_loader, optimizer, use_augmentation=True)
        score = accuracy(finetuned, target_val_x, target_val_y)
        print(f"微调 epoch {epoch + 1}: loss={loss:.4f}, val_acc={score:.3f}")

    # 最终输出同时证明输入/输出 Shape 和全链路可运行。
    sample_logits = finetuned(target_val_x[:4])
    print(f"样例输入 {tuple(target_val_x[:4].shape)} -> logits {tuple(sample_logits.shape)}")


if __name__ == "__main__":
    main()
