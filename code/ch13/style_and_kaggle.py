"""第 13 章：风格损失 + 小型 Kaggle 图像分类流程（合成数据、离线）。

运行：python code/ch13/style_and_kaggle.py --quick --output submission_demo.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


def gram_matrix(features: torch.Tensor) -> torch.Tensor:
    """把空间位置摊平，计算归一化通道相关矩阵。"""

    batch, channels, height, width = features.shape
    # (B,C,H,W) -> (B,C,H*W)，空间排列被丢弃但通道共现被保留。
    flattened = features.reshape(batch, channels, height * width)
    # 与自身转置批量矩阵乘，得到 (B,C,C)。
    gram = flattened @ flattened.transpose(1, 2)
    # 归一化让损失规模不过度依赖通道数和图像尺寸。
    return gram / (channels * height * width)


class FixedFeatures(nn.Module):
    """冻结的小卷积特征器，仅用于解释内容/风格损失。"""

    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Conv2d(3, 8, 3, padding=1, bias=False)
        self.layer2 = nn.Conv2d(8, 8, 3, padding=1, bias=False)
        # 该示例不训练特征器，只优化合成图像像素。
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # 浅层特征更适合描述纹理统计。
        shallow = F.relu(self.layer1(x))
        # 深一层特征作为内容表示。
        deep = F.relu(self.layer2(F.avg_pool2d(shallow, 2)))
        return shallow, deep


def total_variation(image: torch.Tensor) -> torch.Tensor:
    """惩罚相邻像素剧烈变化，让生成图更平滑。"""

    # 竖直相邻像素差。
    vertical = (image[:, :, 1:, :] - image[:, :, :-1, :]).abs().mean()
    # 水平相邻像素差。
    horizontal = (image[:, :, :, 1:] - image[:, :, :, :-1]).abs().mean()
    return vertical + horizontal


def run_style_demo(steps: int) -> None:
    """优化一个小图像，验证内容、风格与 TV 三类损失。"""

    torch.manual_seed(131)
    feature_net = FixedFeatures().eval()
    # 内容图和风格图在真实任务中来自两张图片；这里用合成张量代替。
    content = torch.rand(1, 3, 24, 24)
    style = torch.rand(1, 3, 24, 24)
    # 目标内容/风格不需要梯度，提前 detach 节省图保存。
    with torch.no_grad():
        _, target_content = feature_net(content)
        target_style, _ = feature_net(style)
        target_gram = gram_matrix(target_style)
    # 复制内容图作为起点；它是唯一被优化的“参数”。
    generated = content.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([generated], lr=0.08)

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        current_style, current_content = feature_net(generated)
        # 内容损失保留深层空间布局。
        content_loss = F.mse_loss(current_content, target_content)
        # 风格损失匹配浅层通道相关，不要求像素逐点一致。
        style_loss = F.mse_loss(gram_matrix(current_style), target_gram)
        # TV 作为图像先验抑制高频噪点。
        tv_loss = total_variation(generated)
        loss = content_loss + 20.0 * style_loss + 0.02 * tv_loss
        loss.backward()
        optimizer.step()
        # 像素范围属于约束，更新后投影回 [0,1]。
        with torch.no_grad():
            generated.clamp_(0, 1)
    print(f"风格优化 {steps} 步：总损失={loss.item():.5f}, 输出={tuple(generated.shape)}")


def make_classification_data(count: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """生成三类 RGB 色块，模拟竞赛训练/测试图片。"""

    generator = torch.Generator().manual_seed(seed)
    labels = torch.arange(count) % 3
    images = torch.rand(count, 3, 12, 12, generator=generator) * 0.15
    for index, label in enumerate(labels.tolist()):
        # 第 label 个颜色通道整体更亮，构成可学习类别规则。
        images[index, label] += 0.75
    return images.clamp(0, 1), labels


class ContestCNN(nn.Module):
    """足够小的竞赛分类器。"""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(8, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 (B,3,12,12)，输出 (B,3) logits。
        return self.net(x)


@torch.inference_mode()
def predict_with_tta(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """原图与水平翻转预测取平均，返回类别索引。"""

    model.eval()
    original = model(images).softmax(dim=1)
    flipped = model(torch.flip(images, dims=[3])).softmax(dim=1)
    # 概率平均后再选类别，Shape 从 (N,3) 变为 (N,)。
    return ((original + flipped) / 2).argmax(dim=1)


def run_kaggle_demo(epochs: int, output: Path) -> None:
    """完成划分、训练、验证、TTA 推理和提交 CSV。"""

    # 90 可被 3 个类别整除，便于做严格分层划分。
    sample_count = 90
    images, labels = make_classification_data(sample_count, seed=132)
    # 固定每类最后 6 个样本做验证，相当于简化的分层划分。
    validation_mask = torch.zeros(sample_count, dtype=torch.bool)
    for class_id in range(3):
        class_indices = torch.where(labels == class_id)[0]
        validation_mask[class_indices[-6:]] = True
    train_x, train_y = images[~validation_mask], labels[~validation_mask]
    val_x, val_y = images[validation_mask], labels[validation_mask]
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=18, shuffle=True)

    model = ContestCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    for _ in range(epochs):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
    validation_predictions = predict_with_tta(model, val_x)
    validation_accuracy = (validation_predictions == val_y).float().mean().item()
    print(f"竞赛验证准确率={validation_accuracy:.3f}（只用于流程演示）")

    # 测试集真实场景没有 labels；这里丢弃合成标签模拟盲测。
    test_images, _ = make_classification_data(12, seed=133)
    test_predictions = predict_with_tta(model, test_images)
    # newline='' 避免 Windows 写出空白行，encoding 明确保证跨平台。
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "label"])
        for index, label in enumerate(test_predictions.tolist()):
            writer.writerow([f"test_{index:03d}", label])
    print(f"已写出 {len(test_predictions)} 行预测：{output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="减少优化步数")
    parser.add_argument("--output", type=Path, default=Path("submission_demo.csv"))
    args = parser.parse_args()
    # quick 仍跑完整链路，只缩小迭代预算。
    run_style_demo(steps=2 if args.quick else 12)
    run_kaggle_demo(epochs=1 if args.quick else 5, output=args.output)


if __name__ == "__main__":
    main()
