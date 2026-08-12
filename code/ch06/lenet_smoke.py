"""第 6 章：LeNet 的完整、离线、快速训练示例。

运行：
    python code/ch06/lenet_smoke.py
    python code/ch06/lenet_smoke.py --epochs 8 --device cpu

合成任务：判断图像中的亮条纹是竖直还是水平。这个任务很小，但数据、
DataLoader、前向、损失、反向、更新和评估链路都是完整的。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset


class LeNet(nn.Module):
    """适配 28×28 单通道图像的 LeNet；默认使用更易快速训练的 ReLU。"""

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        activation = nn.ReLU  # 若要复现历史结构，可把这里改成 nn.Sigmoid。
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),  # (N,1,28,28) -> (N,6,28,28)
            activation(),  # 原始 LeNet 用 sigmoid；ReLU 让这个 smoke test 更快收敛。
            nn.AvgPool2d(kernel_size=2, stride=2),  # -> (N,6,14,14)
            nn.Conv2d(6, 16, kernel_size=5),  # -> (N,16,10,10)
            activation(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # -> (N,16,5,5)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),  # (N,16,5,5) -> (N,400)，只展平非批量维。
            nn.Linear(16 * 5 * 5, 120),  # 将局部特征汇总为全局表示。
            activation(),
            nn.Linear(120, 84),
            activation(),
            nn.Linear(84, num_classes),  # 返回 logits，不在末尾手动加 softmax。
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)  # 卷积编码器保留并逐步压缩空间结构。
        return self.classifier(x)  # 分类头把编码结果变成每类一个分数。


def make_stripe_data(num_samples: int, seed: int) -> tuple[Tensor, Tensor]:
    """生成带噪声的横/竖亮条纹图，标签 0=竖直，1=水平。"""
    generator = torch.Generator().manual_seed(seed)  # 使用局部生成器确保可复现。
    images = 0.10 * torch.randn(num_samples, 1, 28, 28, generator=generator)
    labels = torch.randint(0, 2, (num_samples,), generator=generator)  # CE 需要 long 类别索引。
    positions = torch.randint(5, 22, (num_samples,), generator=generator)
    for index, (label, position) in enumerate(zip(labels.tolist(), positions.tolist())):
        if label == 0:
            images[index, :, :, position - 2 : position + 2] += 1.0  # 竖直条纹。
        else:
            images[index, :, position - 2 : position + 2, :] += 1.0  # 水平条纹。
    return images.clamp(0.0, 1.0), labels


def trace_shapes(model: LeNet, sample: Tensor) -> None:
    """逐层打印 Shape；先预测尺寸，再用它核对网络。"""
    x = sample
    print("逐层 Shape：")
    for index, layer in enumerate(model.features):
        x = layer(x)  # 每次只通过一层，便于定位是哪一层改变了 Shape。
        print(f"  features[{index}] {layer.__class__.__name__:10s} -> {tuple(x.shape)}")
    for index, layer in enumerate(model.classifier):
        x = layer(x)
        print(f"  classifier[{index}] {layer.__class__.__name__:10s} -> {tuple(x.shape)}")


@dataclass
class Metrics:
    loss: float
    accuracy: float


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> Metrics:
    """训练或评估一轮；传入 optimizer 表示训练，否则表示评估。"""
    is_training = optimizer is not None
    model.train(is_training)  # 控制训练/评估模式；虽然本模型没有 BN/Dropout，也保留规范写法。
    loss_sum = 0.0
    correct = 0
    sample_count = 0

    context = torch.enable_grad() if is_training else torch.inference_mode()
    with context:  # 评估时关闭梯度记录，减少时间和显存开销。
        for features, targets in loader:
            features = features.to(device)  # 输入、标签和模型必须位于同一设备。
            targets = targets.to(device)
            logits = model(features)  # 前向：得到 (N,2) 原始类别分数。
            loss = loss_fn(logits, targets)  # 交叉熵直接接 logits 和 (N,) 的 long 标签。

            if is_training:
                optimizer.zero_grad(set_to_none=True)  # 清除上一批残留梯度。
                loss.backward()  # 沿计算图反向计算每个参数的梯度。
                optimizer.step()  # 优化器读取 .grad，真正更新参数。

            batch_size = targets.shape[0]
            loss_sum += loss.item() * batch_size  # 将批均值还原为批总和，避免末批加权错误。
            correct += (logits.argmax(dim=1) == targets).sum().item()  # 统计正确样本数。
            sample_count += batch_size

    return Metrics(loss=loss_sum / sample_count, accuracy=correct / sample_count)


def main(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)  # 固定参数初始化和 DataLoader 打乱顺序。
    requested = args.device
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)

    train_x, train_y = make_stripe_data(args.train_samples, seed=args.seed)
    test_x, test_y = make_stripe_data(args.test_samples, seed=args.seed + 1)
    train_loader = DataLoader(
        TensorDataset(train_x, train_y),  # Dataset 定义第 i 个输入与标签如何配对。
        batch_size=args.batch_size,
        shuffle=True,  # 每轮重排训练样本，避免固定批次顺序。
    )
    test_loader = DataLoader(
        TensorDataset(test_x, test_y),
        batch_size=args.batch_size,
        shuffle=False,  # 评估不需要随机顺序，结果更容易复现。
    )

    model = LeNet(num_classes=2).to(device)  # 创建模型并搬到目标设备。
    loss_fn = nn.CrossEntropyLoss()  # 内部稳定地组合 log-softmax 与负对数似然。
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)  # 管理全部可训练参数。

    trace_shapes(model, train_x[:2].to(device))
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, loss_fn, device, optimizer)
        test_metrics = run_epoch(model, test_loader, loss_fn, device)
        print(
            f"epoch {epoch:02d} | train loss {train_metrics.loss:.4f}, "
            f"acc {train_metrics.accuracy:.3f} | test acc {test_metrics.accuracy:.3f}"
        )

    assert test_metrics.accuracy >= 0.90, "合成任务准确率异常，请检查训练链路"
    print(f"smoke test 通过，设备={device}，测试准确率={test_metrics.accuracy:.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LeNet 离线合成数据 smoke test")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--train-samples", type=int, default=512)
    parser.add_argument("--test-samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
