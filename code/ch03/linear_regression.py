"""第三章：线性回归的从零实现与 PyTorch 简洁实现。

运行：
    python linear_regression.py
    python linear_regression.py --implementation scratch
    python linear_regression.py --implementation concise --epochs 5
"""

from __future__ import annotations

import argparse
import random
from collections.abc import Iterator

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int) -> None:
    """固定 Python 与 PyTorch 随机数，方便复现实验。"""
    random.seed(seed)
    torch.manual_seed(seed)


def synthetic_data(
    true_w: Tensor,
    true_b: float,
    num_examples: int,
    noise_std: float = 0.01,
) -> tuple[Tensor, Tensor]:
    """生成 y = Xw + b + 噪声，返回 X:(N,d)、y:(N,1)。"""
    features = torch.randn(num_examples, true_w.numel())
    labels = features @ true_w.reshape(-1, 1) + true_b
    labels += torch.randn_like(labels) * noise_std
    return features, labels


def data_iter(
    batch_size: int,
    features: Tensor,
    labels: Tensor,
) -> Iterator[tuple[Tensor, Tensor]]:
    """手写小批量迭代器；每个 epoch 都会重新打乱样本顺序。"""
    indices = list(range(len(features)))
    random.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_indices = torch.tensor(indices[start : start + batch_size])
        yield features[batch_indices], labels[batch_indices]


def linreg(features: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    """线性模型：X:(B,d) @ w:(d,1) + b:(1,) -> y_hat:(B,1)。"""
    return features @ weight + bias


def squared_loss(predictions: Tensor, targets: Tensor) -> Tensor:
    """返回逐样本平方损失，Shape 为 (B,1)。"""
    targets = targets.reshape_as(predictions)
    return 0.5 * (predictions - targets) ** 2


def manual_sgd(parameters: list[Tensor], learning_rate: float) -> None:
    """使用已经算好的平均损失梯度更新参数，并清除旧梯度。"""
    with torch.no_grad():
        for parameter in parameters:
            if parameter.grad is None:
                continue
            parameter -= learning_rate * parameter.grad
            parameter.grad = None


def report_result(name: str, weight: Tensor, bias: Tensor, true_w: Tensor, true_b: float) -> None:
    learned_w = weight.detach().reshape_as(true_w)
    learned_b = bias.detach().item()
    print(f"[{name}] 学到的 w: {learned_w.tolist()}")
    print(f"[{name}] 学到的 b: {learned_b:.4f}")
    print(f"[{name}] w 最大绝对误差: {(true_w - learned_w).abs().max().item():.6f}")
    print(f"[{name}] b 绝对误差: {abs(true_b - learned_b):.6f}")


def train_from_scratch(
    features: Tensor,
    labels: Tensor,
    true_w: Tensor,
    true_b: float,
    batch_size: int,
    learning_rate: float,
    epochs: int,
) -> None:
    """手写模型、损失和 SGD，自动求导仍由 PyTorch 完成。"""
    weight = torch.randn(true_w.numel(), 1) * 0.01
    weight.requires_grad_()
    bias = torch.zeros(1, requires_grad=True)

    for epoch in range(epochs):
        for batch_features, batch_labels in data_iter(batch_size, features, labels):
            predictions = linreg(batch_features, weight, bias)
            loss = squared_loss(predictions, batch_labels).mean()
            loss.backward()
            manual_sgd([weight, bias], learning_rate)

        with torch.inference_mode():
            epoch_loss = squared_loss(linreg(features, weight, bias), labels).mean()
        print(f"[从零实现] epoch {epoch + 1:02d}, loss {epoch_loss.item():.8f}")

    report_result("从零实现", weight, bias, true_w, true_b)


def train_concise(
    features: Tensor,
    labels: Tensor,
    true_w: Tensor,
    true_b: float,
    batch_size: int,
    learning_rate: float,
    epochs: int,
) -> None:
    """使用 DataLoader、nn.Linear、MSELoss 与 Optimizer。"""
    data_loader = DataLoader(
        TensorDataset(features, labels),
        batch_size=batch_size,
        shuffle=True,
    )
    model = nn.Sequential(nn.Linear(true_w.numel(), 1))
    nn.init.normal_(model[0].weight, mean=0.0, std=0.01)
    nn.init.zeros_(model[0].bias)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        for batch_features, batch_labels in data_loader:
            predictions = model(batch_features)
            loss = loss_fn(predictions, batch_labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.inference_mode():
            epoch_loss = loss_fn(model(features), labels)
        print(f"[简洁实现] epoch {epoch + 1:02d}, loss {epoch_loss.item():.8f}")

    report_result("简洁实现", model[0].weight, model[0].bias, true_w, true_b)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="线性回归：从零实现与简洁实现")
    parser.add_argument("--implementation", choices=("scratch", "concise", "both"), default="both")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-examples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    true_w = torch.tensor([2.0, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, args.num_examples)
    assert features.shape == (args.num_examples, 2)
    assert labels.shape == (args.num_examples, 1)

    print(f"数据 Shape: X={tuple(features.shape)}, y={tuple(labels.shape)}")
    if args.implementation in {"scratch", "both"}:
        train_from_scratch(
            features,
            labels,
            true_w,
            true_b,
            args.batch_size,
            args.learning_rate,
            args.epochs,
        )
    if args.implementation in {"concise", "both"}:
        train_concise(
            features,
            labels,
            true_w,
            true_b,
            args.batch_size,
            args.learning_rate,
            args.epochs,
        )


if __name__ == "__main__":
    main()
