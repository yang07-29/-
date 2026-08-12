"""第三章：Softmax 回归的从零实现与 PyTorch 简洁实现。

无需下载数据的快速验证：
    python softmax_regression.py --smoke-test --implementation both --epochs 2

使用 Fashion-MNIST：
    python softmax_regression.py --implementation concise --epochs 10
"""

from __future__ import annotations

import argparse
import random
from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset


NUM_INPUTS = 28 * 28
NUM_CLASSES = 10


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_softmax(logits: Tensor) -> Tensor:
    """把每行 logits 转成概率；减去最大值可避免 exp 上溢。"""
    shifted = logits - logits.max(dim=1, keepdim=True).values
    exp_values = shifted.exp()
    return exp_values / exp_values.sum(dim=1, keepdim=True)


def cross_entropy_from_logits(logits: Tensor, targets: Tensor) -> Tensor:
    """稳定的逐样本交叉熵，等价于 -log softmax(logits)[真实类别]。"""
    row_index = torch.arange(targets.numel(), device=targets.device)
    true_logits = logits[row_index, targets]
    return torch.logsumexp(logits, dim=1) - true_logits


def accuracy_count(logits: Tensor, targets: Tensor) -> int:
    """返回预测正确的样本个数，而不是批准确率。"""
    return int((logits.argmax(dim=1) == targets).sum().item())


def make_smoke_test_loaders(batch_size: int, seed: int) -> tuple[DataLoader, DataLoader]:
    """构造可学习的假图像，供离线快速检查完整训练链路。"""
    generator = torch.Generator().manual_seed(seed)
    teacher = torch.randn(NUM_INPUTS, NUM_CLASSES, generator=generator)

    def make_dataset(size: int) -> TensorDataset:
        images = torch.randn(size, 1, 28, 28, generator=generator)
        flat = images.reshape(size, -1)
        logits = flat @ teacher + 0.1 * torch.randn(size, NUM_CLASSES, generator=generator)
        labels = logits.argmax(dim=1)
        return TensorDataset(images, labels)

    train_loader = DataLoader(make_dataset(1024), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(make_dataset(256), batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def load_fashion_mnist(batch_size: int, data_root: str) -> tuple[DataLoader, DataLoader]:
    """首次运行会把 Fashion-MNIST 下载到 data_root。"""
    try:
        from torchvision import datasets, transforms
    except ImportError as error:
        raise RuntimeError("缺少 torchvision，请先运行：pip install torch torchvision") from error

    transform = transforms.ToTensor()
    train_dataset = datasets.FashionMNIST(data_root, train=True, transform=transform, download=True)
    test_dataset = datasets.FashionMNIST(data_root, train=False, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


@torch.inference_mode()
def evaluate(
    forward: Callable[[Tensor], Tensor],
    data_loader: DataLoader,
    device: torch.device,
) -> float:
    correct, total = 0, 0
    for features, targets in data_loader:
        features, targets = features.to(device), targets.to(device)
        logits = forward(features)
        correct += accuracy_count(logits, targets)
        total += targets.numel()
    return correct / total


def train_from_scratch(
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    learning_rate: float,
    epochs: int,
) -> None:
    """手写参数、前向、交叉熵与 SGD。"""
    weight = (torch.randn(NUM_INPUTS, NUM_CLASSES, device=device) * 0.01).requires_grad_()
    bias = torch.zeros(NUM_CLASSES, device=device, requires_grad=True)

    def forward(features: Tensor) -> Tensor:
        flat = features.reshape(features.shape[0], -1)
        return flat @ weight + bias

    for epoch in range(epochs):
        loss_sum, correct, total = 0.0, 0, 0

        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            logits = forward(features)
            loss_vector = cross_entropy_from_logits(logits, targets)
            loss = loss_vector.mean()
            loss.backward()

            with torch.no_grad():
                for parameter in (weight, bias):
                    parameter -= learning_rate * parameter.grad
                    parameter.grad = None

            sample_count = targets.numel()
            loss_sum += loss_vector.detach().sum().item()
            correct += accuracy_count(logits.detach(), targets)
            total += sample_count

        test_accuracy = evaluate(forward, test_loader, device)
        print(
            f"[从零实现] epoch {epoch + 1:02d}, "
            f"loss {loss_sum / total:.4f}, "
            f"train acc {correct / total:.4f}, test acc {test_accuracy:.4f}"
        )


def train_concise(
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    learning_rate: float,
    epochs: int,
) -> None:
    """使用 Flatten、Linear、CrossEntropyLoss 与 Optimizer。"""
    model = nn.Sequential(nn.Flatten(), nn.Linear(NUM_INPUTS, NUM_CLASSES)).to(device)
    nn.init.normal_(model[1].weight, mean=0.0, std=0.01)
    nn.init.zeros_(model[1].bias)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        loss_sum, correct, total = 0.0, 0, 0

        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            logits = model(features)
            loss = loss_fn(logits, targets)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            sample_count = targets.numel()
            loss_sum += loss.item() * sample_count
            correct += accuracy_count(logits.detach(), targets)
            total += sample_count

        model.eval()
        test_accuracy = evaluate(model, test_loader, device)
        print(
            f"[简洁实现] epoch {epoch + 1:02d}, "
            f"loss {loss_sum / total:.4f}, "
            f"train acc {correct / total:.4f}, test acc {test_accuracy:.4f}"
        )

    print("参数 Shape:", tuple(model[1].weight.shape), tuple(model[1].bias.shape))
    print("参数总数:", sum(parameter.numel() for parameter in model.parameters()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Softmax 回归：从零实现与简洁实现")
    parser.add_argument("--implementation", choices=("scratch", "concise", "both"), default="concise")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--smoke-test", action="store_true", help="使用离线假数据快速验证")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA，但当前 PyTorch 无法使用 CUDA")
    return torch.device(name)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)

    if args.smoke_test:
        train_loader, test_loader = make_smoke_test_loaders(args.batch_size, args.seed)
        print("数据：离线可学习假数据（仅用于检查代码）")
    else:
        train_loader, test_loader = load_fashion_mnist(args.batch_size, args.data_root)
        print("数据：Fashion-MNIST")
    print("设备：", device)

    sample_features, sample_targets = next(iter(train_loader))
    assert sample_features.ndim == 4 and sample_features.shape[1:] == (1, 28, 28)
    assert sample_targets.ndim == 1 and sample_targets.dtype == torch.long
    print(f"单批 Shape: X={tuple(sample_features.shape)}, y={tuple(sample_targets.shape)}")

    if args.implementation in {"scratch", "both"}:
        train_from_scratch(train_loader, test_loader, device, args.learning_rate, args.epochs)
    if args.implementation in {"concise", "both"}:
        train_concise(train_loader, test_loader, device, args.learning_rate, args.epochs)


if __name__ == "__main__":
    main()
