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
    # 固定 Python 随机数，避免辅助逻辑每次产生不同顺序。
    random.seed(seed)
    # 固定 CPU 上的 PyTorch 随机数。
    torch.manual_seed(seed)
    # 若使用 GPU，再固定所有 CUDA 设备的随机数。
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_softmax(logits: Tensor) -> Tensor:
    """把每行 logits 转成概率；减去最大值可避免 exp 上溢。"""
    # 每个样本单独找最大 logit；keepdim=True 保留 (B,1) 以便按行广播。
    shifted = logits - logits.max(dim=1, keepdim=True).values
    # 平移后所有输入都不大于 0，所以指数最大为 1，不会向上溢出。
    exp_values = shifted.exp()
    # 每一行除以本行指数和，使每个样本的类别概率之和为 1。
    return exp_values / exp_values.sum(dim=1, keepdim=True)


def cross_entropy_from_logits(logits: Tensor, targets: Tensor) -> Tensor:
    """稳定的逐样本交叉熵，等价于 -log softmax(logits)[真实类别]。"""
    # 创建 [0,1,...,B-1]，用于和 targets 配对选择真实类别位置。
    row_index = torch.arange(targets.numel(), device=targets.device)
    # 高级索引一次取出每个样本真实类别的 logit，Shape 为 (B,)。
    true_logits = logits[row_index, targets]
    # logsumexp 在对数域稳定计算 log(sum(exp(logits)))，避免先求极小概率。
    return torch.logsumexp(logits, dim=1) - true_logits


def accuracy_count(logits: Tensor, targets: Tensor) -> int:
    """返回预测正确的样本个数，而不是批准确率。"""
    # argmax 在类别维选择最高分；比较后求和得到本批正确个数。
    return int((logits.argmax(dim=1) == targets).sum().item())


def make_smoke_test_loaders(batch_size: int, seed: int) -> tuple[DataLoader, DataLoader]:
    """构造可学习的假图像，供离线快速检查完整训练链路。"""
    # 使用独立生成器，不影响训练参数初始化的全局随机状态。
    generator = torch.Generator().manual_seed(seed)
    # teacher 相当于隐藏的正确分类器，Shape 为 (784,10)。
    teacher = torch.randn(NUM_INPUTS, NUM_CLASSES, generator=generator)

    def make_dataset(size: int) -> TensorDataset:
        # 模拟 size 张单通道 28×28 图像，Shape 为 (N,1,28,28)。
        images = torch.randn(size, 1, 28, 28, generator=generator)
        # 线性分类器需要向量输入，所以展平为 (N,784)。
        flat = images.reshape(size, -1)
        # 用 teacher 生成可学习的类别分数，并加入一点噪声。
        logits = flat @ teacher + 0.1 * torch.randn(size, NUM_CLASSES, generator=generator)
        # 最高分所在类别作为标签，Shape 为 (N,)，dtype 自动为 long。
        labels = logits.argmax(dim=1)
        # Dataset 把同一索引处的图像和标签组成一个样本。
        return TensorDataset(images, labels)

    # 训练集需要打乱，避免每轮使用完全相同的批次组合。
    train_loader = DataLoader(make_dataset(1024), batch_size=batch_size, shuffle=True)
    # 测试集不更新参数，保持固定顺序更容易复现与定位样本。
    test_loader = DataLoader(make_dataset(256), batch_size=batch_size, shuffle=False)
    # 返回接口与真实 Fashion-MNIST 完全一致，训练代码无需改动。
    return train_loader, test_loader


def load_fashion_mnist(batch_size: int, data_root: str) -> tuple[DataLoader, DataLoader]:
    """首次运行会把 Fashion-MNIST 下载到 data_root。"""
    try:
        from torchvision import datasets, transforms
    except ImportError as error:
        raise RuntimeError("缺少 torchvision，请先运行：pip install torch torchvision") from error

    # ToTensor 把图像转换为 (C,H,W) 浮点张量，并把像素缩放到 [0,1]。
    transform = transforms.ToTensor()
    # 首次运行自动下载 60000 张训练图像。
    train_dataset = datasets.FashionMNIST(data_root, train=True, transform=transform, download=True)
    # 测试集有 10000 张图像，只用于评估泛化能力。
    test_dataset = datasets.FashionMNIST(data_root, train=False, transform=transform, download=True)
    # 训练阶段启用 shuffle，让相邻批次组合随 epoch 变化。
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # 测试阶段不需要打乱，也不会影响整集准确率。
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # 两个 DataLoader 都产出 X:(B,1,28,28)、y:(B,)。
    return train_loader, test_loader


@torch.inference_mode()
def evaluate(
    forward: Callable[[Tensor], Tensor],
    data_loader: DataLoader,
    device: torch.device,
) -> float:
    # 累加“正确个数”和“样本总数”，避免不同大小 batch 带来偏差。
    correct, total = 0, 0
    # 逐批读取验证或测试数据。
    for features, targets in data_loader:
        # 输入与标签必须和参数处在同一设备。
        features, targets = features.to(device), targets.to(device)
        # forward 可以是手写函数，也可以是 nn.Module。
        logits = forward(features)
        # 累加本批预测正确的样本个数。
        correct += accuracy_count(logits, targets)
        # y.numel() 就是当前批的真实样本数，兼容最后一个小批次。
        total += targets.numel()
    # 所有批次结束后只除一次，得到严格按样本加权的准确率。
    return correct / total


def train_from_scratch(
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    learning_rate: float,
    epochs: int,
) -> None:
    """手写参数、前向、交叉熵与 SGD。"""
    # 每个像素到每个类别都有一个权重，Shape 为 (784,10)。
    weight = (torch.randn(NUM_INPUTS, NUM_CLASSES, device=device) * 0.01).requires_grad_()
    # 每个类别各有一个偏置，Shape 为 (10,)。
    bias = torch.zeros(NUM_CLASSES, device=device, requires_grad=True)

    def forward(features: Tensor) -> Tensor:
        # 保留批量维，把单张图的 1×28×28 展成 784 维向量。
        flat = features.reshape(features.shape[0], -1)
        # (B,784)@(784,10)+(10,) -> logits:(B,10)。
        return flat @ weight + bias

    # 外层循环控制完整遍历训练集的次数。
    for epoch in range(epochs):
        # 每个 epoch 重新从 0 累加，三个变量都是总量而非批平均值。
        loss_sum, correct, total = 0.0, 0, 0

        # 逐批读取图像和类别索引。
        for features, targets in train_loader:
            # 数据和参数必须放到同一 CPU 或 GPU。
            features, targets = features.to(device), targets.to(device)
            # 1. Forward：输出原始 logits，不在模型末尾调用 Softmax。
            logits = forward(features)
            # 2. Loss：稳定计算每个样本的交叉熵，Shape 为 (B,)。
            loss_vector = cross_entropy_from_logits(logits, targets)
            # 对本批样本取平均，得到可直接 backward 的标量。
            loss = loss_vector.mean()
            # 3. Backward：把 weight、bias 的梯度写进各自 .grad。
            loss.backward()

            # 4. Update：参数更新不应被 autograd 记录。
            with torch.no_grad():
                # 对权重和偏置执行完全相同的 SGD 规则。
                for parameter in (weight, bias):
                    # loss 已取 mean，所以这里不再除 batch size。
                    parameter -= learning_rate * parameter.grad
                    # 清除旧梯度；否则下一批会在当前梯度上继续累加。
                    parameter.grad = None

            # 记录当前真实批量大小，兼容最后一个小批次。
            sample_count = targets.numel()
            # loss_vector 是逐样本损失，先求和再累加到整个 epoch。
            loss_sum += loss_vector.detach().sum().item()
            # logits 已用于完成 backward；统计指标前 detach 更清楚地表达不求梯度。
            correct += accuracy_count(logits.detach(), targets)
            # 累加实际样本数，后面作为损失和准确率的共同分母。
            total += sample_count

        # 使用独立测试集评估泛化能力；evaluate 已由 inference_mode 装饰。
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
    # Flatten: (B,1,28,28)->(B,784)；Linear: (B,784)->(B,10)。
    model = nn.Sequential(nn.Flatten(), nn.Linear(NUM_INPUTS, NUM_CLASSES)).to(device)
    # 单层分类器使用小正态权重，避免初始 logits 绝对值过大。
    nn.init.normal_(model[1].weight, mean=0.0, std=0.01)
    # 十个类别偏置都从 0 开始。
    nn.init.zeros_(model[1].bias)

    # CrossEntropyLoss 直接接 logits，内部稳定完成 log-softmax 与 NLL。
    loss_fn = nn.CrossEntropyLoss()
    # SGD 自动管理 model.parameters() 中注册的权重和偏置。
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # 外层循环控制训练轮数。
    for epoch in range(epochs):
        # 切到训练模式；以后加入 Dropout/BatchNorm 时这行会影响行为。
        model.train()
        # 每轮重新统计总损失、正确数和样本数。
        loss_sum, correct, total = 0.0, 0, 0

        # DataLoader 每次返回 X:(B,1,28,28)、y:(B,)。
        for features, targets in train_loader:
            # 保证数据和模型位于同一设备。
            features, targets = features.to(device), targets.to(device)
            # 1. Forward：模型直接输出 (B,10) logits。
            logits = model(features)
            # 2. Loss：默认 reduction='mean'，返回本批平均损失标量。
            loss = loss_fn(logits, targets)

            # 3. 清除旧梯度；设置 None 通常更省内存写入。
            optimizer.zero_grad(set_to_none=True)
            # 4. 反向传播只计算梯度，不修改参数。
            loss.backward()
            # 5. 优化器读取 .grad，并按 SGD 规则修改参数。
            optimizer.step()

            # 获取当前实际批量大小，不能假定最后一批仍等于 batch_size。
            sample_count = targets.numel()
            # loss 是批平均值，乘样本数后才可与其他批正确汇总。
            loss_sum += loss.item() * sample_count
            # 准确率不可导，只做统计；detach 避免把指标逻辑挂在计算图上。
            correct += accuracy_count(logits.detach(), targets)
            # 累加当前批样本数。
            total += sample_count

        # 评估前切换模块行为；虽无 Dropout/BN，仍保留通用模板。
        model.eval()
        # evaluate 内部关闭梯度，并按全体测试样本统计准确率。
        test_accuracy = evaluate(model, test_loader, device)
        print(
            f"[简洁实现] epoch {epoch + 1:02d}, "
            f"loss {loss_sum / total:.4f}, "
            f"train acc {correct / total:.4f}, test acc {test_accuracy:.4f}"
        )

    # Linear 权重保存为 (输出维,输入维)=(10,784)，偏置为 (10,)。
    print("参数 Shape:", tuple(model[1].weight.shape), tuple(model[1].bias.shape))
    # 总参数量为 10×784 个权重加 10 个偏置，即 7850。
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
    # auto 优先使用 CUDA；没有可用 GPU 时自动退回 CPU。
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 用户明确要求 CUDA 时快速失败，避免稍后出现含糊的设备错误。
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA，但当前 PyTorch 无法使用 CUDA")
    # cpu 或可用的 cuda 字符串在这里转换为 torch.device。
    return torch.device(name)


def main() -> None:
    # 读取命令行参数，方便切换实现、数据和设备。
    args = parse_args()
    # 固定实验随机性，便于复现损失和准确率变化。
    set_seed(args.seed)
    # 根据参数与硬件能力选择训练设备。
    device = choose_device(args.device)

    # smoke-test 不访问网络，适合先验证程序结构。
    if args.smoke_test:
        train_loader, test_loader = make_smoke_test_loaders(args.batch_size, args.seed)
        print("数据：离线可学习假数据（仅用于检查代码）")
    # 正式示例下载并加载 Fashion-MNIST。
    else:
        train_loader, test_loader = load_fashion_mnist(args.batch_size, args.data_root)
        print("数据：Fashion-MNIST")
    # 明确打印设备，排查 CPU/GPU 混用问题。
    print("设备：", device)

    # 在训练前先取一批数据，验证数据接口是否满足模型预期。
    sample_features, sample_targets = next(iter(train_loader))
    # 图像必须是 (B,1,28,28)。
    assert sample_features.ndim == 4 and sample_features.shape[1:] == (1, 28, 28)
    # CrossEntropyLoss 需要一维 long 类型类别索引。
    assert sample_targets.ndim == 1 and sample_targets.dtype == torch.long
    # 主动显示批量 Shape，方便学习者把代码与公式对上。
    print(f"单批 Shape: X={tuple(sample_features.shape)}, y={tuple(sample_targets.shape)}")

    # 根据命令行参数运行从零实现。
    if args.implementation in {"scratch", "both"}:
        train_from_scratch(train_loader, test_loader, device, args.learning_rate, args.epochs)
    # 根据命令行参数运行框架简洁实现。
    if args.implementation in {"concise", "both"}:
        train_concise(train_loader, test_loader, device, args.learning_rate, args.epochs)


if __name__ == "__main__":
    main()
