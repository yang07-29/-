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
    # 固定 Python 自带随机库；data_iter 的 shuffle 使用它。
    random.seed(seed)
    # 固定 PyTorch 随机数；特征、噪声和参数初始化使用它。
    torch.manual_seed(seed)


def synthetic_data(
    true_w: Tensor,
    true_b: float,
    num_examples: int,
    noise_std: float = 0.01,
) -> tuple[Tensor, Tensor]:
    """生成 y = Xw + b + 噪声，返回 X:(N,d)、y:(N,1)。"""
    # 生成 N 个样本、d 个特征，所以 features 的 Shape 是 (N, d)。
    features = torch.randn(num_examples, true_w.numel())
    # 先把 true_w 从 (d,) 变为列向量 (d,1)，矩阵乘法得到 (N,1)。
    labels = features @ true_w.reshape(-1, 1) + true_b
    # 为每个标签加入同 Shape 的小高斯噪声，模拟现实中的测量误差。
    labels += torch.randn_like(labels) * noise_std
    # 返回特征与标签；后续会用 assert 固定两者的 Shape 契约。
    return features, labels


def data_iter(
    batch_size: int,
    features: Tensor,
    labels: Tensor,
) -> Iterator[tuple[Tensor, Tensor]]:
    """手写小批量迭代器；每个 epoch 都会重新打乱样本顺序。"""
    # 先创建 [0, 1, ..., N-1]，不直接打乱原始张量。
    indices = list(range(len(features)))
    # 每次调用 data_iter 都重新洗牌，相当于每个 epoch 改变样本顺序。
    random.shuffle(indices)
    # start 每次前进 batch_size；最后一批可以小于 batch_size。
    for start in range(0, len(indices), batch_size):
        # 取本批样本下标，并转成可用于张量高级索引的整数张量。
        batch_indices = torch.tensor(indices[start : start + batch_size])
        # yield 返回一批后暂停函数；下一次迭代会从此处继续。
        yield features[batch_indices], labels[batch_indices]


def linreg(features: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    """线性模型：X:(B,d) @ w:(d,1) + b:(1,) -> y_hat:(B,1)。"""
    # bias 会沿批量维广播；输出中的每一行对应一个样本的预测。
    return features @ weight + bias


def squared_loss(predictions: Tensor, targets: Tensor) -> Tensor:
    """返回逐样本平方损失，Shape 为 (B,1)。"""
    # 把标签强制变成与预测相同的 Shape，阻止 (B,1)-(B,) 广播成 (B,B)。
    targets = targets.reshape_as(predictions)
    # 暂不聚合，保留每个样本的损失；1/2 能让求导后的系数更简洁。
    return 0.5 * (predictions - targets) ** 2


def manual_sgd(parameters: list[Tensor], learning_rate: float) -> None:
    """使用已经算好的平均损失梯度更新参数，并清除旧梯度。"""
    # 更新参数不是模型的一部分，因此禁止 autograd 为更新动作建立计算图。
    with torch.no_grad():
        # weight 和 bias 都使用相同的 SGD 更新规则。
        for parameter in parameters:
            # 没参与当前损失计算的参数可能没有梯度，遇到它就跳过。
            if parameter.grad is None:
                continue
            # loss 已经取 mean，所以这里不再除以 batch_size。
            parameter -= learning_rate * parameter.grad
            # PyTorch 默认累加梯度；设为 None 让下一批从新梯度开始。
            parameter.grad = None


def report_result(name: str, weight: Tensor, bias: Tensor, true_w: Tensor, true_b: float) -> None:
    # detach 仅用于报告结果，避免后续打印操作仍挂在计算图上。
    learned_w = weight.detach().reshape_as(true_w)
    # item 把只有一个元素的张量转换成 Python 浮点数。
    learned_b = bias.detach().item()
    # 直接展示学到的参数，便于和真实参数肉眼比较。
    print(f"[{name}] 学到的 w: {learned_w.tolist()}")
    print(f"[{name}] 学到的 b: {learned_b:.4f}")
    # 最大绝对误差比只看 loss 更直接地验证是否找回真实权重。
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
    # 权重 Shape 是 (d,1)；小随机数初始化让初始预测接近 0。
    weight = torch.randn(true_w.numel(), 1) * 0.01
    # weight 由上一步运算产生，显式开启梯度追踪后成为待学习参数。
    weight.requires_grad_()
    # 偏置只有一个元素，并从 0 开始学习。
    bias = torch.zeros(1, requires_grad=True)

    # 外层循环表示完整看过数据集多少遍。
    for epoch in range(epochs):
        # 内层循环每次只拿一个 mini-batch，Shape 分别为 (B,d)、(B,1)。
        for batch_features, batch_labels in data_iter(batch_size, features, labels):
            # 1. Forward：用当前参数产生预测 (B,1)。
            predictions = linreg(batch_features, weight, bias)
            # 2. Loss：先得到逐样本损失，再取均值得到可反传的标量。
            loss = squared_loss(predictions, batch_labels).mean()
            # 3. Backward：计算 dloss/dweight 与 dloss/dbias，并写入 .grad。
            loss.backward()
            # 4. Update：读取 .grad 更新参数，同时清除本批梯度。
            manual_sgd([weight, bias], learning_rate)

        # epoch 结束后的整集评估不需要计算梯度。
        with torch.inference_mode():
            # 使用同一个损失定义计算整集平均损失，便于比较各 epoch。
            epoch_loss = squared_loss(linreg(features, weight, bias), labels).mean()
        # item 把标量张量转为 Python 数值，格式化到小数点后 8 位。
        print(f"[从零实现] epoch {epoch + 1:02d}, loss {epoch_loss.item():.8f}")

    # 用已知真实参数核对训练结果，这是合成数据实验最重要的验收步骤。
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
    # TensorDataset 定义“同一索引处的特征和标签组成一个样本”。
    data_loader = DataLoader(
        TensorDataset(features, labels),
        # 每次返回 batch_size 个样本，最后一批可以更小。
        batch_size=batch_size,
        # 训练阶段每个 epoch 自动打乱样本。
        shuffle=True,
    )
    # 一个 Linear 就是线性回归：输入 d 个特征，输出 1 个连续值。
    model = nn.Sequential(nn.Linear(true_w.numel(), 1))
    # 用很小的正态随机数初始化权重，避免初始预测过大。
    nn.init.normal_(model[0].weight, mean=0.0, std=0.01)
    # 偏置从 0 开始即可。
    nn.init.zeros_(model[0].bias)

    # MSELoss 默认返回当前批次的平均平方误差标量。
    loss_fn = nn.MSELoss()
    # 优化器接管 model.parameters() 中已经注册的 weight 与 bias。
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # 外层循环控制训练轮数。
    for epoch in range(epochs):
        # 切换训练模式；本模型无 Dropout/BN，但保留标准习惯。
        model.train()
        # DataLoader 自动完成打乱、切片和批量堆叠。
        for batch_features, batch_labels in data_loader:
            # 1. Forward：Linear 内部执行 X @ weight.T + bias。
            predictions = model(batch_features)
            # 2. Loss：比较 (B,1) 的预测和标签，得到标量平均损失。
            loss = loss_fn(predictions, batch_labels)

            # 3. 清除上一个批次留下的梯度；None 通常比写零更省内存。
            optimizer.zero_grad(set_to_none=True)
            # 4. 沿计算图反向计算每个参数的梯度。
            loss.backward()
            # 5. SGD 读取 .grad 并真正修改参数。
            optimizer.step()

        # 切换评估模式；更深模型中的 Dropout/BN 会因此改变行为。
        model.eval()
        # 评估阶段既不需要计算图，也不需要版本计数。
        with torch.inference_mode():
            # 一次前向计算全部训练样本的平均损失。
            epoch_loss = loss_fn(model(features), labels)
        # 展示训练是否按预期收敛。
        print(f"[简洁实现] epoch {epoch + 1:02d}, loss {epoch_loss.item():.8f}")

    # Linear 的权重内部保存为 (out_features,in_features)，report_result 会整理 Shape。
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
    # 读取命令行参数，方便不改源码就切换实现与超参数。
    args = parse_args()
    # 固定随机性，使每次运行结果接近，便于学习和调试。
    set_seed(args.seed)

    # 真实参数只用于造数据和最终核对，训练过程不能直接读取答案。
    true_w = torch.tensor([2.0, -3.4])
    true_b = 4.2
    # 生成 X:(N,2) 与 y:(N,1)。
    features, labels = synthetic_data(true_w, true_b, args.num_examples)
    # 尽早用断言固定数据接口；Shape 错误会在训练前直接暴露。
    assert features.shape == (args.num_examples, 2)
    assert labels.shape == (args.num_examples, 1)

    # 先打印 Shape，再看训练曲线；这是排错时最值得养成的习惯。
    print(f"数据 Shape: X={tuple(features.shape)}, y={tuple(labels.shape)}")
    # 根据参数选择是否运行从零实现。
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
    # 根据参数选择是否运行框架简洁实现。
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
