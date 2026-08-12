"""第 8 章：用滞后特征做一维时间序列预测。

这个例子对应 8.1“序列模型”。它不下载数据，运行后会同时展示：
1. 怎样把一条序列切成监督学习样本；
2. 单步预测为什么通常比递归多步预测稳定；
3. 时间序列为什么不能随意随机切分训练集和测试集。

运行：
    python code/ch08/sequence_forecasting.py --epochs 120
"""

from __future__ import annotations

import argparse

import torch
from torch import nn


def make_series(num_points: int) -> torch.Tensor:
    """生成含周期、慢趋势和小噪声的合成时间序列。"""
    # 固定随机种子，使每次运行都能复现实验。
    generator = torch.Generator().manual_seed(7)
    # time 的 Shape 是 (N,)，每个位置代表一个时间步。
    time = torch.arange(num_points, dtype=torch.float32)
    # 主周期负责可预测结构，慢周期模拟长期变化。
    clean = torch.sin(0.05 * time) + 0.35 * torch.sin(0.011 * time + 0.8)
    # 噪声让任务更接近真实观测，而不是死记一条完美曲线。
    noise = 0.04 * torch.randn(num_points, generator=generator)
    return clean + noise


def make_lagged_dataset(series: torch.Tensor, tau: int) -> tuple[torch.Tensor, torch.Tensor]:
    """把 [x_0,...,x_{N-1}] 变成“过去 tau 步 -> 下一步”的样本。"""
    # unfold 会创建滑动窗口；windows 的 Shape 是 (N-tau, tau+1)。
    windows = series.unfold(dimension=0, size=tau + 1, step=1)
    # 每行前 tau 个值是特征，Shape 为 (N-tau, tau)。
    features = windows[:, :-1].contiguous()
    # 每行最后一个值是标签；保留末维后 Shape 为 (N-tau, 1)。
    labels = windows[:, -1:].contiguous()
    return features, labels


def train_model(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    epochs: int,
) -> nn.Module:
    """在前段时间窗口上训练一个很小的 MLP。"""
    # 输入宽度就是 tau，输出是“下一个值”这 1 个连续数。
    model = nn.Sequential(
        nn.Linear(train_features.shape[1], 16),
        nn.ReLU(),
        nn.Linear(16, 1),
    )
    # 回归任务用均方误差衡量预测值与真实值的距离。
    loss_fn = nn.MSELoss()
    # Adam 只负责读取梯度并更新参数，梯度仍由 backward 计算。
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for epoch in range(1, epochs + 1):
        # 前向：predictions 与 train_labels 都是 (B, 1)，不会误广播。
        predictions = model(train_features)
        # Loss：把一批逐样本误差聚合成标量。
        loss = loss_fn(predictions, train_labels)
        # 清除上一轮留下的梯度，避免 PyTorch 默认累加梯度。
        optimizer.zero_grad(set_to_none=True)
        # 反向：为每个可训练参数计算当前损失的梯度。
        loss.backward()
        # 更新：Adam 根据参数的 .grad 真正修改参数值。
        optimizer.step()

        if epoch == 1 or epoch % max(1, epochs // 4) == 0:
            print(f"epoch={epoch:03d}, train_mse={loss.item():.6f}")
    return model


@torch.inference_mode()
def recursive_forecast(
    model: nn.Module,
    initial_window: torch.Tensor,
    num_predictions: int,
) -> torch.Tensor:
    """把上一步预测塞回输入，递归地产生多步预测。"""
    # window 的 Shape 是 (1, tau)，始终保留 batch 维。
    window = initial_window.reshape(1, -1).clone()
    predictions: list[torch.Tensor] = []
    for _ in range(num_predictions):
        # next_value 的 Shape 是 (1, 1)。
        next_value = model(window)
        predictions.append(next_value.squeeze(0))
        # 丢掉最旧值，并把“模型自己的猜测”拼到窗口末尾。
        window = torch.cat((window[:, 1:], next_value), dim=1)
    # 拼接后得到 (num_predictions,)，便于和真实序列逐点比较。
    return torch.cat(predictions, dim=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="一维时间序列预测 smoke test")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--num-points", type=int, default=800)
    parser.add_argument("--tau", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(7)

    series = make_series(args.num_points)
    features, labels = make_lagged_dataset(series, args.tau)

    # 按时间切分：较早窗口用于训练，较晚窗口用于测试，避免未来泄漏。
    split = int(0.7 * features.shape[0])
    train_features, train_labels = features[:split], labels[:split]
    test_features, test_labels = features[split:], labels[split:]

    print(
        "Shape:",
        f"series={tuple(series.shape)}",
        f"train_X={tuple(train_features.shape)}",
        f"train_y={tuple(train_labels.shape)}",
    )
    model = train_model(train_features, train_labels, args.epochs)

    model.eval()
    with torch.inference_mode():
        # 单步预测始终使用真实的过去 tau 步作为输入。
        one_step = model(test_features)
        one_step_mse = nn.functional.mse_loss(one_step, test_labels).item()

    horizon = min(args.horizon, test_labels.shape[0])
    # 递归预测从测试段第一个真实窗口起步，之后只吃自己的输出。
    recursive = recursive_forecast(model, test_features[0], horizon)
    recursive_target = test_labels[:horizon, 0]
    recursive_mse = nn.functional.mse_loss(recursive, recursive_target).item()

    assert one_step.shape == test_labels.shape
    assert torch.isfinite(one_step).all() and torch.isfinite(recursive).all()
    print(f"one_step_mse={one_step_mse:.6f}")
    print(f"recursive_{horizon}_step_mse={recursive_mse:.6f}")
    print("前 5 个递归预测:", [round(x, 3) for x in recursive[:5].tolist()])
    print("前 5 个真实值:  ", [round(x, 3) for x in recursive_target[:5].tolist()])
    print("说明：多步误差通常更大，因为每一步都会把旧误差带到下一步。")


if __name__ == "__main__":
    main()
