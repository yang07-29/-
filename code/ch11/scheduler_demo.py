"""第 11 章：在同一离线分类任务上比较固定、阶梯和余弦学习率。

运行：python code/ch11/scheduler_demo.py --epochs 20
只依赖 PyTorch，不下载数据。
"""

from __future__ import annotations

import argparse
import copy

import torch
from torch import nn


def make_classification_data(num_examples: int = 900) -> tuple[torch.Tensor, torch.Tensor]:
    features = 2.0 * torch.rand(num_examples, 2) - 1.0  # (N,2)，在正方形内采样。
    radius = features.square().sum(dim=1)  # (N,)，计算到原点的平方距离。
    labels = (radius > 0.45).long()  # (N,)，圆外为 1、圆内为 0。
    return features, labels  # 返回离线合成数据。


class TinyMLP(nn.Module):
    """两层 MLP，足以学习非线性圆形边界。"""

    def __init__(self) -> None:
        super().__init__()  # 注册网络层。
        self.network = nn.Sequential(  # 组合成可调用模块。
            nn.Linear(2, 24),  # (B,2) -> (B,24)。
            nn.ReLU(),  # 引入非线性。
            nn.Linear(24, 2),  # (B,24) -> (B,2) 类别 logits。
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)  # 前向不改变参数，只建立计算图。


def train_one_schedule(
    name: str,
    initial_state: dict[str, torch.Tensor],
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    epochs: int,
) -> tuple[float, list[float]]:
    model = TinyMLP()  # 每个调度策略使用新的模型对象。
    model.load_state_dict(copy.deepcopy(initial_state))  # 恢复完全相同的初始参数。
    optimizer = torch.optim.SGD(model.parameters(), lr=0.4, momentum=0.9)  # 优化器和初始 lr 相同。

    if name == "fixed":  # 固定学习率不需要 scheduler。
        scheduler = None  # 每轮 lr 都保持 0.4。
    elif name == "step":  # 每隔固定轮数乘衰减系数。
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.3)
    elif name == "cosine":  # 平滑地从初始学习率退火到 eta_min。
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.01)
    else:  # 未知名称应尽早失败。
        raise ValueError(f"未知调度策略: {name}")

    learning_rates: list[float] = []  # 保存每轮实际用于更新的学习率。
    batch_size = 64  # 小批量在吞吐量与梯度噪声间折中。
    model.train()  # 开启训练模式。

    for _ in range(epochs):  # 每轮遍历一次训练集。
        learning_rates.append(optimizer.param_groups[0]["lr"])  # 在更新前记录本轮 lr。
        permutation = torch.randperm(train_x.shape[0])  # 每轮重新打乱样本。
        for start in range(0, train_x.shape[0], batch_size):  # 按 batch 取数据。
            indices = permutation[start : start + batch_size]  # 当前 batch 索引。
            batch_x = train_x[indices]  # (B,2)。
            batch_y = train_y[indices]  # (B,)。

            optimizer.zero_grad(set_to_none=True)  # 清除上批梯度，参数不变。
            logits = model(batch_x)  # (B,2)，前向建立计算图。
            loss = nn.functional.cross_entropy(logits, batch_y)  # 标量分类损失。
            loss.backward()  # 计算所有参数梯度。
            optimizer.step()  # 用当前学习率和动量真正更新参数。

        if scheduler is not None:  # 固定策略不调用调度器。
            scheduler.step()  # epoch 结束后更新下一轮学习率。

    model.eval()  # 进入评估模式。
    with torch.inference_mode():  # 不建立计算图，节省内存。
        predictions = model(test_x).argmax(dim=1)  # (N_test,)，取最大 logit 类别。
        accuracy = predictions.eq(test_y).float().mean().item()  # 标量准确率。

    print(  # 同时报告性能与 lr 轨迹关键点。
        f"{name:6s} accuracy={accuracy:.3f} "
        f"lr(first/middle/last)={learning_rates[0]:.3f}/"
        f"{learning_rates[len(learning_rates)//2]:.3f}/{learning_rates[-1]:.3f}"
    )
    return accuracy, learning_rates  # 返回结果供断言和进一步作图。


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="比较学习率调度器")  # 命令行入口。
    parser.add_argument("--epochs", type=int, default=20)  # 默认轮数足以观察退火。
    return parser.parse_args()  # 解析参数。


def main() -> None:
    args = parse_args()  # 读取用户配置。
    torch.manual_seed(31)  # 固定数据与初始参数。
    torch.set_num_threads(1)  # 小模型使用单线程减少开销。
    features, labels = make_classification_data()  # (900,2)、(900,)。
    train_x, test_x = features[:700], features[700:]  # 固定训练/测试划分。
    train_y, test_y = labels[:700], labels[700:]  # 标签使用相同切分。
    initial_state = TinyMLP().state_dict()  # 所有策略共享同一起点。

    results = {}  # 保存三种策略的结果。
    for name in ("fixed", "step", "cosine"):  # 逐一训练，避免共享 optimizer 状态。
        torch.manual_seed(32)  # 每种策略使用相同 batch 打乱序列。
        results[name] = train_one_schedule(  # 执行完整训练、评估与记录。
            name, initial_state, train_x, train_y, test_x, test_y, args.epochs
        )

    assert all(0.0 <= accuracy <= 1.0 for accuracy, _ in results.values())  # 准确率范围检查。
    assert results["step"][1][-1] < results["step"][1][0]  # StepLR 必须实际降低学习率。
    assert results["cosine"][1][-1] < results["cosine"][1][0]  # 余弦退火也必须降低学习率。
    print("调度器实验通过：调度器只改变 optimizer 的 lr，不替代 backward 或 step。")


if __name__ == "__main__":  # 直接运行文件时才执行训练。
    main()
