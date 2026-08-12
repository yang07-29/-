"""4.8 比较初始化对深层激活与梯度尺度的影响。

直接运行：python code/ch04/initialization_stability.py
"""

from collections.abc import Callable

import torch
from torch import nn


def build_network(
    depth: int,
    width: int,
    activation_factory: Callable[[], nn.Module],
    initialize: Callable[[torch.Tensor], None],
) -> nn.Sequential:
    """构造等宽深层网络，并按给定规则初始化。"""
    layers: list[nn.Module] = []
    for _ in range(depth):
        # 每个 Linear 都保持 (B,width) -> (B,width)。
        linear = nn.Linear(width, width)
        # 初始化只改数值，不改变 weight.shape == (width,width)。
        initialize(linear.weight)
        nn.init.zeros_(linear.bias)
        layers.extend([linear, activation_factory()])
    return nn.Sequential(*layers)


def diagnose(name: str, model: nn.Sequential, X: torch.Tensor) -> dict[str, float]:
    """记录首尾激活标准差与第一层梯度标准差。"""
    # 复制输入，避免不同模型共享输入梯度状态。
    H = X.detach().clone().requires_grad_(True)
    activation_stds: list[float] = []
    for layer in model:
        # forward：H 的 Shape 始终是 (B,width)。
        H = layer(H)
        if isinstance(layer, (nn.ReLU, nn.Sigmoid, nn.Tanh)):
            activation_stds.append(H.detach().std().item())

    # loss 归约为标量，才能直接 backward。
    loss = H.square().mean()
    # 模型是新建的，梯度初始为 None；这里直接反传。
    loss.backward()
    # 第一个参数是第一层 Linear.weight。
    first_weight = next(model.parameters())
    gradient_std = first_weight.grad.std().item()
    values = {
        "first_activation_std": activation_stds[0],
        "last_activation_std": activation_stds[-1],
        "first_gradient_std": gradient_std,
    }
    print(
        f"{name:18s} first_act={values['first_activation_std']:.3e} "
        f"last_act={values['last_activation_std']:.3e} "
        f"first_grad={values['first_gradient_std']:.3e}"
    )
    return values


def symmetry_demo() -> None:
    """验证隐藏单元全零初始化会收到相同梯度。"""
    layer = nn.Linear(4, 3)
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
    # X.shape == (5,4)，layer(X).shape == (5,3)。
    X = torch.randn(5, 4)
    # 三个输出神经元的处境完全对称。
    loss = layer(X).sum()
    loss.backward()
    same = torch.allclose(layer.weight.grad[0], layer.weight.grad[1])
    print("零初始化后不同神经元梯度相同：", same)
    assert same


def main() -> None:
    torch.manual_seed(23)
    # 一个批次 128 个样本，每个样本 64 维。
    X = torch.randn(128, 64)
    depth, width = 16, 64

    # Xavier 常用于近线性或对称激活；这里搭配 Tanh。
    tanh_xavier = build_network(
        depth, width, nn.Tanh, nn.init.xavier_uniform_
    )
    # He/Kaiming 针对 ReLU 丢弃约一半负激活的方差变化。
    relu_he = build_network(
        depth,
        width,
        nn.ReLU,
        lambda weight: nn.init.kaiming_normal_(weight, nonlinearity="relu"),
    )
    # std=1 对 64 维输入过大，用来演示激活/梯度爆炸风险。
    relu_too_large = build_network(
        depth, width, nn.ReLU, lambda weight: nn.init.normal_(weight, std=1.0)
    )

    tanh_values = diagnose("Tanh + Xavier", tanh_xavier, X)
    he_values = diagnose("ReLU + He", relu_he, X)
    large_values = diagnose("ReLU + std=1", relu_too_large, X)
    symmetry_demo()

    # smoke test 只验证明显过大的初始化确实比 He 产生更大的末层尺度。
    assert large_values["last_activation_std"] > he_values["last_activation_std"]
    assert torch.isfinite(torch.tensor(list(tanh_values.values()))).all()


if __name__ == "__main__":
    main()
