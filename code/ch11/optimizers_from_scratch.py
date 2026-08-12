"""第 11 章：SGD、动量、AdaGrad、RMSProp、Adadelta、Adam 从零实现对照。

运行：python code/ch11/optimizers_from_scratch.py
只依赖 PyTorch，不下载数据。
"""

from __future__ import annotations

from collections.abc import Callable

import torch


TensorState = dict[str, torch.Tensor | int]  # 优化器状态可能包含张量或整数时间步。
UpdateFunction = Callable[[torch.Tensor, torch.Tensor, TensorState], None]  # 统一更新函数签名。


def objective(parameters: torch.Tensor) -> torch.Tensor:
    """一个狭长椭圆等高线的凸目标，最优点是 (0,0)。"""
    return 0.1 * parameters[0].square() + 2.0 * parameters[1].square()  # 标量损失。


@torch.no_grad()
def sgd_update(parameters: torch.Tensor, gradient: torch.Tensor, state: TensorState) -> None:
    learning_rate = 0.4  # SGD 每个方向使用同一个固定步长。
    parameters.add_(gradient, alpha=-learning_rate)  # theta <- theta - eta*g。


@torch.no_grad()
def momentum_update(parameters: torch.Tensor, gradient: torch.Tensor, state: TensorState) -> None:
    learning_rate, beta = 0.15, 0.9  # beta 决定历史速度保留比例。
    velocity = state.setdefault("velocity", torch.zeros_like(parameters))  # 首次创建 v_0=0。
    assert isinstance(velocity, torch.Tensor)  # 帮助类型检查器理解状态类型。
    velocity.mul_(beta).add_(gradient)  # v_t <- beta*v_(t-1)+g_t。
    parameters.add_(velocity, alpha=-learning_rate)  # theta <- theta-eta*v_t。


@torch.no_grad()
def adagrad_update(parameters: torch.Tensor, gradient: torch.Tensor, state: TensorState) -> None:
    learning_rate, epsilon = 0.8, 1e-6  # epsilon 防止分母为 0。
    accumulator = state.setdefault("squares", torch.zeros_like(parameters))  # s_0=0。
    assert isinstance(accumulator, torch.Tensor)  # 明确状态是张量。
    accumulator.addcmul_(gradient, gradient)  # s_t <- s_(t-1)+g_t^2。
    adjusted_gradient = gradient / (accumulator.sqrt() + epsilon)  # 历史梯度大的维度步长变小。
    parameters.add_(adjusted_gradient, alpha=-learning_rate)  # 执行坐标级自适应更新。


@torch.no_grad()
def rmsprop_update(parameters: torch.Tensor, gradient: torch.Tensor, state: TensorState) -> None:
    learning_rate, beta, epsilon = 0.2, 0.9, 1e-6  # beta 控制平方梯度移动平均窗口。
    average_square = state.setdefault("squares", torch.zeros_like(parameters))  # s_0=0。
    assert isinstance(average_square, torch.Tensor)  # 明确状态类型。
    average_square.mul_(beta).addcmul_(gradient, gradient, value=1.0 - beta)  # EMA(g^2)。
    adjusted_gradient = gradient / (average_square.sqrt() + epsilon)  # 按近期尺度归一化。
    parameters.add_(adjusted_gradient, alpha=-learning_rate)  # 更新参数。


@torch.no_grad()
def adadelta_update(parameters: torch.Tensor, gradient: torch.Tensor, state: TensorState) -> None:
    rho, epsilon = 0.9, 1e-5  # Adadelta 通常不需要显式全局学习率。
    average_square_grad = state.setdefault("grad_squares", torch.zeros_like(parameters))  # E[g^2]_0。
    average_square_delta = state.setdefault("delta_squares", torch.zeros_like(parameters))  # E[delta^2]_0。
    assert isinstance(average_square_grad, torch.Tensor)  # 明确第一份状态类型。
    assert isinstance(average_square_delta, torch.Tensor)  # 明确第二份状态类型。
    average_square_grad.mul_(rho).addcmul_(gradient, gradient, value=1.0 - rho)  # 更新 E[g^2]。
    rms_delta = (average_square_delta + epsilon).sqrt()  # 过去参数更新的 RMS。
    rms_gradient = (average_square_grad + epsilon).sqrt()  # 当前梯度尺度的 RMS。
    delta = -(rms_delta / rms_gradient) * gradient  # 让更新量具有类似参数的尺度。
    parameters.add_(delta)  # theta <- theta+delta。
    average_square_delta.mul_(rho).addcmul_(delta, delta, value=1.0 - rho)  # 更新 E[delta^2]。


@torch.no_grad()
def adam_update(parameters: torch.Tensor, gradient: torch.Tensor, state: TensorState) -> None:
    learning_rate, beta1, beta2, epsilon = 0.2, 0.9, 0.999, 1e-8  # Adam 常用超参数。
    first_moment = state.setdefault("first", torch.zeros_like(parameters))  # m_0=0。
    second_moment = state.setdefault("second", torch.zeros_like(parameters))  # v_0=0。
    step = int(state.get("step", 0)) + 1  # 时间步从 1 开始，供偏差修正使用。
    state["step"] = step  # 把新时间步写回状态。
    assert isinstance(first_moment, torch.Tensor)  # 明确一阶矩类型。
    assert isinstance(second_moment, torch.Tensor)  # 明确二阶矩类型。
    first_moment.mul_(beta1).add_(gradient, alpha=1.0 - beta1)  # m_t=beta1*m+(1-beta1)g。
    second_moment.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)  # v_t 的 EMA。
    corrected_first = first_moment / (1.0 - beta1**step)  # 修正初期 m 偏向 0。
    corrected_second = second_moment / (1.0 - beta2**step)  # 修正初期 v 偏向 0。
    direction = corrected_first / (corrected_second.sqrt() + epsilon)  # 动量方向除以坐标尺度。
    parameters.add_(direction, alpha=-learning_rate)  # theta <- theta-eta*direction。


def optimize(name: str, update: UpdateFunction, steps: int = 60) -> tuple[torch.Tensor, list[float]]:
    parameters = torch.tensor([-5.0, -2.0], requires_grad=True)  # 所有算法从同一点开始。
    state: TensorState = {}  # 每个算法拥有独立状态，不能互相污染。
    losses: list[float] = []  # 保存轨迹用于检查是否真正收敛。

    for _ in range(steps):  # 每次循环执行一个完整优化步。
        loss = objective(parameters)  # 前向得到标量损失并建立计算图。
        if parameters.grad is not None:  # 第一步之前 grad 为 None。
            parameters.grad.zero_()  # 清除累积梯度，不改变参数本身。
        loss.backward()  # 计算当前点的解析梯度。
        gradient = parameters.grad.detach().clone()  # 复制梯度，状态更新不应进入计算图。
        update(parameters, gradient, state)  # 在 no_grad 下真正改变参数。
        losses.append(loss.item())  # 保存脱离计算图的数值。

    print(  # 用统一格式便于横向比较。
        f"{name:9s} final_theta={parameters.detach().tolist()} "
        f"initial_loss={losses[0]:.4f} final_loss={losses[-1]:.6f}"
    )
    return parameters.detach(), losses  # 返回轨迹供断言与复用。


def main() -> None:
    torch.manual_seed(23)  # 当前目标无随机性，保留种子作为实验模板。
    algorithms: list[tuple[str, UpdateFunction]] = [  # 固定对照顺序。
        ("SGD", sgd_update),
        ("Momentum", momentum_update),
        ("AdaGrad", adagrad_update),
        ("RMSProp", rmsprop_update),
        ("Adadelta", adadelta_update),
        ("Adam", adam_update),
    ]

    print("目标函数 f(x,y)=0.1*x^2+2*y^2，最优点=(0,0)\n")
    for name, update in algorithms:  # 对每个优化器运行同一目标和步数。
        _, losses = optimize(name, update)  # 从零更新，不调用 torch.optim。
        assert torch.isfinite(torch.tensor(losses)).all()  # 任一步都不允许 NaN/Inf。
        assert min(losses[-5:]) < losses[0]  # 末段至少明显优于起点。

    print("\n所有从零优化器均完成更新；轨迹差异来自状态与坐标缩放，而非不同目标。")


if __name__ == "__main__":  # 直接运行文件时才执行实验。
    main()
