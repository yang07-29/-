"""4.7 用一个标量计算图观察 forward、backward 与梯度累加。

直接运行：python code/ch04/autograd_graph.py
"""

import torch


def build_loss(
    x: torch.Tensor, y: torch.Tensor, w: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """构建 x -> z -> a -> loss 的本轮计算图。"""
    # z 是非叶子张量，表示线性层预激活。
    z = x * w + b
    # ReLU 在 z>0 时让梯度通过，在 z<0 时截断。
    activation = torch.relu(z)
    # 标量平方损失可直接调用 backward。
    loss = (activation - y).square() / 2
    return z, loss


def main() -> None:
    # x、y 是常量，不需要梯度。
    x = torch.tensor(2.0)
    y = torch.tensor(5.0)
    # w、b 是叶子参数，需要梯度。
    w = torch.tensor(1.5, requires_grad=True)
    b = torch.tensor(0.5, requires_grad=True)

    # 1. Forward：建立第一张动态图。
    z, loss = build_loss(x, y, w, b)
    # 非叶子 z 默认不保留 .grad；调试时显式调用 retain_grad。
    z.retain_grad()
    print(f"前向值：z={z.item():.3f}, loss={loss.item():.3f}")

    # 2. Loss 已在 forward 中得到标量。
    # 3. zero_grad：第一次梯度为 None，不需要清理。
    # 4. backward：链式法则把梯度传到 w、b，并保留调试用 z.grad。
    loss.backward()
    print(f"第一次反传：dw={w.grad.item():.3f}, db={b.grad.item():.3f}")
    print(f"中间节点：dz={z.grad.item():.3f}")
    first_dw = w.grad.item()

    # 再次前向会建立一张新图；旧图已在 backward 后释放。
    _, second_loss = build_loss(x, y, w, b)
    # 不清梯度直接 backward，贡献会继续加到 w.grad 与 b.grad。
    second_loss.backward()
    print(f"未清零再反传：累计 dw={w.grad.item():.3f}")
    assert torch.isclose(w.grad, torch.tensor(2 * first_dw))

    # 3. zero_grad：手写参数可把 grad 设为 None。
    w.grad = None
    b.grad = None
    # 重新前向，建立更新前的第三张图。
    _, third_loss = build_loss(x, y, w, b)
    # 4. backward：现在 grad 只包含本轮贡献。
    third_loss.backward()
    print(f"清理后单次 dw={w.grad.item():.3f}")

    # 5. step：参数更新不应该继续被 autograd 记录。
    with torch.no_grad():
        w -= 0.1 * w.grad
        b -= 0.1 * b.grad
    print(f"更新后：w={w.item():.3f}, b={b.item():.3f}")

    # detach 只切断特定张量的梯度路径。
    detached_w = (3 * w).detach()
    detached_loss = (detached_w * b).square()
    b.grad = None
    detached_loss.backward()
    print("detach 后 w 未收到新梯度，b 仍有梯度：", b.grad.item())


if __name__ == "__main__":
    main()
