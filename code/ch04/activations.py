"""4.1 激活函数：验证“没有非线性，多层仍是一层”。

直接运行：python code/ch04/activations.py
只依赖 PyTorch，不下载数据。
"""

import torch


def merge_linear_layers() -> None:
    """用数值实验验证两个仿射层可以合并。"""
    # 固定随机种子，保证每次运行结果一致。
    torch.manual_seed(0)
    # X: (B, d) = (4, 3)，4 个样本、每个样本 3 个特征。
    X = torch.randn(4, 3)
    # 第一层参数：把 3 维输入映射为 5 维隐藏表示。
    W1 = torch.randn(3, 5)
    b1 = torch.randn(5)
    # 第二层参数：把 5 维隐藏表示映射为 2 维输出。
    W2 = torch.randn(5, 2)
    b2 = torch.randn(2)

    # 没有激活函数时，两层前向是 (XW1+b1)W2+b2。
    two_layers = (X @ W1 + b1) @ W2 + b2
    # 按分配律合并权重，新权重 Shape 为 (3, 2)。
    merged_W = W1 @ W2
    # 偏置也要经过第二层权重，不能只写 b1+b2。
    merged_b = b1 @ W2 + b2
    # 合并后只剩一个仿射变换，输出 Shape 仍为 (4, 2)。
    one_layer = X @ merged_W + merged_b

    # 浮点运算有舍入误差，比较最大绝对误差而不是要求逐位相等。
    error = (two_layers - one_layer).abs().max().item()
    print(f"无激活时的最大合并误差：{error:.3e}")
    assert torch.allclose(two_layers, one_layer, atol=1e-5)


def inspect_activation_gradients() -> None:
    """在几个代表点观察激活值与导数。"""
    # x 是叶子张量；requires_grad=True 表示需要对它求梯度。
    x = torch.tensor([-5.0, -1.0, 0.0, 1.0, 5.0], requires_grad=True)
    functions = {
        "ReLU": torch.relu,
        "Sigmoid": torch.sigmoid,
        "Tanh": torch.tanh,
    }

    for name, function in functions.items():
        # 每个函数使用独立叶子，避免多个 backward 的梯度互相累加。
        current_x = x.detach().clone().requires_grad_(True)
        # 激活是逐元素运算，所以 y.shape 与 current_x.shape 都是 (5,)。
        y = function(current_x)
        # backward 需要标量；sum 把 5 个输出归约为一个标量。
        y.sum().backward()
        # current_x.grad 就是每个采样点的一阶导数。
        print(f"\n{name}")
        print("激活值：", y.detach().round(decimals=4).tolist())
        print("导数值：", current_x.grad.round(decimals=4).tolist())


def trace_mlp_shapes() -> None:
    """手写一次 MLP 前向，专门追踪 Shape。"""
    torch.manual_seed(1)
    # X: (B, d) = (8, 4)。
    X = torch.randn(8, 4)
    # W1: (d, h) = (4, 6)，b1: (h,) = (6,)。
    W1 = torch.randn(4, 6)
    b1 = torch.zeros(6)
    # W2: (h, q) = (6, 3)，b2: (q,) = (3,)。
    W2 = torch.randn(6, 3)
    b2 = torch.zeros(3)

    # 隐藏层预激活 Z: (8, 4) @ (4, 6) -> (8, 6)。
    Z = X @ W1 + b1
    # ReLU 不改变 Shape，只改变数值。
    H = torch.relu(Z)
    # 输出 logits: (8, 6) @ (6, 3) -> (8, 3)。
    logits = H @ W2 + b2
    print("\nShape 流：", tuple(X.shape), "->", tuple(H.shape), "->", tuple(logits.shape))
    assert logits.shape == (8, 3)


def main() -> None:
    merge_linear_layers()
    inspect_activation_gradients()
    trace_mlp_shapes()
    print("\n4.1 smoke test 通过。")


if __name__ == "__main__":
    main()
