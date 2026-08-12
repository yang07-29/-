"""4.2 只用张量与 autograd 实现两层 MLP。

直接运行：python code/ch04/mlp_from_scratch.py
使用合成三分类数据，便于离线复制运行。
"""

import math

import torch
from torch.utils.data import DataLoader, TensorDataset


def make_data(samples: int = 600) -> tuple[torch.Tensor, torch.Tensor]:
    """生成三团二维点；X 为 (N, 2)，y 为 (N,)。"""
    torch.manual_seed(7)
    # 每个类别的中心不同，因此一个小 MLP 可以学会分类。
    centers = torch.tensor([[-2.0, -1.5], [2.0, -1.0], [0.0, 2.0]])
    # y: (N,)，每个值是 0、1、2 中的类别索引。
    y = torch.arange(samples) % 3
    # 根据标签选择中心，再叠加高斯噪声；X: (N, 2)。
    X = centers[y] + 0.65 * torch.randn(samples, 2)
    # 打乱样本，避免 DataLoader 未打乱时连续看到同一模式。
    order = torch.randperm(samples)
    return X[order], y[order]


def initialize_parameters(
    inputs: int, hidden: int, outputs: int
) -> list[torch.Tensor]:
    """用适配 ReLU 的 He 尺度初始化两层参数。"""
    # 先完成乘法，再 requires_grad_，可确保 W1 是叶子张量。
    W1 = (torch.randn(inputs, hidden) * math.sqrt(2 / inputs)).requires_grad_()
    # 偏置从零开始；Shape 为 (hidden,)。
    b1 = torch.zeros(hidden, requires_grad=True)
    # 输出层也给出小随机值；Shape 为 (hidden, outputs)。
    W2 = (torch.randn(hidden, outputs) * math.sqrt(2 / hidden)).requires_grad_()
    # 输出偏置 Shape 为 (outputs,)。
    b2 = torch.zeros(outputs, requires_grad=True)
    return [W1, b1, W2, b2]


def forward(X: torch.Tensor, params: list[torch.Tensor]) -> torch.Tensor:
    """X(B,2) -> H(B,16) -> logits(B,3)。"""
    W1, b1, W2, b2 = params
    # 第一层仿射变换，Z 的 Shape 为 (B, 16)。
    Z = X @ W1 + b1
    # ReLU 提供非线性；H 的 Shape 仍为 (B, 16)。
    H = Z.clamp_min(0)
    # 返回 logits，不先做 Softmax；Shape 为 (B, 3)。
    return H @ W2 + b2


def cross_entropy(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """稳定地返回每个样本的交叉熵，Shape 为 (B,)。"""
    # logsumexp 同时处理“取指数、求和、取对数”的数值稳定性。
    log_probs = logits - logits.logsumexp(dim=1, keepdim=True)
    # arange 选择每一行；y 选择该样本真实类别的列。
    rows = torch.arange(y.numel(), device=y.device)
    return -log_probs[rows, y]


def sgd_step(params: list[torch.Tensor], learning_rate: float) -> None:
    """loss 已取 mean，因此直接按学习率更新，不再除 batch。"""
    # 参数更新本身不应该进入下一轮计算图。
    with torch.no_grad():
        for parameter in params:
            # step：θ <- θ - η * grad。
            parameter.add_(parameter.grad, alpha=-learning_rate)
            # 等价于 set_to_none：释放旧梯度，下一轮重新创建。
            parameter.grad = None


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    """返回当前批次正确率。"""
    # argmax 后 Shape 从 (B,3) 变为 (B,)。
    predictions = logits.argmax(dim=1)
    return (predictions == y).float().mean().item()


def main() -> None:
    # 准备离线数据；最后一批可能小于 batch_size，代码不依赖固定批大小。
    X, y = make_data()
    loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)
    # 2 -> 16 -> 3，对应四组可学习参数。
    params = initialize_parameters(inputs=2, hidden=16, outputs=3)
    learning_rate = 0.15

    for epoch in range(40):
        loss_sum = 0.0
        correct = 0
        total = 0
        for batch_X, batch_y in loader:
            # 1. Forward：得到未归一化 logits，Shape 为 (B, 3)。
            logits = forward(batch_X, params)
            # 2. Loss：先得到 (B,)，再 mean 成标量。
            loss = cross_entropy(logits, batch_y).mean()
            # 3. zero_grad：首次循环梯度本来就是 None；之后由 sgd_step 清理。
            # 4. backward：沿本轮计算图计算并累加四组参数的梯度。
            loss.backward()
            # 5. step：在 no_grad 中更新参数，同时把梯度设回 None。
            sgd_step(params, learning_rate)

            # 指标按真实样本数累计，避免最后一批权重错误。
            batch_size = batch_y.numel()
            loss_sum += loss.item() * batch_size
            correct += int((logits.argmax(dim=1) == batch_y).sum())
            total += batch_size

        if epoch in {0, 9, 19, 39}:
            print(
                f"epoch={epoch + 1:02d} loss={loss_sum / total:.4f} "
                f"accuracy={correct / total:.3f}"
            )

    # 评估不需要计算梯度，inference_mode 比单独 no_grad 更适合纯推理。
    with torch.inference_mode():
        final_logits = forward(X, params)
        final_accuracy = accuracy(final_logits, y)
    print(f"最终训练集准确率：{final_accuracy:.3f}")
    assert final_accuracy > 0.90


if __name__ == "__main__":
    main()
