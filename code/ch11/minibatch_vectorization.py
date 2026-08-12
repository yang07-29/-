"""第 11 章：小批量梯度噪声与向量化为什么更高效。

运行：python code/ch11/minibatch_vectorization.py
只依赖 PyTorch，不下载数据。
"""

from __future__ import annotations

from time import perf_counter

import torch


def make_regression_data(num_examples: int, num_features: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    true_weights = torch.linspace(-1.0, 1.0, num_features)  # (D,)，生成数据的真实权重。
    features = torch.randn(num_examples, num_features)  # (N,D)，每行一个样本。
    noise = 0.05 * torch.randn(num_examples)  # (N,)，加入小量观测噪声。
    labels = features @ true_weights + 0.3 + noise  # (N,)，线性回归目标。
    return features, labels, true_weights  # 返回数据与真值便于验证。


def per_sample_gradient(features: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """教学版逐样本循环：数学正确，但不能发挥批量矩阵乘法优势。"""
    gradient = torch.zeros_like(weights)  # (D,)，累加每个样本贡献。
    for row in range(features.shape[0]):  # Python 循环逐个处理样本。
        prediction = torch.dot(features[row], weights)  # 标量预测。
        error = prediction - labels[row]  # 标量残差。
        gradient += 2.0 * error * features[row]  # (D,)，平方损失对 w 的梯度。
    return gradient / features.shape[0]  # 对 batch 求平均。


def vectorized_gradient(features: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """向量化版：一次矩阵运算同时计算整批样本。"""
    predictions = features @ weights  # (B,D)@(D,) -> (B,)。
    errors = predictions - labels  # (B,)，逐样本残差。
    return 2.0 * features.T @ errors / features.shape[0]  # (D,B)@(B,) -> (D,)。


def estimate_gradient_noise(
    features: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    batch_size: int,
    repeats: int = 120,
) -> float:
    """重复抽样，估计小批量梯度相对全数据梯度的均方偏差。"""
    full_gradient = vectorized_gradient(features, labels, weights)  # (D,)，作为低噪声参照。
    deviations = []  # 收集每次随机 batch 的梯度误差。
    for _ in range(repeats):  # 重复抽样观察方差。
        indices = torch.randint(0, features.shape[0], (batch_size,))  # (B,)，有放回抽样。
        batch_gradient = vectorized_gradient(features[indices], labels[indices], weights)  # (D,)。
        deviations.append((batch_gradient - full_gradient).square().mean())  # 标量 MSE。
    return torch.stack(deviations).mean().item()  # 返回 Python 数值方便打印。


def train_minibatch(
    features: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    epochs: int = 12,
) -> tuple[torch.Tensor, float]:
    """用手写小批量 SGD 训练线性回归。"""
    weights = torch.zeros(features.shape[1], requires_grad=True)  # (D,)，从零初始化。
    bias = torch.zeros((), requires_grad=True)  # 标量偏置。
    learning_rate = 0.08  # 所有 batch size 使用同一基础学习率做直观对照。

    for _ in range(epochs):  # 多次遍历整个数据集。
        permutation = torch.randperm(features.shape[0])  # 每轮打乱，避免固定批次偏差。
        for start in range(0, features.shape[0], batch_size):  # 依次切出小批量。
            indices = permutation[start : start + batch_size]  # 当前 batch 的索引。
            batch_x = features[indices]  # (B,D)。
            batch_y = labels[indices]  # (B,)。

            predictions = batch_x @ weights + bias  # (B,)，前向建立计算图。
            loss = (predictions - batch_y).square().mean()  # 标量均方误差。
            if weights.grad is not None:  # 第一步前梯度为空。
                weights.grad.zero_()  # 清除 w 的累积梯度。
                bias.grad.zero_()  # 清除 b 的累积梯度。
            loss.backward()  # 得到本 batch 的随机梯度。
            with torch.no_grad():  # 参数更新不应被 autograd 记录。
                weights -= learning_rate * weights.grad  # 真正改变权重。
                bias -= learning_rate * bias.grad  # 真正改变偏置。

    with torch.inference_mode():  # 评估时关闭计算图。
        final_loss = ((features @ weights + bias - labels).square().mean()).item()  # 全数据 MSE。
    return weights.detach(), final_loss  # 返回训练结果。


def main() -> None:
    torch.manual_seed(29)  # 固定数据与抽样序列。
    torch.set_num_threads(1)  # 小矩阵减少线程调度噪声。
    features, labels, true_weights = make_regression_data(2048, 64)  # N=2048、D=64。
    probe_weights = torch.randn(64)  # 在同一点比较两种梯度计算。

    loop_gradient = per_sample_gradient(features[:512], labels[:512], probe_weights)  # Python 循环版。
    batch_gradient = vectorized_gradient(features[:512], labels[:512], probe_weights)  # 矩阵版。
    max_difference = (loop_gradient - batch_gradient).abs().max().item()  # 数值误差应很小。
    print("[正确性] 循环与向量化梯度最大差:", f"{max_difference:.2e}")
    assert torch.allclose(loop_gradient, batch_gradient, atol=2e-4, rtol=1e-5)  # 验证公式一致。

    start = perf_counter()  # 开始测循环版耗时。
    for _ in range(10):  # 重复多次减少计时抖动。
        per_sample_gradient(features[:512], labels[:512], probe_weights)  # 逐样本执行。
    loop_seconds = perf_counter() - start  # 记录循环总时间。

    start = perf_counter()  # 开始测向量化版耗时。
    for _ in range(10):  # 使用相同重复次数。
        vectorized_gradient(features[:512], labels[:512], probe_weights)  # 单次矩阵运算。
    vector_seconds = perf_counter() - start  # 记录向量化总时间。
    print(f"[效率] 循环={loop_seconds:.4f}s 向量化={vector_seconds:.4f}s 加速比={loop_seconds / vector_seconds:.1f}x")

    print("\n[梯度噪声：越小越接近全数据梯度]")
    for batch_size in (1, 8, 64, 512):  # 从纯 SGD 到较大 mini-batch。
        noise = estimate_gradient_noise(features, labels, probe_weights, batch_size)  # 估计梯度方差。
        print(f"batch_size={batch_size:3d} gradient_mse={noise:.6f}")

    learned_weights, final_loss = train_minibatch(features, labels, batch_size=32)  # 手写 mini-batch SGD。
    parameter_error = (learned_weights - true_weights).abs().mean().item()  # 与真值比较。
    print(f"\n[训练] final_mse={final_loss:.6f} mean_weight_error={parameter_error:.6f}")
    assert final_loss < 0.01  # 合成线性任务应被可靠拟合。
    print("小批量与向量化实验通过。")


if __name__ == "__main__":  # 直接运行脚本时才执行实验。
    main()
