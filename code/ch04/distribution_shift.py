"""4.9 用合成数据区分协变量偏移与概念偏移。

直接运行：python code/ch04/distribution_shift.py
"""

import torch
from torch import nn


def sample(
    samples: int,
    mean: tuple[float, float],
    concept_changed: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """采样 X(N,2)；可选择是否改变 p(y|x)。"""
    # 改 mean 只改变 p(x)，不自动改变标签机制。
    X = torch.randn(samples, 2) + torch.tensor(mean)
    if concept_changed:
        # 概念偏移：同一个 x 的判别规则发生改变。
        score = -X[:, 0] + 0.7 * X[:, 1]
    else:
        # 源域和协变量偏移目标域共享此标签规则。
        score = X[:, 0] + 0.7 * X[:, 1]
    # y.shape == (N,)，用于 CrossEntropyLoss 的类别索引。
    y = (score > 0).long()
    return X, y


def gaussian_density_ratio(
    X: torch.Tensor,
    target_mean: tuple[float, float],
    source_mean: tuple[float, float] = (0.0, 0.0),
) -> torch.Tensor:
    """计算同单位协方差高斯的 q(x)/p(x)，返回 (N,)。"""
    source = torch.tensor(source_mean, dtype=X.dtype)
    target = torch.tensor(target_mean, dtype=X.dtype)
    # 先算 log 密度比，减少直接计算小概率密度的下溢。
    log_ratio = -0.5 * (X - target).square().sum(dim=1)
    log_ratio += 0.5 * (X - source).square().sum(dim=1)
    # 裁剪极端权重，降低少数样本主导梯度的方差。
    return log_ratio.exp().clamp(max=15.0)


def train(
    X: torch.Tensor,
    y: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> nn.Linear:
    """训练二分类线性模型。"""
    model = nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    # reduction='none' 保留每样本损失 (N,)，才能做重要性加权。
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    for _ in range(220):
        model.train()
        # 1. Forward：logits.shape == (N,2)。
        logits = model(X)
        # 2. Loss：per_sample_loss.shape == (N,)。
        per_sample_loss = loss_fn(logits, y)
        if sample_weight is None:
            loss = per_sample_loss.mean()
        else:
            # 加权平均除以权重和，维持可解释的损失尺度。
            loss = (per_sample_loss * sample_weight).sum() / sample_weight.sum()
        # 3. zero_grad：清除上一轮参数梯度。
        optimizer.zero_grad(set_to_none=True)
        # 4. backward：加权系数会同步改变每个样本的梯度贡献。
        loss.backward()
        # 5. step：更新分类边界。
        optimizer.step()
    return model


def accuracy(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    """在评估模式下计算准确率。"""
    model.eval()
    with torch.inference_mode():
        predictions = model(X).argmax(dim=1)
        return (predictions == y).float().mean().item()


def main() -> None:
    torch.manual_seed(3)
    # 源域训练数据：p_source(x)。
    train_X, train_y = sample(350, mean=(0.0, 0.0))
    # 同分布测试集。
    source_X, source_y = sample(1500, mean=(0.0, 0.0))
    # 协变量偏移：p(x) 变，但 p(y|x) 没变。
    covariate_X, covariate_y = sample(1500, mean=(1.8, 0.0))
    # 概念偏移：p(y|x) 变，输入分布可以不变。
    concept_X, concept_y = sample(1500, mean=(0.0, 0.0), concept_changed=True)

    baseline = train(train_X, train_y)
    weights = gaussian_density_ratio(train_X, target_mean=(1.8, 0.0))
    reweighted = train(train_X, train_y, sample_weight=weights)

    print(f"baseline / source：{accuracy(baseline, source_X, source_y):.3f}")
    print(
        "baseline / covariate shift："
        f"{accuracy(baseline, covariate_X, covariate_y):.3f}"
    )
    print(
        "reweighted / covariate shift："
        f"{accuracy(reweighted, covariate_X, covariate_y):.3f}"
    )
    concept_accuracy = accuracy(baseline, concept_X, concept_y)
    print(f"baseline / concept shift：{concept_accuracy:.3f}")
    print("重要性加权不能修复已经改变的 p(y|x)。")
    assert concept_accuracy < 0.50


if __name__ == "__main__":
    main()
