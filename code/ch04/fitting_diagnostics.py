"""4.4 用学习曲线区分欠拟合、合适拟合与过拟合。

直接运行：python code/ch04/fitting_diagnostics.py
"""

import torch
from torch import nn


def make_polynomial_data(
    samples: int = 160, max_degree: int = 20
) -> tuple[torch.Tensor, torch.Tensor]:
    """生成真实规律为三次多项式的带噪数据。"""
    torch.manual_seed(42)
    # x.shape == (N,1)，限制在 [-1,1] 可避免高次幂数值爆炸。
    x = 2 * torch.rand(samples, 1) - 1
    # powers.shape == (1,20)，用于广播生成 0~19 次幂。
    powers = torch.arange(max_degree).reshape(1, -1)
    # features.shape == (N,20)。
    features = x**powers
    # 真实模型只使用前四列：常数、一次、二次、三次。
    true_w = torch.zeros(max_degree, 1)
    true_w[:4, 0] = torch.tensor([5.0, 1.2, -3.4, 5.6])
    # labels.shape == (N,1)；显式保留列维，避免 (B,1)-(B,) 广播成 (B,B)。
    # 较明显的观测噪声配合极小训练集，便于观察高容量模型记忆噪声。
    labels = features @ true_w + 0.50 * torch.randn(samples, 1)
    return features.float(), labels.float()


def mse(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    """在评估态计算 MSE。"""
    model.eval()
    with torch.inference_mode():
        # model(X) 与 y 都是 (B,1)，不存在广播歧义。
        return nn.functional.mse_loss(model(X), y).item()


def train_degree(
    train_X: torch.Tensor,
    train_y: torch.Tensor,
    valid_X: torch.Tensor,
    valid_y: torch.Tensor,
    degree: int,
    epochs: int = 2500,
) -> tuple[float, float]:
    """只使用前 degree 个多项式特征训练线性回归。"""
    # degree=1 只有常数项，degree=4 正好覆盖真实规律，degree=20 容量很高。
    model = nn.Linear(degree, 1, bias=False)
    nn.init.normal_(model.weight, std=0.01)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

    # 小数据直接全批训练，重点放在容量而不是数据加载代码。
    used_train_X = train_X[:, :degree]
    used_valid_X = valid_X[:, :degree]
    for _ in range(epochs):
        model.train()
        # 1. Forward：prediction.shape == (N_train,1)。
        prediction = model(used_train_X)
        # 2. Loss：prediction 和 train_y Shape 完全一致。
        loss = nn.functional.mse_loss(prediction, train_y)
        # 3. zero_grad：清掉上一轮梯度。
        optimizer.zero_grad(set_to_none=True)
        # 4. backward：求当前训练损失对权重的梯度。
        loss.backward()
        # 5. step：更新多项式系数。
        optimizer.step()

    return mse(model, used_train_X, train_y), mse(model, used_valid_X, valid_y)


def diagnose(train_loss: float, valid_loss: float) -> str:
    """给出教学用诊断；真实项目还需结合曲线、基线与业务尺度。"""
    gap = valid_loss - train_loss
    if train_loss > 0.2 and valid_loss > 0.2:
        return "更像欠拟合：训练集本身都没学好"
    if gap > max(0.05, 2 * train_loss):
        return "更像过拟合：训练好，但验证差距明显"
    return "当前容量较合适：训练与验证都较好"


def main() -> None:
    features, labels = make_polynomial_data()
    # 只用 12 个样本训练，其余验证；极小训练集放大高容量模型的过拟合风险。
    train_X, valid_X = features[:12], features[12:]
    train_y, valid_y = labels[:12], labels[12:]

    for degree in (1, 4, 20):
        train_loss, valid_loss = train_degree(
            train_X, train_y, valid_X, valid_y, degree
        )
        print(
            f"degree={degree:02d} train={train_loss:.5f} "
            f"valid={valid_loss:.5f} | {diagnose(train_loss, valid_loss)}"
        )

    # 最重要的 smoke test：正确容量应明显优于只含常数项的模型。
    underfit = train_degree(train_X, train_y, valid_X, valid_y, degree=1)[1]
    suitable = train_degree(train_X, train_y, valid_X, valid_y, degree=4)[1]
    assert suitable < underfit


if __name__ == "__main__":
    main()
