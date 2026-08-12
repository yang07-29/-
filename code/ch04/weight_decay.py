"""4.5 权重衰减：在小样本高维回归中比较有无 L2 约束。

直接运行：python code/ch04/weight_decay.py
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def make_data(
    train_samples: int = 24,
    valid_samples: int = 160,
    features: int = 120,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """构造“样本少、维度高”的易过拟合场景。"""
    torch.manual_seed(19)
    # 真实权重都很小；true_w.shape == (D,1)。
    true_w = torch.full((features, 1), 0.02)

    def sample(size: int) -> tuple[torch.Tensor, torch.Tensor]:
        # X.shape == (N,D)。
        X = torch.randn(size, features)
        # y.shape == (N,1)，与模型输出对齐，防止广播陷阱。
        y = X @ true_w + 0.05 + 0.05 * torch.randn(size, 1)
        return X, y

    train_X, train_y = sample(train_samples)
    valid_X, valid_y = sample(valid_samples)
    return train_X, train_y, valid_X, valid_y


def evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    """返回评估集 MSE。"""
    model.eval()
    with torch.inference_mode():
        return nn.functional.mse_loss(model(X), y).item()


def train(weight_decay: float) -> tuple[float, float, float]:
    """训练线性模型，返回训练损失、验证损失和权重范数。"""
    train_X, train_y, valid_X, valid_y = make_data()
    # 输入维度从数据读取，避免把 120 写死在多处。
    model = nn.Linear(train_X.shape[1], 1)
    nn.init.normal_(model.weight, std=0.01)
    nn.init.zeros_(model.bias)

    # 只衰减 weight；bias 参数少，通常不施加同样的 L2 约束。
    optimizer = torch.optim.SGD(
        [
            {"params": [model.weight], "weight_decay": weight_decay},
            {"params": [model.bias], "weight_decay": 0.0},
        ],
        lr=0.01,
    )
    loader = DataLoader(
        TensorDataset(train_X, train_y), batch_size=6, shuffle=True
    )

    for _ in range(180):
        model.train()
        for X, y in loader:
            # 1. Forward：X(B,120) -> prediction(B,1)。
            prediction = model(X)
            # 2. Loss：MSE 为标量；衰减由 optimizer 参数组处理。
            loss = nn.functional.mse_loss(prediction, y)
            # 3. zero_grad：避免跨批梯度累加。
            optimizer.zero_grad(set_to_none=True)
            # 4. backward：MSE 梯度写入 weight.grad 与 bias.grad。
            loss.backward()
            # 5. step：同时执行梯度更新和 weight decay。
            optimizer.step()

    train_loss = evaluate(model, train_X, train_y)
    valid_loss = evaluate(model, valid_X, valid_y)
    # 权重范数越小，说明约束越强；但越小并不自动等于越好。
    weight_norm = model.weight.detach().norm().item()
    return train_loss, valid_loss, weight_norm


def main() -> None:
    no_decay = train(weight_decay=0.0)
    with_decay = train(weight_decay=0.1)
    print(
        "无衰减：train={:.5f} valid={:.5f} ||w||={:.4f}".format(*no_decay)
    )
    print(
        "有衰减：train={:.5f} valid={:.5f} ||w||={:.4f}".format(*with_decay)
    )
    # 这个实验只断言机制：权重衰减应让权重范数更小。
    assert with_decay[2] < no_decay[2]


if __name__ == "__main__":
    main()
