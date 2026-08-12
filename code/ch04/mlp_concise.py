"""4.3 用 nn.Module、Loss 与 Optimizer 实现两层 MLP。

直接运行：python code/ch04/mlp_concise.py
使用合成数据，不下载 Fashion-MNIST。
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def make_loaders() -> tuple[DataLoader, DataLoader]:
    """生成并拆分三分类数据。"""
    torch.manual_seed(11)
    # 三个类别中心；centers.shape == (3, 2)。
    centers = torch.tensor([[-2.0, -1.5], [2.0, -1.0], [0.0, 2.0]])
    # y.shape == (750,)，且 dtype 为 CrossEntropyLoss 需要的 int64。
    y = torch.arange(750) % 3
    # X.shape == (750, 2)。
    X = centers[y] + 0.7 * torch.randn(750, 2)
    order = torch.randperm(len(X))
    X, y = X[order], y[order]
    # 600 个训练样本，150 个验证样本。
    train_set = TensorDataset(X[:600], y[:600])
    valid_set = TensorDataset(X[600:], y[600:])
    return (
        DataLoader(train_set, batch_size=64, shuffle=True),
        DataLoader(valid_set, batch_size=128, shuffle=False),
    )


def build_model() -> nn.Sequential:
    """构造 2 -> 32 -> 3 的串行 MLP。"""
    model = nn.Sequential(
        nn.Linear(2, 32),  # (B,2) -> (B,32)
        nn.ReLU(),         # Shape 不变，加入非线性
        nn.Linear(32, 3),  # (B,32) -> logits(B,3)
    )
    # ReLU 前的权重使用 He/Kaiming 初始化。
    nn.init.kaiming_normal_(model[0].weight, nonlinearity="relu")
    nn.init.zeros_(model[0].bias)
    # 输出层没有 ReLU，使用 Xavier 保持方差。
    nn.init.xavier_uniform_(model[2].weight)
    nn.init.zeros_(model[2].bias)
    return model


def evaluate(model: nn.Module, loader: DataLoader) -> tuple[float, float]:
    """返回按样本加权的平均 loss 与 accuracy。"""
    # eval 控制 Dropout/BatchNorm 等模块行为，但不会自动关闭 autograd。
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    loss_sum = 0.0
    correct = 0
    total = 0
    # inference_mode 关闭梯度跟踪，适合只读评估。
    with torch.inference_mode():
        for X, y in loader:
            # logits.shape == (B,3)。
            logits = model(X)
            # CrossEntropyLoss 输入 logits 和 y(B,)，不要先 Softmax。
            loss = loss_fn(logits, y)
            batch_size = y.numel()
            loss_sum += loss.item() * batch_size
            correct += int((logits.argmax(dim=1) == y).sum())
            total += batch_size
    return loss_sum / total, correct / total


def main() -> None:
    train_loader, valid_loader = make_loaders()
    model = build_model()
    # 类别不平衡时可传 weight=(3,)；此合成数据平衡，所以使用默认权重。
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.15)

    for epoch in range(35):
        # train 打开训练态；即使当前网络没有 Dropout，也应形成固定习惯。
        model.train()
        for X, y in train_loader:
            # 1. Forward：logits.shape == (B,3)。
            logits = model(X)
            # 2. Loss：标量批平均交叉熵。
            loss = loss_fn(logits, y)
            # 3. zero_grad：将旧梯度设为 None，减少无意义内存写入。
            optimizer.zero_grad(set_to_none=True)
            # 4. backward：把梯度累加到每个 Parameter.grad。
            loss.backward()
            # 5. step：优化器读取 grad 并原地更新参数。
            optimizer.step()

        if epoch in {0, 9, 19, 34}:
            valid_loss, valid_accuracy = evaluate(model, valid_loader)
            print(
                f"epoch={epoch + 1:02d} val_loss={valid_loss:.4f} "
                f"val_accuracy={valid_accuracy:.3f}"
            )

    _, final_accuracy = evaluate(model, valid_loader)
    assert final_accuracy > 0.90
    print("4.3 smoke test 通过。")


if __name__ == "__main__":
    main()
