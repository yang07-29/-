"""4.6 从零实现 inverted dropout，并验证 train/eval 差异。

直接运行：python code/ch04/dropout.py
"""

import torch
from torch import nn


def dropout_layer(
    X: torch.Tensor, drop_probability: float, training: bool
) -> torch.Tensor:
    """训练时随机置零并除以保留率；评估时恒等映射。"""
    if not 0.0 <= drop_probability <= 1.0:
        raise ValueError("drop_probability 必须位于 [0, 1]")
    # 评估态或 p=0 时不丢弃，直接返回输入。
    if not training or drop_probability == 0.0:
        return X
    # p=1 时保留率为零，需单独处理，避免除零。
    if drop_probability == 1.0:
        return torch.zeros_like(X)

    keep_probability = 1.0 - drop_probability
    # mask.shape 与 X.shape 相同；True 表示该位置保留。
    mask = torch.rand_like(X) < keep_probability
    # 除以 keep_probability，使 E[dropout(X)] = X。
    return mask.to(X.dtype) * X / keep_probability


class ScratchDropoutMLP(nn.Module):
    """在 forward 中读取 self.training 控制 Dropout。"""

    def __init__(self, inputs: int = 10, hidden: int = 24, outputs: int = 3):
        super().__init__()
        # hidden 层把 (B,10) 映射为 (B,24)。
        self.hidden = nn.Linear(inputs, hidden)
        # output 层把 (B,24) 映射为 logits(B,3)。
        self.output = nn.Linear(hidden, outputs)
        self.drop_probability = 0.5

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # 先做 Linear + ReLU，H.shape == (B,24)。
        H = torch.relu(self.hidden(X))
        # model.train()/eval() 会递归改变 self.training。
        H = dropout_layer(H, self.drop_probability, self.training)
        # Dropout 不改变 Shape，输出 logits.shape == (B,3)。
        return self.output(H)


def main() -> None:
    torch.manual_seed(5)
    # 大样本均值用于验证期望保持；X.shape == (20000,)。
    X = torch.ones(20_000)
    dropped = dropout_layer(X, drop_probability=0.5, training=True)
    print(f"训练态均值（应接近 1）：{dropped.mean().item():.4f}")
    assert abs(dropped.mean().item() - 1.0) < 0.05

    model = ScratchDropoutMLP()
    batch = torch.randn(8, 10)

    # train 模式每次重采样 mask，因此两次输出通常不同。
    model.train()
    first_train = model(batch)
    second_train = model(batch)
    print("训练态两次输出不同：", not torch.equal(first_train, second_train))

    # eval 关闭 Dropout；inference_mode 另外关闭 autograd。
    model.eval()
    with torch.inference_mode():
        first_eval = model(batch)
        second_eval = model(batch)
    print("评估态两次输出相同：", torch.equal(first_eval, second_eval))
    assert torch.equal(first_eval, second_eval)


if __name__ == "__main__":
    main()
