"""5.4 无参数层、带参数层、buffer 与梯度验证。

直接运行：python code/ch05/custom_layers.py
"""

import torch
from torch import nn
from torch.nn import functional as F


class CenteredLayer(nn.Module):
    """无参数层：对每个样本的最后一维做中心化。"""

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # keepdim=True 让均值 Shape 为 (B,1)，可明确沿特征维广播。
        sample_mean = X.mean(dim=-1, keepdim=True)
        # X(B,D) - sample_mean(B,1) -> (B,D)。
        return X - sample_mean


class MyLinear(nn.Module):
    """手动注册 weight/bias，再复用 F.linear 的稳定计算。"""

    def __init__(self, inputs: int, outputs: int):
        super().__init__()
        # PyTorch Linear 权重约定为 (out_features, in_features)。
        self.weight = nn.Parameter(torch.empty(outputs, inputs))
        # bias.shape == (outputs,)，前向时沿批量维广播。
        self.bias = nn.Parameter(torch.empty(outputs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # 这里后接 ReLU，所以使用 Kaiming/He 初始化。
        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")
        # 偏置先置零，便于理解。
        nn.init.zeros_(self.bias)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # F.linear 内部计算 X @ weight.T + bias。
        return F.linear(X, self.weight, self.bias)


class AffineFeatureScale(nn.Module):
    """逐特征缩放/平移；展示广播 Parameter 与固定 buffer。"""

    def __init__(self, features: int):
        super().__init__()
        # scale 与 shift 需要学习，所以用 Parameter。
        self.scale = nn.Parameter(torch.ones(features))
        self.shift = nn.Parameter(torch.zeros(features))
        # epsilon 需随模型保存/迁移但不训练，所以用 buffer。
        self.register_buffer("epsilon", torch.tensor(1e-6))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if X.shape[-1] != self.scale.numel():
            raise ValueError("输入最后一维必须与 scale 长度一致")
        # X(B,D) * scale(D,) + shift(D,) -> (B,D)。
        return X * (self.scale + self.epsilon) + self.shift


def main() -> None:
    torch.manual_seed(53)
    # X 是叶子输入；requires_grad=True 用来验证梯度能否穿过自定义层。
    X = torch.randn(5, 4, requires_grad=True)
    model = nn.Sequential(
        MyLinear(4, 7),       # (5,4) -> (5,7)
        nn.ReLU(),            # Shape 不变
        AffineFeatureScale(7),# Shape 不变
        CenteredLayer(),      # 每行均值变为 0
    )

    model.train()
    # 1. Forward：output.shape == (5,7)。
    output = model(X)
    print("输出 Shape：", tuple(output.shape))
    print("逐样本均值：", output.mean(dim=-1).detach().round(decimals=6).tolist())
    # 2. Loss：归约为标量，使用 square 而非 sum，避免中心化结果恒等抵消。
    loss = output.square().mean()
    # 3. zero_grad：第一次 parameter.grad 为 None；输入 X.grad 也为 None。
    model.zero_grad(set_to_none=True)
    # 4. backward：标准张量算子让 autograd 自动穿过自定义层。
    loss.backward()
    # 本示例不训练，只验证梯度，所以没有 optimizer.step。

    print("输入梯度 Shape：", tuple(X.grad.shape))
    for name, parameter in model.named_parameters():
        print(name, "parameter=", tuple(parameter.shape), "grad=", tuple(parameter.grad.shape))
        assert torch.isfinite(parameter.grad).all()

    # state_dict 应包含自定义 Parameter 和持久 buffer。
    keys = list(model.state_dict().keys())
    print("state_dict keys：", keys)
    assert "2.epsilon" in keys
    assert output.shape == (5, 7)
    assert torch.allclose(output.mean(dim=-1), torch.zeros(5), atol=1e-5)


if __name__ == "__main__":
    main()
