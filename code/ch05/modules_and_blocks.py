"""5.1 nn.Module、Sequential、ModuleList 与动态 forward。

直接运行：python code/ch05/modules_and_blocks.py
"""

import torch
from torch import nn


class MLP(nn.Module):
    """最基本的自定义块：__init__ 注册，forward 描述数据流。"""

    def __init__(self, inputs: int = 8, hidden: int = 16, outputs: int = 3):
        # 必须先初始化 nn.Module 内部的注册表。
        super().__init__()
        # 绑定为属性后，hidden 会自动注册为子模块。
        self.hidden = nn.Linear(inputs, hidden)
        # ReLU 没有参数，但仍作为模块注册，便于统一 train/eval 与打印结构。
        self.activation = nn.ReLU()
        # 输出层也是注册树的一部分。
        self.output = nn.Linear(hidden, outputs)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X.shape == (B,8)，隐藏输出 H.shape == (B,16)。
        H = self.activation(self.hidden(X))
        # logits.shape == (B,3)。
        return self.output(H)


class MySequential(nn.Module):
    """从零理解 Sequential：注册子模块并依次调用。"""

    def __init__(self, *modules: nn.Module):
        super().__init__()
        for index, module in enumerate(modules):
            # add_module 把层写入注册树，而不是普通 Python 容器。
            self.add_module(str(index), module)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        for module in self.children():
            # 当前输出成为下一层输入，适合单输入、单输出的直线图。
            X = module(X)
        return X


class ResidualStack(nn.Module):
    """ModuleList 只负责注册，残差数据流由 forward 决定。"""

    def __init__(self, width: int = 8, depth: int = 3):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(width, width), nn.ReLU())
                for _ in range(depth)
            ]
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            # block(X).shape 与 X.shape 都是 (B,8)，所以可以相加。
            X = X + block(X)
        return X


class SharedDynamicBlock(nn.Module):
    """展示权重共享、buffer 与 Python 动态控制流。"""

    def __init__(self, width: int = 8):
        super().__init__()
        # 同一个 Linear 对象会在 forward 中被调用两次，即共享一套参数。
        self.shared = nn.Linear(width, width)
        # fixed_scale 需随模型迁移/保存但不训练，因此注册为 buffer。
        self.register_buffer("fixed_scale", torch.linspace(0.8, 1.2, width))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # 第一次使用共享层，输出 Shape 仍为 (B,8)。
        X = torch.relu(self.shared(X))
        # fixed_scale.shape == (8,)，沿批量维广播。
        X = X * self.fixed_scale
        # 第二次调用同一对象，梯度会汇总到同一 Parameter.grad。
        X = self.shared(X)
        # 动态图只记录本次实际执行的循环次数。
        while X.detach().abs().mean().item() > 1.0:
            X = X / 2
        return X


def main() -> None:
    torch.manual_seed(41)
    # 输入为 4 个样本、每个样本 8 个特征。
    X = torch.randn(4, 8)
    models: list[nn.Module] = [
        MLP(),
        MySequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 3)),
        ResidualStack(),
        SharedDynamicBlock(),
    ]

    for model in models:
        # 调用 model(X)，让 Module.__call__ 处理 hooks 后再进入 forward。
        output = model(X)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        print(
            f"{type(model).__name__:18s} output={tuple(output.shape)} "
            f"parameters={parameter_count}"
        )
        assert torch.isfinite(output).all()

    # ModuleList 中每个 block 都应出现在注册树里。
    names = dict(models[2].named_modules()).keys()
    assert "blocks.0.0" in names


if __name__ == "__main__":
    main()
