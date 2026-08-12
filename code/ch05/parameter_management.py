"""5.2 参数访问、初始化、共享、冻结与 buffer。

直接运行：python code/ch05/parameter_management.py
"""

import torch
from torch import nn


def initialize(module: nn.Module) -> None:
    """由 model.apply 递归调用，只初始化 Linear。"""
    if isinstance(module, nn.Linear):
        # 当前网络使用 ReLU，因此用 He/Kaiming 初始化权重。
        nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class RunningCenter(nn.Module):
    """用 buffer 保存运行均值；它不是可训练参数。"""

    def __init__(self, features: int):
        super().__init__()
        # running_mean 会进 state_dict，也会随 model.to(device) 迁移。
        self.register_buffer("running_mean", torch.zeros(features))
        # batches 是整数计数器，同样属于持久状态。
        self.register_buffer("batches", torch.tensor(0, dtype=torch.long))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.training:
            # 运行统计不是梯度优化对象，更新时关闭 autograd。
            with torch.no_grad():
                # batch_mean.shape == (features,)。
                batch_mean = X.mean(dim=0)
                self.running_mean.mul_(0.9).add_(batch_mean, alpha=0.1)
                self.batches.add_(1)
        # X(B,D) - running_mean(D,) 沿批量维广播，输出仍是 (B,D)。
        return X - self.running_mean


class SharedModel(nn.Module):
    """同一层使用两次，验证共享参数的梯度汇总。"""

    def __init__(self, width: int = 6):
        super().__init__()
        self.shared = nn.Linear(width, width, bias=False)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # 两条使用路径都指向同一 self.shared.weight。
        first = self.shared(X)
        second = self.shared(torch.relu(first))
        return second


def main() -> None:
    torch.manual_seed(43)
    model = nn.Sequential(
        nn.Linear(6, 12),
        nn.ReLU(),
        RunningCenter(12),
        nn.Linear(12, 3),
    )
    # apply 沿注册树递归执行初始化。
    model.apply(initialize)

    print("=== named_parameters：会被优化器发现 ===")
    for name, parameter in model.named_parameters():
        print(name, tuple(parameter.shape), "requires_grad=", parameter.requires_grad)

    print("\n=== named_buffers：保存/迁移但不优化 ===")
    for name, buffer in model.named_buffers():
        print(name, tuple(buffer.shape), buffer.dtype)

    # state_dict 同时包含 Parameter 与持久 buffer，但不包含 forward 代码。
    state_keys = list(model.state_dict().keys())
    print("\nstate_dict keys：", state_keys)
    assert "2.running_mean" in state_keys

    # 冻结第一层：它不再需要梯度，但这与 model.eval() 无关。
    for parameter in model[0].parameters():
        parameter.requires_grad_(False)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.SGD(trainable, lr=0.05)

    X = torch.randn(10, 6)
    y = torch.randint(0, 3, (10,))
    model.train()
    # 1. Forward：logits.shape == (10,3)。
    logits = model(X)
    # 2. Loss：标签 y.shape == (10,)。
    loss = nn.functional.cross_entropy(logits, y)
    # 3. zero_grad：只清理优化器所管理的可训练参数。
    optimizer.zero_grad(set_to_none=True)
    # 4. backward：冻结层不会创建 weight.grad。
    loss.backward()
    # 5. step：只更新未冻结参数。
    optimizer.step()
    assert model[0].weight.grad is None

    # 独立验证共享层：两次使用只有一套权重和一个 grad。
    shared_model = SharedModel()
    shared_output = shared_model(torch.randn(4, 6))
    shared_loss = shared_output.square().mean()
    shared_loss.backward()
    print("共享权重 Shape：", tuple(shared_model.shared.weight.shape))
    print("共享梯度 Shape：", tuple(shared_model.shared.weight.grad.shape))
    assert len(list(shared_model.parameters())) == 1


if __name__ == "__main__":
    main()
