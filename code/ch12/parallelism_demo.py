"""第 12 章：数据并行、模型并行与 DataParallel 的离线演示。

运行：python code/ch12/parallelism_demo.py
"""

from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn import functional as F


class SplitNet(nn.Module):
    """可按层拆分的两段式网络。"""

    def __init__(self) -> None:
        super().__init__()
        # 第一段把 8 维特征编码为 16 维表示。
        self.encoder = nn.Sequential(nn.Linear(8, 16), nn.ReLU())
        # 第二段把表示映射为 3 类 logits。
        self.head = nn.Linear(16, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 (B,8) -> 表示 (B,16) -> logits (B,3)。
        return self.head(self.encoder(x))


def full_batch_gradients(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> list[torch.Tensor]:
    """计算完整 batch 的平均损失梯度，作为参照答案。"""

    # 清除历史梯度，防止 PyTorch 默认累加。
    model.zero_grad(set_to_none=True)
    # 前向得到 (B,3)，并建立计算图。
    logits = model(x)
    # reduction=mean 表示完整 batch 梯度是各样本梯度平均。
    loss = F.cross_entropy(logits, y, reduction="mean")
    # 反向把梯度写入每个 parameter.grad。
    loss.backward()
    # clone 让参照值不受下一次清梯度影响。
    return [parameter.grad.detach().clone() for parameter in model.parameters()]


def simulated_data_parallel(
    base_model: nn.Module, x: torch.Tensor, y: torch.Tensor, replicas: int = 2
) -> list[torch.Tensor]:
    """在单设备上模拟“复制模型—切 batch—平均梯度”。"""

    # 沿 batch 维切分；模型参数不会沿 batch 维切分。
    x_parts = torch.tensor_split(x, replicas, dim=0)
    # 标签必须使用完全相同的切分边界。
    y_parts = torch.tensor_split(y, replicas, dim=0)
    # 每个工作器拿到参数完全相同的模型副本。
    workers = [copy.deepcopy(base_model) for _ in range(replicas)]
    # 为每个参数准备一个与参数同 Shape 的梯度和。
    reduced = [torch.zeros_like(parameter) for parameter in base_model.parameters()]
    # 总样本数用于处理最后一份 batch 较小的情况。
    total_examples = len(x)

    for worker, x_part, y_part in zip(workers, x_parts, y_parts):
        # 本地工作器只看自己的 (B_i,8) 数据。
        logits = worker(x_part)
        # 本地使用 mean，稍后必须按 B_i/B 加权才能还原全局 mean。
        local_loss = F.cross_entropy(logits, y_part, reduction="mean")
        # 反向只把梯度写入当前副本。
        local_loss.backward()
        # 不等长切片时，不能简单除以 replicas。
        weight = len(x_part) / total_examples
        for target, parameter in zip(reduced, worker.parameters()):
            # 加权规约等价于对全 batch 样本梯度求平均。
            target.add_(parameter.grad, alpha=weight)
    return reduced


def model_parallel_forward(model: SplitNet, x: torch.Tensor) -> torch.Tensor:
    """演示按层放置设备；只有一块设备时仍走相同逻辑。"""

    # 若至少两块 GPU，则两段分别放 cuda:0 和 cuda:1。
    first = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    second = torch.device("cuda:1" if torch.cuda.device_count() >= 2 else first)
    # 参数必须显式搬到各自执行设备。
    model.encoder.to(first)
    model.head.to(second)
    # 输入先进入 encoder 所在设备。
    hidden = model.encoder(x.to(first))
    # 跨设备边界搬运激活；两设备相同时这一步近似无操作。
    hidden = hidden.to(second)
    # 输出 Shape 为 (B,3)，计算图跨越设备复制操作。
    return model.head(hidden)


def safe_data_parallel_step(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    """DataParallel 安全演示：0/1/多 GPU 均可运行。"""

    # 有 CUDA 就把主副本放在 cuda:0，否则保持 CPU。
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # DataParallel 在单 GPU/CPU 上退化为普通调用，代码路径仍可验证。
    wrapped = nn.DataParallel(model.to(device))
    # 优化器必须接收包装后可见的参数。
    optimizer = torch.optim.SGD(wrapped.parameters(), lr=0.05)
    # 梯度清零是一次参数更新的起点。
    optimizer.zero_grad(set_to_none=True)
    # DataParallel 在多 GPU 时自动 scatter、replicate、gather。
    logits = wrapped(x.to(device))
    # 标签也要与汇总后的 logits 位于同一主设备。
    loss = F.cross_entropy(logits, y.to(device))
    # autograd 会把各副本梯度汇总到主副本。
    loss.backward()
    # 只更新主副本参数；下一次 forward 再复制新参数。
    optimizer.step()
    return float(loss.detach().cpu())


def main() -> None:
    # 固定参数、输入与标签，便于验证并行等价性。
    torch.manual_seed(12)
    # 故意使用奇数 batch，检查加权平均而不是盲目除以副本数。
    x = torch.randn(15, 8)
    # 三分类标签 Shape 为 (15,)。
    y = torch.randint(0, 3, (15,))
    # 两份模型从完全相同参数起步。
    reference_model = SplitNet()
    parallel_model = copy.deepcopy(reference_model)

    # 完整 batch 梯度是正确性参照。
    expected = full_batch_gradients(reference_model, x, y)
    # 单设备模拟两个数据并行工作器。
    actual = simulated_data_parallel(parallel_model, x, y, replicas=2)
    # 比较每个参数张量的最大误差。
    errors = [(left - right).abs().max().item() for left, right in zip(expected, actual)]
    print(f"数据并行规约与完整 batch 的最大梯度误差: {max(errors):.3e}")
    assert max(errors) < 1e-6

    # 模型并行示例只做前向，不改变参数。
    split_output = model_parallel_forward(copy.deepcopy(reference_model), x)
    print(f"模型并行输出 Shape: {tuple(split_output.shape)}，设备: {split_output.device}")

    # DataParallel 示例真正完成一次反向和参数更新。
    loss = safe_data_parallel_step(copy.deepcopy(reference_model), x, y)
    print(f"DataParallel 单步损失: {loss:.4f}，可见 GPU 数: {torch.cuda.device_count()}")
    print("结论：数据并行切样本并规约梯度；模型并行切层并传递激活。")


if __name__ == "__main__":
    main()
