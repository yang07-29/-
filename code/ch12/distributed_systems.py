"""第 12 章：参数服务器与环形 all-reduce 的纯张量模拟。

运行：python code/ch12/distributed_systems.py
"""

from __future__ import annotations

import torch


def parameter_server_average(worker_gradients: list[torch.Tensor]) -> torch.Tensor:
    """模拟所有工作器把梯度推送到中心服务器再取回平均值。"""

    # stack 后 Shape 从 (P,) 变为 (N,P)，N 是工作器数。
    stacked = torch.stack(worker_gradients, dim=0)
    # 中心服务器沿工作器维求平均，结果 Shape 回到 (P,)。
    return stacked.mean(dim=0)


def ring_allreduce(worker_gradients: list[torch.Tensor]) -> list[torch.Tensor]:
    """用 reduce-scatter + all-gather 模拟环形 all-reduce。"""

    world_size = len(worker_gradients)
    if world_size < 2:
        # 单工作器无需通信，但返回 clone 避免调用方意外共享存储。
        return [worker_gradients[0].clone()]
    if worker_gradients[0].numel() % world_size != 0:
        raise ValueError("为简化演示，参数长度必须能被工作器数整除")

    # 每个工作器把自己的梯度切成 N 块；chunks[r][c] 位于工作器 r。
    chunks = [list(torch.chunk(gradient.clone(), world_size)) for gradient in worker_gradients]

    # 第一阶段：每轮把一块发给右邻居，并在接收端累加。
    for step in range(world_size - 1):
        # 先复制消息，模拟所有工作器并行发送而非边改边读。
        messages = [chunks[rank][(rank - step) % world_size].clone() for rank in range(world_size)]
        for rank in range(world_size):
            # rank 从左邻居 rank-1 收到对应块。
            source = (rank - 1) % world_size
            receive_index = (rank - step - 1) % world_size
            # 原地相加后，该块多汇总了一个工作器的贡献。
            chunks[rank][receive_index].add_(messages[source])

    # 第二阶段：每个 rank 已拥有一块完整和，再沿环传播这些结果。
    for step in range(world_size - 1):
        # reduce-scatter 后 rank 完整拥有的块索引为 rank+1（模 N）。
        messages = [
            chunks[rank][(rank - step + 1) % world_size].clone()
            for rank in range(world_size)
        ]
        for rank in range(world_size):
            # 从左邻居接收下一块已规约结果。
            source = (rank - 1) % world_size
            receive_index = (rank - step) % world_size
            # all-gather 是覆盖，不再相加。
            chunks[rank][receive_index] = messages[source]

    # 拼回 (P,) 并除以 N，得到平均梯度。
    return [torch.cat(parts) / world_size for parts in chunks]


def stale_parameter_server_demo() -> None:
    """展示异步参数服务器中“旧梯度”为什么会改变优化轨迹。"""

    # 标量参数初始值为 4，目标损失可想成 (w-target)^2/2。
    weight = torch.tensor(4.0)
    # 两个工作器在同一个旧版本 w=4 上分别对目标 1 和 3 求梯度。
    old_gradients = [weight - 1.0, weight - 3.0]
    # 服务器先应用工作器 0 的梯度，参数版本已经前进。
    learning_rate = 0.2
    weight = weight - learning_rate * old_gradients[0]
    # 工作器 1 仍推送基于旧 w=4 的梯度；这就是陈旧性。
    stale_weight = weight - learning_rate * old_gradients[1]
    # 同步方案先平均同版本梯度，再只更新一次。
    synchronous_weight = torch.tensor(4.0) - learning_rate * torch.stack(old_gradients).mean()
    print(f"异步旧梯度后的 w={stale_weight.item():.3f}；同步平均后的 w={synchronous_weight.item():.3f}")


def main() -> None:
    # 四个工作器、每个 8 维梯度，满足可均分为四块。
    torch.manual_seed(12)
    gradients = [torch.randn(8) for _ in range(4)]
    # 中心服务器给出参照平均值。
    central = parameter_server_average(gradients)
    # 环形算法让每个工作器最终都拿到同一个平均值。
    ring_results = ring_allreduce(gradients)
    # 检查每个 rank 与中心参照的最大误差。
    max_error = max((result - central).abs().max().item() for result in ring_results)
    print(f"ring all-reduce 与中心平均最大误差: {max_error:.3e}")
    assert max_error < 1e-6

    # P 表示总参数字节数，下面只比较每个工作器的理论通信量。
    world_size = len(gradients)
    parameter_bytes = gradients[0].numel() * gradients[0].element_size()
    server_bytes = 2 * parameter_bytes
    ring_bytes = 2 * parameter_bytes * (world_size - 1) / world_size
    print(f"每工作器理论流量：参数服务器约 {server_bytes:.0f} B，环形约 {ring_bytes:.0f} B")
    print("注意：参数服务器的中心节点还要承受所有工作器的聚合流量。")
    stale_parameter_server_demo()


if __name__ == "__main__":
    main()
