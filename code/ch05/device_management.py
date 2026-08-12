"""5.6 CPU/GPU 自适应训练、迁移与正确计时。

直接运行：python code/ch05/device_management.py
无 CUDA 时自动回退 CPU，同样完成 smoke test。
"""

import time

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def choose_device(index: int = 0) -> torch.device:
    """指定 GPU 存在则使用，否则回退 CPU。"""
    if torch.cuda.is_available() and index < torch.cuda.device_count():
        return torch.device(f"cuda:{index}")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    """CUDA 异步执行，准确计时前后需要同步；CPU 不需要。"""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    torch.manual_seed(61)
    device = choose_device()
    print("PyTorch：", torch.__version__)
    print("CUDA 可用：", torch.cuda.is_available())
    print("本次设备：", device)

    # model.to 会递归迁移注册 Parameter 和 buffer。
    model = nn.Sequential(
        nn.Linear(10, 24),
        nn.ReLU(),
        nn.Linear(24, 3),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # 数据集保留在 CPU；每个批次按需迁移，避免一次占满 GPU。
    cpu_X = torch.randn(256, 10)
    cpu_y = torch.randint(0, 3, (256,))
    loader = DataLoader(
        TensorDataset(cpu_X, cpu_y),
        batch_size=32,
        shuffle=True,
        # page-locked memory 主要服务 CPU -> CUDA 异步拷贝。
        pin_memory=(device.type == "cuda"),
    )

    model.train()
    synchronize(device)
    start = time.perf_counter()
    for X, y in loader:
        # tensor.to 可能返回新张量，必须接住返回值。
        X = X.to(device, non_blocking=(device.type == "cuda"))
        y = y.to(device, non_blocking=(device.type == "cuda"))
        # 1. Forward：X(B,10) -> logits(B,3)，都在同一 device。
        logits = model(X)
        # 2. Loss：标签也必须在同一 device。
        loss = nn.functional.cross_entropy(logits, y)
        # 3. zero_grad：清理参数旧梯度。
        optimizer.zero_grad(set_to_none=True)
        # 4. backward：梯度张量与参数位于同一 device。
        loss.backward()
        # 5. step：设备上的优化器状态和参数一起更新。
        optimizer.step()
    synchronize(device)
    elapsed = time.perf_counter() - start
    print(f"一个 epoch 用时：{elapsed * 1000:.2f} ms")

    # eval 控制模块行为；inference_mode 关闭 autograd。
    model.eval()
    with torch.inference_mode():
        sample = cpu_X[:5].to(device)
        # 预测结果移回 CPU，便于交给 NumPy、日志或业务代码。
        prediction = model(sample).argmax(dim=1).cpu()
    print("预测：", prediction.tolist(), "| device=", prediction.device)
    assert prediction.device.type == "cpu"
    assert next(model.parameters()).device == device

    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device) / 2**20
        reserved = torch.cuda.memory_reserved(device) / 2**20
        print(f"allocated={allocated:.1f} MB, reserved={reserved:.1f} MB")
        print("reserved 较大不自动等于泄漏，它包含缓存分配器保留空间。")


if __name__ == "__main__":
    main()
