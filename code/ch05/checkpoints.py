"""5.5 保存并恢复可继续训练的 checkpoint。

直接运行：python code/ch05/checkpoints.py
默认把临时 checkpoint 写入系统临时目录，测试结束后自动删除。
"""

from pathlib import Path
import random
import tempfile

import torch
from torch import nn


def build_model() -> nn.Sequential:
    """模型结构代码不会被 state_dict 保存，加载前必须重建。"""
    return nn.Sequential(
        nn.Linear(6, 12),  # (B,6) -> (B,12)
        nn.ReLU(),         # Shape 不变
        nn.Linear(12, 3),  # (B,12) -> logits(B,3)
    )


def one_training_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    X: torch.Tensor,
    y: torch.Tensor,
) -> float:
    """执行通用五步训练法并返回 loss。"""
    model.train()
    # 1. Forward：logits.shape == (B,3)。
    logits = model(X)
    # 2. Loss：y.shape == (B,)，dtype 为 int64。
    loss = nn.functional.cross_entropy(logits, y)
    # 3. zero_grad：清理旧梯度。
    optimizer.zero_grad(set_to_none=True)
    # 4. backward：产生每个参数的 grad。
    loss.backward()
    # 5. step：Adam 会同时更新参数及内部一、二阶矩状态。
    optimizer.step()
    return loss.item()


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
) -> None:
    """保存恢复训练所需的核心状态。"""
    checkpoint = {
        # 只保存模型张量状态，不 pickle 整个模型对象。
        "model_state": model.state_dict(),
        # Adam 动量等历史状态决定后续更新轨迹。
        "optimizer_state": optimizer.state_dict(),
        # 记录训练进度，恢复时从下一轮继续。
        "epoch": epoch,
        # 随机状态帮助恢复随机序列。
        "python_random_state": random.getstate(),
        "torch_random_state": torch.random.get_rng_state(),
        # 结构超参数用于重建兼容模型。
        "config": {"inputs": 6, "hidden": 12, "classes": 3},
    }
    torch.save(checkpoint, path)


def load_checkpoint(path: Path) -> tuple[nn.Module, torch.optim.Optimizer, int]:
    """在 CPU 上恢复结构、模型权重、优化器与进度。"""
    # weights_only=True 限制为张量和安全基本类型；只加载可信文件仍是原则。
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    if config != {"inputs": 6, "hidden": 12, "classes": 3}:
        raise ValueError("checkpoint 配置与当前 build_model 不兼容")

    # 先重建相同结构，state_dict 本身不包含 forward 代码。
    model = build_model()
    # strict=True 要求键名和 Shape 全部匹配。
    result = model.load_state_dict(checkpoint["model_state"], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("checkpoint 键不完整")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    return model, optimizer, int(checkpoint["epoch"])


def main() -> None:
    torch.manual_seed(59)
    random.seed(59)
    # 固定验证输入可用于比较保存前后输出。
    X = torch.randn(16, 6)
    y = torch.randint(0, 3, (16,))
    model = build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss = one_training_step(model, optimizer, X, y)
    print(f"保存前训练 loss：{loss:.4f}")

    # TemporaryDirectory 在退出时自动清理，不污染仓库。
    with tempfile.TemporaryDirectory(prefix="d2l_checkpoint_") as directory:
        path = Path(directory) / "checkpoint.pt"
        save_checkpoint(path, model, optimizer, epoch=1)
        restored_model, restored_optimizer, epoch = load_checkpoint(path)

        # 比较时同时切换 eval，并关闭 autograd。
        model.eval()
        restored_model.eval()
        with torch.inference_mode():
            original_output = model(X)
            restored_output = restored_model(X)
        max_error = (original_output - restored_output).abs().max().item()
        print(f"恢复 epoch={epoch}，最大输出误差={max_error:.3e}")
        assert max_error == 0.0

        # 再执行一步，验证恢复的优化器确实可以继续训练。
        resumed_loss = one_training_step(restored_model, restored_optimizer, X, y)
        print(f"恢复后继续训练 loss：{resumed_loss:.4f}")


if __name__ == "__main__":
    main()
