"""第 12 章：执行模式、正确计时与硬件/吞吐诊断。

离线运行：python code/ch12/performance_basics.py --quick
尝试更激进的编译：python code/ch12/performance_basics.py --backend inductor
"""

from __future__ import annotations

import argparse
import os
import platform
import time
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn


@dataclass
class Timing:
    """保存一次基准测试的可复核结果。"""

    seconds: float
    samples_per_second: float


class TinyMLP(nn.Module):
    """足够小、CPU 也能迅速运行的性能实验网络。"""

    def __init__(self, width: int = 128) -> None:
        super().__init__()
        # 两层线性层提供主要矩阵乘工作量。
        self.net = nn.Sequential(
            nn.Linear(width, width * 2),
            nn.ReLU(),
            nn.Linear(width * 2, width),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入和输出均为 (B, width)，这里只构建前向计算图。
        return self.net(x)


def synchronize(device: torch.device) -> None:
    """等待设备完成队列中的工作；CPU 本身是同步路径。"""

    # CUDA kernel 默认异步入队，读取墙钟前后都必须设障碍器。
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def benchmark(
    function: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    warmup: int,
    steps: int,
) -> Timing:
    """同步计时，并把总时间换算为样本吞吐。"""

    # 预热会触发内存分配、缓存填充以及可能的图编译。
    for _ in range(warmup):
        function(x)
    # 确保预热工作不混入正式计时。
    synchronize(x.device)
    # perf_counter 是适合短耗时测量的单调高分辨率时钟。
    started = time.perf_counter()
    # 重复运行可摊薄 Python 计时本身的噪声。
    for _ in range(steps):
        function(x)
    # 等到最后一个设备任务真正结束才停止计时。
    synchronize(x.device)
    elapsed = time.perf_counter() - started
    # 吞吐量必须用真正处理的样本数除以同步后的总时间。
    throughput = x.shape[0] * steps / elapsed
    return Timing(seconds=elapsed, samples_per_second=throughput)


def compile_with_fallback(
    model: nn.Module, backend: str
) -> tuple[Callable[[torch.Tensor], torch.Tensor], str]:
    """返回编译模型；环境不支持时明确回退到 eager。"""

    # 老版本 PyTorch 可能根本没有 torch.compile。
    if not hasattr(torch, "compile"):
        return model, "eager（当前 PyTorch 没有 torch.compile）"
    try:
        # backend=eager 仍会捕获图，适合快速演示；inductor 才尝试生成优化代码。
        compiled = torch.compile(model, backend=backend)
        return compiled, f"torch.compile(backend={backend!r})"
    except Exception as error:  # pragma: no cover - 取决于本机编译工具链
        # 性能功能失败不应妨碍正确性实验继续运行。
        return model, f"eager（编译创建失败：{type(error).__name__}）"


def print_hardware(device: torch.device) -> None:
    """打印与吞吐诊断直接相关的环境信息。"""

    print("=== 硬件与运行时 ===")
    print(f"Python/系统: {platform.python_version()} / {platform.system()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CPU 逻辑核心: {os.cpu_count()}，PyTorch 线程: {torch.get_num_threads()}")
    print(f"当前设备: {device}")
    if device.type == "cuda":
        # 属性可帮助判断显存容量、计算能力和设备型号。
        props = torch.cuda.get_device_properties(device)
        gib = props.total_memory / 1024**3
        print(f"GPU: {props.name}，显存: {gib:.2f} GiB，计算能力: {props.major}.{props.minor}")
    else:
        print("CUDA 不可用或被禁用；CPU 路径仍能完整验证计时逻辑。")


def estimate_linear_intensity(batch: int, width: int) -> None:
    """粗估第一层线性层的算术强度，建立 roofline 直觉。"""

    hidden = width * 2
    # 每个乘加近似算 2 FLOPs，第一层总计算量约为 2*B*D*H。
    flops = 2 * batch * width * hidden
    # 极简下界：读输入、权重并写输出；实际还受缓存和中间量影响。
    bytes_moved = 4 * (batch * width + width * hidden + batch * hidden)
    intensity = flops / bytes_moved
    print(f"第一层粗估: {flops / 1e6:.1f} MFLOPs，算术强度约 {intensity:.1f} FLOPs/byte")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="使用更少步数做烟雾测试")
    parser.add_argument("--cpu", action="store_true", help="即使有 GPU 也强制用 CPU")
    parser.add_argument("--backend", default="eager", choices=["eager", "inductor"])
    args = parser.parse_args()

    # 固定随机种子，使 eager 与 compiled 比较面对同一个输入和参数。
    torch.manual_seed(12)
    # 用户可强制 CPU；否则优先选 CUDA。
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    # quick 模式减少正式循环，适合 CI；正常模式结果更稳定。
    warmup, steps = (2, 8) if args.quick else (5, 40)
    # batch 和 width 决定每次矩阵乘规模。
    batch, width = (64, 64) if args.quick else (512, 128)

    print_hardware(device)
    estimate_linear_intensity(batch, width)

    # 创建输入 (B,D)，requires_grad=False，因为这里只测推理。
    x = torch.randn(batch, width, device=device)
    # eval 会关闭 Dropout 等训练态随机行为。
    model = TinyMLP(width).to(device).eval()
    # 先测动态图 eager，作为正确性和速度基线。
    eager_timing = benchmark(model, x, warmup, steps)
    print(f"eager: {eager_timing.samples_per_second:,.0f} samples/s")

    # 构造可编译调用；失败时函数会保留 eager 路径。
    candidate, mode = compile_with_fallback(model, args.backend)
    try:
        # torch.compile 常在第一次真正调用时才编译，因此 benchmark 也要捕获异常。
        compiled_timing = benchmark(candidate, x, warmup, steps)
    except Exception as error:  # pragma: no cover - 依赖本机编译器
        mode = f"eager 回退（首次执行失败：{type(error).__name__}）"
        compiled_timing = benchmark(model, x, warmup, steps)
    print(f"{mode}: {compiled_timing.samples_per_second:,.0f} samples/s")

    # 对同一输入检查语义等价；这里只读取输出，不改变参数。
    with torch.inference_mode():
        eager_output = model(x)
        candidate_output = candidate(x) if "回退" not in mode else eager_output
    max_error = (eager_output - candidate_output).abs().max().item()
    print(f"输出最大绝对误差: {max_error:.3e}")
    print("提示：一次快慢不能下结论；应报告预热、同步、输入 Shape 和多次统计。")


if __name__ == "__main__":
    main()
