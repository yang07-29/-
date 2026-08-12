"""第 16 章：深度学习环境诊断、设备选择与可靠计时。

运行：
    python code/ch16/environment_diagnostics.py
    python code/ch16/environment_diagnostics.py --device cpu --matrix-size 256
    python code/ch16/environment_diagnostics.py --json-report environment-report.json

程序只依赖 PyTorch 和 Python 标准库。它会完成环境盘点、矩阵乘法计时，
并执行一次完整的 forward -> loss -> backward -> step 训练冒烟测试。
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


def choose_device(requested: str) -> torch.device:
    """把 auto/cpu/cuda 请求转换成经过检查的 torch.device。"""
    # auto 表示有 CUDA 就用第一张 GPU，否则安全退回 CPU。
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    # 用户明确要求 CUDA 时，不要悄悄退回 CPU；直接给出可理解的错误。
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求了 CUDA，但当前 PyTorch 看不到可用的 CUDA GPU")
    # 把经过检查的字符串转换为 PyTorch 设备对象。
    return torch.device(requested)


def synchronize(device: torch.device) -> None:
    """GPU 运算是异步提交的；计时边界必须显式同步。"""
    # CPU 运算通常在调用返回前已完成，不需要 CUDA 同步器。
    if device.type == "cuda":
        # 等待当前设备队列中的 CUDA 工作真正完成。
        torch.cuda.synchronize(device)


def collect_environment() -> dict[str, Any]:
    """收集不含密钥、用户名和文件内容的安全环境摘要。"""
    # 只记录操作系统类型与版本，不读取任何凭据或用户文件。
    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "cuda_compiled_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
    }
    # 只有 CUDA 可用时才查询设备属性，避免在 CPU 环境触发驱动错误。
    if torch.cuda.is_available():
        # 为每张可见 GPU 保存名称、计算能力和显存容量。
        report["cuda_devices"] = []
        # 遍历 PyTorch 当前可见的全部 CUDA 设备。
        for index in range(torch.cuda.device_count()):
            # get_device_properties 返回当前 GPU 的静态硬件属性。
            properties = torch.cuda.get_device_properties(index)
            # 把字节换算成 GiB，更符合深度学习选卡时的直觉。
            memory_gib = properties.total_memory / 1024**3
            # 只保存排错和选型真正需要的字段。
            report["cuda_devices"].append(
                {
                    "index": index,
                    "name": properties.name,
                    "compute_capability": f"{properties.major}.{properties.minor}",
                    "total_memory_gib": round(memory_gib, 2),
                }
            )
    # 返回可直接 JSON 序列化的字典。
    return report


def benchmark_matmul(device: torch.device, matrix_size: int, repeats: int) -> dict[str, float]:
    """正确计时同尺寸方阵乘法，并返回延迟和近似吞吐。"""
    # 在目标设备创建两个方阵；Shape 均为 (n,n)。
    left = torch.randn(matrix_size, matrix_size, device=device)
    right = torch.randn(matrix_size, matrix_size, device=device)
    # 预热让内核选择、缓存和可能的编译开销不混入正式计时。
    for _ in range(2):
        _ = left @ right
    # 若在 GPU 上，必须等预热真正结束后再开启计时器。
    synchronize(device)
    # perf_counter 是适合测量短时间间隔的高分辨率单调时钟。
    started_at = time.perf_counter()
    # 重复相同运算以降低一次计时的偶然噪声。
    for _ in range(repeats):
        product = left @ right
    # GPU 调用返回不等于计算完成；停止计时前再次等待设备队列。
    synchronize(device)
    # 计算所有重复的真实墙钟时间。
    elapsed_seconds = time.perf_counter() - started_at
    # 读取一个元素，确保结果确实可用，同时避免打印整个大矩阵。
    checksum = float(product[0, 0].item())
    # 一次 n×n 矩阵乘法近似需要 2n^3 次浮点运算。
    total_flops = 2.0 * matrix_size**3 * repeats
    # 把总浮点运算量除以时间并换算成十进制 GFLOP/s。
    throughput_gflops = total_flops / elapsed_seconds / 1e9
    # 返回便于显示和写入报告的数值指标。
    return {
        "mean_latency_ms": elapsed_seconds / repeats * 1000.0,
        "approx_gflops": throughput_gflops,
        "checksum": checksum,
    }


def training_smoke_test(device: torch.device) -> dict[str, Any]:
    """跑通一个批次的训练五步，并确认参数确实发生变化。"""
    # 固定随机种子，让损失和参数变化在同一环境中可复现。
    torch.manual_seed(2026)
    # 创建 32 个样本、8 个特征的合成输入，Shape 为 (32,8)。
    features = torch.randn(32, 8, device=device)
    # 创建三分类标签，Shape 为 (32,)，dtype 为 long。
    targets = torch.randint(0, 3, (32,), device=device)
    # 小模型把 8 维输入映射到 3 个类别 logits。
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 3)).to(device)
    # 交叉熵直接接收 logits:(32,3) 与类别索引:(32,)。
    loss_fn = nn.CrossEntropyLoss()
    # SGD 管理模型中全部已注册的可训练参数。
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    # 克隆第一层权重，用于更新后验证参数值真的改变。
    weight_before = model[0].weight.detach().clone()

    # 1. Forward：模型产生每个样本的三个原始类别分数。
    logits = model(features)
    # 2. Loss：得到一个标量批平均交叉熵。
    loss = loss_fn(logits, targets)
    # 3. zero_grad：清除可能遗留的旧梯度。
    optimizer.zero_grad(set_to_none=True)
    # 4. backward：把梯度写入每个参数的 .grad。
    loss.backward()
    # 在更新前记录梯度范数，确认反向传播产生了有限信号。
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf"))
    # 5. step：优化器读取梯度并真正修改参数。
    optimizer.step()

    # 比较更新前后的第一层权重；差值应大于零。
    max_parameter_change = (model[0].weight.detach() - weight_before).abs().max().item()
    # 所有检查都使用明确断言，让环境问题在最小案例中快速暴露。
    assert logits.shape == (32, 3), "模型输出 Shape 不符合三分类约定"
    assert torch.isfinite(loss), "损失出现 NaN 或 inf"
    assert torch.isfinite(grad_norm), "梯度范数出现 NaN 或 inf"
    assert max_parameter_change > 0.0, "step 后参数没有变化"
    # 返回可读的训练健康指标。
    return {
        "logits_shape": list(logits.shape),
        "loss": float(loss.item()),
        "gradient_norm": float(grad_norm.item()),
        "max_parameter_change": float(max_parameter_change),
        "passed": True,
    }


def parse_args() -> argparse.Namespace:
    """定义命令行接口；所有默认值都适合普通 CPU 快速运行。"""
    # 创建命令行解析器并给出程序用途。
    parser = argparse.ArgumentParser(description="PyTorch 环境诊断与训练冒烟测试")
    # auto 自动选择 CUDA 或 CPU，也允许用户明确指定。
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    # 512 方阵足以演示计时，同时不会在普通机器上占用大量内存。
    parser.add_argument("--matrix-size", type=int, default=512)
    # 多次重复能够降低计时抖动。
    parser.add_argument("--repeats", type=int, default=5)
    # 可选 JSON 路径便于提交 bug 报告；默认不写文件。
    parser.add_argument("--json-report", type=Path)
    # 返回已经解析并完成类型转换的参数对象。
    return parser.parse_args()


def main() -> None:
    """执行环境盘点、可靠计时和完整训练链路检查。"""
    # 读取用户指定的设备、矩阵尺寸和输出路径。
    args = parse_args()
    # 对非法的非正参数尽早失败，避免产生难懂的底层错误。
    if args.matrix_size <= 0 or args.repeats <= 0:
        raise ValueError("matrix-size 与 repeats 必须是正整数")
    # 选择并验证实际运行设备。
    device = choose_device(args.device)
    # 收集软件与硬件摘要。
    report = collect_environment()
    # 把本次真正使用的设备写入报告。
    report["selected_device"] = str(device)
    # 使用正确同步边界执行矩阵乘法基准。
    report["matmul_benchmark"] = benchmark_matmul(device, args.matrix_size, args.repeats)
    # 跑通训练五步并验证梯度和参数变化。
    report["training_smoke_test"] = training_smoke_test(device)

    # 用中文标题帮助学习者快速定位报告的三个部分。
    print("=== 环境摘要 ===")
    # ensure_ascii=False 保留中文，indent=2 提升可读性。
    print(json.dumps(report, ensure_ascii=False, indent=2))
    # 只有用户明确提供路径时才把报告写入磁盘。
    if args.json_report is not None:
        # 创建目标文件的父目录，exist_ok=True 允许目录已经存在。
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        # 以 UTF-8 写入，不包含密码、令牌或用户文件内容。
        args.json_report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        # 明确显示最终路径，方便用户附到 bug 报告中。
        print(f"报告已写入：{args.json_report.resolve()}")


if __name__ == "__main__":
    # 直接运行脚本时进入主流程；被 import 时不会自动执行。
    main()
