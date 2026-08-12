"""第 6 章：卷积、通道与池化的最小可运行实验。

运行：
    python code/ch06/convolution_basics.py

程序不下载数据。它先手写核心运算，再与 PyTorch 的高效实现逐项核对。
这里故意使用很小的张量，让读者可以打印并手算每个结果。
"""

from __future__ import annotations

import argparse

import torch
from torch import Tensor
from torch.nn import functional as F


def corr2d(x: Tensor, kernel: Tensor) -> Tensor:
    """计算二维互相关；x 和 kernel 都是不含批量、通道维的二维张量。"""
    if x.ndim != 2 or kernel.ndim != 2:
        raise ValueError("corr2d 期望两个二维张量")
    kernel_h, kernel_w = kernel.shape  # 卷积窗口本身的高和宽。
    out_h = x.shape[0] - kernel_h + 1  # 无填充、步幅 1 时的输出高度。
    out_w = x.shape[1] - kernel_w + 1  # 无填充、步幅 1 时的输出宽度。
    if out_h <= 0 or out_w <= 0:
        raise ValueError("卷积核不能大于输入")
    output = torch.zeros((out_h, out_w), dtype=x.dtype, device=x.device)
    for row in range(out_h):  # 窗口从上到下滑动。
        for col in range(out_w):  # 窗口从左到右滑动。
            patch = x[row : row + kernel_h, col : col + kernel_w]  # 取当前局部区域。
            output[row, col] = (patch * kernel).sum()  # 对应元素相乘后求和。
    return output


def corr2d_multi_in(x: Tensor, kernel: Tensor) -> Tensor:
    """多输入通道互相关：各通道分别计算，再沿输入通道求和。"""
    if x.ndim != 3 or kernel.ndim != 3:
        raise ValueError("输入与卷积核都应为 (C_in, H, W)")
    if x.shape[0] != kernel.shape[0]:
        raise ValueError("输入通道数必须与卷积核的输入通道数一致")
    per_channel = [corr2d(x_channel, k_channel) for x_channel, k_channel in zip(x, kernel)]
    return torch.stack(per_channel, dim=0).sum(dim=0)  # 通道信息在这里被融合。


def corr2d_multi_in_out(x: Tensor, kernels: Tensor) -> Tensor:
    """多输入、多输出通道互相关。

    x:       (C_in, H, W)
    kernels: (C_out, C_in, K_h, K_w)
    返回:    (C_out, H_out, W_out)
    """
    if kernels.ndim != 4:
        raise ValueError("kernels 应为 (C_out, C_in, K_h, K_w)")
    maps = [corr2d_multi_in(x, one_output_kernel) for one_output_kernel in kernels]
    return torch.stack(maps, dim=0)  # 每组卷积核生成一个输出通道。


def corr2d_1x1(x: Tensor, kernels: Tensor) -> Tensor:
    """把 1×1 卷积改写为每个空间位置共享的矩阵乘法。"""
    channels_in, height, width = x.shape
    channels_out = kernels.shape[0]
    x_flat = x.reshape(channels_in, height * width)  # 每一列代表一个空间位置。
    weights = kernels.reshape(channels_out, channels_in)  # 只混合通道，不看邻居。
    output = weights @ x_flat  # 同一权重矩阵作用于所有空间位置。
    return output.reshape(channels_out, height, width)


def pool2d(x: Tensor, window: tuple[int, int], mode: str = "max") -> Tensor:
    """手写二维最大池化或平均池化；使用步幅 1，且不填充。"""
    if x.ndim != 2:
        raise ValueError("pool2d 的演示输入应为二维张量")
    pool_h, pool_w = window
    out_h = x.shape[0] - pool_h + 1
    out_w = x.shape[1] - pool_w + 1
    output = torch.empty((out_h, out_w), dtype=x.dtype, device=x.device)
    for row in range(out_h):
        for col in range(out_w):
            patch = x[row : row + pool_h, col : col + pool_w]  # 当前池化窗口。
            if mode == "max":
                output[row, col] = patch.max()  # 只保留最强响应。
            elif mode == "avg":
                output[row, col] = patch.mean()  # 汇总局部平均水平。
            else:
                raise ValueError("mode 只能是 'max' 或 'avg'")
    return output


def conv_output_size(size: int, kernel: int, padding: int = 0, stride: int = 1) -> int:
    """计算单个空间方向的卷积输出尺寸。"""
    return (size + 2 * padding - kernel) // stride + 1


def run_demo(seed: int) -> None:
    """运行所有可手算实验，并用断言验证实现。"""
    torch.manual_seed(seed)  # 固定随机数，方便复现实验。

    # 一条竖直亮带：左边缘和右边缘可以被差分核检测出来。
    image = torch.zeros((6, 8), dtype=torch.float32)
    image[:, 2:6] = 1.0
    edge_kernel = torch.tensor([[1.0, -1.0]])
    edge_map = corr2d(image, edge_kernel)
    pytorch_edge = F.conv2d(
        image[None, None],  # 给二维图像补上批量维 N 和通道维 C。
        edge_kernel[None, None],  # 给二维核补上输出、输入通道维。
    )[0, 0]
    torch.testing.assert_close(edge_map, pytorch_edge)

    # 多通道核的 Shape 是 (C_out, C_in, K_h, K_w)。
    multi_x = torch.randn(3, 5, 6)
    multi_k = torch.randn(4, 3, 2, 3)
    manual_multi = corr2d_multi_in_out(multi_x, multi_k)
    torch_multi = F.conv2d(multi_x[None], multi_k)[0]  # F.conv2d 还需要批量维。
    torch.testing.assert_close(manual_multi, torch_multi, rtol=1e-5, atol=1e-6)

    # 1×1 卷积与逐位置的通道矩阵乘法完全等价。
    pointwise_k = torch.randn(5, 3, 1, 1)
    pointwise_matrix = corr2d_1x1(multi_x, pointwise_k)
    pointwise_torch = F.conv2d(multi_x[None], pointwise_k)[0]
    torch.testing.assert_close(pointwise_matrix, pointwise_torch, rtol=1e-5, atol=1e-6)

    # 池化通常不融合通道；这里只对一个二维通道展示窗口动作。
    pool_x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    manual_max = pool2d(pool_x, (2, 2), mode="max")
    manual_avg = pool2d(pool_x, (2, 2), mode="avg")
    torch_max = F.max_pool2d(pool_x[None, None], kernel_size=2, stride=1)[0, 0]
    torch_avg = F.avg_pool2d(pool_x[None, None], kernel_size=2, stride=1)[0, 0]
    torch.testing.assert_close(manual_max, torch_max)
    torch.testing.assert_close(manual_avg, torch_avg)

    # padding=1、kernel=3、stride=2：7×7 会变为 4×4。
    predicted_size = conv_output_size(size=7, kernel=3, padding=1, stride=2)
    shape_probe = F.conv2d(torch.randn(1, 2, 7, 7), torch.randn(4, 2, 3, 3), padding=1, stride=2)
    assert shape_probe.shape == (1, 4, predicted_size, predicted_size)

    print("二维边缘响应：")
    print(edge_map)
    print(f"多通道卷积: (3, 5, 6) -> {tuple(manual_multi.shape)}")
    print(f"1x1 卷积:    (3, 5, 6) -> {tuple(pointwise_matrix.shape)}")
    print("最大池化结果：\n", manual_max)
    print("平均池化结果：\n", manual_avg)
    print(f"padding/stride 公式预测输出边长：{predicted_size}")
    print("全部对照测试通过。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="卷积基础的可手算 PyTorch 实验")
    parser.add_argument("--seed", type=int, default=7, help="随机种子")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_demo(args.seed)
