"""演示标准卷积怎样把 2 个输入通道组合成 16 个输出通道。

这不是训练程序，而是固定权重的可手算实验。运行后会打印输入、权重、
输出 Shape，以及 16 个确定的输出值，并对照标准卷积与分组卷积。
"""

import torch
from torch import nn


def demonstrate_two_to_sixteen() -> None:
    """固定一个 2→16 的 1×1 卷积，让全部输出都可以手算。"""

    # NCHW=(1,2,1,1)：一张图、两个通道、高宽都为 1。
    x = torch.tensor([[[[10.0]], [[20.0]]]])

    # 创建 16 套输出权重；每套都必须读取两个输入通道。
    conv = nn.Conv2d(
        in_channels=2,
        out_channels=16,
        kernel_size=1,
        bias=False,
    )

    # 这里是教学演示，所以人为写入容易手算的权重，不进行训练。
    with torch.no_grad():
        # 第 o 个输出对输入通道 0 使用权重 o+1，即依次为 1 到 16。
        conv.weight[:, 0, 0, 0] = torch.arange(1.0, 17.0)
        # 16 个输出对输入通道 1 都使用权重 1。
        conv.weight[:, 1, 0, 0] = 1.0

    # 标准 Module 调用执行前向：每个输出都把两个通道的贡献相加。
    y = conv(x)

    # 预期第 o 个输出是 10×(o+1)+20，因此依次为 30、40、…、180。
    expected = torch.arange(30.0, 181.0, 10.0).reshape(1, 16, 1, 1)

    # 不只检查 Shape，还逐项检查实际数值是否与手算结果一致。
    torch.testing.assert_close(y, expected)

    print("=== 2 个输入通道 → 16 个输出通道 ===")
    print("输入 Shape:", tuple(x.shape))
    print("权重 Shape:", tuple(conv.weight.shape))
    print("输出 Shape:", tuple(y.shape))
    print("16 个输出值:", y.flatten().tolist())


def compare_standard_and_grouped_convolution() -> None:
    """对照 6→16 标准卷积和 groups=2 分组卷积的连接范围。"""

    # 默认 groups=1：每个输出通道都读取全部 6 个输入通道。
    standard = nn.Conv2d(6, 16, kernel_size=3, bias=True)

    # 权重应为 (输出16, 输入6, 核高3, 核宽3)。
    assert tuple(standard.weight.shape) == (16, 6, 3, 3)

    # 参数包括 16×6×3×3 个权重和 16 个输出偏置。
    standard_parameters = sum(parameter.numel() for parameter in standard.parameters())
    assert standard_parameters == 880

    # groups=2：6 个输入通道分成两组，每个输出只读取所属组的 3 个通道。
    grouped = nn.Conv2d(6, 16, kernel_size=3, groups=2, bias=True)

    # 因为每个输出只连接 6/2=3 个输入通道，所以权重第二维变为 3。
    assert tuple(grouped.weight.shape) == (16, 3, 3, 3)

    # 分组卷积参数为 16×3×3×3 个权重，再加 16 个偏置。
    grouped_parameters = sum(parameter.numel() for parameter in grouped.parameters())
    assert grouped_parameters == 448

    print("\n=== 标准卷积与分组卷积对照 ===")
    print("标准 6→16 权重 Shape:", tuple(standard.weight.shape))
    print("标准 6→16 含偏置参数量:", standard_parameters)
    print("groups=2 时权重 Shape:", tuple(grouped.weight.shape))
    print("groups=2 时含偏置参数量:", grouped_parameters)


if __name__ == "__main__":
    demonstrate_two_to_sixteen()
    compare_standard_and_grouped_convolution()
