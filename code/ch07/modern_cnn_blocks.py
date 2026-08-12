"""第 7 章：现代 CNN 关键结构的紧凑、可运行 PyTorch 图谱。

运行全部架构的前向、loss、反向和更新 smoke test：
    python code/ch07/modern_cnn_blocks.py --model all --device cpu

只检查某个架构：
    python code/ch07/modern_cnn_blocks.py --model resnet

这些是教学用缩小版：保留 AlexNet、VGG、NiN、GoogLeNet、ResNet、DenseNet
的关键连接方式，但缩小通道数并使用合成输入，从而能在普通 CPU 上快速运行。
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F


NUM_CLASSES = 10


def parameter_count(model: nn.Module) -> int:
    """统计可训练参数量，便于比较架构的参数效率。"""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


class AlexNetSmall(nn.Module):
    """缩小版 AlexNet：更宽更深的卷积编码器 + 大型全连接分类头。"""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3),  # 先用大窗口快速扩大感受野。
            nn.ReLU(),  # AlexNet 以 ReLU 替代 sigmoid，加快深层网络训练。
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 降低空间分辨率。
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 连续 3×3 卷积逐步加工特征。
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((2, 2)),  # 固定分类头收到的尺寸，允许不同输入分辨率。
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),  # (N,64,2,2) -> (N,256)。
            nn.Linear(64 * 2 * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),  # 训练时随机屏蔽激活，缓解全连接头过拟合。
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),  # 返回 logits，交给交叉熵。
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)  # 卷积部分从像素提取空间特征。
        return self.classifier(x)  # 稠密部分完成类别判别。


def vgg_block(num_convs: int, channels_in: int, channels_out: int) -> nn.Sequential:
    """VGG 块：重复同尺寸 3×3 卷积，最后统一减半空间尺寸。"""
    layers: list[nn.Module] = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        channels_in = channels_out  # 下一层收到的是前一层的输出通道。
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))  # 每个块只在末尾下采样一次。
    return nn.Sequential(*layers)


class VGGSmall(nn.Module):
    """用可复用 VGG 块搭建的缩小版网络。"""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        configuration = ((1, 32), (1, 64), (2, 128), (2, 256))
        blocks: list[nn.Module] = []
        channels_in = 1
        for num_convs, channels_out in configuration:
            blocks.append(vgg_block(num_convs, channels_in, channels_out))
            channels_in = channels_out
        self.features = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 把每个通道汇聚成一个数。
            nn.Flatten(),  # (N,256,1,1) -> (N,256)。
            nn.Linear(256, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.features(x))


def nin_block(
    channels_in: int,
    channels_out: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
) -> nn.Sequential:
    """NiN 块：空间卷积后接两个 1×1 卷积，增加逐像素非线性。"""
    return nn.Sequential(
        nn.Conv2d(channels_in, channels_out, kernel_size, stride, padding),
        nn.ReLU(),
        nn.Conv2d(channels_out, channels_out, kernel_size=1),  # 只混合当前位置的通道。
        nn.ReLU(),
        nn.Conv2d(channels_out, channels_out, kernel_size=1),
        nn.ReLU(),
    )


class NiNSmall(nn.Module):
    """缩小版 NiN：末尾直接生成类别通道，再做全局平均池化。"""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nin_block(1, 32, kernel_size=5, stride=2, padding=2),
            nn.MaxPool2d(3, stride=2, padding=1),
            nin_block(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Dropout(0.2),
            nin_block(64, num_classes, kernel_size=3, padding=1),  # 一个通道对应一个类别。
            nn.AdaptiveAvgPool2d((1, 1)),  # 每个类别图取全局平均。
            nn.Flatten(),  # (N,C,1,1) -> (N,C)。
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Inception(nn.Module):
    """Inception 块：四条不同感受野的分支并行，最后沿通道拼接。"""

    def __init__(
        self,
        channels_in: int,
        c1: int,
        c2: tuple[int, int],
        c3: tuple[int, int],
        c4: int,
    ) -> None:
        super().__init__()
        self.branch1 = nn.Sequential(nn.Conv2d(channels_in, c1, kernel_size=1), nn.ReLU())
        self.branch2 = nn.Sequential(
            nn.Conv2d(channels_in, c2[0], kernel_size=1),  # 先压缩通道，降低 3×3 成本。
            nn.ReLU(),
            nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(channels_in, c3[0], kernel_size=1),  # 先降维，再使用更大窗口。
            nn.ReLU(),
            nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),  # 池化分支也保留同样 H、W。
            nn.Conv2d(channels_in, c4, kernel_size=1),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        outputs = (self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x))
        return torch.cat(outputs, dim=1)  # 分支的 H、W 必须一致；输出通道数相加。


class GoogLeNetSmall(nn.Module):
    """由 Inception 并行块组成的缩小版 GoogLeNet。"""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            Inception(32, 16, (16, 24), (8, 8), 8),  # 输出 16+24+8+8=56 通道。
            Inception(56, 24, (16, 32), (8, 16), 8),  # 输出 24+32+16+8=80 通道。
            nn.MaxPool2d(2, stride=2),
            Inception(80, 32, (24, 40), (8, 16), 8),  # 输出 96 通道。
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(96, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Residual(nn.Module):
    """ResNet 基本残差块：学习 F(x)，输出 F(x)+x。"""

    def __init__(self, channels_in: int, channels_out: int, stride: int = 1) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(channels_out),  # 按通道规范化卷积激活。
            nn.ReLU(),
            nn.Conv2d(channels_out, channels_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels_out),
        )
        if stride != 1 or channels_in != channels_out:
            self.skip: nn.Module = nn.Sequential(
                nn.Conv2d(channels_in, channels_out, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels_out),  # 投影捷径同时对齐通道和空间尺寸。
            )
        else:
            self.skip = nn.Identity()  # Shape 已一致时让输入原样通过。
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        residual = self.main(x)  # 主路学习“需要在 x 上改多少”。
        shortcut = self.skip(x)  # 捷径提供基线，并保证可相加的 Shape。
        return self.relu(residual + shortcut)  # 相加不是拼接，Shape 必须完全一致。


class ResNetSmall(nn.Module):
    """使用残差块的紧凑 ResNet。"""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            Residual(32, 32),  # 尺寸不变，捷径是 Identity。
            Residual(32, 64, stride=2),  # 主路与捷径都把 H、W 减半并升通道。
            Residual(64, 64),
            Residual(64, 128, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class DenseLayer(nn.Module):
    """DenseNet 的单层变换；只生成 growth_rate 个新通道。"""

    def __init__(self, channels_in: int, growth_rate: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(channels_in),
            nn.ReLU(),
            nn.Conv2d(channels_in, growth_rate, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)  # 返回新特征；旧特征由 DenseBlock 负责保留。


class DenseBlock(nn.Module):
    """把每一层新特征追加到通道维，后续层能看到此前所有特征。"""

    def __init__(self, num_layers: int, channels_in: int, growth_rate: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            DenseLayer(channels_in + index * growth_rate, growth_rate) for index in range(num_layers)
        )

    def forward(self, x: Tensor) -> Tensor:
        features = [x]  # 列表中始终保留块输入和所有历史新特征。
        for layer in self.layers:
            new_feature = layer(torch.cat(features, dim=1))  # 当前层读取全部历史通道。
            features.append(new_feature)
        return torch.cat(features, dim=1)  # 通道数每层增加 growth_rate。


def transition_block(channels_in: int, channels_out: int) -> nn.Sequential:
    """DenseNet 过渡层：1×1 压缩通道，平均池化压缩空间。"""
    return nn.Sequential(
        nn.BatchNorm2d(channels_in),
        nn.ReLU(),
        nn.Conv2d(channels_in, channels_out, kernel_size=1, bias=False),
        nn.AvgPool2d(kernel_size=2, stride=2),
    )


class DenseNetSmall(nn.Module):
    """由稠密块与过渡层组成的紧凑 DenseNet。"""

    def __init__(self, num_classes: int = NUM_CLASSES, growth_rate: int = 12) -> None:
        super().__init__()
        channels = 24
        block1 = DenseBlock(num_layers=3, channels_in=channels, growth_rate=growth_rate)
        channels += 3 * growth_rate  # DenseBlock 只拼接，所以能精确计算输出通道。
        transition1 = transition_block(channels, channels // 2)
        channels //= 2
        block2 = DenseBlock(num_layers=3, channels_in=channels, growth_rate=growth_rate)
        channels += 3 * growth_rate
        self.net = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, padding=1, bias=False),
            block1,
            transition1,
            block2,
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(channels, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


MODEL_FACTORIES: dict[str, Callable[[], nn.Module]] = {
    "alexnet": AlexNetSmall,
    "vgg": VGGSmall,
    "nin": NiNSmall,
    "googlenet": GoogLeNetSmall,
    "resnet": ResNetSmall,
    "densenet": DenseNetSmall,
}


def check_batch_norm() -> None:
    """用公式手算训练态 BN，并与 PyTorch 输出核对。"""
    x = torch.randn(4, 3, 5, 5)
    gamma = torch.tensor([1.0, 1.5, 0.5]).reshape(1, 3, 1, 1)  # 每通道一个缩放参数。
    beta = torch.tensor([0.0, -0.2, 0.3]).reshape(1, 3, 1, 1)  # 每通道一个平移参数。
    mean = x.mean(dim=(0, 2, 3), keepdim=True)  # 卷积 BN 在 N、H、W 上统计。
    variance = x.var(dim=(0, 2, 3), unbiased=False, keepdim=True)
    manual = gamma * (x - mean) / torch.sqrt(variance + 1e-5) + beta
    pytorch = F.batch_norm(
        x,
        running_mean=None,
        running_var=None,
        weight=gamma.flatten(),
        bias=beta.flatten(),
        training=True,
        eps=1e-5,
    )
    torch.testing.assert_close(manual, pytorch, rtol=1e-4, atol=1e-5)


def one_training_step(name: str, model: nn.Module, device: torch.device, seed: int) -> None:
    """对一个架构执行完整训练步，并验证参数真的改变。"""
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(2, 1, 64, 64, generator=generator).to(device)  # 合成一批灰度图。
    targets = torch.randint(0, NUM_CLASSES, (2,), generator=generator).to(device)  # 类别索引 Shape=(N,)。
    model = model.to(device)
    model.train()  # 启用 Dropout，并让 BatchNorm 使用当前批次统计量。
    loss_fn = nn.CrossEntropyLoss()  # 直接接收 logits，不提前做 softmax。
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02)

    first_parameter = next(model.parameters())
    before = first_parameter.detach().clone()  # 保存更新前参数，稍后核对 step 是否生效。
    logits = model(features)  # 前向：所有模型都必须返回 (N,10)。
    assert logits.shape == (2, NUM_CLASSES), f"{name} 输出 Shape 错误: {tuple(logits.shape)}"
    loss = loss_fn(logits, targets)  # 将模型分数与真实类别比较。
    optimizer.zero_grad(set_to_none=True)  # 清除旧梯度，防止跨批次累加。
    loss.backward()  # 反向传播，把梯度写入每个参数的 .grad。
    assert first_parameter.grad is not None  # 确认梯度链路到达最前面的参数。
    optimizer.step()  # 沿负梯度方向更新参数。
    assert not torch.equal(before, first_parameter.detach()), f"{name} 参数没有更新"

    model.eval()  # 评估态关闭 Dropout，BN 改用运行统计量。
    with torch.inference_mode():  # 推理不记录梯度，节省内存。
        evaluated = model(features)
    assert torch.isfinite(evaluated).all(), f"{name} 推理出现 NaN/Inf"
    print(
        f"{name:9s} | params={parameter_count(model):>8,d} | "
        f"output={tuple(logits.shape)} | loss={loss.item():.4f} | 更新通过"
    )


def main(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    requested = args.device
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    check_batch_norm()
    print("手算 BatchNorm 与 PyTorch 对照通过。")

    names = list(MODEL_FACTORIES) if args.model == "all" else [args.model]
    for index, name in enumerate(names):
        model = MODEL_FACTORIES[name]()  # 每次重新创建模型，避免不同架构共享状态。
        one_training_step(name, model, device, seed=args.seed + index)
    print(f"全部 smoke test 通过，设备={device}。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="现代 CNN 结构图谱与训练步 smoke test")
    parser.add_argument("--model", choices=("all", *MODEL_FACTORIES.keys()), default="all")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
