"""第 16 章：可复现、可恢复、可验证的最小实验模板。

运行：
    python code/ch16/reproducible_experiment.py
    python code/ch16/reproducible_experiment.py --epochs 8 --output-dir runs/demo

默认在临时目录保存 checkpoint 并自动清理；明确传入 --output-dir 时保留文件。
"""

from __future__ import annotations

import argparse
import json
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset, random_split


@dataclass(frozen=True)
class Config:
    """把影响结果的超参数集中保存，避免散落在脚本各处。"""

    seed: int = 42
    epochs: int = 5
    batch_size: int = 64
    learning_rate: float = 0.02
    train_size: int = 768
    valid_size: int = 256


def set_seed(seed: int) -> None:
    """固定本示例使用的 Python、CPU 与 CUDA 随机数。"""
    # 固定 Python 随机库，覆盖可能使用 random 的数据逻辑。
    random.seed(seed)
    # 固定 PyTorch CPU 随机数，覆盖参数和合成数据生成。
    torch.manual_seed(seed)
    # 只有 CUDA 可用时才固定所有 GPU 的随机数流。
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dataset(config: Config) -> TensorDataset:
    """生成一个边界清楚、无需联网的三分类数据集。"""
    # 使用独立生成器，使造数据不改变后续模型初始化随机状态。
    generator = torch.Generator().manual_seed(config.seed)
    # 样本总数由训练集与验证集大小共同决定。
    total_size = config.train_size + config.valid_size
    # 每个样本有四个标准正态特征，Shape 为 (N,4)。
    features = torch.randn(total_size, 4, generator=generator)
    # teacher 权重 Shape 为 (4,3)，代表隐藏的真实三分类规则。
    teacher = torch.tensor(
        [[1.2, -0.7, 0.1], [-0.5, 1.1, 0.3], [0.7, 0.2, -1.0], [0.1, -0.4, 1.2]]
    )
    # 用线性规则产生 logits，并加入少量噪声避免任务过于机械。
    logits = features @ teacher + 0.08 * torch.randn(total_size, 3, generator=generator)
    # 每行最高分位置成为 long 类型类别索引，Shape 为 (N,)。
    labels = logits.argmax(dim=1)
    # TensorDataset 固定第 i 行特征与第 i 个标签的配对关系。
    return TensorDataset(features, labels)


def make_loaders(config: Config) -> tuple[DataLoader, DataLoader]:
    """用固定随机种子切分数据，并创建训练/验证迭代器。"""
    # 生成完整合成数据集。
    dataset = make_dataset(config)
    # 独立切分生成器保证训练/验证索引每次一致。
    split_generator = torch.Generator().manual_seed(config.seed + 1)
    # random_split 按给定大小返回互不重叠的两个子集。
    train_set, valid_set = random_split(
        dataset,
        [config.train_size, config.valid_size],
        generator=split_generator,
    )
    # 独立的 DataLoader 生成器让 shuffle 顺序也能够复现。
    loader_generator = torch.Generator().manual_seed(config.seed + 2)
    # 训练集每个 epoch 打乱，drop_last=False 保留最后一个小批次。
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        generator=loader_generator,
    )
    # 验证集不需要打乱，固定顺序更方便定位错误样本。
    valid_loader = DataLoader(valid_set, batch_size=config.batch_size, shuffle=False)
    # 返回相同 `(features, labels)` 接口的两个迭代器。
    return train_loader, valid_loader


def build_model(device: torch.device) -> nn.Module:
    """创建一个小型 MLP，并立即搬到目标设备。"""
    # Sequential 清晰表达 4维输入 -> 16维隐藏 -> 3类 logits。
    model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 3))
    # 模型参数、输入和标签必须在同一设备。
    return model.to(device)


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """按样本数正确汇总验证准确率。"""
    # eval 改变 Dropout/BatchNorm 等模块行为；本模型仍保留标准模板。
    model.eval()
    # 正确个数和总样本数都从零累加。
    correct, total = 0, 0
    # 遍历全部验证批次。
    for features, labels in loader:
        # 输入和标签搬到模型所在设备。
        features, labels = features.to(device), labels.to(device)
        # 前向得到 Shape 为 (B,3) 的 logits。
        logits = model(features)
        # argmax 选择预测类别，并累计正确样本数。
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        # 使用真实批量大小，兼容最后一批不足 batch_size。
        total += labels.numel()
    # 所有批次结束后只除一次，得到严格样本加权准确率。
    return correct / total


def train(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    device: torch.device,
    config: Config,
) -> tuple[torch.optim.Optimizer, list[dict[str, float]]]:
    """执行标准训练五步，并返回优化器和可序列化历史。"""
    # 交叉熵直接接收三类 logits 与 long 类别索引。
    loss_fn = nn.CrossEntropyLoss()
    # AdamW 同时演示优化器状态和解耦权重衰减。
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    # 历史记录只保存普通 Python 数值，便于 JSON 序列化。
    history: list[dict[str, float]] = []

    # 每个 epoch 完整遍历一次训练集。
    for epoch in range(1, config.epochs + 1):
        # train 切换模型训练行为。
        model.train()
        # 累加批损失总量与样本数，正确处理最后一个小批次。
        loss_sum, sample_count = 0.0, 0
        # DataLoader 按固定但逐轮变化的乱序返回批次。
        for features, labels in train_loader:
            # 数据搬到同一计算设备。
            features, labels = features.to(device), labels.to(device)
            # 1. Forward：输出 (B,3) logits。
            logits = model(features)
            # 2. Loss：得到当前批次平均交叉熵标量。
            loss = loss_fn(logits, labels)
            # 3. zero_grad：清掉旧梯度，设为 None 减少内存写入。
            optimizer.zero_grad(set_to_none=True)
            # 4. backward：沿计算图求出每个参数的梯度。
            loss.backward()
            # 5. step：AdamW 读取梯度与内部动量状态，更新参数。
            optimizer.step()
            # 当前批次真实大小可能小于配置的 batch_size。
            batch_size = labels.numel()
            # 批平均损失乘样本数后再跨批累加。
            loss_sum += loss.item() * batch_size
            # 累加本批真实样本数。
            sample_count += batch_size

        # 每轮结束后在独立验证集上评估泛化能力。
        valid_accuracy = evaluate(model, valid_loader, device)
        # 保存 epoch、样本加权损失和验证准确率。
        metrics = {
            "epoch": float(epoch),
            "train_loss": loss_sum / sample_count,
            "valid_accuracy": valid_accuracy,
        }
        # 追加历史记录，后面写入 checkpoint 和 JSON。
        history.append(metrics)
        # 输出紧凑训练进度。
        print(
            f"epoch={epoch:02d} train_loss={metrics['train_loss']:.4f} "
            f"valid_accuracy={valid_accuracy:.3f}"
        )
    # 返回带内部状态的优化器和完整训练历史。
    return optimizer, history


def save_artifacts(
    output_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: Config,
    history: list[dict[str, float]],
) -> Path:
    """保存恢复训练所需状态，并用 JSON 保存人类可读元数据。"""
    # 创建输出目录；parents=True 允许父目录尚不存在。
    output_dir.mkdir(parents=True, exist_ok=True)
    # checkpoint 使用稳定、描述性的文件名。
    checkpoint_path = output_dir / "checkpoint.pt"
    # state_dict 保存模型参数与优化器动量，不保存不可移植的整个对象。
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(config),
            "history": history,
            "torch_version": torch.__version__,
        },
        checkpoint_path,
    )
    # JSON 元数据可以不用 PyTorch 就直接阅读和做版本比较。
    metadata = {"config": asdict(config), "history": history, "torch_version": torch.__version__}
    # 使用 UTF-8 和缩进写出人类可读实验记录。
    (output_dir / "metrics.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    # 返回 checkpoint 路径供加载验证。
    return checkpoint_path


def verify_checkpoint(checkpoint_path: Path, device: torch.device, valid_loader: DataLoader) -> float:
    """重新创建模型并加载 state_dict，验证文件不是“只保存没测试”。"""
    # weights_only=False 是因为文件包含配置和历史等普通 Python 对象。
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # 重新创建一个参数随机的新模型。
    restored_model = build_model(device)
    # 加载已训练参数；strict=True 默认要求键完全一致。
    restored_model.load_state_dict(checkpoint["model"])
    # 用恢复后的模型重新计算验证准确率。
    restored_accuracy = evaluate(restored_model, valid_loader, device)
    # checkpoint 至少应包含一次训练历史。
    assert checkpoint["history"], "checkpoint 缺少训练历史"
    # 恢复后的指标应与保存前最后一次记录一致到浮点容差。
    expected_accuracy = checkpoint["history"][-1]["valid_accuracy"]
    assert abs(restored_accuracy - expected_accuracy) < 1e-12, "恢复后指标与保存前不一致"
    # 返回恢复模型的准确率用于最终报告。
    return restored_accuracy


def parse_args() -> argparse.Namespace:
    """读取实验配置和可选持久化目录。"""
    # 创建命令行解析器。
    parser = argparse.ArgumentParser(description="可复现 PyTorch 实验模板")
    # 允许从命令行调整最常用的实验参数。
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    # auto 会自动选择 GPU 或 CPU。
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    # 省略时使用自动清理的临时目录，避免一次演示污染仓库。
    parser.add_argument("--output-dir", type=Path)
    # 返回解析结果。
    return parser.parse_args()


def run_experiment(args: argparse.Namespace, output_dir: Path) -> None:
    """把数据、模型、训练、保存和恢复串成一个可审计实验。"""
    # 将命令行参数冻结为配置对象。
    config = Config(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    # 固定随机状态后再创建数据和模型。
    set_seed(config.seed)
    # auto 优先 CUDA，否则使用 CPU。
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    # auto 且无 CUDA 时显式退回 CPU。
    if device_name == "auto":
        device_name = "cpu"
    # 明确请求不可用 CUDA 时快速失败。
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求了 CUDA，但当前环境没有可用 CUDA 设备")
    # 构造实际 torch.device。
    device = torch.device(device_name)
    # 固定索引创建训练和验证 DataLoader。
    train_loader, valid_loader = make_loaders(config)
    # 创建模型并搬到设备。
    model = build_model(device)
    # 训练并获取优化器状态与指标历史。
    optimizer, history = train(model, train_loader, valid_loader, device, config)
    # 保存模型、优化器、配置与历史。
    checkpoint_path = save_artifacts(output_dir, model, optimizer, config, history)
    # 真正重新加载文件并验证恢复结果。
    restored_accuracy = verify_checkpoint(checkpoint_path, device, valid_loader)
    # 合成任务在默认设置下应明显优于三分类随机基线 1/3。
    assert restored_accuracy >= 0.80, "准确率异常，请检查环境或训练链路"
    # 输出最终验收信息和产物位置。
    print(f"checkpoint 恢复验证通过：accuracy={restored_accuracy:.3f}")
    print(f"实验产物目录：{output_dir.resolve()}")


def main() -> None:
    """默认临时运行，也支持用户明确保留实验产物。"""
    # 解析本次运行参数。
    args = parse_args()
    # 用户给出目录时保留 checkpoint 与指标文件。
    if args.output_dir is not None:
        run_experiment(args, args.output_dir)
        return
    # 默认使用 TemporaryDirectory，程序结束后自动清理演示产物。
    with tempfile.TemporaryDirectory(prefix="d2l-ch16-") as temporary_dir:
        # 把临时目录字符串包装成 Path 交给统一实验流程。
        run_experiment(args, Path(temporary_dir))
        # 退出上下文后自动删除临时文件，不污染当前仓库。
        print("未指定 --output-dir，临时实验产物已在退出时清理")


if __name__ == "__main__":
    # 直接运行脚本时执行完整实验；import 时只暴露函数与配置类。
    main()
