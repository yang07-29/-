"""5.3 LazyLinear 的参数实体化时机与输入契约。

直接运行：python code/ch05/lazy_initialization.py
"""

import torch
from torch import nn
from torch.nn.parameter import UninitializedParameter


def print_parameter_state(model: nn.Module, stage: str) -> None:
    """安全打印 Lazy 参数是否已经实体化。"""
    print(f"\n=== {stage} ===")
    for name, parameter in model.named_parameters():
        if isinstance(parameter, UninitializedParameter):
            print(name, "尚未初始化")
        else:
            print(name, tuple(parameter.shape))


def initialize_materialized(module: nn.Module) -> None:
    """只在参数已有真实 Shape 后执行自定义初始化。"""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def main() -> None:
    torch.manual_seed(47)
    model = nn.Sequential(
        # 只声明 out_features=16；in_features 留到首次前向推断。
        nn.LazyLinear(16),
        nn.ReLU(),
        nn.Linear(16, 3),
    )
    print_parameter_state(model, "首次前向之前")
    assert isinstance(model[0].weight, UninitializedParameter)

    # representative_batch.shape == (8,5)，最后一维 5 是输入契约。
    representative_batch = torch.randn(8, 5)
    # dry run：X(8,5) -> H(8,16) -> logits(8,3)。
    dry_output = model(representative_batch)
    print("dry run 输出 Shape：", tuple(dry_output.shape))
    print_parameter_state(model, "首次前向之后")
    assert model[0].weight.shape == (16, 5)

    # 参数实体化后再执行自定义初始化。
    model.apply(initialize_materialized)
    # 初始化原地改变参数；训练前重新前向，建立使用新数值的计算图。
    logits = model(representative_batch)
    labels = torch.randint(0, 3, (8,))
    # 稳妥做法：实体化和自定义初始化完成后，再创建优化器。
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    # 1. Forward 已得到 logits(8,3)。
    # 2. Loss：CrossEntropyLoss 消费 logits 和 labels(8,)。
    loss = nn.functional.cross_entropy(logits, labels)
    # 3. zero_grad：新优化器的梯度为 None，仍保持统一训练模板。
    optimizer.zero_grad(set_to_none=True)
    # 4. backward：计算已实体化参数的梯度。
    loss.backward()
    # 5. step：更新参数。
    optimizer.step()

    # LazyLinear 只推断一次；之后最后一维从 5 变 7 会矩阵乘法失败。
    try:
        model(torch.randn(8, 7))
    except RuntimeError as error:
        print("输入维度改变后的预期错误：", str(error).splitlines()[0])
    else:
        raise AssertionError("输入契约改变后本应报错")


if __name__ == "__main__":
    main()
