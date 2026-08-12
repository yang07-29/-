"""4.10 Kaggle 房价流程的离线缩小版。

直接运行：python code/ch04/house_price_pipeline.py
脚本用合成表格数据演示：预处理、K 折、log-RMSE、全量重训和预测。
不下载数据，也不写文件。
"""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class TableData:
    """数值特征与类别索引已经拆开的轻量表格。"""

    numeric: torch.Tensor      # (N,3)：面积、房龄、卧室数
    city: torch.Tensor         # (N,)：城市类别索引
    price: torch.Tensor | None # (N,1)：测试集可为 None


def make_synthetic_tables() -> tuple[TableData, TableData]:
    """生成训练表和测试表，模拟房价竞赛输入。"""
    torch.manual_seed(31)

    def sample(size: int, with_price: bool) -> TableData:
        # 三个数值列，暂时处于不同量纲。
        area = torch.randint(55, 220, (size,)).float()
        age = torch.randint(0, 45, (size,)).float()
        bedrooms = torch.randint(1, 6, (size,)).float()
        numeric = torch.stack([area, age, bedrooms], dim=1)
        # 三个城市类别：0、1、2。
        city = torch.randint(0, 3, (size,))
        if not with_price:
            return TableData(numeric, city, None)
        # 价格始终为正；城市和面积影响明显。
        city_bonus = torch.tensor([0.0, 35.0, 70.0])[city]
        price = 45.0 + 1.8 * area - 0.7 * age + 9.0 * bedrooms + city_bonus
        price = (price + 8.0 * torch.randn(size)).clamp_min(10.0).reshape(-1, 1)
        return TableData(numeric, city, price)

    return sample(180, with_price=True), sample(30, with_price=False)


class Preprocessor:
    """只在训练折上拟合均值/标准差，避免验证信息泄漏。"""

    def __init__(self, categories: int = 3):
        self.categories = categories
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None

    def fit(self, numeric: torch.Tensor) -> "Preprocessor":
        # mean/std.shape == (3,)，统计量只来自当前训练折。
        self.mean = numeric.mean(dim=0)
        self.std = numeric.std(dim=0).clamp_min(1e-6)
        return self

    def transform(self, numeric: torch.Tensor, city: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise RuntimeError("必须先在训练数据上调用 fit")
        # 标准化后 numeric_scaled.shape == (N,3)。
        numeric_scaled = (numeric - self.mean) / self.std
        # one_hot.shape == (N,3)，训练/验证/测试列空间固定一致。
        one_hot = nn.functional.one_hot(city, num_classes=self.categories).float()
        # 合并后 X.shape == (N,6)。
        return torch.cat([numeric_scaled, one_hot], dim=1)


def log_rmse(model: nn.Module, X: torch.Tensor, price: torch.Tensor) -> float:
    """计算 log-RMSE；模型直接预测 log(price)。"""
    model.eval()
    with torch.inference_mode():
        # prediction_log 与 target_log 都是 (N,1)。
        prediction_log = model(X)
        target_log = price.log()
        return torch.sqrt(nn.functional.mse_loss(prediction_log, target_log)).item()


def train_model(X: torch.Tensor, price: torch.Tensor, epochs: int = 180) -> nn.Linear:
    """训练一个线性基线；真实竞赛可替换为更强模型。"""
    model = nn.Linear(X.shape[1], 1)
    nn.init.xavier_uniform_(model.weight)
    nn.init.zeros_(model.bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=1e-4)
    target_log = price.log()

    for _ in range(epochs):
        model.train()
        # 1. Forward：X(N,6) -> prediction_log(N,1)。
        prediction_log = model(X)
        # 2. Loss：在 log 空间优化 MSE，与 log-RMSE 排序一致。
        loss = nn.functional.mse_loss(prediction_log, target_log)
        # 3. zero_grad：清理上一轮梯度。
        optimizer.zero_grad(set_to_none=True)
        # 4. backward：计算线性层权重与偏置梯度。
        loss.backward()
        # 5. step：AdamW 更新参数并解耦权重衰减。
        optimizer.step()
    return model


def k_fold(table: TableData, folds: int = 5) -> tuple[float, float]:
    """每折重新拟合预处理器和模型，返回均值与标准差。"""
    if table.price is None:
        raise ValueError("K 折需要标签")
    indices = torch.randperm(len(table.numeric))
    fold_indices = torch.tensor_split(indices, folds)
    scores: list[float] = []

    for fold in range(folds):
        valid_index = fold_indices[fold]
        train_index = torch.cat(
            [fold_indices[i] for i in range(folds) if i != fold]
        )
        # 每折只在该折训练子集 fit，验证子集不能参与均值/标准差估计。
        processor = Preprocessor().fit(table.numeric[train_index])
        train_X = processor.transform(table.numeric[train_index], table.city[train_index])
        valid_X = processor.transform(table.numeric[valid_index], table.city[valid_index])
        # 每折都创建新模型，不能沿用上一折权重。
        model = train_model(train_X, table.price[train_index])
        score = log_rmse(model, valid_X, table.price[valid_index])
        scores.append(score)
        print(f"fold={fold + 1} log-RMSE={score:.4f}")

    score_tensor = torch.tensor(scores)
    return score_tensor.mean().item(), score_tensor.std(unbiased=False).item()


def main() -> None:
    train_table, test_table = make_synthetic_tables()
    mean_score, std_score = k_fold(train_table)
    print(f"5 折 log-RMSE：{mean_score:.4f} ± {std_score:.4f}")

    # 超参数确定后，在全部训练数据上重新 fit 预处理器和模型。
    processor = Preprocessor().fit(train_table.numeric)
    full_train_X = processor.transform(train_table.numeric, train_table.city)
    test_X = processor.transform(test_table.numeric, test_table.city)
    assert train_table.price is not None
    final_model = train_model(full_train_X, train_table.price)

    final_model.eval()
    with torch.inference_mode():
        # 模型输出 log(price)，exp 后得到正数房价预测；Shape 为 (N_test,1)。
        test_predictions = final_model(test_X).exp()
    print("测试集前 5 个预测：", test_predictions[:5, 0].round().tolist())
    assert torch.isfinite(test_predictions).all()
    assert (test_predictions > 0).all()


if __name__ == "__main__":
    main()
