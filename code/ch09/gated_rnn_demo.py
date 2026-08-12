"""第 9 章：GRU、LSTM、深层与双向循环网络的可运行对照。

本程序不下载数据。它先手算一个 GRU/LSTM 时间步，观察门值与 Shape；
再用四种 PyTorch 循环网络完成一个小型序列分类任务。

运行：
    python code/ch09/gated_rnn_demo.py --epochs 30
快速检查：
    python code/ch09/gated_rnn_demo.py --epochs 2
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


def new_weight(input_size: int, output_size: int) -> torch.Tensor:
    """创建演示用权重；除以输入维度平方根以控制初始尺度。"""
    return torch.randn(input_size, output_size) / input_size**0.5


@dataclass
class GRUParameters:
    W_xz: torch.Tensor
    W_hz: torch.Tensor
    b_z: torch.Tensor
    W_xr: torch.Tensor
    W_hr: torch.Tensor
    b_r: torch.Tensor
    W_xh: torch.Tensor
    W_hh: torch.Tensor
    b_h: torch.Tensor


def make_gru_parameters(input_size: int, hidden_size: int) -> GRUParameters:
    return GRUParameters(
        new_weight(input_size, hidden_size),
        new_weight(hidden_size, hidden_size),
        torch.zeros(hidden_size),
        new_weight(input_size, hidden_size),
        new_weight(hidden_size, hidden_size),
        torch.zeros(hidden_size),
        new_weight(input_size, hidden_size),
        new_weight(hidden_size, hidden_size),
        torch.zeros(hidden_size),
    )


def gru_step(
    X: torch.Tensor,
    H_previous: torch.Tensor,
    p: GRUParameters,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """手写一个 GRU 时间步，返回新状态、更新门和重置门。"""
    # 更新门 Z:(B,H)：越接近 1，越倾向保留旧状态。
    Z = torch.sigmoid(X @ p.W_xz + H_previous @ p.W_hz + p.b_z)
    # 重置门 R:(B,H)：决定计算候选状态时读取多少旧记忆。
    R = torch.sigmoid(X @ p.W_xr + H_previous @ p.W_hr + p.b_r)
    # 候选状态先用 R 过滤旧状态，再与当前输入共同生成。
    H_tilde = torch.tanh(X @ p.W_xh + (R * H_previous) @ p.W_hh + p.b_h)
    # 按元素插值：Z 控制旧状态与候选状态各占多少。
    H = Z * H_previous + (1.0 - Z) * H_tilde
    return H, Z, R


@dataclass
class LSTMParameters:
    W_xi: torch.Tensor
    W_hi: torch.Tensor
    b_i: torch.Tensor
    W_xf: torch.Tensor
    W_hf: torch.Tensor
    b_f: torch.Tensor
    W_xo: torch.Tensor
    W_ho: torch.Tensor
    b_o: torch.Tensor
    W_xc: torch.Tensor
    W_hc: torch.Tensor
    b_c: torch.Tensor


def make_lstm_parameters(input_size: int, hidden_size: int) -> LSTMParameters:
    return LSTMParameters(
        new_weight(input_size, hidden_size),
        new_weight(hidden_size, hidden_size),
        torch.zeros(hidden_size),
        new_weight(input_size, hidden_size),
        new_weight(hidden_size, hidden_size),
        torch.ones(hidden_size),  # 正遗忘门偏置让初始模型更愿意保留记忆。
        new_weight(input_size, hidden_size),
        new_weight(hidden_size, hidden_size),
        torch.zeros(hidden_size),
        new_weight(input_size, hidden_size),
        new_weight(hidden_size, hidden_size),
        torch.zeros(hidden_size),
    )


def lstm_step(
    X: torch.Tensor,
    H_previous: torch.Tensor,
    C_previous: torch.Tensor,
    p: LSTMParameters,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """手写一个 LSTM 时间步，显式展示三扇门与记忆元。"""
    # 输入门 I 决定候选记忆有多少写入 C。
    I = torch.sigmoid(X @ p.W_xi + H_previous @ p.W_hi + p.b_i)
    # 遗忘门 F 决定旧记忆 C_previous 有多少继续保留。
    F_gate = torch.sigmoid(X @ p.W_xf + H_previous @ p.W_hf + p.b_f)
    # 输出门 O 决定记忆元有多少暴露为隐状态 H。
    O = torch.sigmoid(X @ p.W_xo + H_previous @ p.W_ho + p.b_o)
    # 候选记忆 C_tilde 的值域由 tanh 压到 [-1,1]。
    C_tilde = torch.tanh(X @ p.W_xc + H_previous @ p.W_hc + p.b_c)
    # 加法记忆通道是 LSTM 缓解长程梯度问题的关键。
    C = F_gate * C_previous + I * C_tilde
    # 隐状态是“经过输出门筛选后的记忆”，而不是记忆元本身。
    H = O * torch.tanh(C)
    return H, C, (I, F_gate, O)


class SequenceClassifier(nn.Module):
    """统一封装 GRU/LSTM、层数和方向数，方便观察 Shape。"""

    def __init__(
        self,
        kind: str,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        recurrent_class = nn.GRU if kind == "gru" else nn.LSTM
        self.kind = kind
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.recurrent = recurrent_class(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            # Dropout 只加在相邻循环层之间，因此单层时必须为 0。
            dropout=0.15 if num_layers > 1 else 0.0,
        )
        directions = 2 if bidirectional else 1
        self.classifier = nn.Linear(hidden_size * directions, 2)

    def forward(
        self,
        X: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
        # X 的 Shape 是 (B,T,D)，因为创建循环层时设置了 batch_first=True。
        outputs, state = self.recurrent(X)
        # LSTM state=(H,C)，GRU state=H；分类时都取最终隐藏状态 H。
        hidden = state[0] if self.kind == "lstm" else state

        if self.bidirectional:
            # hidden[-2] 是最后一层正向最终状态，hidden[-1] 是反向最终状态。
            representation = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            # 单向模型取最后一层的最终状态，Shape 为 (B,H)。
            representation = hidden[-1]
        # logits 的 Shape 为 (B,2)，直接交给 CrossEntropyLoss。
        logits = self.classifier(representation)
        return logits, outputs, state


def make_classification_data(
    num_examples: int,
    num_steps: int,
    input_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """标签依赖整段序列，模型必须累积多个时间步的信息。"""
    generator = torch.Generator().manual_seed(23)
    # X:(N,T,D)，每个样本是一段 T 步、每步 D 个特征的序列。
    X = torch.randn(num_examples, num_steps, input_size, generator=generator)
    # 前半段和后半段共同决定标签，不能只看最后一个输入点。
    score = X[:, :, 0].mean(dim=1) + 0.6 * X[:, -4:, 1].mean(dim=1)
    y = (score > 0).to(torch.long)
    return X, y


def train_classifier(
    model: SequenceClassifier,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int,
) -> tuple[float, tuple[int, ...], tuple[int, ...]]:
    """训练一个变体并返回准确率和关键 Shape。"""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.015)
    loss_fn = nn.CrossEntropyLoss()
    split = int(0.8 * X.shape[0])
    train_X, test_X = X[:split], X[split:]
    train_y, test_y = y[:split], y[split:]

    for _ in range(epochs):
        model.train()
        # Forward：循环层先处理整个序列，分类头再读最终状态。
        logits, _, _ = model(train_X)
        # Loss：分类标签必须是 (B,) 的 long 索引。
        loss = loss_fn(logits, train_y)
        optimizer.zero_grad(set_to_none=True)
        # Backward 会自动沿 T 个时间步和 L 个层反向传播。
        loss.backward()
        # clip_grad_norm_ 返回裁剪前范数，并在过大时原地缩放梯度。
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    model.eval()
    with torch.inference_mode():
        logits, outputs, state = model(test_X)
        accuracy = (logits.argmax(dim=1) == test_y).float().mean().item()
        hidden = state[0] if model.kind == "lstm" else state
    return accuracy, tuple(outputs.shape), tuple(hidden.shape)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="门控循环网络 smoke test")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--num-examples", type=int, default=480)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(23)

    batch_size, input_size, hidden_size = 4, 3, 5
    X_step = torch.randn(batch_size, input_size)
    H0 = torch.zeros(batch_size, hidden_size)
    C0 = torch.zeros(batch_size, hidden_size)

    H_gru, update_gate, reset_gate = gru_step(
        X_step, H0, make_gru_parameters(input_size, hidden_size)
    )
    H_lstm, C_lstm, gates = lstm_step(
        X_step, H0, C0, make_lstm_parameters(input_size, hidden_size)
    )
    assert update_gate.min() >= 0 and update_gate.max() <= 1
    assert all(gate.min() >= 0 and gate.max() <= 1 for gate in gates)
    print("手写 GRU:", f"H={tuple(H_gru.shape)}", f"Z均值={update_gate.mean():.3f}", f"R均值={reset_gate.mean():.3f}")
    print("手写 LSTM:", f"H={tuple(H_lstm.shape)}", f"C={tuple(C_lstm.shape)}")

    X, y = make_classification_data(args.num_examples, num_steps=12, input_size=input_size)
    configurations = {
        "GRU": dict(kind="gru", num_layers=1, bidirectional=False),
        "LSTM": dict(kind="lstm", num_layers=1, bidirectional=False),
        "Deep-GRU": dict(kind="gru", num_layers=2, bidirectional=False),
        "BiLSTM": dict(kind="lstm", num_layers=1, bidirectional=True),
    }
    for name, config in configurations.items():
        model = SequenceClassifier(
            input_size=input_size,
            hidden_size=hidden_size,
            **config,
        )
        accuracy, output_shape, hidden_shape = train_classifier(model, X, y, args.epochs)
        print(
            f"{name:8s} accuracy={accuracy:.3f}, "
            f"outputs={output_shape}, hidden={hidden_shape}"
        )
        assert 0.0 <= accuracy <= 1.0

    print("提醒：BiLSTM 使用了未来位置，只适合完整序列已知的任务，不能直接用于因果生成。")


if __name__ == "__main__":
    main()
