"""第 8 章：字符级 RNN 语言模型——从零实现与 PyTorch 简洁实现。

程序内置一小段英文字符语料，不需要联网或下载数据。默认依次训练：
1. 手写循环计算的 ScratchRNNLM；
2. 使用 nn.RNN 的 ConciseRNNLM。

运行：
    python code/ch08/char_rnn_language_model.py --implementation both --epochs 40
快速检查：
    python code/ch08/char_rnn_language_model.py --implementation both --epochs 2
"""

from __future__ import annotations

import argparse
import math
import random
import time
from collections.abc import Iterator

import torch
from torch import nn
from torch.nn import functional as F


TEXT = (
    "time traveller smiled and said time is a river and we are travellers "
    "the machine remembers every little step and predicts the next letter "
    "a recurrent state carries yesterday into today and then moves forward "
)


class CharVocab:
    """字符与整数索引之间的双向映射。"""

    def __init__(self, text: str) -> None:
        # 排序让索引映射稳定，便于复现实验和排错。
        self.idx_to_token = sorted(set(text))
        self.token_to_idx = {token: index for index, token in enumerate(self.idx_to_token)}

    def __len__(self) -> int:
        return len(self.idx_to_token)

    def encode(self, text: str) -> list[int]:
        return [self.token_to_idx[character] for character in text]

    def decode(self, indices: list[int]) -> str:
        return "".join(self.idx_to_token[index] for index in indices)


def sequential_batches(
    corpus: list[int],
    batch_size: int,
    num_steps: int,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """相邻小批量沿时间连续，因此可以跨批携带隐状态。"""
    # 丢弃不能整齐放进 batch 行的少量词元。
    usable = ((len(corpus) - 1) // batch_size) * batch_size
    # Xs 与 Ys 只错开一位：当前位置的标签就是下一个字符。
    Xs = torch.tensor(corpus[:usable], dtype=torch.long).reshape(batch_size, -1)
    Ys = torch.tensor(corpus[1 : usable + 1], dtype=torch.long).reshape(batch_size, -1)
    # 每次沿列方向截取 num_steps；X、Y 的 Shape 都是 (B, T)。
    for start in range(0, Xs.shape[1] - num_steps + 1, num_steps):
        yield Xs[:, start : start + num_steps], Ys[:, start : start + num_steps]


def random_batches(
    corpus: list[int],
    batch_size: int,
    num_steps: int,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """随机打乱子序列起点；每批都应重新初始化隐状态。"""
    # 每个 offset 指向一个长度为 num_steps 的输入子序列。
    offsets = list(range(0, len(corpus) - num_steps, num_steps))
    random.shuffle(offsets)
    for start in range(0, len(offsets), batch_size):
        batch_offsets = offsets[start : start + batch_size]
        if len(batch_offsets) < batch_size:
            break
        # 标签窗口相对输入窗口整体右移一个字符。
        X = torch.tensor([corpus[i : i + num_steps] for i in batch_offsets])
        Y = torch.tensor([corpus[i + 1 : i + num_steps + 1] for i in batch_offsets])
        yield X, Y


class ScratchRNNLM(nn.Module):
    """只借助张量运算和 autograd 手写的 Elman RNN 语言模型。"""

    def __init__(self, vocab_size: int, num_hiddens: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.num_hiddens = num_hiddens

        def parameter(*shape: int) -> nn.Parameter:
            # 小随机初始化防止所有隐藏单元从完全相同的状态开始学习。
            return nn.Parameter(torch.randn(*shape) * 0.01)

        # W_xh 把 one-hot 字符从 V 维投影到 H 维。
        self.W_xh = parameter(vocab_size, num_hiddens)
        # W_hh 把上一个隐状态从 H 维传到当前 H 维。
        self.W_hh = parameter(num_hiddens, num_hiddens)
        self.b_h = nn.Parameter(torch.zeros(num_hiddens))
        # W_hq 把当前隐状态映射为 V 个“下一字符”logits。
        self.W_hq = parameter(num_hiddens, vocab_size)
        self.b_q = nn.Parameter(torch.zeros(vocab_size))

    def begin_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        # 从零实现使用 (B, H)；它代表每条序列各自的一份记忆。
        return torch.zeros(batch_size, self.num_hiddens, device=device)

    def forward(
        self,
        inputs: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # inputs 原 Shape 为 (B, T)，值是字符索引。
        # 转置后 one_hot 的 Shape 为 (T, B, V)，方便按时间步循环。
        one_hot = F.one_hot(inputs.T, self.vocab_size).to(torch.float32)
        outputs: list[torch.Tensor] = []
        hidden = state
        for X_t in one_hot:
            # X_t:(B,V)，hidden:(B,H)，当前隐状态仍为 (B,H)。
            hidden = torch.tanh(X_t @ self.W_xh + hidden @ self.W_hh + self.b_h)
            # 当前时间步为每个样本输出 V 个 logits，Shape 为 (B,V)。
            logits_t = hidden @ self.W_hq + self.b_q
            outputs.append(logits_t)
        # 时间优先拼接为 (T*B,V)，与 Y.T.reshape(-1) 完全同序。
        logits = torch.cat(outputs, dim=0)
        return logits, hidden


class ConciseRNNLM(nn.Module):
    """使用 nn.RNN 完成循环层，显式保留语言模型输出层。"""

    def __init__(self, vocab_size: int, num_hiddens: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.num_hiddens = num_hiddens
        # batch_first=False，因此循环层期待 (T,B,V)。
        self.rnn = nn.RNN(vocab_size, num_hiddens)
        self.output = nn.Linear(num_hiddens, vocab_size)

    def begin_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        # nn.RNN 的状态多一维“层数×方向数”，此处为 (1,B,H)。
        return torch.zeros(1, batch_size, self.num_hiddens, device=device)

    def forward(
        self,
        inputs: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 字符索引 (B,T) -> one-hot (T,B,V)。
        one_hot = F.one_hot(inputs.T, self.vocab_size).to(torch.float32)
        # outputs:(T,B,H)，new_state:(1,B,H)。
        outputs, new_state = self.rnn(one_hot, state)
        # Linear 只看最后一维；先展平为 (T*B,H)，再得到 (T*B,V)。
        logits = self.output(outputs.reshape(-1, self.num_hiddens))
        return logits, new_state


def detach_state(state: torch.Tensor) -> torch.Tensor:
    """切断跨小批量计算图，但保留状态数值作为下一批的记忆。"""
    return state.detach()


def grad_clipping(model: nn.Module, theta: float) -> float:
    """按全局 L2 范数缩放梯度方向，返回裁剪前范数。"""
    parameters = [p for p in model.parameters() if p.grad is not None]
    # 所有参数梯度平方和再开方，得到一个全局标量范数。
    norm = torch.sqrt(sum(torch.sum(p.grad.detach() ** 2) for p in parameters))
    if norm > theta:
        # 所有梯度乘同一比例，所以只缩短向量，不改变整体方向。
        scale = theta / (norm + 1e-12)
        for parameter in parameters:
            parameter.grad.mul_(scale)
    return float(norm)


def train_epoch(
    model: nn.Module,
    corpus: list[int],
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    batch_size: int,
    num_steps: int,
    random_sampling: bool,
) -> tuple[float, float, float]:
    """训练一轮，返回困惑度、每秒词元数和最大裁剪前梯度范数。"""
    model.train()
    iterator_fn = random_batches if random_sampling else sequential_batches
    state: torch.Tensor | None = None
    total_loss = 0.0
    total_tokens = 0
    max_grad_norm = 0.0
    start_time = time.perf_counter()

    for X, Y in iterator_fn(corpus, batch_size, num_steps):
        if state is None or random_sampling:
            # 随机批彼此不连续，不能把上一批的记忆误接到这一批。
            state = model.begin_state(X.shape[0], X.device)  # type: ignore[attr-defined]
        else:
            # 顺序批在数值上延续状态，但必须截断旧计算图，形成 truncated BPTT。
            state = detach_state(state)

        # 标签 Y:(B,T) 转为时间优先的一维 (T*B,)，与 logits 的拼接顺序一致。
        targets = Y.T.reshape(-1)
        # Forward：logits:(T*B,V)，state 保存最后一个时间步的记忆。
        logits, state = model(X, state)
        # CrossEntropyLoss 接收原始 logits 与 long 类型类别索引。
        loss = loss_fn(logits, targets)

        optimizer.zero_grad(set_to_none=True)
        # BPTT 会沿展开的 T 个时间步把梯度传回共享参数。
        loss.backward()
        # 在 step 前裁剪，防止爆炸梯度把参数一步推飞。
        grad_norm = grad_clipping(model, theta=1.0)
        max_grad_norm = max(max_grad_norm, grad_norm)
        optimizer.step()

        total_loss += loss.item() * targets.numel()
        total_tokens += targets.numel()

    mean_loss = total_loss / total_tokens
    # 困惑度是平均交叉熵的指数；1 表示每次都完全确定地猜对。
    perplexity = math.exp(min(mean_loss, 20.0))
    speed = total_tokens / max(time.perf_counter() - start_time, 1e-9)
    return perplexity, speed, max_grad_norm


@torch.inference_mode()
def predict(
    prefix: str,
    num_predictions: int,
    model: nn.Module,
    vocab: CharVocab,
) -> str:
    """先用 prefix 预热隐状态，再自回归生成后续字符。"""
    model.eval()
    state = model.begin_state(batch_size=1, device=torch.device("cpu"))  # type: ignore[attr-defined]
    outputs = [vocab.token_to_idx[prefix[0]]]

    for character in prefix[1:]:
        # 送入前一个已知字符，只更新状态；此时预测结果先不采用。
        X = torch.tensor([[outputs[-1]]], dtype=torch.long)
        _, state = model(X, state)
        outputs.append(vocab.token_to_idx[character])

    for _ in range(num_predictions):
        # 推理阶段把上一步输出当成下一步输入，这就是自回归生成。
        X = torch.tensor([[outputs[-1]]], dtype=torch.long)
        logits, state = model(X, state)
        # logits 为 (1,V)，直接 argmax；类别排序不需要显式 softmax。
        outputs.append(int(logits[-1].argmax()))
    return vocab.decode(outputs)


def build_model(name: str, vocab_size: int, num_hiddens: int) -> nn.Module:
    if name == "scratch":
        return ScratchRNNLM(vocab_size, num_hiddens)
    if name == "concise":
        return ConciseRNNLM(vocab_size, num_hiddens)
    raise ValueError(f"未知实现：{name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="字符级 RNN 语言模型")
    parser.add_argument("--implementation", choices=("scratch", "concise", "both"), default="both")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--num-hiddens", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.5)
    parser.add_argument("--random-sampling", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(11)
    random.seed(11)

    # 重复小语料只是为了让 smoke test 有足够训练样本，不代表真实语料质量。
    text = TEXT * 35
    vocab = CharVocab(text)
    corpus = vocab.encode(text)
    names = ("scratch", "concise") if args.implementation == "both" else (args.implementation,)

    first_X, first_Y = next(sequential_batches(corpus, args.batch_size, args.num_steps))
    print(f"vocab={len(vocab)}, corpus_tokens={len(corpus)}")
    print(f"X.shape={tuple(first_X.shape)}, Y.shape={tuple(first_Y.shape)}")
    print("第一行标签是否等于输入右移一位:", bool(torch.equal(first_X[0, 1:], first_Y[0, :-1])))

    for name in names:
        print(f"\n--- implementation={name} ---")
        model = build_model(name, len(vocab), args.num_hiddens)
        # 中等学习率配合全局梯度裁剪，兼顾从零版和框架版的稳定性。
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(1, args.epochs + 1):
            perplexity, speed, max_norm = train_epoch(
                model,
                corpus,
                optimizer,
                loss_fn,
                args.batch_size,
                args.num_steps,
                args.random_sampling,
            )
            if epoch == 1 or epoch % max(1, args.epochs // 4) == 0:
                sample = predict("time ", 35, model, vocab)
                print(
                    f"epoch={epoch:03d}, ppl={perplexity:.3f}, "
                    f"tokens/s={speed:.0f}, max_grad_norm={max_norm:.3f}"
                )
                print("sample:", sample)

        # 最低限度的 smoke test：输出能生成且所有参数保持有限。
        generated = predict("time ", 20, model, vocab)
        assert len(generated) == len("time ") + 20
        assert all(torch.isfinite(parameter).all() for parameter in model.parameters())


if __name__ == "__main__":
    main()
