"""第 15 章：同一份离线情感数据训练 BiRNN 与 textCNN。

完整链路：合成评论 -> 词表/补齐 -> 两种序列编码器 -> 交叉熵 -> 更新 -> 评估。

运行：
    python code/ch15/sentiment_models.py
快速检查：
    python code/ch15/sentiment_models.py --epochs 5
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset


POSITIVE = ("great", "wonderful", "excellent", "enjoyable", "amazing")
NEGATIVE = ("bad", "awful", "boring", "terrible", "disappointing")
NOUNS = ("movie", "story", "acting", "music", "ending")
TEMPLATES = (
    "this {noun} is {sentiment}",
    "i think the {noun} feels {sentiment}",
    "the {noun} was really {sentiment}",
    "what a {sentiment} {noun}",
)


def make_examples(size: int, seed: int) -> list[tuple[str, int]]:
    """构造平衡二分类评论；标签 1=积极，0=消极。"""
    rng = random.Random(seed)
    examples: list[tuple[str, int]] = []
    for index in range(size):
        label = index % 2
        sentiment = rng.choice(POSITIVE if label == 1 else NEGATIVE)
        sentence = rng.choice(TEMPLATES).format(noun=rng.choice(NOUNS), sentiment=sentiment)
        examples.append((sentence, label))
    rng.shuffle(examples)
    return examples


class Vocab:
    def __init__(self, examples: list[tuple[str, int]]) -> None:
        tokens = sorted({token for sentence, _ in examples for token in sentence.split()})
        self.idx_to_token = ["<pad>", "<unk>"] + tokens
        self.token_to_idx = {token: index for index, token in enumerate(self.idx_to_token)}

    def __len__(self) -> int:
        return len(self.idx_to_token)

    def encode(self, sentence: str) -> list[int]:
        unknown = self.token_to_idx["<unk>"]
        return [self.token_to_idx.get(token, unknown) for token in sentence.lower().split()]


class SentimentDataset(Dataset[tuple[list[int], int]]):
    def __init__(self, examples: list[tuple[str, int]], vocab: Vocab) -> None:
        self.items = [(vocab.encode(sentence), label) for sentence, label in examples]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[list[int], int]:
        return self.items[index]


@dataclass
class Batch:
    tokens: torch.Tensor
    lengths: torch.Tensor
    labels: torch.Tensor


def collate(examples: list[tuple[list[int], int]]) -> Batch:
    """在当前批量内补齐；lengths 保留补齐前长度。"""
    max_len = max(len(tokens) for tokens, _ in examples)
    rows = [tokens + [0] * (max_len - len(tokens)) for tokens, _ in examples]
    # tokens:(B,T)，dtype 必须为 long，才能查嵌入表。
    tokens = torch.tensor(rows, dtype=torch.long)
    # lengths:(B,)，BiRNN 用它跳过 padding。
    lengths = torch.tensor([len(tokens_) for tokens_, _ in examples], dtype=torch.long)
    # labels:(B,)，CrossEntropyLoss 需要类别索引而非 one-hot。
    labels = torch.tensor([label for _, label in examples], dtype=torch.long)
    return Batch(tokens, lengths, labels)


class BiRNNClassifier(nn.Module):
    """双向 LSTM：连接最后一层正向/反向最终状态做分类。"""

    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.encoder = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_size * 2, 2)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # tokens:(B,T) -> embedded:(B,T,E)。
        embedded = self.embedding(tokens)
        # pack 让循环网络不把右侧 <pad> 当成真实词元。
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        # h_n:(2,B,H)：第 0/1 行分别是正向与反向最终隐状态。
        _, (h_n, _) = self.encoder(packed)
        # representation:(B,2H)，把两个方向看到的信息连接起来。
        representation = torch.cat((h_n[-2], h_n[-1]), dim=1)
        # logits:(B,2)，不要在交叉熵之前手动 softmax。
        return self.classifier(representation)


class TextCNNClassifier(nn.Module):
    """多尺度一维卷积 + 最大时间汇聚的 textCNN。"""

    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        channels: int,
        kernel_sizes: tuple[int, ...] = (2, 3, 4),
    ) -> None:
        super().__init__()
        # 可训练嵌入学习下游任务特征。
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        # 常量嵌入模拟“预训练后冻结”的通道。
        self.constant_embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.constant_embedding.weight.requires_grad_(False)
        # 每个卷积核宽度负责寻找一种 n-gram 尺度。
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_size * 2, channels, kernel_size) for kernel_size in kernel_sizes]
        )
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(channels * len(kernel_sizes), 2)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        del lengths  # textCNN 通过全局汇聚得到固定长度，本例无需循环有效长度。
        # 两张嵌入表各输出 (B,T,E)，在特征维拼成 (B,T,2E)。
        embedded = torch.cat((self.embedding(tokens), self.constant_embedding(tokens)), dim=2)
        # Conv1d 约定输入 (B,C,T)，所以把特征维换到通道位置。
        channels_first = embedded.transpose(1, 2)
        pooled_features: list[torch.Tensor] = []
        for conv in self.convs:
            # conv 输出 (B,C,T-K+1)，ReLU 保留正模式响应。
            feature_map = F.relu(conv(channels_first))
            # 沿时间维取最大值，得到“这种模式是否在任意位置出现” (B,C)。
            pooled = feature_map.max(dim=2).values
            pooled_features.append(pooled)
        # 多尺度特征连接为 (B,C*len(K))。
        representation = torch.cat(pooled_features, dim=1)
        # dropout 只在训练态随机失活，输出 logits:(B,2)。
        return self.classifier(self.dropout(representation))


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader[Batch]) -> tuple[float, float]:
    """评估不建图、不更新参数，返回平均损失与准确率。"""
    model.eval()
    loss_sum = 0.0
    correct = 0
    count = 0
    for batch in loader:
        logits = model(batch.tokens, batch.lengths)
        loss_sum += float(F.cross_entropy(logits, batch.labels, reduction="sum"))
        correct += int((logits.argmax(dim=1) == batch.labels).sum())
        count += batch.labels.numel()
    return loss_sum / count, correct / count


def train_model(
    name: str,
    model: nn.Module,
    train_loader: DataLoader[Batch],
    test_loader: DataLoader[Batch],
    epochs: int,
    lr: float,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    first_loss: float | None = None
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            # Forward：把变长文本批量转成 (B,2) 分类分数。
            logits = model(batch.tokens, batch.lengths)
            # Loss：比较 logits 与 (B,) 标签，建立标量计算图。
            loss = F.cross_entropy(logits, batch.labels)
            # Zero grad：避免梯度跨批量累加。
            optimizer.zero_grad(set_to_none=True)
            # Backward：梯度流过分类头与编码器；冻结嵌入不会得到梯度。
            loss.backward()
            # Update：改变所有 requires_grad=True 的参数。
            optimizer.step()
        test_loss, accuracy = evaluate(model, test_loader)
        if first_loss is None:
            first_loss = test_loss
        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 3) == 0:
            print(f"{name:7s} epoch={epoch:02d}, test_loss={test_loss:.4f}, accuracy={accuracy:.3f}")
    assert accuracy >= 0.85, f"{name} 未学会小数据，请检查预处理或梯度"


def main(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    examples = make_examples(args.examples, args.seed)
    split = int(len(examples) * 0.8)
    train_examples, test_examples = examples[:split], examples[split:]
    # 词表只用训练集建立，测试集未知词会落到 <unk>。
    vocab = Vocab(train_examples)
    train_loader = DataLoader(
        SentimentDataset(train_examples, vocab),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        SentimentDataset(test_examples, vocab),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    birnn = BiRNNClassifier(len(vocab), embed_size=24, hidden_size=24)
    textcnn = TextCNNClassifier(len(vocab), embed_size=16, channels=16)
    train_model("BiRNN", birnn, train_loader, test_loader, args.epochs, args.lr)
    train_model("textCNN", textcnn, train_loader, test_loader, args.epochs, args.lr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="离线 BiRNN 与 textCNN 情感分类")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--examples", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=23)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
