"""第 15 章：微型 BERT 先做 MLM，再为 NLI 添加下游头并整体微调。

本程序完全离线。它用合成文本演示“预训练骨干 + 新任务头”的参数流，
并额外打印序列分类、词元标注、抽取式问答三种输出 Shape。

运行：
    python code/ch15/mini_bert_finetuning.py
快速检查：
    python code/ch15/mini_bert_finetuning.py --pretrain-steps 10 --epochs 6
"""

from __future__ import annotations

import argparse
import copy
import random
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


SPECIAL = ("<pad>", "<unk>", "<cls>", "<sep>", "<mask>")
ANIMALS = ("cat", "dog", "bird", "horse")
ACTIONS = ("runs", "sleeps", "jumps", "eats")
COLORS = ("black", "white", "brown", "small")


def build_pair_examples(size_per_class: int, seed: int) -> list[tuple[str, str, int]]:
    """标签 0/1/2 分别表示蕴涵、矛盾、中性。"""
    rng = random.Random(seed)
    items: list[tuple[str, str, int]] = []
    for _ in range(size_per_class):
        animal = rng.choice(ANIMALS)
        action = rng.choice(ACTIONS)
        color = rng.choice(COLORS)
        premise = f"a {color} {animal} {action} outside"
        # 前缀词是刻意加入的“易学关系提示”，让 CPU 冒烟测试在少量 epoch 内稳定收敛；
        # 它也提醒读者：真实数据若有类似提示，模型可能靠捷径而不是真正推理。
        items.append((premise, f"yes the {animal} {action}", 0))
        items.append((premise, f"no the {animal} does not {action}", 1))
        other = rng.choice([candidate for candidate in ACTIONS if candidate != action])
        items.append((premise, f"maybe the {animal} {other}", 2))
    rng.shuffle(items)
    return items


class Vocab:
    def __init__(self, examples: list[tuple[str, str, int]]) -> None:
        words = sorted({word for premise, hypothesis, _ in examples for text in (premise, hypothesis) for word in text.split()})
        self.idx_to_token = list(SPECIAL) + words
        self.token_to_idx = {token: index for index, token in enumerate(self.idx_to_token)}

    def __len__(self) -> int:
        return len(self.idx_to_token)

    def __getitem__(self, token: str) -> int:
        return self.token_to_idx.get(token, self.token_to_idx["<unk>"])

    def encode(self, tokens: list[str]) -> list[int]:
        return [self[token] for token in tokens]


@dataclass
class EncodedPair:
    tokens: torch.Tensor
    segments: torch.Tensor
    valid_mask: torch.Tensor
    label: int


def encode_pair(
    premise: str,
    hypothesis: str,
    label: int,
    vocab: Vocab,
    max_len: int,
) -> EncodedPair:
    """构造 <cls> premise <sep> hypothesis <sep> 并补齐。"""
    premise_tokens = premise.split()
    hypothesis_tokens = hypothesis.split()
    tokens = ["<cls>"] + premise_tokens + ["<sep>"] + hypothesis_tokens + ["<sep>"]
    segments = [0] * (len(premise_tokens) + 2) + [1] * (len(hypothesis_tokens) + 1)
    tokens = tokens[:max_len]
    segments = segments[:max_len]
    valid_length = len(tokens)
    padding = max_len - valid_length
    token_ids = vocab.encode(tokens) + [vocab["<pad>"]] * padding
    segment_ids = segments + [0] * padding
    valid_mask = [True] * valid_length + [False] * padding
    return EncodedPair(
        tokens=torch.tensor(token_ids, dtype=torch.long),
        segments=torch.tensor(segment_ids, dtype=torch.long),
        valid_mask=torch.tensor(valid_mask, dtype=torch.bool),
        label=label,
    )


class PairDataset(Dataset[EncodedPair]):
    def __init__(self, examples: list[tuple[str, str, int]], vocab: Vocab, max_len: int) -> None:
        self.items = [encode_pair(a, b, label, vocab, max_len) for a, b, label in examples]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> EncodedPair:
        return self.items[index]


@dataclass
class PairBatch:
    tokens: torch.Tensor
    segments: torch.Tensor
    valid_mask: torch.Tensor
    labels: torch.Tensor


def collate_pairs(items: list[EncodedPair]) -> PairBatch:
    return PairBatch(
        tokens=torch.stack([item.tokens for item in items]),
        segments=torch.stack([item.segments for item in items]),
        valid_mask=torch.stack([item.valid_mask for item in items]),
        labels=torch.tensor([item.label for item in items], dtype=torch.long),
    )


class TinyBERTEncoder(nn.Module):
    """下游任务共享的双向 Transformer 骨干。"""

    def __init__(self, vocab_size: int, hidden_size: int, num_heads: int, max_len: int) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.segment_embedding = nn.Embedding(2, hidden_size)
        self.position_embedding = nn.Embedding(max_len, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 2,
            dropout=0.1,
            batch_first=True,
            norm_first=False,
        )
        # 关闭原型 nested tensor 路径，避免不同 PyTorch 小版本产生额外警告。
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=2,
            enable_nested_tensor=False,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        segments: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        sequence_length = tokens.shape[1]
        # positions:(S,)；查询位置嵌入后广播到 B 个样本。
        positions = torch.arange(sequence_length, device=tokens.device)
        # 三种嵌入均为 (B,S,H)，相加后隐藏宽度保持 H。
        x = self.token_embedding(tokens)
        x = x + self.segment_embedding(segments)
        x = x + self.position_embedding(positions).unsqueeze(0)
        # Transformer 返回每个位置的上下文表示 encoded:(B,S,H)。
        return self.transformer(x, src_key_padding_mask=~valid_mask)


class MLMPretrainer(nn.Module):
    """临时预训练外壳；训练完成后只保留 encoder。"""

    def __init__(self, encoder: TinyBERTEncoder, hidden_size: int, vocab_size: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.mlm_head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.Linear(hidden_size, vocab_size))

    def forward(self, tokens: torch.Tensor, segments: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(tokens, segments, valid_mask)
        # mlm_logits:(B,S,V)。
        return self.mlm_head(encoded)


class SequenceClassifier(nn.Module):
    """序列级任务：读取 <cls> 表示并输出每个句子对一个标签。"""

    def __init__(self, encoder: TinyBERTEncoder, hidden_size: int, num_classes: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(nn.Dropout(0.1), nn.Linear(hidden_size, num_classes))

    def forward(self, tokens: torch.Tensor, segments: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(tokens, segments, valid_mask)
        # encoded[:,0] 是每个样本 <cls> 的 (B,H) 表示；logits:(B,C)。
        return self.classifier(encoded[:, 0, :])


class TokenTagger(nn.Module):
    """词元级标注头：同一个线性层作用于所有位置。"""

    def __init__(self, hidden_size: int, num_tags: int) -> None:
        super().__init__()
        self.output = nn.Linear(hidden_size, num_tags)

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        # (B,S,H) -> (B,S,num_tags)。
        return self.output(encoded)


class QuestionAnsweringHead(nn.Module):
    """抽取式问答头：每个位置分别得到 start/end 分数。"""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.output = nn.Linear(hidden_size, 2)

    def forward(self, encoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # span_logits:(B,S,2)，最后一维的两列分别是 start/end。
        span_logits = self.output(encoded)
        return span_logits[..., 0], span_logits[..., 1]


def build_mlm_batch(
    encoded_examples: list[EncodedPair],
    vocab: Vocab,
    batch_size: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """从句子对随机抽样，并把每条序列的一个普通词替换为 <mask>。"""
    example_indices = torch.randint(len(encoded_examples), (batch_size,), generator=generator)
    tokens = torch.stack([encoded_examples[int(index)].tokens for index in example_indices]).clone()
    segments = torch.stack([encoded_examples[int(index)].segments for index in example_indices])
    valid_mask = torch.stack([encoded_examples[int(index)].valid_mask for index in example_indices])
    # labels:(B,S)，仅被选位置存原词，其余 -100 不参与交叉熵。
    labels = torch.full_like(tokens, -100)
    for row in range(batch_size):
        candidates = torch.nonzero(
            valid_mask[row]
            & (tokens[row] != vocab["<cls>"])
            & (tokens[row] != vocab["<sep>"]),
            as_tuple=False,
        ).squeeze(1)
        choice = int(torch.randint(len(candidates), (1,), generator=generator))
        position = int(candidates[choice])
        labels[row, position] = tokens[row, position]
        tokens[row, position] = vocab["<mask>"]
    return tokens, segments, valid_mask, labels


def pretrain_encoder(
    encoder: TinyBERTEncoder,
    examples: list[EncodedPair],
    vocab: Vocab,
    hidden_size: int,
    steps: int,
    seed: int,
) -> None:
    """用 MLM 做很短的领域预训练；演示权重来源，不追求大模型质量。"""
    model = MLMPretrainer(encoder, hidden_size, len(vocab))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.004)
    generator = torch.Generator().manual_seed(seed)
    first_loss: float | None = None
    for step in range(1, steps + 1):
        tokens, segments, valid_mask, labels = build_mlm_batch(examples, vocab, 32, generator)
        # Forward：得到所有位置的词表分数 (B,S,V)。
        logits = model(tokens, segments, valid_mask)
        # Loss：只在每条序列被遮住的一个位置计算。
        loss = F.cross_entropy(logits.reshape(-1, len(vocab)), labels.reshape(-1), ignore_index=-100)
        # 清梯度，避免与上一步相加。
        optimizer.zero_grad(set_to_none=True)
        # 反向传播会同时训练 MLM 头和共享编码器。
        loss.backward()
        # 预训练参数更新。
        optimizer.step()
        if first_loss is None:
            first_loss = float(loss.detach())
        if step == 1 or step == steps or step % max(1, steps // 3) == 0:
            print(f"pretrain step={step:03d}, mlm_loss={float(loss.detach()):.4f}")
    assert first_loss is not None


@torch.no_grad()
def evaluate(model: SequenceClassifier, loader: DataLoader[PairBatch]) -> float:
    model.eval()
    correct = total = 0
    for batch in loader:
        logits = model(batch.tokens, batch.segments, batch.valid_mask)
        correct += int((logits.argmax(dim=1) == batch.labels).sum())
        total += batch.labels.numel()
    return correct / total


def finetune(
    encoder: TinyBERTEncoder,
    train_loader: DataLoader[PairBatch],
    test_loader: DataLoader[PairBatch],
    hidden_size: int,
    epochs: int,
) -> SequenceClassifier:
    model = SequenceClassifier(encoder, hidden_size, num_classes=3)
    # 骨干用较小学习率防止快速忘掉预训练；新分类头可用较大学习率。
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": 0.001},
            {"params": model.classifier.parameters(), "lr": 0.006},
        ],
        weight_decay=0.01,
    )
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            # Forward：<cls> 表示经新任务头得到 (B,3)。
            logits = model(batch.tokens, batch.segments, batch.valid_mask)
            # Loss：监督标签来自 NLI，而非预训练文本自身。
            loss = F.cross_entropy(logits, batch.labels)
            # Zero grad：同时清理骨干和分类头梯度。
            optimizer.zero_grad(set_to_none=True)
            # Backward：梯度不仅训练新头，也微调整个 BERT 编码器。
            loss.backward()
            # Update：两个参数组按不同学习率变化。
            optimizer.step()
        accuracy = evaluate(model, test_loader)
        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 4) == 0:
            print(f"finetune epoch={epoch:02d}, loss={float(loss.detach()):.4f}, test_accuracy={accuracy:.3f}")
    assert accuracy >= 0.80, "微调准确率过低，请检查 segment、<cls> 或标签"
    return model


@torch.no_grad()
def show_head_shapes(model: SequenceClassifier, batch: PairBatch, hidden_size: int) -> None:
    """同一份 BERT 表示接不同小头，即可适配不同输出粒度。"""
    model.eval()
    encoded = model.encoder(batch.tokens, batch.segments, batch.valid_mask)
    sequence_logits = model.classifier(encoded[:, 0])
    tag_logits = TokenTagger(hidden_size, num_tags=5)(encoded)
    start_logits, end_logits = QuestionAnsweringHead(hidden_size)(encoded)
    print("序列分类 Shape：", tuple(sequence_logits.shape))
    print("词元标注 Shape：", tuple(tag_logits.shape))
    print("问答起点/终点 Shape：", tuple(start_logits.shape), tuple(end_logits.shape))


def main(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    examples = build_pair_examples(args.size_per_class, args.seed)
    split = int(len(examples) * 0.8)
    train_examples, test_examples = examples[:split], examples[split:]
    vocab = Vocab(train_examples)
    train_set = PairDataset(train_examples, vocab, args.max_len)
    test_set = PairDataset(test_examples, vocab, args.max_len)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_pairs)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_pairs)

    encoder = TinyBERTEncoder(len(vocab), args.hidden_size, num_heads=4, max_len=args.max_len)
    # 保存初始化权重只用于证明预训练确实改过骨干参数。
    before = copy.deepcopy(encoder.token_embedding.weight.detach())
    pretrain_encoder(encoder, train_set.items, vocab, args.hidden_size, args.pretrain_steps, args.seed)
    change = (encoder.token_embedding.weight.detach() - before).abs().mean()
    print(f"预训练后 token embedding 平均绝对变化={float(change):.6f}")
    model = finetune(encoder, train_loader, test_loader, args.hidden_size, args.epochs)
    show_head_shapes(model, next(iter(test_loader)), args.hidden_size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="离线微型 BERT 预训练与 NLI 微调")
    parser.add_argument("--pretrain-steps", type=int, default=45)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--size-per-class", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--max-len", type=int, default=16)
    parser.add_argument("--seed", type=int, default=37)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
