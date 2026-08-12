"""第 15 章：用合成 NLI 数据训练可分解注意力模型。

模型严格经过 Attend -> Compare -> Aggregate 三步，并显式遮蔽 padding。

运行：
    python code/ch15/decomposable_attention.py
快速检查：
    python code/ch15/decomposable_attention.py --epochs 6
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


LABELS = {"entailment": 0, "contradiction": 1, "neutral": 2}
ANIMALS = ("cat", "dog", "bird", "horse")
ACTIONS = ("runs", "sleeps", "jumps", "eats")
COLORS = ("black", "white", "brown", "small")


def make_examples(size_per_class: int, seed: int) -> list[tuple[str, str, int]]:
    """构造蕴涵、矛盾、中性三类句子对。"""
    rng = random.Random(seed)
    examples: list[tuple[str, str, int]] = []
    for _ in range(size_per_class):
        animal = rng.choice(ANIMALS)
        action = rng.choice(ACTIONS)
        color = rng.choice(COLORS)
        premise = f"a {color} {animal} {action} outside"
        # 去掉修饰语仍由前提推出，属于蕴涵。
        examples.append((premise, f"the {animal} {action}", LABELS["entailment"]))
        # 显式否定同一事件，属于矛盾。
        examples.append((premise, f"the {animal} does not {action}", LABELS["contradiction"]))
        # 换成另一动作，前提既不能证明也不能否定，属于中性。
        other_actions = [candidate for candidate in ACTIONS if candidate != action]
        examples.append((premise, f"the {animal} {rng.choice(other_actions)}", LABELS["neutral"]))
    rng.shuffle(examples)
    return examples


class Vocab:
    def __init__(self, examples: list[tuple[str, str, int]]) -> None:
        words = sorted({word for premise, hypothesis, _ in examples for text in (premise, hypothesis) for word in text.split()})
        self.idx_to_token = ["<pad>", "<unk>"] + words
        self.token_to_idx = {token: index for index, token in enumerate(self.idx_to_token)}

    def __len__(self) -> int:
        return len(self.idx_to_token)

    def encode(self, text: str) -> list[int]:
        unknown = self.token_to_idx["<unk>"]
        return [self.token_to_idx.get(word, unknown) for word in text.split()]


class NLIDataset(Dataset[tuple[list[int], list[int], int]]):
    def __init__(self, examples: list[tuple[str, str, int]], vocab: Vocab) -> None:
        self.items = [(vocab.encode(a), vocab.encode(b), label) for a, b, label in examples]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[list[int], list[int], int]:
        return self.items[index]


@dataclass
class Batch:
    premise: torch.Tensor
    hypothesis: torch.Tensor
    premise_mask: torch.Tensor
    hypothesis_mask: torch.Tensor
    labels: torch.Tensor


def collate(items: list[tuple[list[int], list[int], int]]) -> Batch:
    """分别补齐前提与假设；二者长度不必相同。"""
    max_premise = max(len(premise) for premise, _, _ in items)
    max_hypothesis = max(len(hypothesis) for _, hypothesis, _ in items)
    premise_rows = [premise + [0] * (max_premise - len(premise)) for premise, _, _ in items]
    hypothesis_rows = [hypothesis + [0] * (max_hypothesis - len(hypothesis)) for _, hypothesis, _ in items]
    premise = torch.tensor(premise_rows, dtype=torch.long)
    hypothesis = torch.tensor(hypothesis_rows, dtype=torch.long)
    # mask:(B,T)，True 表示真实词，False 表示 pad。
    premise_mask = premise != 0
    hypothesis_mask = hypothesis != 0
    labels = torch.tensor([label for _, _, label in items], dtype=torch.long)
    return Batch(premise, hypothesis, premise_mask, hypothesis_mask, labels)


def mlp(input_size: int, hidden_size: int) -> nn.Sequential:
    """各时间步共享同一 MLP，最后一维变换，前两维保持。"""
    return nn.Sequential(
        nn.Dropout(0.1),
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
    )


class DecomposableAttention(nn.Module):
    """对齐、比较、聚合三步可分解注意力。"""

    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        # f 把单词映射到注意力匹配空间。
        self.attend_mlp = mlp(embed_size, hidden_size)
        # g 比较“原词 + 对齐词”，输入维为 2E。
        self.compare_mlp = mlp(embed_size * 2, hidden_size)
        # h 汇总前提和假设两个 H 维向量，输入维为 2H。
        self.aggregate_mlp = mlp(hidden_size * 2, hidden_size)
        self.output = nn.Linear(hidden_size, 3)

    def forward(
        self,
        premise: torch.Tensor,
        hypothesis: torch.Tensor,
        premise_mask: torch.Tensor,
        hypothesis_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # A:(B,M,E)，B:(B,N,E)。
        A = self.embedding(premise)
        B = self.embedding(hypothesis)
        # f_A:(B,M,H)，f_B:(B,N,H)，f 只需计算 M+N 次表示。
        f_A = self.attend_mlp(A)
        f_B = self.attend_mlp(B)
        # scores:(B,M,N)，一次 bmm 得到所有词对的匹配分数。
        scores = torch.bmm(f_A, f_B.transpose(1, 2))

        # 对每个前提词，在假设词维 N 上 softmax；先屏蔽假设 pad。
        scores_for_beta = scores.masked_fill(~hypothesis_mask.unsqueeze(1), -1e9)
        attention_to_B = F.softmax(scores_for_beta, dim=2)
        # beta:(B,M,E)，是与每个前提词软对齐的假设表示。
        beta = torch.bmm(attention_to_B, B)

        # 对每个假设词，在前提词维 M 上 softmax；先屏蔽前提 pad。
        scores_for_alpha = scores.transpose(1, 2).masked_fill(~premise_mask.unsqueeze(1), -1e9)
        attention_to_A = F.softmax(scores_for_alpha, dim=2)
        # alpha:(B,N,E)，是与每个假设词软对齐的前提表示。
        alpha = torch.bmm(attention_to_A, A)

        # 比较原词与软对齐词；输出 V_A:(B,M,H)、V_B:(B,N,H)。
        V_A = self.compare_mlp(torch.cat((A, beta), dim=2))
        V_B = self.compare_mlp(torch.cat((B, alpha), dim=2))
        # padding 位置必须在求和前清零，否则 MLP 偏置会伪造证据。
        V_A = V_A * premise_mask.unsqueeze(2)
        V_B = V_B * hypothesis_mask.unsqueeze(2)
        # 分别沿词元维求和，得到两个 (B,H) 句子证据向量。
        summed_A = V_A.sum(dim=1)
        summed_B = V_B.sum(dim=1)
        # 聚合后 logits:(B,3)，对应蕴涵/矛盾/中性。
        aggregate = self.aggregate_mlp(torch.cat((summed_A, summed_B), dim=1))
        logits = self.output(aggregate)
        return logits, attention_to_B


@torch.no_grad()
def evaluate(model: DecomposableAttention, loader: DataLoader[Batch]) -> float:
    model.eval()
    correct = total = 0
    for batch in loader:
        logits, _ = model(batch.premise, batch.hypothesis, batch.premise_mask, batch.hypothesis_mask)
        correct += int((logits.argmax(dim=1) == batch.labels).sum())
        total += batch.labels.numel()
    return correct / total


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    examples = make_examples(args.size_per_class, args.seed)
    split = int(len(examples) * 0.8)
    train_examples, test_examples = examples[:split], examples[split:]
    vocab = Vocab(train_examples)
    train_loader = DataLoader(NLIDataset(train_examples, vocab), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(NLIDataset(test_examples, vocab), batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    model = DecomposableAttention(len(vocab), embed_size=24, hidden_size=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            # Forward：同时得到分类分数和可检查的注意力矩阵。
            logits, attention = model(
                batch.premise,
                batch.hypothesis,
                batch.premise_mask,
                batch.hypothesis_mask,
            )
            # Shape 守卫能尽早发现 softmax 轴或转置写错。
            assert attention.shape[:2] == batch.premise.shape
            # Loss：三分类交叉熵建立标量计算图。
            loss = F.cross_entropy(logits, batch.labels)
            # Zero grad：本批从干净梯度开始。
            optimizer.zero_grad(set_to_none=True)
            # Backward：梯度穿过聚合、比较、注意力与嵌入层。
            loss.backward()
            # Update：Adam 改变模型参数。
            optimizer.step()
        accuracy = evaluate(model, test_loader)
        if epoch == 1 or epoch == args.epochs or epoch % max(1, args.epochs // 4) == 0:
            print(f"epoch={epoch:02d}, loss={float(loss.detach()):.4f}, test_accuracy={accuracy:.3f}")
    assert accuracy >= 0.80, "NLI 准确率过低，请检查标签、mask 或 softmax 轴"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="离线可分解注意力 NLI")
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--size-per-class", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.006)
    parser.add_argument("--seed", type=int, default=29)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
