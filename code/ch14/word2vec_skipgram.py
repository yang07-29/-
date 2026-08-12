"""第 14 章：用小语料从零训练 Skip-Gram + 负采样词向量。

这个程序不下载数据，重点打通完整链路：
语料 -> 词表 -> 中心词/上下文词 -> 负样本 -> 点积打分 -> BCE 损失 -> 更新词向量。

运行：
    python code/ch14/word2vec_skipgram.py
快速检查：
    python code/ch14/word2vec_skipgram.py --epochs 20
"""

from __future__ import annotations

import argparse
import math
import random
from collections import Counter

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


# 小语料故意重复若干“语义角色相似、上下文也相似”的句式。
BASE_SENTENCES = [
    "king rules kingdom royal palace",
    "queen rules kingdom royal palace",
    "man works city human adult",
    "woman works city human adult",
    "prince lives kingdom royal palace",
    "princess lives kingdom royal palace",
    "dog likes pet animal home",
    "cat likes pet animal home",
    "apple sweet fruit food fresh",
    "orange sweet fruit food fresh",
]


class SkipGramDataset(Dataset[tuple[int, int, torch.Tensor]]):
    """每个样本包含一个中心词、一个正上下文词和 K 个负上下文词。"""

    def __init__(self, samples: list[tuple[int, int, list[int]]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[int, int, torch.Tensor]:
        center, positive, negatives = self.samples[index]
        # negatives:(K,)，必须是 long，才能作为 Embedding 的索引。
        return center, positive, torch.tensor(negatives, dtype=torch.long)


class SkipGramNegativeSampling(nn.Module):
    """中心词表和上下文词表分开存参数，正是 Skip-Gram 的两套向量。"""

    def __init__(self, vocab_size: int, embed_size: int) -> None:
        super().__init__()
        # center_embedding.weight:(V,D)，存词作为“中心词”时的向量 v_c。
        self.center_embedding = nn.Embedding(vocab_size, embed_size)
        # context_embedding.weight:(V,D)，存词作为“上下文词”时的向量 u_o。
        self.context_embedding = nn.Embedding(vocab_size, embed_size)
        # 较小初始化让训练初期的 sigmoid 不容易直接饱和。
        nn.init.normal_(self.center_embedding.weight, mean=0.0, std=0.05)
        nn.init.normal_(self.context_embedding.weight, mean=0.0, std=0.05)

    def forward(
        self,
        centers: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # centers:(B,) -> center_vectors:(B,D)。
        center_vectors = self.center_embedding(centers)
        # positives:(B,) -> positive_vectors:(B,D)。
        positive_vectors = self.context_embedding(positives)
        # negatives:(B,K) -> negative_vectors:(B,K,D)。
        negative_vectors = self.context_embedding(negatives)
        # 正例逐维乘后求和，得到 u_o^T v_c；positive_logits:(B,)。
        positive_logits = (center_vectors * positive_vectors).sum(dim=1)
        # bmm 同时计算 K 个负例点积；先补成 (B,1,D)，结果再压成 (B,K)。
        negative_logits = torch.bmm(
            center_vectors.unsqueeze(1),
            negative_vectors.transpose(1, 2),
        ).squeeze(1)
        # logits 尚未过 sigmoid；BCEWithLogitsLoss 会稳定地完成 sigmoid + 对数损失。
        return positive_logits, negative_logits


def tokenize_corpus(repeats: int) -> list[list[str]]:
    """把句子拆成词元；重复只是让离线演示有足够的共现样本。"""
    sentences = BASE_SENTENCES * repeats
    return [sentence.lower().split() for sentence in sentences]


def build_vocab(sentences: list[list[str]]) -> tuple[list[str], dict[str, int], Counter[str]]:
    """按字母序构造可复现词表，并返回词频。"""
    counts = Counter(token for sentence in sentences for token in sentence)
    idx_to_token = sorted(counts)
    token_to_idx = {token: index for index, token in enumerate(idx_to_token)}
    return idx_to_token, token_to_idx, counts


def keep_probability(count: int, total: int, threshold: float = 1e-3) -> float:
    """高频词下采样保留概率；大语料中可削弱无信息高频词的支配。"""
    frequency = count / total
    # 这是 word2vec 常见近似形式；最终截到 1，避免把稀有词“保留超过 100%”。
    return min(1.0, math.sqrt(threshold / frequency))


def make_positive_pairs(
    sentences: list[list[str]],
    token_to_idx: dict[str, int],
    max_window: int,
    seed: int,
) -> list[tuple[int, int]]:
    """为每次中心词随机采样窗口宽度，并提取所有正中心－上下文对。"""
    rng = random.Random(seed)
    pairs: list[tuple[int, int]] = []
    for sentence in sentences:
        indices = [token_to_idx[token] for token in sentence]
        for center_position, center in enumerate(indices):
            # 每个中心词随机窗口能让近邻位置被采样更多次，贴近正式实现。
            window = rng.randint(1, max_window)
            left = max(0, center_position - window)
            right = min(len(indices), center_position + window + 1)
            for context_position in range(left, right):
                if context_position != center_position:
                    pairs.append((center, indices[context_position]))
    return pairs


def attach_negative_samples(
    pairs: list[tuple[int, int]],
    counts: Counter[str],
    idx_to_token: list[str],
    num_negatives: int,
    seed: int,
) -> list[tuple[int, int, list[int]]]:
    """按词频的 0.75 次幂采噪声词，并排除当前正上下文词。"""
    generator = torch.Generator().manual_seed(seed)
    # weights:(V,)，高频词更常成为负例，但 0.75 次幂会稍微压平极高频词。
    weights = torch.tensor([counts[token] ** 0.75 for token in idx_to_token], dtype=torch.float)
    samples: list[tuple[int, int, list[int]]] = []
    for center, positive in pairs:
        negatives: list[int] = []
        while len(negatives) < num_negatives:
            candidate = int(torch.multinomial(weights, 1, generator=generator).item())
            # 正上下文不能同时充当本样本的负例，否则标签会互相打架。
            if candidate != positive:
                negatives.append(candidate)
        samples.append((center, positive, negatives))
    return samples


def negative_sampling_loss(
    positive_logits: torch.Tensor,
    negative_logits: torch.Tensor,
) -> torch.Tensor:
    """实现 -log sigma(pos) - sum log sigma(-neg)，最后对批量求均值。"""
    # 正例目标为 1；softplus(-x) 等价于稳定版 -log(sigmoid(x))。
    positive_loss = F.softplus(-positive_logits)
    # 负例目标为 0；softplus(x) 等价于稳定版 -log(sigmoid(-x))。
    negative_loss = F.softplus(negative_logits).sum(dim=1)
    # 每个中心－正上下文对的 K+1 项先相加，再在 B 个样本上平均。
    return (positive_loss + negative_loss).mean()


@torch.no_grad()
def nearest_neighbors(
    query: str,
    model: SkipGramNegativeSampling,
    token_to_idx: dict[str, int],
    idx_to_token: list[str],
    top_k: int = 3,
) -> list[tuple[str, float]]:
    """用中心词向量余弦相似度查近邻；评估阶段不建立计算图。"""
    embeddings = F.normalize(model.center_embedding.weight, dim=1)
    query_index = token_to_idx[query]
    similarities = embeddings @ embeddings[query_index]
    # 多取一个再排除查询词自己（它与自己的余弦相似度为 1）。
    indices = similarities.topk(top_k + 1).indices.tolist()
    return [
        (idx_to_token[index], float(similarities[index]))
        for index in indices
        if index != query_index
    ][:top_k]


def train(args: argparse.Namespace) -> None:
    # 固定随机数，保证初始化、打乱顺序和输出可复现。
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    # sentences 是“句子列表”，每个句子又是词元列表。
    sentences = tokenize_corpus(args.repeats)
    # 建立 index <-> token 映射以及负采样所需词频。
    idx_to_token, token_to_idx, counts = build_vocab(sentences)
    # 正样本只来自真实窗口，不需要人工标签。
    positive_pairs = make_positive_pairs(sentences, token_to_idx, args.window, args.seed)
    # 每个正样本附 K 个噪声词，避免对完整词表做 softmax。
    samples = attach_negative_samples(
        positive_pairs,
        counts,
        idx_to_token,
        args.negatives,
        args.seed,
    )
    # DataLoader 自动把 center/positive 堆成 (B,)，negative 堆成 (B,K)。
    loader = DataLoader(SkipGramDataset(samples), batch_size=args.batch_size, shuffle=True)
    # 模型参数只有两张 (V,D) 嵌入表。
    model = SkipGramNegativeSampling(len(idx_to_token), args.embed_size)
    # Adam 根据反向传播得到的稀疏访问梯度更新两张嵌入表。
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    first_loss: float | None = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        seen = 0
        for centers, positives, negatives in loader:
            # Forward：得到正例 (B,) 与负例 (B,K) 原始分数。
            positive_logits, negative_logits = model(centers, positives, negatives)
            # Loss：把正点积推高、负点积压低，建立标量计算图。
            loss = negative_sampling_loss(positive_logits, negative_logits)
            # 清掉上一批梯度；参数本身此时尚未改变。
            optimizer.zero_grad(set_to_none=True)
            # Backward：为两张 Embedding 表中本批访问过的行计算梯度。
            loss.backward()
            # Update：Adam 真正改变参数，完成一次学习。
            optimizer.step()
            batch_size = centers.shape[0]
            epoch_loss += float(loss.detach()) * batch_size
            seen += batch_size
        mean_loss = epoch_loss / seen
        if first_loss is None:
            first_loss = mean_loss
        if epoch == 1 or epoch == args.epochs or epoch % max(1, args.epochs // 4) == 0:
            print(f"epoch={epoch:03d}, loss={mean_loss:.4f}")

    # 用断言检查训练链路真的学到了，而不只是“程序没有报错”。
    assert first_loss is not None and mean_loss < first_loss, "损失没有下降，请检查样本或学习率"
    print(f"vocab_size={len(idx_to_token)}, positive_pairs={len(positive_pairs)}")
    print("king 的近邻：", nearest_neighbors("king", model, token_to_idx, idx_to_token))
    print("dog 的近邻：", nearest_neighbors("dog", model, token_to_idx, idx_to_token))
    total = sum(counts.values())
    example = idx_to_token[0]
    print(f"下采样示例：P(保留 {example!r})={keep_probability(counts[example], total):.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="离线 Skip-Gram + 负采样演示")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--window", type=int, default=2)
    parser.add_argument("--negatives", type=int, default=4)
    parser.add_argument("--embed-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
