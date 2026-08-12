"""第 14 章：离线演示 GloVe、BPE、词相似度与类比。

运行：
    python code/ch14/glove_bpe_analogy.py
快速检查：
    python code/ch14/glove_bpe_analogy.py --epochs 40
"""

from __future__ import annotations

import argparse
from collections import Counter

import torch
from torch import nn
from torch.nn import functional as F


SENTENCES = [
    "king queen royal palace kingdom",
    "prince princess royal palace kingdom",
    "king man male human",
    "queen woman female human",
    "prince boy male human",
    "princess girl female human",
    "paris france capital europe",
    "berlin germany capital europe",
    "tokyo japan capital asia",
    "beijing china capital asia",
] * 40


class TinyGloVe(nn.Module):
    """GloVe 为中心/上下文各学习向量和偏置。"""

    def __init__(self, vocab_size: int, embed_size: int) -> None:
        super().__init__()
        # center/context:(V,D)，分别对应 v_i 与 u_j。
        self.center = nn.Embedding(vocab_size, embed_size)
        self.context = nn.Embedding(vocab_size, embed_size)
        # 两个标量偏置表形状都是 (V,1)。
        self.center_bias = nn.Embedding(vocab_size, 1)
        self.context_bias = nn.Embedding(vocab_size, 1)
        # 统一做小尺度初始化，避免初始预测离 log(X_ij) 太远。
        for parameter in self.parameters():
            nn.init.normal_(parameter, mean=0.0, std=0.05)

    def forward(self, centers: torch.Tensor, contexts: torch.Tensor) -> torch.Tensor:
        # centers/contexts:(N,) -> v/u:(N,D)。
        center_vectors = self.center(centers)
        context_vectors = self.context(contexts)
        # 每行点积给出 (N,)，再加两个 (N,) 偏置。
        dot = (center_vectors * context_vectors).sum(dim=1)
        prediction = dot + self.center_bias(centers).squeeze(1)
        prediction = prediction + self.context_bias(contexts).squeeze(1)
        # prediction 近似的不是概率，而是 log 共现次数。
        return prediction


def build_cooccurrence(
    sentences: list[str],
    window: int,
) -> tuple[list[str], dict[str, int], torch.Tensor]:
    """统计距离加权共现矩阵 X；邻得越近，贡献越大。"""
    tokenized = [sentence.split() for sentence in sentences]
    idx_to_token = sorted({token for sentence in tokenized for token in sentence})
    token_to_idx = {token: index for index, token in enumerate(idx_to_token)}
    # X:(V,V)，X[i,j] 表示 j 出现在 i 窗口内的加权次数。
    matrix = torch.zeros(len(idx_to_token), len(idx_to_token))
    for sentence in tokenized:
        indices = [token_to_idx[token] for token in sentence]
        for center_position, center in enumerate(indices):
            left = max(0, center_position - window)
            right = min(len(indices), center_position + window + 1)
            for context_position in range(left, right):
                if context_position == center_position:
                    continue
                distance = abs(center_position - context_position)
                # 距离 1 贡献 1，距离 2 贡献 1/2。
                matrix[center, indices[context_position]] += 1.0 / distance
    return idx_to_token, token_to_idx, matrix


def glove_weight(counts: torch.Tensor, x_max: float, alpha: float) -> torch.Tensor:
    """低频共现按幂函数降权，高于 x_max 后权重封顶为 1。"""
    return torch.clamp((counts / x_max) ** alpha, max=1.0)


def train_glove(args: argparse.Namespace) -> tuple[TinyGloVe, list[str], dict[str, int]]:
    torch.manual_seed(args.seed)
    idx_to_token, token_to_idx, matrix = build_cooccurrence(SENTENCES, args.window)
    # 只训练 X_ij > 0 的位置，避免对 log(0) 做无意义运算。
    centers, contexts = torch.nonzero(matrix > 0, as_tuple=True)
    counts = matrix[centers, contexts]
    # target:(N,) 是每个非零共现次数的自然对数。
    targets = counts.log()
    # weights:(N,) 抑制极稀有共现，同时不让高频项无限支配损失。
    weights = glove_weight(counts, args.x_max, alpha=0.75)
    model = TinyGloVe(len(idx_to_token), args.embed_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    first_loss: float | None = None
    for epoch in range(1, args.epochs + 1):
        # Forward：prediction:(N,)，一次用完全部非零共现对。
        prediction = model(centers, contexts)
        # Loss：加权平方误差，标量会建立完整计算图。
        loss = (weights * (prediction - targets).square()).mean()
        # Zero grad：清除上一轮累计梯度。
        optimizer.zero_grad(set_to_none=True)
        # Backward：求向量表和偏置表的梯度。
        loss.backward()
        # Update：同时更新 v_i、u_j、b_i、c_j。
        optimizer.step()
        if first_loss is None:
            first_loss = float(loss.detach())
        if epoch == 1 or epoch == args.epochs or epoch % max(1, args.epochs // 4) == 0:
            print(f"glove epoch={epoch:03d}, loss={float(loss.detach()):.5f}")
    assert first_loss is not None and float(loss.detach()) < first_loss, "GloVe 损失没有下降"
    return model, idx_to_token, token_to_idx


def final_embeddings(model: TinyGloVe) -> torch.Tensor:
    """实际使用时常把中心向量与上下文向量相加，减少角色不对称。"""
    return model.center.weight.detach() + model.context.weight.detach()


def most_similar(
    token: str,
    embeddings: torch.Tensor,
    token_to_idx: dict[str, int],
    idx_to_token: list[str],
    top_k: int = 3,
) -> list[tuple[str, float]]:
    """余弦近邻；向量先归一化后，矩阵乘就是批量余弦相似度。"""
    normalized = F.normalize(embeddings, dim=1)
    query_index = token_to_idx[token]
    scores = normalized @ normalized[query_index]
    candidates = scores.topk(top_k + 1).indices.tolist()
    return [
        (idx_to_token[index], float(scores[index]))
        for index in candidates
        if index != query_index
    ][:top_k]


def analogy(
    a: str,
    b: str,
    c: str,
    embeddings: torch.Tensor,
    token_to_idx: dict[str, int],
    idx_to_token: list[str],
) -> tuple[str, float]:
    """求解 a:b :: c:?，查询向量为 e_b - e_a + e_c。"""
    normalized = F.normalize(embeddings, dim=1)
    query = embeddings[token_to_idx[b]] - embeddings[token_to_idx[a]]
    query = query + embeddings[token_to_idx[c]]
    query = F.normalize(query.unsqueeze(0), dim=1).squeeze(0)
    scores = normalized @ query
    # 排除题目中已经出现的三个词，避免返回平凡答案。
    for token in (a, b, c):
        scores[token_to_idx[token]] = -float("inf")
    index = int(scores.argmax())
    return idx_to_token[index], float(scores[index])


def word_symbols(word: str) -> tuple[str, ...]:
    """BPE 初始把词拆成字符，并用 </w> 显式标记词尾。"""
    return tuple(list(word) + ["</w>"])


def merge_pair(symbols: tuple[str, ...], pair: tuple[str, str]) -> tuple[str, ...]:
    """把一个词中的指定相邻符号对同时合并。"""
    merged: list[str] = []
    index = 0
    while index < len(symbols):
        if index + 1 < len(symbols) and (symbols[index], symbols[index + 1]) == pair:
            merged.append(symbols[index] + symbols[index + 1])
            index += 2
        else:
            merged.append(symbols[index])
            index += 1
    return tuple(merged)


def learn_bpe(word_counts: Counter[str], num_merges: int) -> tuple[dict[str, tuple[str, ...]], list[tuple[str, str]]]:
    """每轮合并带权频率最高的相邻符号对。"""
    vocabulary = {word: word_symbols(word) for word in word_counts}
    merge_rules: list[tuple[str, str]] = []
    for _ in range(num_merges):
        pair_counts: Counter[tuple[str, str]] = Counter()
        for word, symbols in vocabulary.items():
            for left, right in zip(symbols, symbols[1:]):
                pair_counts[(left, right)] += word_counts[word]
        if not pair_counts:
            break
        best_pair = pair_counts.most_common(1)[0][0]
        merge_rules.append(best_pair)
        vocabulary = {word: merge_pair(symbols, best_pair) for word, symbols in vocabulary.items()}
    return vocabulary, merge_rules


def apply_bpe(word: str, merge_rules: list[tuple[str, str]]) -> tuple[str, ...]:
    """按训练所得规则顺序切分新词；未见词仍可退回到字符级。"""
    symbols = word_symbols(word)
    for pair in merge_rules:
        symbols = merge_pair(symbols, pair)
    return symbols


def main(args: argparse.Namespace) -> None:
    model, idx_to_token, token_to_idx = train_glove(args)
    embeddings = final_embeddings(model)
    print("king 的近邻：", most_similar("king", embeddings, token_to_idx, idx_to_token))
    print("类比 man:woman :: king:? ->", analogy("man", "woman", "king", embeddings, token_to_idx, idx_to_token))

    # BPE 词频中的 high/higher/highest 会促使高频片段 h-i-g-h 被逐步合并。
    word_counts = Counter({"low": 5, "lower": 2, "newest": 6, "widest": 3, "high": 8, "higher": 5, "highest": 4})
    learned_vocab, rules = learn_bpe(word_counts, num_merges=args.bpe_merges)
    print("BPE 前 8 条合并规则：", rules[:8])
    print("训练词 highest 的切分：", learned_vocab["highest"])
    print("未见词 highly 的切分：", apply_bpe("highly", rules))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="离线 GloVe、BPE 与类比演示")
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--embed-size", type=int, default=16)
    parser.add_argument("--window", type=int, default=2)
    parser.add_argument("--x-max", type=float, default=10.0)
    parser.add_argument("--lr", type=float, default=0.04)
    parser.add_argument("--bpe-merges", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
