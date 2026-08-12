"""第 14 章：在离线小语料上完成微型 BERT 的 MLM + NSP 预训练。

本程序用于理解数据与 Shape，不追求真实 BERT 的规模或效果。它完整包含：
句子对构造、[CLS]/[SEP]、segment id、15% 掩蔽、80/10/10 替换、
Transformer 编码、MLM/NSP 两个头、联合损失、反向传播与文本表示。

运行：
    python code/ch14/mini_bert_pretraining.py
快速检查：
    python code/ch14/mini_bert_pretraining.py --epochs 15
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


SPECIAL_TOKENS = ("<pad>", "<unk>", "<cls>", "<sep>", "<mask>")

# 每个段落至少两句，这样相邻句关系可以直接从无标签文本中产生。
PARAGRAPHS = [
    ["a cat sits on the mat", "the cat watches a bird", "the bird flies away"],
    ["a dog runs in the park", "the dog chases a ball", "the owner smiles"],
    ["students read a book", "the book explains neural networks", "students write notes"],
    ["rain falls from clouds", "the street becomes wet", "people carry umbrellas"],
    ["the chef cuts vegetables", "the chef cooks a meal", "the family eats dinner"],
]


class Vocab:
    """最小词表；特殊词元索引固定，便于 padding 与预训练任务。"""

    def __init__(self, paragraphs: list[list[str]]) -> None:
        words = sorted({word for paragraph in paragraphs for sentence in paragraph for word in sentence.split()})
        self.idx_to_token = list(SPECIAL_TOKENS) + words
        self.token_to_idx = {token: index for index, token in enumerate(self.idx_to_token)}

    def __len__(self) -> int:
        return len(self.idx_to_token)

    def __getitem__(self, token: str) -> int:
        return self.token_to_idx.get(token, self.token_to_idx["<unk>"])

    def encode(self, tokens: list[str]) -> list[int]:
        return [self[token] for token in tokens]


@dataclass
class PretrainingExample:
    token_ids: list[int]
    segment_ids: list[int]
    mlm_labels: list[int]
    nsp_label: int


def make_bert_tokens(tokens_a: list[str], tokens_b: list[str]) -> tuple[list[str], list[int]]:
    """拼出 <cls> A <sep> B <sep>，并给 A/B 分配 0/1 片段号。"""
    tokens = ["<cls>"] + tokens_a + ["<sep>"] + tokens_b + ["<sep>"]
    segments = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
    return tokens, segments


def replace_for_mlm(
    tokens: list[str],
    vocab: Vocab,
    rng: random.Random,
) -> tuple[list[str], list[int]]:
    """选择约 15% 普通词元，并按 80/10/10 规则改写输入。"""
    candidates = [index for index, token in enumerate(tokens) if token not in {"<cls>", "<sep>"}]
    rng.shuffle(candidates)
    # 小句也至少预测一个词，否则 MLM 这一样本没有梯度。
    num_predictions = max(1, round(len(candidates) * 0.15))
    chosen = set(candidates[:num_predictions])
    corrupted = tokens.copy()
    # -100 是 CrossEntropyLoss 的 ignore_index：未选位置不参与 MLM 损失。
    labels = [-100] * len(tokens)
    normal_tokens = vocab.idx_to_token[len(SPECIAL_TOKENS) :]
    for position in chosen:
        original = tokens[position]
        labels[position] = vocab[original]
        draw = rng.random()
        if draw < 0.8:
            corrupted[position] = "<mask>"
        elif draw < 0.9:
            corrupted[position] = rng.choice(normal_tokens)
        # 剩余 10% 保持 original；标签仍要求模型预测原词。
    return corrupted, labels


def build_examples(vocab: Vocab, repeats: int, seed: int) -> list[PretrainingExample]:
    """一半使用真相邻句，一半随机替换第二句，自动得到 NSP 标签。"""
    rng = random.Random(seed)
    all_sentences = [sentence for paragraph in PARAGRAPHS for sentence in paragraph]
    examples: list[PretrainingExample] = []
    for _ in range(repeats):
        for paragraph in PARAGRAPHS:
            for index in range(len(paragraph) - 1):
                sentence_a = paragraph[index].split()
                if rng.random() < 0.5:
                    sentence_b = paragraph[index + 1].split()
                    nsp_label = 1
                else:
                    # 避免随机句恰好还是原来的真下一句。
                    candidates = [s for s in all_sentences if s != paragraph[index + 1]]
                    sentence_b = rng.choice(candidates).split()
                    nsp_label = 0
                tokens, segments = make_bert_tokens(sentence_a, sentence_b)
                corrupted, mlm_labels = replace_for_mlm(tokens, vocab, rng)
                examples.append(
                    PretrainingExample(
                        token_ids=vocab.encode(corrupted),
                        segment_ids=segments,
                        mlm_labels=mlm_labels,
                        nsp_label=nsp_label,
                    )
                )
    rng.shuffle(examples)
    return examples


class BertPretrainingDataset(Dataset[PretrainingExample]):
    def __init__(self, examples: list[PretrainingExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> PretrainingExample:
        return self.examples[index]


@dataclass
class Batch:
    token_ids: torch.Tensor
    segment_ids: torch.Tensor
    valid_mask: torch.Tensor
    mlm_labels: torch.Tensor
    nsp_labels: torch.Tensor


def make_collate_fn(pad_index: int):
    """动态补齐批量；四个序列张量共享同一个最大长度 S。"""

    def collate(examples: list[PretrainingExample]) -> Batch:
        max_len = max(len(example.token_ids) for example in examples)
        token_rows: list[list[int]] = []
        segment_rows: list[list[int]] = []
        label_rows: list[list[int]] = []
        valid_rows: list[list[bool]] = []
        for example in examples:
            padding = max_len - len(example.token_ids)
            token_rows.append(example.token_ids + [pad_index] * padding)
            segment_rows.append(example.segment_ids + [0] * padding)
            label_rows.append(example.mlm_labels + [-100] * padding)
            valid_rows.append([True] * len(example.token_ids) + [False] * padding)
        # token_ids/segment_ids/mlm_labels/valid_mask 的 Shape 都是 (B,S)。
        return Batch(
            token_ids=torch.tensor(token_rows, dtype=torch.long),
            segment_ids=torch.tensor(segment_rows, dtype=torch.long),
            valid_mask=torch.tensor(valid_rows, dtype=torch.bool),
            mlm_labels=torch.tensor(label_rows, dtype=torch.long),
            nsp_labels=torch.tensor([example.nsp_label for example in examples], dtype=torch.long),
        )

    return collate


class TinyBERT(nn.Module):
    """小型双向 Transformer 编码器，加 MLM 与 NSP 两个预训练头。"""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_heads: int,
        num_layers: int,
        max_len: int,
    ) -> None:
        super().__init__()
        # 三种嵌入相加后 Shape 都是 (B,S,H)。
        self.token_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.segment_embedding = nn.Embedding(2, hidden_size)
        self.position_embedding = nn.Embedding(max_len, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 2,
            dropout=0.1,
            batch_first=True,
            norm_first=False,
        )
        # 关闭原型 nested tensor 路径，让教学脚本在不同 PyTorch 小版本下输出更稳定。
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        # MLM 在每个位置输出 V 类；实际损失只读取被选中的位置。
        self.mlm_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, vocab_size),
        )
        # NSP 只读取 <cls> 位置，输出“非下一句/真下一句”两个类别。
        self.nsp_head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 2))

    def forward(
        self,
        token_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, sequence_length = token_ids.shape
        # positions:(S,) 会被广播到批量维，给相同位置查同一位置嵌入。
        positions = torch.arange(sequence_length, device=token_ids.device)
        # x:(B,S,H)，逐元素相加而不是拼接，所以隐藏维不变。
        x = self.token_embedding(token_ids)
        x = x + self.segment_embedding(segment_ids)
        x = x + self.position_embedding(positions).unsqueeze(0)
        # src_key_padding_mask 中 True 表示“屏蔽”；因此要对 valid_mask 取反。
        encoded = self.encoder(x, src_key_padding_mask=~valid_mask)
        # mlm_logits:(B,S,V)，包含所有位置的词表分数。
        mlm_logits = self.mlm_head(encoded)
        # nsp_logits:(B,2)，每个句子对只预测一个关系标签。
        nsp_logits = self.nsp_head(encoded[:, 0, :])
        return encoded, mlm_logits, nsp_logits


def compute_loss(
    mlm_logits: torch.Tensor,
    nsp_logits: torch.Tensor,
    mlm_labels: torch.Tensor,
    nsp_labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """联合损失 L= L_MLM + L_NSP；两项都按各自有效样本平均。"""
    vocab_size = mlm_logits.shape[-1]
    # 展平 (B,S,V)->(B*S,V)，-100 位置自动忽略。
    mlm_loss = F.cross_entropy(mlm_logits.reshape(-1, vocab_size), mlm_labels.reshape(-1), ignore_index=-100)
    # NSP 是普通二分类交叉熵，输入 (B,2)，标签 (B,)。
    nsp_loss = F.cross_entropy(nsp_logits, nsp_labels)
    total_loss = mlm_loss + nsp_loss
    return total_loss, mlm_loss, nsp_loss


@torch.no_grad()
def encode_pair(
    model: TinyBERT,
    vocab: Vocab,
    sentence_a: str,
    sentence_b: str,
) -> torch.Tensor:
    """返回 <cls> 的上下文表示 (H,)，演示预训练后如何抽取文本向量。"""
    tokens, segments = make_bert_tokens(sentence_a.split(), sentence_b.split())
    token_ids = torch.tensor([vocab.encode(tokens)], dtype=torch.long)
    segment_ids = torch.tensor([segments], dtype=torch.long)
    valid_mask = torch.ones_like(token_ids, dtype=torch.bool)
    model.eval()
    encoded, _, _ = model(token_ids, segment_ids, valid_mask)
    return encoded[0, 0]


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    vocab = Vocab(PARAGRAPHS)
    examples = build_examples(vocab, args.repeats, args.seed)
    loader = DataLoader(
        BertPretrainingDataset(examples),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=make_collate_fn(vocab["<pad>"]),
    )
    model = TinyBERT(
        vocab_size=len(vocab),
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_len=64,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    first_loss: float | None = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = mlm_sum = nsp_sum = 0.0
        num_batches = 0
        for batch in loader:
            # Forward：encoded:(B,S,H)，MLM:(B,S,V)，NSP:(B,2)。
            _, mlm_logits, nsp_logits = model(batch.token_ids, batch.segment_ids, batch.valid_mask)
            # Loss：用自动标签同时监督词元恢复和句子关系。
            loss, mlm_loss, nsp_loss = compute_loss(
                mlm_logits,
                nsp_logits,
                batch.mlm_labels,
                batch.nsp_labels,
            )
            # 清梯度；使用 set_to_none 能少一次全零写入。
            optimizer.zero_grad(set_to_none=True)
            # 反向传播经过两个任务头，共同更新 Transformer 编码器。
            loss.backward()
            # 控制微型 Transformer 的偶发大梯度。
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # AdamW 更新模型参数，权重衰减与梯度更新解耦。
            optimizer.step()
            loss_sum += float(loss.detach())
            mlm_sum += float(mlm_loss.detach())
            nsp_sum += float(nsp_loss.detach())
            num_batches += 1
        mean_loss = loss_sum / num_batches
        if first_loss is None:
            first_loss = mean_loss
        if epoch == 1 or epoch == args.epochs or epoch % max(1, args.epochs // 4) == 0:
            print(
                f"epoch={epoch:03d}, total={mean_loss:.4f}, "
                f"mlm={mlm_sum / num_batches:.4f}, nsp={nsp_sum / num_batches:.4f}"
            )
    assert first_loss is not None and mean_loss < first_loss, "联合损失没有下降"

    # 相同句子 A 配不同句子 B，<cls> 表示会因上下文变化而改变。
    vector_1 = encode_pair(model, vocab, "a cat sits on the mat", "the cat watches a bird")
    vector_2 = encode_pair(model, vocab, "a cat sits on the mat", "people carry umbrellas")
    cosine = F.cosine_similarity(vector_1, vector_2, dim=0)
    print(f"examples={len(examples)}, vocab_size={len(vocab)}, cls_shape={tuple(vector_1.shape)}")
    print(f"两组文本对 <cls> 表示的余弦相似度={float(cosine):.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="离线微型 BERT MLM + NSP 预训练")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--repeats", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
