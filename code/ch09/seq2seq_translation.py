"""第 9 章：离线可运行的编码器－解码器、seq2seq 与束搜索。

任务是把英文数字词翻译为中文数字词，例如：
    "one four two" -> "一 四 二"

语料由程序合成，因此不用下载大型机器翻译数据。这个小任务只用于打通
预处理、teacher forcing、遮蔽损失、自回归预测、BLEU 与 beam search 链路。

运行：
    python code/ch09/seq2seq_translation.py --epochs 30
快速检查：
    python code/ch09/seq2seq_translation.py --epochs 2 --num-examples 256
"""

from __future__ import annotations

import argparse
import math
import random
from collections import Counter
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset


SOURCE_WORDS = ("zero", "one", "two", "three", "four", "five")
TARGET_WORDS = ("零", "一", "二", "三", "四", "五")
TRANSLATION = dict(zip(SOURCE_WORDS, TARGET_WORDS, strict=True))
SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>")


class Vocab:
    """机器翻译用词表；特殊词元索引固定，便于遮蔽和解码。"""

    def __init__(self, tokens: list[str]) -> None:
        ordered = list(SPECIAL_TOKENS) + sorted(set(tokens) - set(SPECIAL_TOKENS))
        self.idx_to_token = ordered
        self.token_to_idx = {token: index for index, token in enumerate(ordered)}

    def __len__(self) -> int:
        return len(self.idx_to_token)

    def __getitem__(self, token: str) -> int:
        return self.token_to_idx.get(token, self.unk)

    @property
    def pad(self) -> int:
        return self.token_to_idx["<pad>"]

    @property
    def bos(self) -> int:
        return self.token_to_idx["<bos>"]

    @property
    def eos(self) -> int:
        return self.token_to_idx["<eos>"]

    @property
    def unk(self) -> int:
        return self.token_to_idx["<unk>"]

    def encode(self, tokens: list[str]) -> list[int]:
        return [self[token] for token in tokens]

    def decode(self, indices: list[int], drop_special: bool = True) -> list[str]:
        result: list[str] = []
        for index in indices:
            token = self.idx_to_token[index]
            if token == "<eos>":
                break
            if not drop_special or token not in SPECIAL_TOKENS:
                result.append(token)
        return result


def build_examples(num_examples: int, seed: int = 31) -> list[tuple[list[str], list[str]]]:
    """构造 1～4 词的平行语料；目标序列与源序列逐词对应。"""
    rng = random.Random(seed)
    examples: list[tuple[list[str], list[str]]] = []
    for _ in range(num_examples):
        length = rng.randint(1, 4)
        source = [rng.choice(SOURCE_WORDS) for _ in range(length)]
        target = [TRANSLATION[word] for word in source]
        examples.append((source, target))
    return examples


class TranslationDataset(Dataset[tuple[list[int], list[int]]]):
    def __init__(self, examples: list[tuple[list[str], list[str]]], src_vocab: Vocab, tgt_vocab: Vocab) -> None:
        self.items: list[tuple[list[int], list[int]]] = []
        for source, target in examples:
            # 编码器输入在正文后加 eos，让模型明确看到源句终点。
            source_indices = src_vocab.encode(source) + [src_vocab.eos]
            # 解码器目标前加 bos、后加 eos，分别标记生成起点和终点。
            target_indices = [tgt_vocab.bos] + tgt_vocab.encode(target) + [tgt_vocab.eos]
            self.items.append((source_indices, target_indices))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[list[int], list[int]]:
        return self.items[index]


@dataclass
class Batch:
    source: torch.Tensor
    source_valid_lengths: torch.Tensor
    target: torch.Tensor
    target_valid_lengths: torch.Tensor


def make_collate_fn(src_pad: int, tgt_pad: int):
    """返回批处理函数：补齐变长序列，同时保留补齐前的有效长度。"""

    def collate(items: list[tuple[list[int], list[int]]]) -> Batch:
        source_tensors = [torch.tensor(source, dtype=torch.long) for source, _ in items]
        target_tensors = [torch.tensor(target, dtype=torch.long) for _, target in items]
        # 有效长度必须在 padding 前记录，否则模型分不清真实词元和补位符。
        source_lengths = torch.tensor([len(source) for source, _ in items])
        target_lengths = torch.tensor([len(target) for _, target in items])
        # pad_sequence 输出 (B,S_max) 与 (B,T_max)。
        source = pad_sequence(source_tensors, batch_first=True, padding_value=src_pad)
        target = pad_sequence(target_tensors, batch_first=True, padding_value=tgt_pad)
        return Batch(source, source_lengths, target, target_lengths)

    return collate


class Encoder(nn.Module):
    """把变长源句压缩为最终循环状态。"""

    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int, pad_index: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_index)
        self.rnn = nn.GRU(embed_size, hidden_size, batch_first=True)

    def forward(self, source: torch.Tensor, valid_lengths: torch.Tensor) -> torch.Tensor:
        # source:(B,S) -> embedded:(B,S,E)。
        embedded = self.embedding(source)
        # pack 让 GRU 跳过 pad；lengths 在 PyTorch 中需位于 CPU。
        packed = pack_padded_sequence(
            embedded,
            valid_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        # final_state 的 Shape 是 (L=1,B,H)，它作为源句上下文摘要。
        _, final_state = self.rnn(packed)
        return final_state


class Decoder(nn.Module):
    """以上一目标词、旧状态和固定上下文预测下一个目标词。"""

    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int, pad_index: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_index)
        # 每步输入由目标词嵌入 E 与编码器上下文 H 拼接而成。
        self.rnn = nn.GRU(embed_size + hidden_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        tokens: torch.Tensor,
        state: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # tokens:(B,T) -> embedded:(B,T,E)。
        embedded = self.embedding(tokens)
        # context 原为 (B,H)，在 T 维复制后变成 (B,T,H)。
        repeated_context = context.unsqueeze(1).expand(-1, tokens.shape[1], -1)
        # 拼接后 decoder_inputs:(B,T,E+H)。
        decoder_inputs = torch.cat((embedded, repeated_context), dim=2)
        # outputs:(B,T,H)，new_state:(1,B,H)。
        outputs, new_state = self.rnn(decoder_inputs, state)
        # logits:(B,T,V_tgt)，不要在 CrossEntropyLoss 前手动 softmax。
        logits = self.output(outputs)
        return logits, new_state


class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self,
        source: torch.Tensor,
        source_valid_lengths: torch.Tensor,
        decoder_inputs: torch.Tensor,
    ) -> torch.Tensor:
        # 编码器最终状态既用于初始化解码器，也作为每一步固定上下文。
        encoder_state = self.encoder(source, source_valid_lengths)
        context = encoder_state[-1]
        logits, _ = self.decoder(decoder_inputs, encoder_state, context)
        return logits


def masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_lengths: torch.Tensor,
) -> torch.Tensor:
    """只平均真实目标词元的损失，不让 pad 稀释结果。"""
    # 交叉熵要求类别维位于第 1 维，因此 (B,T,V) -> (B,V,T)。
    token_losses = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none")
    # positions:(1,T)，与 valid_lengths:(B,1) 广播得到 mask:(B,T)。
    positions = torch.arange(targets.shape[1], device=targets.device).unsqueeze(0)
    mask = positions < valid_lengths.unsqueeze(1)
    # 先遮住 pad，再除以真实词元总数，避免长短句权重口径混乱。
    return (token_losses * mask).sum() / mask.sum().clamp_min(1)


def train_model(model: Seq2Seq, loader: DataLoader[Batch], epochs: int) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        for batch in loader:
            # teacher forcing：输入 <bos> 和真实目标前缀，不输入最后的 eos。
            decoder_inputs = batch.target[:, :-1]
            # 标签整体左移一位：每个位置学习预测“下一个词”。
            decoder_targets = batch.target[:, 1:]
            # 去掉开头 bos 后，有效标签长度也要减 1。
            decoder_valid_lengths = batch.target_valid_lengths - 1

            # Forward 得到 (B,T,V_tgt) 的原始 logits。
            logits = model(batch.source, batch.source_valid_lengths, decoder_inputs)
            # Loss 忽略每条句子 eos 之后的 pad 位置。
            loss = masked_cross_entropy(logits, decoder_targets, decoder_valid_lengths)
            optimizer.zero_grad(set_to_none=True)
            # Backward 同时训练编码器、解码器、嵌入层和输出层。
            loss.backward()
            # 序列网络更新前做全局梯度裁剪，防止偶发梯度爆炸。
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            token_count = int(decoder_valid_lengths.sum())
            total_loss += loss.item() * token_count
            total_tokens += token_count

        if epoch == 1 or epoch % max(1, epochs // 5) == 0:
            mean_loss = total_loss / total_tokens
            print(f"epoch={epoch:03d}, token_loss={mean_loss:.4f}, ppl={math.exp(mean_loss):.3f}")


def encode_source(sentence: str, vocab: Vocab) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = sentence.lower().strip().split()
    indices = vocab.encode(tokens) + [vocab.eos]
    source = torch.tensor([indices], dtype=torch.long)
    valid_length = torch.tensor([len(indices)], dtype=torch.long)
    return source, valid_length


@torch.inference_mode()
def greedy_decode(
    model: Seq2Seq,
    sentence: str,
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    max_steps: int = 8,
) -> list[str]:
    """每一步只保留概率最大的一个词元。"""
    model.eval()
    source, valid_length = encode_source(sentence, src_vocab)
    state = model.encoder(source, valid_length)
    context = state[-1]
    current = torch.tensor([[tgt_vocab.bos]])
    generated: list[int] = []

    for _ in range(max_steps):
        # 推理时没有真实目标前缀，只能把上一步预测喂回来。
        logits, state = model.decoder(current, state, context)
        next_index = int(logits[0, -1].argmax())
        if next_index == tgt_vocab.eos:
            break
        generated.append(next_index)
        current = torch.tensor([[next_index]])
    return tgt_vocab.decode(generated)


def length_normalized_score(log_probability: float, length: int, alpha: float) -> float:
    """用长度惩罚减少束搜索对过短序列的偏爱。"""
    return log_probability / max(length, 1) ** alpha


@torch.inference_mode()
def beam_search_decode(
    model: Seq2Seq,
    sentence: str,
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    beam_size: int = 3,
    max_steps: int = 8,
    alpha: float = 0.75,
) -> list[str]:
    """保留累计得分最高的 K 条部分序列。"""
    model.eval()
    source, valid_length = encode_source(sentence, src_vocab)
    encoder_state = model.encoder(source, valid_length)
    context = encoder_state[-1]
    # 每个 beam 保存 token 序列、累计对数概率、尚未消费末词前的状态。
    beams: list[tuple[list[int], float, torch.Tensor]] = [
        ([tgt_vocab.bos], 0.0, encoder_state)
    ]

    for _ in range(max_steps):
        candidates: list[tuple[list[int], float, torch.Tensor]] = []
        for tokens, log_probability, state in beams:
            if tokens[-1] == tgt_vocab.eos:
                # 已结束的候选原样保留，不再继续扩展。
                candidates.append((tokens, log_probability, state))
                continue

            current = torch.tensor([[tokens[-1]]])
            logits, new_state = model.decoder(current, state, context)
            log_probs = F.log_softmax(logits[0, -1], dim=0)
            # 多取两个位置，过滤掉不应在正文生成的 pad 与 bos。
            values, indices = torch.topk(log_probs, k=min(beam_size + 2, len(tgt_vocab)))
            for value, index in zip(values.tolist(), indices.tolist(), strict=True):
                if index in (tgt_vocab.pad, tgt_vocab.bos):
                    continue
                candidates.append((tokens + [index], log_probability + value, new_state.clone()))

        # 对部分序列做长度归一化后只保留 K 条。
        beams = sorted(
            candidates,
            key=lambda item: length_normalized_score(item[1], len(item[0]) - 1, alpha),
            reverse=True,
        )[:beam_size]
        if all(tokens[-1] == tgt_vocab.eos for tokens, _, _ in beams):
            break

    best_tokens, _, _ = max(
        beams,
        key=lambda item: length_normalized_score(item[1], len(item[0]) - 1, alpha),
    )
    return tgt_vocab.decode(best_tokens[1:])


def bleu(prediction: list[str], label: list[str], max_n: int = 2) -> float:
    """计算 D2L 风格的简化 BLEU：短句惩罚乘 n-gram 精确率。"""
    if not prediction:
        return 0.0
    # 预测过短时施加 brevity penalty；足够长时该项为 1。
    score = math.exp(min(0.0, 1.0 - len(label) / len(prediction)))
    for n in range(1, min(max_n, len(prediction), len(label)) + 1):
        label_counts = Counter(tuple(label[i : i + n]) for i in range(len(label) - n + 1))
        matches = 0
        for i in range(len(prediction) - n + 1):
            ngram = tuple(prediction[i : i + n])
            if label_counts[ngram] > 0:
                matches += 1
                label_counts[ngram] -= 1
        precision = matches / (len(prediction) - n + 1)
        score *= precision ** (0.5**n)
    return score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="离线 seq2seq 翻译 smoke test")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--num-examples", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--beam-size", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(31)
    random.seed(31)

    examples = build_examples(args.num_examples)
    src_vocab = Vocab(list(SOURCE_WORDS))
    tgt_vocab = Vocab(list(TARGET_WORDS))
    dataset = TranslationDataset(examples, src_vocab, tgt_vocab)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=make_collate_fn(src_vocab.pad, tgt_vocab.pad),
        generator=torch.Generator().manual_seed(31),
    )

    encoder = Encoder(len(src_vocab), embed_size=24, hidden_size=48, pad_index=src_vocab.pad)
    decoder = Decoder(len(tgt_vocab), embed_size=24, hidden_size=48, pad_index=tgt_vocab.pad)
    model = Seq2Seq(encoder, decoder)

    first_batch = next(iter(loader))
    first_logits = model(
        first_batch.source,
        first_batch.source_valid_lengths,
        first_batch.target[:, :-1],
    )
    print(
        "Shape:",
        f"src={tuple(first_batch.source.shape)}",
        f"tgt={tuple(first_batch.target.shape)}",
        f"logits={tuple(first_logits.shape)}",
    )
    assert first_logits.shape[:2] == first_batch.target[:, :-1].shape

    train_model(model, loader, args.epochs)

    tests = (
        ("one four two", ["一", "四", "二"]),
        ("five zero", ["五", "零"]),
        ("three two one four", ["三", "二", "一", "四"]),
    )
    for source, reference in tests:
        greedy = greedy_decode(model, source, src_vocab, tgt_vocab)
        beam = beam_search_decode(
            model,
            source,
            src_vocab,
            tgt_vocab,
            beam_size=args.beam_size,
        )
        print(f"source: {source}")
        print(f"target: {' '.join(reference)}")
        print(f"greedy: {' '.join(greedy)} | BLEU={bleu(greedy, reference):.3f}")
        print(f"beam:   {' '.join(beam)} | BLEU={bleu(beam, reference):.3f}")
        assert len(greedy) <= 8 and len(beam) <= 8
        assert all(token in TARGET_WORDS for token in greedy + beam)


if __name__ == "__main__":
    main()
