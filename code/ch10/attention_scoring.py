"""第 10 章：从零理解注意力评分、遮蔽 softmax 与 Bahdanau 解码一步。

运行：python code/ch10/attention_scoring.py
只依赖 PyTorch，不下载数据。
"""

from __future__ import annotations

import math

import torch
from torch import nn


def masked_softmax(scores: torch.Tensor, valid_lengths: torch.Tensor | None) -> torch.Tensor:
    """在最后一个维度做 softmax，并让无效键得到恰好为 0 的权重。

    scores: (B, Q, K)，B=批量，Q=查询数，K=键值对数。
    valid_lengths: (B,) 表示每个样本所有查询共用的有效键数；
                   或 (B, Q) 表示每个查询自己的有效键数。
    """
    if valid_lengths is None:  # 没有 padding 时直接在键维做普通 softmax。
        return torch.softmax(scores, dim=-1)  # 输出仍为 (B,Q,K)，每行和为 1。

    batch_size, num_queries, num_keys = scores.shape  # 明确三个轴，避免把查询轴当成键轴。
    positions = torch.arange(num_keys, device=scores.device)  # (K,)，记录每个键的位置编号。

    if valid_lengths.ndim == 1:  # (B,) 意味着同一样本内所有查询共享长度。
        valid_lengths = valid_lengths[:, None].expand(batch_size, num_queries)  # 广播成 (B,Q)。
    elif valid_lengths.shape != (batch_size, num_queries):  # 其余 Shape 很可能来自错误广播。
        raise ValueError("valid_lengths 必须是 (B,) 或 (B,Q)")  # 立刻报错比静默掩蔽错位更安全。

    mask = positions.view(1, 1, num_keys) < valid_lengths.unsqueeze(-1)  # (B,Q,K)，True 表示有效键。
    masked_scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)  # 无效位置填极小有限值。
    weights = torch.softmax(masked_scores, dim=-1)  # (B,Q,K)，只在有效位置归一化。
    weights = weights.masked_fill(~mask, 0.0)  # 显式清零，兼顾有效长度为 0 的防御性语义。
    normalizer = weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).eps)  # 防止除零。
    return weights / normalizer  # 有有效键的行和为 1；全无效行保持全 0。


class AdditiveAttention(nn.Module):
    """加性注意力：允许查询维与键维不同。"""

    def __init__(self, query_size: int, key_size: int, hidden_size: int) -> None:
        super().__init__()  # 注册参数与子模块。
        self.query_proj = nn.Linear(query_size, hidden_size, bias=False)  # Dq -> H_a。
        self.key_proj = nn.Linear(key_size, hidden_size, bias=False)  # Dk -> H_a。
        self.score_proj = nn.Linear(hidden_size, 1, bias=False)  # H_a -> 单个兼容性分数。
        self.attention_weights: torch.Tensor | None = None  # 保存权重，便于调试和可视化。

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        valid_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        projected_q = self.query_proj(queries)  # (B,Q,Dq) -> (B,Q,H_a)。
        projected_k = self.key_proj(keys)  # (B,K,Dk) -> (B,K,H_a)。
        joint = torch.tanh(projected_q.unsqueeze(2) + projected_k.unsqueeze(1))  # (B,Q,K,H_a)。
        scores = self.score_proj(joint).squeeze(-1)  # (B,Q,K,H_a) -> (B,Q,K)。
        self.attention_weights = masked_softmax(scores, valid_lengths)  # (B,Q,K)。
        return torch.bmm(self.attention_weights, values)  # (B,Q,K)@(B,K,Dv) -> (B,Q,Dv)。


class ScaledDotProductAttention(nn.Module):
    """缩放点积注意力：查询和键的最后一维必须相同。"""

    def __init__(self) -> None:
        super().__init__()  # 保持标准 nn.Module 接口。
        self.attention_weights: torch.Tensor | None = None  # 保存最近一次注意力权重。

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        valid_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        feature_size = queries.shape[-1]  # d 是缩放因子的来源。
        if keys.shape[-1] != feature_size:  # 点积要求查询和键长度相同。
            raise ValueError("点积注意力要求 queries 与 keys 的最后一维一致")
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(feature_size)  # (B,Q,K)。
        self.attention_weights = masked_softmax(scores, valid_lengths)  # (B,Q,K)。
        return torch.bmm(self.attention_weights, values)  # (B,Q,Dv)。


def nadaraya_watson_demo() -> None:
    """用一维核回归演示“相近的键获得更大权重”。"""
    train_x = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])  # 五个已知自变量充当键。
    train_y = torch.sin(train_x)  # 已知函数值充当值。
    query = torch.tensor([1.6])  # 想估计的新位置充当查询。
    bandwidth = torch.tensor(0.6)  # 带宽越小，注意力越局部。
    scores = -0.5 * ((query[:, None] - train_x[None, :]) / bandwidth).square()  # (1,5)。
    weights = torch.softmax(scores, dim=-1)  # (1,5)，距离近的位置权重大。
    prediction = weights @ train_y[:, None]  # (1,5)@(5,1) -> (1,1)。
    print("\n[Nadaraya-Watson 核回归]")
    print("权重:", weights.squeeze(0).round(decimals=3).tolist())
    print(f"x=1.6 的估计值: {prediction.item():.4f}")


def bahdanau_decoder_step(
    decoder_state: torch.Tensor,
    encoder_outputs: torch.Tensor,
    source_valid_lengths: torch.Tensor,
    attention: AdditiveAttention,
    gru_cell: nn.GRUCell,
) -> tuple[torch.Tensor, torch.Tensor]:
    """演示 Bahdanau 解码器的一个时间步，不承担完整翻译训练。"""
    query = decoder_state.unsqueeze(1)  # (B,H_dec) -> (B,1,H_dec)，当前状态是唯一查询。
    context = attention(query, encoder_outputs, encoder_outputs, source_valid_lengths)  # (B,1,H_enc)。
    context = context.squeeze(1)  # (B,1,H_enc) -> (B,H_enc)。
    next_state = gru_cell(context, decoder_state)  # 上下文与旧状态共同得到 (B,H_dec) 新状态。
    return next_state, context  # 返回状态供下一步递推，同时返回上下文便于检查。


def main() -> None:
    torch.manual_seed(7)  # 固定随机种子，让输出可复现。

    scores = torch.tensor(  # 构造 B=2、Q=2、K=4 的可读分数。
        [[[1.0, 2.0, 3.0, 100.0], [2.0, 1.0, 0.0, 100.0]],
         [[1.0, 2.0, 100.0, 100.0], [3.0, 2.0, 100.0, 100.0]]]
    )
    valid_lengths = torch.tensor([3, 2])  # 第一个样本前三个键有效，第二个前两个有效。
    weights = masked_softmax(scores, valid_lengths)  # (2,2,4)。
    print("[masked softmax]")
    print(weights)
    print("每行权重和:", weights.sum(dim=-1))  # 应全部为 1。
    assert torch.all(weights[0, :, 3] == 0)  # 第一个样本第 4 个键必须被屏蔽。
    assert torch.all(weights[1, :, 2:] == 0)  # 第二个样本第 3、4 个键必须被屏蔽。

    batch_size, num_queries, num_keys = 2, 3, 4  # 统一记录示例规模。
    queries_add = torch.randn(batch_size, num_queries, 5)  # 加性注意力查询为 (2,3,5)。
    keys_add = torch.randn(batch_size, num_keys, 6)  # 键为 (2,4,6)，允许与查询维不同。
    values = torch.randn(batch_size, num_keys, 7)  # 值为 (2,4,7)。
    additive = AdditiveAttention(query_size=5, key_size=6, hidden_size=8)  # 创建可训练评分器。
    output_add = additive(queries_add, keys_add, values, valid_lengths)  # 输出 (2,3,7)。
    print("\n[加性注意力]")
    print("输出 Shape:", tuple(output_add.shape))
    print("权重 Shape:", tuple(additive.attention_weights.shape))
    assert output_add.shape == (batch_size, num_queries, 7)  # 输出查询数不变，宽度来自值。

    queries_dot = torch.randn(batch_size, num_queries, 6)  # 点积查询为 (2,3,6)。
    keys_dot = torch.randn(batch_size, num_keys, 6)  # 点积键也必须是 (2,4,6)。
    dot_product = ScaledDotProductAttention()  # 评分函数无可训练参数。
    output_dot = dot_product(queries_dot, keys_dot, values, valid_lengths)  # 输出 (2,3,7)。
    print("\n[缩放点积注意力]")
    print("输出 Shape:", tuple(output_dot.shape))
    print("被遮蔽位置最大权重:", dot_product.attention_weights[1, :, 2:].max().item())

    encoder_outputs = torch.randn(batch_size, num_keys, 8)  # 模拟编码器所有时间步 (2,4,8)。
    decoder_state = torch.randn(batch_size, 8)  # 模拟解码器当前状态 (2,8)。
    bahdanau_attention = AdditiveAttention(8, 8, 12)  # 用加性评分连接编码器和解码器。
    gru_cell = nn.GRUCell(input_size=8, hidden_size=8)  # 上下文作为本步 GRU 输入。
    next_state, context = bahdanau_decoder_step(  # 执行一个可微分的解码时间步。
        decoder_state, encoder_outputs, valid_lengths, bahdanau_attention, gru_cell
    )
    loss = next_state.square().mean() + context.square().mean()  # 构造标量损失验证计算图。
    loss.backward()  # 梯度应流过注意力评分器和 GRUCell。
    print("\n[Bahdanau 解码一步]")
    print("context / next_state Shape:", tuple(context.shape), tuple(next_state.shape))
    print("attention 参数有梯度:", all(p.grad is not None for p in bahdanau_attention.parameters()))

    nadaraya_watson_demo()  # 最后演示核回归视角。
    print("\n所有注意力评分测试通过。")


if __name__ == "__main__":  # 直接运行文件时执行，import 时不自动跑实验。
    main()
