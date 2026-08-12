"""第 10 章：多头自注意力、因果掩码和正弦位置编码。

运行：python code/ch10/multihead_self_attention.py
只依赖 PyTorch，不下载数据。
"""

from __future__ import annotations

import math

import torch
from torch import nn


def masked_softmax(scores: torch.Tensor, valid_lengths: torch.Tensor | None) -> torch.Tensor:
    """对 (B,Q,K) 的键轴做 padding 或逐查询因果遮蔽。"""
    if valid_lengths is None:  # 无遮蔽时走最直接的路径。
        return torch.softmax(scores, dim=-1)  # (B,Q,K)。
    batch_size, num_queries, num_keys = scores.shape  # 拆开三条语义轴。
    if valid_lengths.ndim == 1:  # (B,) 表示 padding 长度。
        valid_lengths = valid_lengths[:, None].expand(batch_size, num_queries)  # -> (B,Q)。
    positions = torch.arange(num_keys, device=scores.device).view(1, 1, -1)  # (1,1,K)。
    mask = positions < valid_lengths.unsqueeze(-1)  # (B,Q,K)。
    return torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)  # 无效键权重为 0。


def transpose_qkv(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """把头轴并入批量轴：(B,T,D) -> (B*h,T,D/h)。"""
    batch_size, num_steps, hidden_size = x.shape  # 记录原始 Shape。
    if hidden_size % num_heads != 0:  # 每个头必须分到整数个特征。
        raise ValueError("hidden_size 必须能被 num_heads 整除")
    head_size = hidden_size // num_heads  # 每头宽度 d_h。
    x = x.reshape(batch_size, num_steps, num_heads, head_size)  # (B,T,h,d_h)。
    x = x.permute(0, 2, 1, 3)  # (B,h,T,d_h)，把头轴移到序列轴前。
    return x.reshape(batch_size * num_heads, num_steps, head_size)  # (B*h,T,d_h)。


def transpose_output(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """撤销头轴并批：(B*h,T,d_h) -> (B,T,D)。"""
    batch_heads, num_steps, head_size = x.shape  # 读取合并后的 Shape。
    if batch_heads % num_heads != 0:  # 防止错误的头数静默重排数据。
        raise ValueError("合并批量维必须能被 num_heads 整除")
    batch_size = batch_heads // num_heads  # 恢复原批量大小 B。
    x = x.reshape(batch_size, num_heads, num_steps, head_size)  # (B,h,T,d_h)。
    x = x.permute(0, 2, 1, 3)  # (B,T,h,d_h)。
    return x.reshape(batch_size, num_steps, num_heads * head_size)  # (B,T,D)。


class MultiHeadAttention(nn.Module):
    """从线性投影到分头、注意力、拼接与输出投影的完整实现。"""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()  # 注册所有投影参数。
        if hidden_size % num_heads != 0:  # 构造时尽早检查整除关系。
            raise ValueError("hidden_size 必须能被 num_heads 整除")
        self.num_heads = num_heads  # 保存头数，forward 中反复使用。
        self.query_proj = nn.Linear(hidden_size, hidden_size, bias=False)  # D -> D。
        self.key_proj = nn.Linear(hidden_size, hidden_size, bias=False)  # D -> D。
        self.value_proj = nn.Linear(hidden_size, hidden_size, bias=False)  # D -> D。
        self.output_proj = nn.Linear(hidden_size, hidden_size, bias=False)  # 拼头后再混合各头。
        self.dropout = nn.Dropout(dropout)  # 只作用在注意力权重上。
        self.attention_weights: torch.Tensor | None = None  # 保存为 (B,h,Q,K)。

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        valid_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = queries.shape[0]  # 保留 B 以便恢复注意力权重 Shape。
        q = transpose_qkv(self.query_proj(queries), self.num_heads)  # (B*h,Q,d_h)。
        k = transpose_qkv(self.key_proj(keys), self.num_heads)  # (B*h,K,d_h)。
        v = transpose_qkv(self.value_proj(values), self.num_heads)  # (B*h,K,d_h)。

        if valid_lengths is not None:  # 每个头应使用同一个样本/查询有效长度。
            valid_lengths = valid_lengths.repeat_interleave(self.num_heads, dim=0)  # B -> B*h。

        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(q.shape[-1])  # (B*h,Q,K)。
        weights = masked_softmax(scores, valid_lengths)  # (B*h,Q,K)。
        self.attention_weights = weights.reshape(batch_size, self.num_heads, *weights.shape[1:])  # (B,h,Q,K)。
        head_outputs = torch.bmm(self.dropout(weights), v)  # (B*h,Q,d_h)。
        concatenated = transpose_output(head_outputs, self.num_heads)  # (B,Q,D)。
        return self.output_proj(concatenated)  # (B,Q,D)，允许头间信息混合。


class PositionalEncoding(nn.Module):
    """正弦/余弦位置编码，无需训练即可扩展到预设最大长度。"""

    def __init__(self, hidden_size: int, max_length: int = 512) -> None:
        super().__init__()  # 注册缓冲区而非参数。
        positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)  # (T_max,1)。
        even_dims = torch.arange(0, hidden_size, 2, dtype=torch.float32)  # 偶数维索引。
        frequencies = torch.exp(-math.log(10000.0) * even_dims / hidden_size)  # (ceil(D/2),)。
        angles = positions * frequencies.unsqueeze(0)  # (T_max,ceil(D/2))。
        encoding = torch.zeros(max_length, hidden_size)  # (T_max,D)。
        encoding[:, 0::2] = torch.sin(angles)  # 偶数维使用 sin。
        encoding[:, 1::2] = torch.cos(angles[:, : encoding[:, 1::2].shape[1]])  # 奇数维使用 cos。
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)  # (1,T_max,D)，随模型迁移设备。

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] > self.encoding.shape[1]:  # 防止越过预计算最大长度。
            raise ValueError("序列长度超过位置编码 max_length")
        return x + self.encoding[:, : x.shape[1]].to(dtype=x.dtype)  # (B,T,D)+(1,T,D)。


class SelfAttentionBlock(nn.Module):
    """预规范化自注意力块：LayerNorm -> MHA -> 残差。"""

    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()  # 注册子模块。
        self.norm = nn.LayerNorm(hidden_size)  # 每个 token 独立规范化最后一维。
        self.attention = MultiHeadAttention(hidden_size, num_heads, dropout=0.0)  # 自注意力核心。

    def forward(self, x: torch.Tensor, valid_lengths: torch.Tensor | None = None) -> torch.Tensor:
        normalized = self.norm(x)  # (B,T,D) -> (B,T,D)，不跨 token 混合。
        attended = self.attention(normalized, normalized, normalized, valid_lengths)  # Q=K=V 即自注意力。
        return x + attended  # 残差保留原表示并提供短梯度路径。


def main() -> None:
    torch.manual_seed(11)  # 固定参数与输入，方便复现。
    batch_size, num_steps, hidden_size, num_heads = 2, 5, 8, 2  # D=8 可被 h=2 整除。
    token_embeddings = torch.randn(batch_size, num_steps, hidden_size, requires_grad=True)  # (2,5,8)。
    position = PositionalEncoding(hidden_size, max_length=16)  # 创建无训练参数的位置编码。
    positioned = position(token_embeddings)  # (2,5,8)，同词不同位置获得不同向量。

    valid_lengths = torch.tensor([5, 3])  # 第二个样本最后两个 token 是 padding。
    block = SelfAttentionBlock(hidden_size, num_heads)  # 创建多头自注意力残差块。
    output = block(positioned, valid_lengths)  # (2,5,8)。
    loss = output.square().mean()  # 构造标量损失。
    loss.backward()  # 梯度流经残差、注意力投影和 token embedding。

    print("[padding 自注意力]")
    print("输入 / 输出 Shape:", tuple(token_embeddings.shape), tuple(output.shape))
    print("注意力权重 Shape:", tuple(block.attention.attention_weights.shape))
    print("第二个样本 padding 键最大权重:", block.attention.attention_weights[1, :, :, 3:].max().item())
    print("输入梯度范数:", token_embeddings.grad.norm().item())
    assert output.shape == token_embeddings.shape  # 自注意力块保持序列长度和隐藏宽度。
    assert block.attention.attention_weights[1, :, :, 3:].max().item() == 0.0  # padding 键不可被读取。

    causal_lengths = torch.arange(1, num_steps + 1).repeat(batch_size, 1)  # (B,T)，第 t 行只能看前 t 个键。
    causal_attention = MultiHeadAttention(hidden_size, num_heads)  # 单独实例用于因果测试。
    causal_output = causal_attention(positioned.detach(), positioned.detach(), positioned.detach(), causal_lengths)  # (B,T,D)。
    causal_weights = causal_attention.attention_weights  # (B,h,T,T)。
    future_mask = torch.triu(torch.ones(num_steps, num_steps, dtype=torch.bool), diagonal=1)  # 上三角表示未来。
    future_weights = causal_weights.masked_select(future_mask.view(1, 1, num_steps, num_steps))  # 取出所有未来权重。

    print("\n[因果自注意力]")
    print("输出 Shape:", tuple(causal_output.shape))
    print("未来位置最大权重:", future_weights.max().item())
    assert future_weights.max().item() == 0.0  # 任何查询都不能读取右侧未来 token。

    first_token = torch.zeros(1, 1, hidden_size)  # 同一个零向量放在不同位置。
    repeated = first_token.expand(1, 4, hidden_size).clone()  # (1,4,8)，没有位置时四行完全相同。
    encoded = position(repeated)  # 加位置后四行不同。
    print("\n[位置编码]")
    print("位置 0 与位置 1 是否相同:", torch.allclose(encoded[:, 0], encoded[:, 1]))
    print("位置编码是 parameter 吗:", any(name == "encoding" for name, _ in position.named_parameters()))
    print("\n多头、自注意力、因果遮蔽与位置编码测试通过。")


if __name__ == "__main__":  # 直接执行脚本时运行演示。
    main()
