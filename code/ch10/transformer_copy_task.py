"""第 10 章：用 PyTorch Transformer 完成离线“序列复制”小任务。

运行：python code/ch10/transformer_copy_task.py --epochs 8
只依赖 PyTorch；数据在内存中合成，不下载任何文件。
"""

from __future__ import annotations

import argparse
import math

import torch
from torch import nn


PAD_ID = 0  # padding 只负责补齐，不应参与损失。
BOS_ID = 1  # 解码器首输入，表示“开始生成”。
EOS_ID = 2  # 真实监督目标，表示“停止生成”。
FIRST_TOKEN_ID = 3  # 普通数据词元从 3 开始。


class PositionalEncoding(nn.Module):
    """把顺序信息加到词元嵌入中。"""

    def __init__(self, hidden_size: int, max_length: int) -> None:
        super().__init__()  # 初始化 nn.Module。
        positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)  # (T,1)。
        dims = torch.arange(0, hidden_size, 2, dtype=torch.float32)  # 偶数特征维。
        rates = torch.exp(-math.log(10000.0) * dims / hidden_size)  # 不同维使用不同频率。
        angles = positions * rates.unsqueeze(0)  # (T,ceil(D/2))。
        table = torch.zeros(max_length, hidden_size)  # (T,D)。
        table[:, 0::2] = torch.sin(angles)  # 偶数列编码为 sin。
        table[:, 1::2] = torch.cos(angles[:, : table[:, 1::2].shape[1]])  # 奇数列编码为 cos。
        self.register_buffer("table", table.unsqueeze(0), persistent=False)  # (1,T,D)，不是参数但会随设备移动。

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.table[:, : x.shape[1]].to(dtype=x.dtype)  # (B,T,D)+(1,T,D)。


class TinyTransformer(nn.Module):
    """最小但完整的编码器－解码器 Transformer。"""

    def __init__(self, vocab_size: int, hidden_size: int, num_heads: int, num_layers: int, max_length: int) -> None:
        super().__init__()  # 注册参数与子模块。
        self.hidden_size = hidden_size  # embedding 乘 sqrt(D) 时使用。
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=PAD_ID)  # (B,T) -> (B,T,D)。
        self.position = PositionalEncoding(hidden_size, max_length)  # 弥补注意力本身不含顺序的问题。
        self.transformer = nn.Transformer(  # 直接使用 PyTorch 的标准编码器－解码器。
            d_model=hidden_size,  # 每个 token 的特征宽度 D。
            nhead=num_heads,  # 把 D 分成 h 个注意力头。
            num_encoder_layers=num_layers,  # 编码器块堆叠层数。
            num_decoder_layers=num_layers,  # 解码器块堆叠层数。
            dim_feedforward=hidden_size * 2,  # 每位置前馈网络的中间宽度。
            dropout=0.0,  # 小型确定性演示关闭 dropout。
            batch_first=True,  # 输入输出统一为 (B,T,D)。
            norm_first=False,  # 使用经典 post-LN，并避免小示例产生嵌套张量警告。
        )
        self.transformer.encoder.enable_nested_tensor = False  # 小任务关闭实验性嵌套张量快路径。
        self.transformer.encoder.use_nested_tensor = False  # 当前 PyTorch 版本还缓存此开关，显式关闭以避免原型警告。
        self.output = nn.Linear(hidden_size, vocab_size)  # (B,T,D) -> (B,T,V)。

    def forward(self, source: torch.Tensor, decoder_inputs: torch.Tensor) -> torch.Tensor:
        source_padding_mask = source.eq(PAD_ID)  # (B,S)，True 表示编码器不可读取的 pad。
        target_padding_mask = decoder_inputs.eq(PAD_ID)  # (B,T)，True 表示解码器 pad。
        target_length = decoder_inputs.shape[1]  # 获取当前解码长度 T。
        causal_mask = torch.triu(  # (T,T)，True 表示禁止读取的未来位置。
            torch.ones(target_length, target_length, dtype=torch.bool, device=decoder_inputs.device),
            diagonal=1,
        )
        source_emb = self.embedding(source) * math.sqrt(self.hidden_size)  # (B,S,D)，放大到合适尺度。
        source_emb = self.position(source_emb)  # (B,S,D)，加入源位置。
        target_emb = self.embedding(decoder_inputs) * math.sqrt(self.hidden_size)  # (B,T,D)。
        target_emb = self.position(target_emb)  # (B,T,D)，加入目标位置。
        hidden = self.transformer(  # 建立完整计算图，尚未改变任何参数。
            src=source_emb,  # 编码器输入 (B,S,D)。
            tgt=target_emb,  # 解码器输入 (B,T,D)。
            tgt_mask=causal_mask,  # 禁止位置 t 查看未来目标。
            src_key_padding_mask=source_padding_mask,  # 编码器自注意力忽略源 pad。
            tgt_key_padding_mask=target_padding_mask,  # 解码器自注意力忽略目标 pad。
            memory_key_padding_mask=source_padding_mask,  # 交叉注意力也忽略源 pad。
        )
        return self.output(hidden)  # (B,T,V)，每个位置给出下一词分数。


def make_batch(
    batch_size: int,
    max_content_length: int,
    vocab_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """生成不同长度的复制样本：source=[正文,eos]，target=[bos,正文,eos]。"""
    lengths = torch.randint(2, max_content_length + 1, (batch_size,), device=device)  # (B,)。
    source = torch.full((batch_size, max_content_length + 1), PAD_ID, dtype=torch.long, device=device)  # (B,S)。
    target = torch.full((batch_size, max_content_length + 2), PAD_ID, dtype=torch.long, device=device)  # (B,T+1)。
    target[:, 0] = BOS_ID  # 每个目标序列都从 bos 开始。
    for row, length_tensor in enumerate(lengths):  # 逐样本写入不同长度，数据生成不参与模型性能瓶颈。
        length = int(length_tensor.item())  # 把标量张量转为 Python 整数。
        tokens = torch.randint(FIRST_TOKEN_ID, vocab_size, (length,), device=device)  # 采样正文词元。
        source[row, :length] = tokens  # 源正文写入前 length 个位置。
        source[row, length] = EOS_ID  # 源正文后追加 eos。
        target[row, 1 : length + 1] = tokens  # 目标要求原样复制源正文。
        target[row, length + 1] = EOS_ID  # 目标末尾追加可学习的停止符。
    return source, target  # source:(B,S)，target:(B,S+1)。


def train_model(model: TinyTransformer, epochs: int, steps_per_epoch: int, device: torch.device) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)  # Adam 更新所有可训练参数。
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)  # pad 不贡献损失与梯度。
    model.train()  # 启用训练模式；本例虽无 dropout，仍保持正确模板。

    for epoch in range(1, epochs + 1):  # 训练多个 epoch。
        running_loss = 0.0  # 累积当前 epoch 的标量损失。
        for _ in range(steps_per_epoch):  # 每步重新合成一批数据。
            source, target = make_batch(32, 5, 13, device)  # source:(32,6)，target:(32,7)。
            decoder_inputs = target[:, :-1]  # (B,T)，输入为 bos + 正文。
            decoder_targets = target[:, 1:]  # (B,T)，标签为正文 + eos。

            optimizer.zero_grad(set_to_none=True)  # 清掉上一步梯度，参数尚未改变。
            logits = model(source, decoder_inputs)  # 前向得到 (B,T,V)，建立计算图。
            loss = criterion(logits.reshape(-1, logits.shape[-1]), decoder_targets.reshape(-1))  # 标量平均损失。
            loss.backward()  # 计算嵌入、注意力、FFN 与输出层梯度。
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 限制偶发大梯度。
            optimizer.step()  # 这是本轮唯一真正改变参数的语句。
            running_loss += loss.item()  # detach 后记录 Python 数值，不保留计算图。

        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 4) == 0:  # 控制日志频率。
            print(f"epoch={epoch:02d} loss={running_loss / steps_per_epoch:.4f}")


@torch.inference_mode()
def greedy_decode(model: TinyTransformer, source: torch.Tensor, max_steps: int) -> torch.Tensor:
    model.eval()  # 关闭训练期随机行为。
    generated = torch.full((source.shape[0], 1), BOS_ID, dtype=torch.long, device=source.device)  # (B,1)。
    for _ in range(max_steps):  # 设置硬上限，防止模型忘记输出 eos 时无限循环。
        logits = model(source, generated)  # (B,current_T,V)。
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)  # 只取最后位置预测 (B,1)。
        generated = torch.cat((generated, next_token), dim=1)  # 把预测反馈为下一步输入。
        if torch.all(next_token.eq(EOS_ID)):  # 该小演示整批同时 eos 时提前结束。
            break
    return generated  # 包含开头 bos 和已生成词元。


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练一个离线 Transformer 复制任务")  # 创建命令行解析器。
    parser.add_argument("--epochs", type=int, default=25)  # 默认训练轮数可稳定学会示例复制。
    parser.add_argument("--steps-per-epoch", type=int, default=40)  # 每轮重新合成 40 个批次。
    return parser.parse_args()  # 返回用户参数。


def main() -> None:
    args = parse_args()  # 读取命令行参数。
    torch.manual_seed(19)  # 固定随机种子以便复现。
    torch.set_num_threads(1)  # 小张量在单线程下更快且更稳定。
    device = torch.device("cpu")  # 强制 CPU，保证离线机器可直接运行。
    model = TinyTransformer(vocab_size=13, hidden_size=32, num_heads=4, num_layers=1, max_length=16).to(device)  # 创建模型。

    source, target = make_batch(4, 5, 13, device)  # 先做一次 Shape smoke test。
    logits = model(source, target[:, :-1])  # (4,6,13)。
    print("source / target / logits Shape:", tuple(source.shape), tuple(target.shape), tuple(logits.shape))
    assert logits.shape[:2] == target[:, :-1].shape  # 每个解码输入位置都有一个词表分布。

    train_model(model, args.epochs, args.steps_per_epoch, device)  # 执行完整训练循环。

    fixed_source = torch.tensor([[4, 7, 9, EOS_ID, PAD_ID, PAD_ID]], device=device)  # 手工构造可读测试输入。
    prediction = greedy_decode(model, fixed_source, max_steps=7)  # 自回归生成，不再使用真实目标。
    print("测试源序列:", fixed_source.squeeze(0).tolist())
    print("模型生成:", prediction.squeeze(0).tolist())
    print("提示：短训练用于验证机制；增加 epochs 可提高复制稳定性。")


if __name__ == "__main__":  # 直接运行脚本时才启动训练。
    main()
