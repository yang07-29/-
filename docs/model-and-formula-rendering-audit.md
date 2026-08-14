# 模型结构图与数学公式渲染审计

[返回总目录](../README.md) · [查看第 7 章](ch07-modern-convolutional-neural-networks.md) · [查看写作规范](../NOTE_STYLE.md)

## 红色报错到底是什么

页面中的红色提示不是 Python 代码报错，也不是注意力公式本身推导错误。原因是 Markdown 页面使用的数学渲染器禁止了一个命名函数宏，导致整段公式无法转换成数学排版，下面的公式源码才会以灰色原文显示。

处理方式是把命名函数统一写成渲染器支持的直立文本。例如：

$$
\mathrm{head}_h
=\mathrm{Attention}(QW_h^Q,KW_h^K,VW_h^V),
$$

$$
\mathrm{MHA}(Q,K,V)
=\mathrm{Concat}(\mathrm{head}_1,\ldots,\mathrm{head}_H)W^O.
$$

这只是数学排版写法变化，不改变公式含义。`head` 表示一个注意力头，`Concat` 表示沿头的特征维拼接，$W^O$ 再把拼接结果做一次输出投影。

## 本轮公式修复范围

- 全仓替换 11 个 Markdown 文件中的 39 处禁用命名宏；
- 覆盖 Linear、ReLU、PPL、BLEU、softmax、Attention、MHA、Concat、LayerNorm、IoU、cos 等公式；
- 检查独立公式定界符、Markdown 代码围栏和 `details` 折叠标签是否成对；
- 检查公式没有落入代码围栏，也没有继续出现禁用宏。

代码名仍使用反引号，例如 `nn.MultiheadAttention`；数学符号才放在 `$...$` 或独立公式中。二者不再混在一起依赖颜色区分。

## 模型图盘点结果

| 章节 | 模型/结构 | 整体结构是否可追踪 | 本轮处理 |
| --- | --- | --- | --- |
| 第 6 章 | LeNet | 输入到 logits 的逐层 Shape 已完整 | 保留原完整图 |
| 第 7 章 | AlexNet、VGG、NiN、GoogLeNet、ResNet、DenseNet | 原先多为局部机制图，整网比较不足 | 新增六模型全景图与 VGG–NiN 逐层对照图 |
| 第 8 章 | RNN | 时间状态、从零实现和简洁实现可追踪 | 保留原状态图与 Shape 图 |
| 第 9 章 | GRU、LSTM、深层/双向 RNN、seq2seq | 门控单元和编码器—解码器路径可追踪 | 保留原八张结构/例子图 |
| 第 10 章 | 多头注意力、Transformer | 分头、拼接、编码器和因果解码器可追踪 | 修复公式渲染，保留完整 Transformer 图 |
| 第 14 章 | word2vec、BERT | 训练目标、输入嵌入和共享编码器可追踪 | 保留原十张结构/过程图 |
| 第 15 章 | BiRNN、textCNN、可分解注意力、BERT 下游头 | 输入到任务头的数据流可追踪 | 保留原七张结构/例子图 |

## 第 7 章新增图怎样看

1. 先看[六类现代 CNN 完整结构总览](../assets/visuals/ch07/7-0-cnn-model-atlas.svg)，只回答每个模型的数据从哪里进、在哪里下采样、怎样合并、如何输出；
2. 再看[VGG 与 NiN 完整架构对照](../assets/visuals/ch07/7-3-vgg-vs-nin-architecture.svg)，重点比较 VGG 的 `Flatten + FC` 与 NiN 的“类别通道 + GAP”；
3. 最后回到各小节原有小图手算 ReLU、小卷积、GAP、拼接、BN、残差相加和 DenseNet 通道增长。

两张新增图根据 [D2L 第 7 章](https://zh-v2.d2l.ai/chapter_convolutional-modern/index.html)、[VGG 小节](https://zh-v2.d2l.ai/chapter_convolutional-modern/vgg.html)和 [NiN 小节](https://zh-v2.d2l.ai/chapter_convolutional-modern/nin.html)做原创重绘，不直接使用课程截图。

## 自动检查结果

- 禁用数学宏：0；未闭合独立公式：0；公式误入代码围栏：0；
- 未闭合 Markdown 代码围栏：0；未闭合折叠答案：0；本地链接缺失：0；
- SVG 共 115 张，XML 解析错误：0；两张新增结构图另用浏览器实际渲染截图检查过；
- `code/` 下全部 Python 文件通过 `compileall` 语法检查。
