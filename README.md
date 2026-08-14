# 复习动手学深度学习知识点

这是一套面向复习与实践的 PyTorch 白话精读笔记。内容不照抄教材，而是沿着“先讲清楚为什么，再解释数学与 Shape，最后把机制对应到可运行代码”的顺序整理。

![从注意力到深度学习工具的学习路线](assets/images/deep-learning-journey-ch10-16.png)

这里的“白话”不是少讲细节，而是每个正式小节都尽量用一个小输入把过程走到具体输出。图只辅助例子；完整程序放在各章代码目录，关键训练与算法步骤配中文注释。

整套笔记同时面向三种场景：新手第一次学习、三个月后快速恢复、面试前机制追问。[查看全仓“三个月后仍能看懂”标准](docs/three-month-review-standard.md)。章节与知识范围统一对照[《动手学深度学习》中文第二版](https://zh-v2.d2l.ai/)。

## 章节导航

| 章节 | 主要问题 | 笔记 | 完整代码 |
| --- | --- | --- | --- |
| 第三章：线性神经网络 | 模型怎样预测、计算误差并学会参数 | [阅读](docs/ch03-linear-neural-networks.md) | [打开](code/ch03/) |
| 第四章：多层感知机 | 非线性、过拟合与正则化 | [阅读](docs/ch04-multilayer-perceptrons.md) | [打开](code/ch04/) |
| 第五章：深度学习计算 | 怎样组织层、参数、设备与模型文件 | [阅读](docs/ch05-deep-learning-computation.md) | [打开](code/ch05/) |
| 第六章：卷积神经网络 | 卷积怎样利用图像的局部结构 | [阅读](docs/ch06-convolutional-neural-networks.md) | [打开](code/ch06/) |
| 第七章：现代卷积神经网络 | 经典 CNN 架构为什么这样演化 | [阅读](docs/ch07-modern-convolutional-neural-networks.md) | [打开](code/ch07/) |
| 第八章：循环神经网络 | 模型怎样携带并更新序列记忆 | [阅读](docs/ch08-recurrent-neural-networks.md) | [打开](code/ch08/) |
| 第九章：现代循环神经网络 | 门控记忆与编码器-解码器怎样工作 | [阅读](docs/ch09-modern-recurrent-neural-networks.md) | [打开](code/ch09/) |
| 第十章：注意力机制 | 模型怎样按查询选择性读取信息 | [阅读](docs/ch10-attention-mechanisms.md) | [打开](code/ch10/) |
| 第十一章：优化算法 | 梯度、动量、自适应步长和学习率怎样配合 | [阅读](docs/ch11-optimization-algorithms.md) | [打开](code/ch11/) |
| 第十二章：计算性能 | 编译、异步、多卡与参数服务器怎样提高效率 | [阅读](docs/ch12-computational-performance.md) | [打开](code/ch12/) |
| 第十三章：计算机视觉 | 增广、检测、分割、风格迁移与微调怎样落地 | [阅读](docs/ch13-computer-vision.md) | [打开](code/ch13/) |
| 第十四章：自然语言处理预训练 | 词向量、子词与 BERT 怎样学习语言表示 | [阅读](docs/ch14-nlp-pretraining.md) | [打开](code/ch14/) |
| 第十五章：自然语言处理应用 | 情感分类、自然语言推断与 BERT 微调 | [阅读](docs/ch15-nlp-applications.md) | [打开](code/ch15/) |
| 第十六章：深度学习工具 | Notebook、远程训练、硬件、复现与贡献 | [阅读](docs/ch16-tools-for-deep-learning.md) | [打开](code/ch16/) |

## Hot 100 算法迁移

出现经典算法或数据结构的章节，会安排真正相关的 [LeetCode 热题 100](https://leetcode.cn/studyplan/top-100-liked/)。目前共 24 题；章内注明题号、中文题名、官方直达链接，并给出白话推导、复杂度、易错点和逐行中文注释的完整答案。

[打开 24 题章节—算法学习目标总表](docs/leetcode-hot100-learning-map.md) · [打开本次新增 16 题完整解析](docs/leetcode-hot100-expanded-practice.md)

主动回忆、课程经典问答与面试八股已按章全量审阅；第 3～16 章现有 372 道折叠问答，其中 47 道集中整理 D2L 教材与课程高价值追问，并采用“结论、机制、工程影响、误区、追问”的深答标准。[查看审阅记录与各章题数](docs/active-recall-interview-audit.md) · [查看逐章长期复习覆盖审计](docs/three-month-review-audit.md)

模型图和数学公式另有页面显示审计：模型章节必须区分整网结构图与局部机制图，公式只使用当前页面可安全渲染的写法。[查看模型图与公式渲染审计](docs/model-and-formula-rendering-audit.md)

## 阅读方式

1. 先读每章开头的“一句话主线”，建立整体地图。
2. 跟着每节“小输入 → 逐步过程 → 具体输出”的例子手算一遍。
3. 再看机制图，弄清数据、状态或梯度怎样流动；图不是正文的替代品。
4. 遇到公式时，先看符号和 Shape，再看推导。
5. 正文重点解释代码在训练流程中的职责；完整程序统一放在 `code/`。
6. 运行代码后，先遮住答案完成章末主动回忆，不要只看损失是否下降。

每章至少包含 12 道折叠答案的主动回忆题，覆盖概念解释、Shape 推演、代码推演和故障诊断。详细规则见[笔记写作规范](NOTE_STYLE.md)。

## 运行环境

```bash
pip install -r requirements.txt
```

各章程序优先提供合成数据或 smoke test，因此可以先在不下载大型数据集的情况下检查 Shape、前向、反向和更新链路。第三章示例：

```bash
python code/ch03/linear_regression.py
python code/ch03/softmax_regression.py --smoke-test --implementation both --epochs 2
```

Hot 100 配套答案也能直接离线自测，例如：

```bash
python code/ch06/hot100_sliding_window_maximum.py
python code/ch10/hot100_top_k_frequent.py
python code/ch11/jump_game.py
python code/ch15/hot100_word_break.py
```

> 学习建议：先读解析，再运行简洁实现；确认结果后，回头阅读从零实现。这样更容易把“框架 API”和“底层训练逻辑”对上号。

## 内容边界

这些笔记用于解释与复习，不替代教材。章节脉络参考[《动手学深度学习》中文官方文档](https://zh-v2.d2l.ai/)，所有正文均为面向学习的原创转述；代码是可运行的教学实现，并以中文注释解释关键步骤。
