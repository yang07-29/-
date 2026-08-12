# LeetCode Hot 100：与深度学习章节配套的算法迁移地图

> 官方题单：[LeetCode 热题 100](https://leetcode.cn/studyplan/top-100-liked/)<br>
> 使用原则：只选和章节机制有真实连接的题；题意在章内用原创语言概括，不复制题库正文。

[返回总目录](../README.md)

## 怎样使用这张表

不要在读完一章后立刻背答案。按下面四步做：

1. 先点“原题”读清输入、输出和约束。
2. 遮住章内答案，用一个最小样例手推状态变化。
3. 写出循环不变量或状态定义，再开始写代码。
4. 运行章内自测，最后用自己的话解释“为什么不是另一个算法”。

## 学习目标总表

| 完成 | 对应章节 | Hot 100 原题 | 本题要练会什么 | 与本章的连接 |
| --- | --- | --- | --- | --- |
| [ ] | [第 6 章：卷积神经网络](ch06-convolutional-neural-networks.md) | [239. 滑动窗口最大值](https://leetcode.cn/problems/sliding-window-maximum/) | 用单调队列保存“仍可能成为最大值”的候选；达到 $O(n)$ | 卷积、池化和滑动窗口都在局部邻域移动；算法题额外训练窗口状态怎样增量维护 |
| [ ] | [第 8 章：循环神经网络](ch08-recurrent-neural-networks.md) | [3. 无重复字符的最长子串](https://leetcode.cn/problems/longest-substring-without-repeating-characters/) | 定义左右边界和窗口不变量；字符重复时一次跳到合法位置 | 序列模型按时间更新状态；滑动窗口也只保留回答当前问题所需的历史摘要 |
| [ ] | [第 9 章：现代循环神经网络](ch09-modern-recurrent-neural-networks.md) | [215. 数组中的第 K 个最大元素](https://leetcode.cn/problems/kth-largest-element-in-an-array/) | 理解 Top-k、堆或快速选择，以及候选规模与计算量的权衡 | 束搜索每步保留高分候选；本题训练怎样找 Top-k，但它不包含序列累计概率 |
| [ ] | [第 11 章：优化算法](ch11-optimization-algorithms.md) | [55. 跳跃游戏](https://leetcode.cn/problems/jump-game/) | 用“当前最远可达位置”建立贪心不变量，并证明局部信息足够 | 两者都逐步更新状态；贪心可行性证明与梯度下降的连续优化不是同一种方法 |
| [ ] | [第 12 章：计算性能](ch12-computational-performance.md) | [146. LRU 缓存](https://leetcode.cn/problems/lru-cache/) | 用哈希表加双向链表实现平均 $O(1)$ 查询、更新和淘汰 | 硬件缓存利用局部性减少慢速访存；题目把“最近使用”策略变成可操作的数据结构 |
| [ ] | [第 13 章：计算机视觉](ch13-computer-vision.md) | [200. 岛屿数量](https://leetcode.cn/problems/number-of-islands/) | 用 DFS/BFS 遍历二维网格的四邻域连通分量 | 图像与分割标签也是二维网格；本题训练像素邻接与区域连通，但不是学习式语义分割 |
| [ ] | [第 14 章：自然语言处理预训练](ch14-nlp-pretraining.md) | [208. 实现 Trie（前缀树）](https://leetcode.cn/problems/implement-trie-prefix-tree/) | 让公共前缀共享路径，区分“完整词”与“存在此前缀” | 词元、子词和前缀处理都涉及字符串结构；Trie 是离散索引，不是词向量模型 |
| [ ] | [第 15 章：自然语言处理应用](ch15-nlp-applications.md) | [139. 单词拆分](https://leetcode.cn/problems/word-break/) | 定义 <code>dp[i]</code> 表示前缀可拆分，并从更短前缀转移 | 分词与序列决策需要处理多种边界；动态规划精确枚举可行切分，不等同于神经网络概率预测 |

## 建议验收标准

每题完成后，不以“通过提交”为唯一标准。至少应能回答：

- 状态或数据结构里保存的每个量代表什么？
- 每处理一个新元素，哪些状态必须改变，哪些可以丢掉？
- 为什么算法不会漏掉正确答案？
- 时间、空间复杂度怎样从循环和数据结构推出？
- 它与对应深度学习机制共享的只是哪个抽象？两者在哪一步开始不同？

## 暂不配题的章节

第 3～5、7、10、16 章目前不强行配题。这些章当然可以延伸到很多算法，但正文没有出现足够直接的 Hot 100 经典算法映射。与其用牵强类比增加记忆负担，不如先把模型机制和代码路径讲透。

后续若章节正文新增真正相关的算法，会同时更新本表、章内专题和可运行答案。
