# Hot 100 新增 16 题：白话推导与代码解析

> 原题统一来自 [LeetCode 热题 100](https://leetcode.cn/studyplan/top-100-liked/)。每题给出官方直达链接与 CodeFun 练习入口；这里只写原创摘要，不复制题面。

[返回 24 题总表](leetcode-hot100-learning-map.md) · [返回总目录](../README.md)

## 第 6 章：二维局部结构

### 48. 旋转图像

- 原题：[LeetCode 48](https://leetcode.cn/problems/rotate-image/) · [CodeFun P4040](https://codefun2000.com/p/P4040)
- 学习目标：把二维坐标变换拆成可验证的原地操作。
- 小例子：`[[1,2],[3,4]]` 先沿主对角线转置成 `[[1,3],[2,4]]`，再反转每一行得到 `[[3,1],[4,2]]`。
- 不变量：转置后 `(r,c)` 与 `(c,r)` 已交换；行反转只改变列方向。两步合起来正好顺时针 90°。
- 复杂度：时间 $O(n^2)$，除循环变量外额外空间 $O(1)$。
- 易错点：只转置会得到镜像；用 `matrix = rotated` 不算按题意原地修改。
- 完整代码：[hot100_rotate_image.py](../code/ch06/hot100_rotate_image.py)。重点看“只遍历主对角线上方”和“逐行反转”两段中文注释。

### 73. 矩阵置零

- 原题：[LeetCode 73](https://leetcode.cn/problems/set-matrix-zeroes/) · [CodeFun P4013](https://codefun2000.com/p/P4013)
- 学习目标：用矩阵自身的首行、首列充当标记数组，把额外空间降到 $O(1)$。
- 小例子：中间元素为 0 时，先把它所在行的首元素、所在列的首元素置 0；第二遍据此清空对应内部元素。
- 不变量：第一遍只记录“哪些行列应清零”，不能边发现边清整行，否则新产生的 0 会污染标记。
- 复杂度：时间 $O(mn)$，额外空间 $O(1)$。
- 易错点：`matrix[0][0]` 同时属于首行首列，必须另用两个布尔量保存它们原本是否含 0。
- 完整代码：[hot100_set_matrix_zeroes.py](../code/ch06/hot100_set_matrix_zeroes.py)。

## 第 8 章：序列历史摘要

### 560. 和为 K 的子数组

- 原题：[LeetCode 560](https://leetcode.cn/problems/subarray-sum-equals-k/) · [CodeFun P4009](https://codefun2000.com/p/P4009)
- 学习目标：把“枚举每个连续区间”改写为“查询过去出现过多少个目标前缀和”。
- 小例子：当前前缀和为 5、目标 `k=3`，只要过去出现过前缀和 2，二者之间的子数组和就是 3。
- 状态定义：哈希表 `frequency[p]` 表示当前下标之前，前缀和 `p` 出现次数；先查询 `prefix-k`，再登记当前 prefix。
- 复杂度：期望时间 $O(n)$，空间 $O(n)$。
- 易错点：初始必须放入 `{0:1}`，否则从下标 0 开始的合法子数组会漏掉；不能用滑动窗口处理含负数的通用情况。
- 完整代码：[hot100_subarray_sum_equals_k.py](../code/ch08/hot100_subarray_sum_equals_k.py)。

### 739. 每日温度

- 原题：[LeetCode 739](https://leetcode.cn/problems/daily-temperatures/) · [CodeFun P4081](https://codefun2000.com/p/P4081)
- 学习目标：理解“尚未等到更大值”的历史为何适合单调栈。
- 小例子：读到 75 时，栈顶 71 和 73 都得到答案；仍比 75 高的历史继续等待。
- 不变量：栈内保存尚未解决的下标，其温度从栈底到栈顶单调不增；新温度更高时持续弹栈结算。
- 复杂度：每个下标最多入栈、出栈一次，时间 $O(n)$，空间 $O(n)$。
- 易错点：栈里必须存下标，答案需要计算天数差；相同温度不算“更高”。
- 完整代码：[hot100_daily_temperatures.py](../code/ch08/hot100_daily_temperatures.py)。

## 第 9 章：双序列状态转移

### 1143. 最长公共子序列

- 原题：[LeetCode 1143](https://leetcode.cn/problems/longest-common-subsequence/) · [CodeFun P4099](https://codefun2000.com/p/P4099)
- 学习目标：用 `dp[i][j]` 表示两个前缀的最优答案，而不是贪心匹配眼前字符。
- 转移：末字符相同，答案是 `dp[i-1][j-1]+1`；不同则取删去任一侧末字符后的较大值。
- 小例子：`abc` 与 `ac` 的末尾 `c` 相同，于是问题缩为 `ab` 与 `a`，长度再加 1。
- 复杂度：时间、空间均为 $O(mn)$；只求长度时可滚动压缩空间。
- 易错点：子序列无需连续；初始化多留一行一列可统一处理空前缀。
- 完整代码：[hot100_longest_common_subsequence.py](../code/ch09/hot100_longest_common_subsequence.py)。

### 72. 编辑距离

- 原题：[LeetCode 72](https://leetcode.cn/problems/edit-distance/) · [CodeFun P4100](https://codefun2000.com/p/P4100)
- 学习目标：把插入、删除、替换分别对应到清楚的前缀状态。
- 状态：`dp[i][j]` 是前 `i` 个字符变成前 `j` 个字符的最少操作数；末字符不同取删除、插入、替换三者最小再加 1。
- 边界：空串变长度 `j` 的串需要 `j` 次插入；长度 `i` 的串变空串需 `i` 次删除。
- 复杂度：时间、空间 $O(mn)$。
- 易错点：不要把“替换”写成同时做一次删除和一次插入；索引 `i-1` 对应状态长度 `i`。
- 完整代码：[hot100_edit_distance.py](../code/ch09/hot100_edit_distance.py)。

## 第 10 章：Top-k 选择

### 347. 前 K 个高频元素

- 原题：[LeetCode 347](https://leetcode.cn/problems/top-k-frequent-elements/) · [CodeFun P4028](https://codefun2000.com/p/P4028)
- 学习目标：先统计，再按频次组织候选；区分“频率”与“元素值”。
- 小例子：频次 `{1:3, 2:2, 3:1}` 放入下标为 3、2、1 的桶，从高频桶向下收集前 k 个。
- 不变量：桶下标就是出现次数，因此逆序扫描天然按频率从高到低。
- 复杂度：统计与扫描总时间 $O(n)$，空间 $O(n)$。
- 易错点：题目允许多个等价顺序，测试不要强行要求固定组内次序。
- 完整代码：[hot100_top_k_frequent.py](../code/ch10/hot100_top_k_frequent.py)。注意它与注意力 Top-k 筛选只共享候选选择抽象，并不等于注意力计算。

## 第 11 章：可证明的贪心状态

### 45. 跳跃游戏 II

- 原题：[LeetCode 45](https://leetcode.cn/problems/jump-game-ii/) · [CodeFun P4084](https://codefun2000.com/p/P4084)
- 学习目标：把当前一次跳跃能覆盖的区间看成 BFS 的一层。
- 状态：`current_end` 是当前跳数可到的最右边界，`farthest` 是扫描这一层时下一层能达到的最远位置。
- 过程：扫描到 `current_end` 才把跳数加一，并令新边界等于 farthest；不是每个位置都加一。
- 复杂度：时间 $O(n)$，空间 $O(1)$。
- 易错点：这是离散可达性的贪心证明，不是“每次跳最远位置”也不是梯度下降。
- 完整代码：[hot100_jump_game_ii.py](../code/ch11/hot100_jump_game_ii.py)。

### 121. 买卖股票的最佳时机

- 原题：[LeetCode 121](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock/) · [CodeFun P4029](https://codefun2000.com/p/P4029)
- 学习目标：用最小历史价格概括所有可能买点。
- 不变量：处理今天价格前，`min_price` 是过去最低价；今天卖出的最好利润只能是 `today-min_price`。
- 小例子：价格 `[7,1,5]` 扫到 5 时，过去最低为 1，候选利润为 4。
- 复杂度：时间 $O(n)$，空间 $O(1)$。
- 易错点：必须先买后卖，不能拿未来最低价配过去最高价；全程下跌答案为 0。
- 完整代码：[hot100_best_time_stock.py](../code/ch11/hot100_best_time_stock.py)。

## 第 12 章：堆与在线数据结构

### 23. 合并 K 个升序链表

- 原题：[LeetCode 23](https://leetcode.cn/problems/merge-k-sorted-lists/) · [CodeFun P4051](https://codefun2000.com/p/P4051)
- 学习目标：堆里只保留每条链表当前的最小候选，实现多路归并。
- 不变量：堆顶一定是所有尚未输出节点中的全局最小值；弹出后只需把同链表下一节点入堆。
- 复杂度：总节点数为 $N$、链表数为 $k$ 时，时间 $O(N\log k)$，堆空间 $O(k)$。
- 易错点：值相同的节点不可直接比较，Python 堆元组中加入唯一序号作稳定破同值字段。
- 完整代码：[hot100_merge_k_sorted_lists.py](../code/ch12/hot100_merge_k_sorted_lists.py)，包含节点构造和输出辅助函数。

### 295. 数据流的中位数

- 原题：[LeetCode 295](https://leetcode.cn/problems/find-median-from-data-stream/) · [CodeFun P4083](https://codefun2000.com/p/P4083)
- 学习目标：用两个堆动态维护“较小一半”与“较大一半”。
- 不变量：大根堆 `lower` 的所有数不大于小根堆 `upper`；两堆大小相等或 lower 多一个。
- 查询：奇数个数取 lower 堆顶；偶数个数取两个堆顶平均。
- 复杂度：插入 $O(\log n)$，查询 $O(1)$，空间 $O(n)$。
- 易错点：Python 用负数模拟大根堆；每次插入后都要恢复大小平衡。
- 完整代码：[hot100_median_data_stream.py](../code/ch12/hot100_median_data_stream.py)。

## 第 13 章：二维网格搜索

### 994. 腐烂的橘子

- 原题：[LeetCode 994](https://leetcode.cn/problems/rotting-oranges/) · [CodeFun P4020](https://codefun2000.com/p/P4020)
- 学习目标：多源 BFS 如何模拟同时扩散，并把队列层数转换为分钟数。
- 初始化：所有腐烂位置一起进入第 0 层队列，同时统计 fresh 数量。
- 不变量：一次处理当前队列长度，代表完整的一分钟；新腐烂位置进入下一层。
- 复杂度：每格最多访问一次，时间、空间均为 $O(mn)$。
- 易错点：没有新鲜橘子时答案是 0；循环结束 fresh 仍非零说明有隔离区域，应返回 -1。
- 完整代码：[hot100_rotting_oranges.py](../code/ch13/hot100_rotting_oranges.py)。

### 240. 搜索二维矩阵 II

- 原题：[LeetCode 240](https://leetcode.cn/problems/search-a-2d-matrix-ii/) · [CodeFun P4041](https://codefun2000.com/p/P4041)
- 学习目标：从右上角利用“左边更小、下边更大”每步排除一行或一列。
- 过程：当前值太大就左移，太小就下移；这两种移动都不会丢失可能答案。
- 复杂度：最多移动 $m+n$ 次，时间 $O(m+n)$，空间 $O(1)$。
- 易错点：从左上角出发时，右和下都更大，无法唯一决定方向。
- 完整代码：[hot100_search_2d_matrix_ii.py](../code/ch13/hot100_search_2d_matrix_ii.py)。

## 第 14 章：离散文本签名

### 49. 字母异位词分组

- 原题：[LeetCode 49](https://leetcode.cn/problems/group-anagrams/) · [CodeFun P4001](https://codefun2000.com/p/P4001)
- 学习目标：为“字符组成相同”设计与顺序无关、可哈希的等价类签名。
- 过程：对每个小写词统计 26 个字符次数，计数元组相同的词进入同一组。
- 复杂度：总字符数为 $L$ 时，时间 $O(L)$，分组空间 $O(L)$。
- 易错点：列表不能作字典键，要转元组；输出分组顺序通常不重要，自测需做规范化比较。
- 完整代码：[hot100_group_anagrams.py](../code/ch14/hot100_group_anagrams.py)。它与词元统计相关，但不是词向量学习。

## 第 15 章：固定长度滑动窗口

### 438. 找到字符串中所有字母异位词

- 原题：[LeetCode 438](https://leetcode.cn/problems/find-all-anagrams-in-a-string/) · [CodeFun P4008](https://codefun2000.com/p/P4008)
- 学习目标：窗口右端加入、左端移除时增量维护频次，而不是每次重新统计。
- 不变量：从形成首个长度 `len(p)` 的窗口起，`window` 始终恰好描述当前固定长度片段。
- 复杂度：时间 $O(|s|+|p|)$；小写字母表固定时额外空间 $O(1)$。
- 易错点：只有窗口长度恰好等于目标长度才能比较；答案记录窗口左边界，不是右边界。
- 完整代码：[hot100_find_all_anagrams.py](../code/ch15/hot100_find_all_anagrams.py)。

## 第 16 章：栈与工程路径

### 71. 简化路径

- 原题：[LeetCode 71](https://leetcode.cn/problems/simplify-path/) · [CodeFun P4818](https://codefun2000.com/p/P4818)
- 学习目标：把目录切换规则变成栈状态机。
- 过程：空片段和 `.` 忽略，`..` 在栈非空时弹出，其他片段作为普通目录压栈，最后用 `/` 连接。
- 不变量：栈从底到顶始终是处理完当前前缀后的规范绝对路径。
- 复杂度：时间、空间均为 $O(n)$。
- 易错点：`...` 是普通目录名，不等于 `..`；在根目录执行 `..` 仍停在根目录。
- 完整代码：[hot100_simplify_path.py](../code/ch16/hot100_simplify_path.py)。

## 面试验收：每题必须能回答

- 状态或数据结构中每个量的准确含义是什么？
- 循环开始或结束时保持什么不变量，为什么不会漏解？
- 复杂度如何从“每个元素进出几次”或“堆大小”推出？
- 哪个边界样例最容易让代码失败？
- 如果面试官要求 ACM 输入输出，怎样保留核心函数并在外层解析输入？

题目选择遵循 `hw-ask` 与 `hw-interview` 的本地 Hot 100 索引；AI 岗手撕也可能直接考 Softmax、自注意力等模型组件，因此本仓库把算法题与章节八股分开训练，避免把二者混成同一种能力。
