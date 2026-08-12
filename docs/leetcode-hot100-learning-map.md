# LeetCode Hot 100：与深度学习章节配套的算法迁移地图

> 官方题单：[LeetCode 热题 100](https://leetcode.cn/studyplan/top-100-liked/)<br>
> 每题同时给出 LeetCode 原题与 CodeFun 练习入口，题意只做原创摘要，不复制题库正文。

[返回总目录](../README.md) · [打开新增 16 题完整解析](leetcode-hot100-expanded-practice.md)

## 学习方法

1. 先打开原题，自己用一个小样例手推。
2. 写出状态、循环不变量或数据结构职责，再写代码。
3. 运行本仓库自测，解释每一处状态变化为何正确。
4. 最后回答“它与本章机制只在哪个抽象上相似，哪里并不相同”。

## 24 题学习目标总表

| 章节 | Hot 100 原题 | CodeFun | 学习目标 | 完整代码 |
| --- | --- | --- | --- | --- |
| 第 6 章 | [239. 滑动窗口最大值](https://leetcode.cn/problems/sliding-window-maximum/) | — | 单调队列与窗口不变量 | [代码](../code/ch06/hot100_sliding_window_maximum.py) |
| 第 6 章 | [48. 旋转图像](https://leetcode.cn/problems/rotate-image/) | [练习](https://codefun2000.com/p/P4040) | 原地矩阵变换与坐标映射 | [代码](../code/ch06/hot100_rotate_image.py) |
| 第 6 章 | [73. 矩阵置零](https://leetcode.cn/problems/set-matrix-zeroes/) | [练习](https://codefun2000.com/p/P4013) | 用首行首列压缩标记空间 | [代码](../code/ch06/hot100_set_matrix_zeroes.py) |
| 第 8 章 | [3. 无重复字符的最长子串](https://leetcode.cn/problems/longest-substring-without-repeating-characters/) | — | 滑动窗口状态摘要 | [代码](../code/ch08/hot100_longest_substring.py) |
| 第 8 章 | [560. 和为 K 的子数组](https://leetcode.cn/problems/subarray-sum-equals-k/) | [练习](https://codefun2000.com/p/P4009) | 前缀和与历史频次 | [代码](../code/ch08/hot100_subarray_sum_equals_k.py) |
| 第 8 章 | [739. 每日温度](https://leetcode.cn/problems/daily-temperatures/) | [练习](https://codefun2000.com/p/P4081) | 单调栈保存未决历史 | [代码](../code/ch08/hot100_daily_temperatures.py) |
| 第 9 章 | [215. 数组中的第 K 个最大元素](https://leetcode.cn/problems/kth-largest-element-in-an-array/) | — | Top-k 候选维护 | [代码](../code/ch09/hot100_kth_largest.py) |
| 第 9 章 | [1143. 最长公共子序列](https://leetcode.cn/problems/longest-common-subsequence/) | [练习](https://codefun2000.com/p/P4099) | 双序列动态规划 | [代码](../code/ch09/hot100_longest_common_subsequence.py) |
| 第 9 章 | [72. 编辑距离](https://leetcode.cn/problems/edit-distance/) | [练习](https://codefun2000.com/p/P4100) | 序列前缀状态与三种转移 | [代码](../code/ch09/hot100_edit_distance.py) |
| 第 10 章 | [347. 前 K 个高频元素](https://leetcode.cn/problems/top-k-frequent-elements/) | [练习](https://codefun2000.com/p/P4028) | 频次统计与桶式 Top-k | [代码](../code/ch10/hot100_top_k_frequent.py) |
| 第 11 章 | [55. 跳跃游戏](https://leetcode.cn/problems/jump-game/) | — | 可达性贪心不变量 | [代码](../code/ch11/jump_game.py) |
| 第 11 章 | [45. 跳跃游戏 II](https://leetcode.cn/problems/jump-game-ii/) | [练习](https://codefun2000.com/p/P4084) | 分层边界与最少跳数 | [代码](../code/ch11/hot100_jump_game_ii.py) |
| 第 11 章 | [121. 买卖股票的最佳时机](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock/) | [练习](https://codefun2000.com/p/P4029) | 单遍扫描维护最小前缀 | [代码](../code/ch11/hot100_best_time_stock.py) |
| 第 12 章 | [146. LRU 缓存](https://leetcode.cn/problems/lru-cache/) | — | 哈希表加双向链表 | [代码](../code/ch12/lru_cache.py) |
| 第 12 章 | [23. 合并 K 个升序链表](https://leetcode.cn/problems/merge-k-sorted-lists/) | [练习](https://codefun2000.com/p/P4051) | 小根堆维护多路候选 | [代码](../code/ch12/hot100_merge_k_sorted_lists.py) |
| 第 12 章 | [295. 数据流的中位数](https://leetcode.cn/problems/find-median-from-data-stream/) | [练习](https://codefun2000.com/p/P4083) | 双堆维护动态平衡 | [代码](../code/ch12/hot100_median_data_stream.py) |
| 第 13 章 | [200. 岛屿数量](https://leetcode.cn/problems/number-of-islands/) | — | 二维网格连通分量 | [代码](../code/ch13/number_of_islands.py) |
| 第 13 章 | [994. 腐烂的橘子](https://leetcode.cn/problems/rotting-oranges/) | [练习](https://codefun2000.com/p/P4020) | 多源 BFS 与分层时间 | [代码](../code/ch13/hot100_rotting_oranges.py) |
| 第 13 章 | [240. 搜索二维矩阵 II](https://leetcode.cn/problems/search-a-2d-matrix-ii/) | [练习](https://codefun2000.com/p/P4041) | 利用行列单调性消元 | [代码](../code/ch13/hot100_search_2d_matrix_ii.py) |
| 第 14 章 | [208. 实现 Trie](https://leetcode.cn/problems/implement-trie-prefix-tree/) | — | 前缀共享与终止标记 | [代码](../code/ch14/hot100_trie.py) |
| 第 14 章 | [49. 字母异位词分组](https://leetcode.cn/problems/group-anagrams/) | [练习](https://codefun2000.com/p/P4001) | 设计可哈希的等价类签名 | [代码](../code/ch14/hot100_group_anagrams.py) |
| 第 15 章 | [139. 单词拆分](https://leetcode.cn/problems/word-break/) | — | 前缀动态规划 | [代码](../code/ch15/hot100_word_break.py) |
| 第 15 章 | [438. 找到字符串中所有字母异位词](https://leetcode.cn/problems/find-all-anagrams-in-a-string/) | [练习](https://codefun2000.com/p/P4008) | 固定窗口频次更新 | [代码](../code/ch15/hot100_find_all_anagrams.py) |
| 第 16 章 | [71. 简化路径](https://leetcode.cn/problems/simplify-path/) | [练习](https://codefun2000.com/p/P4818) | 栈模拟路径状态 | [代码](../code/ch16/hot100_simplify_path.py) |

## 每章题数

| 章节 | 题数 | 本次新增 |
| --- | ---: | ---: |
| 第 6、8、9、11、12、13 章 | 各 3 题 | 各 2 题 |
| 第 10、16 章 | 各 1 题 | 各 1 题 |
| 第 14、15 章 | 各 2 题 | 各 1 题 |
| **合计** | **24 题** | **16 题** |

第 3～5、7 章不强行配题：这些章节当前内容没有足够直接的 Hot 100 算法映射。数量不是目标，能说清状态、不变量、复杂度和边界才算完成。
