"""LeetCode Hot 100 #72：编辑距离。

原题：https://leetcode.cn/problems/edit-distance/
练习：https://codefun2000.com/p/P4100
运行：python code/ch09/hot100_edit_distance.py
"""

from __future__ import annotations


class Solution:
    """用动态规划计算插入、删除、替换的最少次数。"""

    def minDistance(self, word1: str, word2: str) -> int:
        """返回把 word1 变成 word2 所需的最少编辑次数。"""
        # dp[i][j] 表示 word1 前 i 个字符变成 word2 前 j 个字符的代价。
        dp = [[0] * (len(word2) + 1) for _ in range(len(word1) + 1)]
        # 非空前缀变为空串，只能连续删除 i 次。
        for i in range(len(word1) + 1):
            dp[i][0] = i
        # 空串变成非空前缀，只能连续插入 j 次。
        for j in range(len(word2) + 1):
            dp[0][j] = j

        # 从短前缀逐步扩展到完整字符串。
        for i in range(1, len(word1) + 1):
            for j in range(1, len(word2) + 1):
                # 末尾字符相同，无需新操作，直接继承左上角。
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    # 删除 word1 末尾、插入 word2 末尾、替换末尾三选一。
                    delete = dp[i - 1][j]
                    insert = dp[i][j - 1]
                    replace = dp[i - 1][j - 1]
                    # 当前执行一次操作，再接此前最省的方案。
                    dp[i][j] = 1 + min(delete, insert, replace)

        # 完整前缀对应最终答案。
        return dp[-1][-1]


def run_tests() -> None:
    """覆盖组合编辑、空串、相同字符串和纯替换。"""
    solver = Solution()
    assert solver.minDistance("horse", "ros") == 3
    assert solver.minDistance("intention", "execution") == 5
    assert solver.minDistance("", "abc") == 3
    assert solver.minDistance("same", "same") == 0
    print("#72 编辑距离：全部测试通过")


if __name__ == "__main__":
    run_tests()
