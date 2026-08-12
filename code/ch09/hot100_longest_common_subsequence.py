"""LeetCode Hot 100 #1143：最长公共子序列。

原题：https://leetcode.cn/problems/longest-common-subsequence/
练习：https://codefun2000.com/p/P4099
运行：python code/ch09/hot100_longest_common_subsequence.py
"""

from __future__ import annotations


class Solution:
    """用二维动态规划比较两个序列前缀。"""

    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        """返回两个字符串最长公共子序列的长度。"""
        # dp[i][j] 表示 text1 前 i 个字符与 text2 前 j 个字符的答案。
        dp = [[0] * (len(text2) + 1) for _ in range(len(text1) + 1)]

        # i、j 从 1 开始，让第 0 行和第 0 列自然表示空前缀。
        for i in range(1, len(text1) + 1):
            for j in range(1, len(text2) + 1):
                # 当前两个末尾字符相同，可以接在更短前缀的公共子序列后。
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                # 末尾字符不同，至少丢掉其中一个，再取两种情况较大值。
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        # 右下角覆盖两个完整字符串。
        return dp[-1][-1]


def run_tests() -> None:
    """覆盖部分匹配、完全相同、无匹配和空串。"""
    solver = Solution()
    assert solver.longestCommonSubsequence("abcde", "ace") == 3
    assert solver.longestCommonSubsequence("abc", "abc") == 3
    assert solver.longestCommonSubsequence("abc", "def") == 0
    assert solver.longestCommonSubsequence("", "abc") == 0
    print("#1143 最长公共子序列：全部测试通过")


if __name__ == "__main__":
    run_tests()
