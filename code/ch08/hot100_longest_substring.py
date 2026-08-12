"""LeetCode Hot 100 #3：无重复字符的最长子串。

原题：https://leetcode.cn/problems/longest-substring-without-repeating-characters/
运行：python code/ch08/hot100_longest_substring.py
"""

from __future__ import annotations


class Solution:
    """用滑动窗口维护“窗口内没有重复字符”这一不变量。"""

    def lengthOfLongestSubstring(self, s: str) -> int:
        """返回不含重复字符的最长连续子串长度。"""
        # left 是当前合法窗口的左边界，窗口为 s[left:right+1]。
        left = 0
        # best 保存目前见过的最大合法窗口长度。
        best = 0
        # last_seen[字符] 保存该字符最近一次出现的下标。
        last_seen: dict[str, int] = {}

        # right 逐个扫描新字符，相当于序列状态随时间更新。
        for right, character in enumerate(s):
            # 只有字符上次出现在当前窗口内，才会破坏“不重复”条件。
            if character in last_seen and last_seen[character] >= left:
                # 左边界直接跳到重复字符旧位置的下一格，无需一格格移动。
                left = last_seen[character] + 1

            # 当前字符成为它自己的最新出现位置。
            last_seen[character] = right
            # 当前合法窗口长度是右边界减左边界再加一。
            current_length = right - left + 1
            # 用当前窗口更新历史最优长度。
            best = max(best, current_length)

        # 空字符串时循环不执行，best 自然保持 0。
        return best


def run_tests() -> None:
    """覆盖重复、全相同、空串和边界回退陷阱。"""
    # 创建 LeetCode 风格解题对象。
    solver = Solution()
    # abc 是最长合法连续窗口，长度为 3。
    assert solver.lengthOfLongestSubstring("abcabcbb") == 3
    # 全部字符相同，只能保留一个。
    assert solver.lengthOfLongestSubstring("bbbbb") == 1
    # wke 是连续子串；pwke 虽不重复但不是连续子串。
    assert solver.lengthOfLongestSubstring("pwwkew") == 3
    # 空串没有字符，答案为 0。
    assert solver.lengthOfLongestSubstring("") == 0
    # abba 用于验证 left 不能被旧位置错误地向左拉回。
    assert solver.lengthOfLongestSubstring("abba") == 2
    # 打印清晰的离线验收信息。
    print("#3 无重复字符的最长子串：全部测试通过")


if __name__ == "__main__":
    # 直接运行文件时执行自测。
    run_tests()
