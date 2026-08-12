"""LeetCode Hot 100 #438：找到字符串中所有字母异位词。

原题：https://leetcode.cn/problems/find-all-anagrams-in-a-string/
练习入口：https://codefun2000.com/p/P4008
"""


class Solution:
    def findAnagrams(self, s: str, p: str) -> list[int]:
        window_size = len(p)
        if window_size > len(s):
            return []

        need = [0] * 26
        window = [0] * 26

        # 统计目标串 p 需要的字母频次。
        for char in p:
            need[ord(char) - ord("a")] += 1

        answer: list[int] = []
        for right, char in enumerate(s):
            # 新字符进入窗口右端。
            window[ord(char) - ord("a")] += 1

            # 当窗口长度超过 |p| 时，让最左字符离开窗口。
            if right >= window_size:
                left_char = s[right - window_size]
                window[ord(left_char) - ord("a")] -= 1

            # 只有固定长度窗口的频次完全一致，才是字母异位词。
            if right >= window_size - 1 and window == need:
                answer.append(right - window_size + 1)

        return answer


def run_tests() -> None:
    solver = Solution()
    assert solver.findAnagrams("cbaebabacd", "abc") == [0, 6]
    assert solver.findAnagrams("abab", "ab") == [0, 1, 2]
    assert solver.findAnagrams("a", "ab") == []
    print("#438 找到字符串中所有字母异位词：全部测试通过")


if __name__ == "__main__":
    run_tests()
