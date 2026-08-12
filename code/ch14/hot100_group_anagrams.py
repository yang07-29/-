"""LeetCode Hot 100 #49：字母异位词分组。

原题：https://leetcode.cn/problems/group-anagrams/
练习入口：https://codefun2000.com/p/P4001
"""

from collections import defaultdict


class Solution:
    def groupAnagrams(self, strs: list[str]) -> list[list[str]]:
        # 键是 26 个字母的计数；值是拥有同一计数的原字符串列表。
        groups: dict[tuple[int, ...], list[str]] = defaultdict(list)

        for word in strs:
            counts = [0] * 26
            for char in word:
                # ord(char) - ord('a') 把 a..z 映射到 0..25。
                counts[ord(char) - ord("a")] += 1
            # 列表不可哈希，转成元组后才能作为字典键。
            signature = tuple(counts)
            groups[signature].append(word)

        # 题目不要求组间顺序，直接返回全部分组即可。
        return list(groups.values())


def normalized(groups: list[list[str]]) -> list[list[str]]:
    """只用于自测：消除题目不关心的组内、组间顺序。"""
    return sorted(sorted(group) for group in groups)


def run_tests() -> None:
    solver = Solution()
    actual = solver.groupAnagrams(["eat", "tea", "tan", "ate", "nat", "bat"])
    expected = [["bat"], ["nat", "tan"], ["ate", "eat", "tea"]]
    assert normalized(actual) == normalized(expected)
    assert normalized(solver.groupAnagrams([""])) == [[""]]
    print("#49 字母异位词分组：全部测试通过")


if __name__ == "__main__":
    run_tests()
