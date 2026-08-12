"""LeetCode Hot 100 #560：和为 K 的子数组。

原题：https://leetcode.cn/problems/subarray-sum-equals-k/
练习：https://codefun2000.com/p/P4009
运行：python code/ch08/hot100_subarray_sum_equals_k.py
"""

from __future__ import annotations


class Solution:
    """用前缀和频次把“找连续区间”改成“找历史状态”。"""

    def subarraySum(self, nums: list[int], k: int) -> int:
        """返回元素和恰好为 k 的连续子数组数量。"""
        # prefix 表示从数组起点到当前位置的累计和。
        prefix = 0
        # frequency[s] 表示此前出现过多少次前缀和 s。
        frequency: dict[int, int] = {0: 1}
        # 前缀和 0 先出现一次，才能统计从下标 0 开始的合法区间。
        answer = 0

        # 按序读取每个数，像序列模型一样增量更新状态。
        for value in nums:
            # 加入当前值，得到新的右端点前缀和。
            prefix += value
            # 若历史前缀为 prefix-k，两者之差对应区间和 k。
            answer += frequency.get(prefix - k, 0)
            # 当前前缀进入历史，供后续右端点查询。
            frequency[prefix] = frequency.get(prefix, 0) + 1

        # 返回所有不同起止位置的合法连续区间数量。
        return answer


def run_tests() -> None:
    """覆盖正数、负数、零和从首位开始的区间。"""
    solver = Solution()
    assert solver.subarraySum([1, 1, 1], 2) == 2
    assert solver.subarraySum([1, 2, 3], 3) == 2
    assert solver.subarraySum([1, -1, 0], 0) == 3
    assert solver.subarraySum([], 0) == 0
    print("#560 和为 K 的子数组：全部测试通过")


if __name__ == "__main__":
    run_tests()
