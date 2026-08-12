"""LeetCode Hot 100 #347：前 K 个高频元素。

原题：https://leetcode.cn/problems/top-k-frequent-elements/
练习：https://codefun2000.com/p/P4028
运行：python code/ch10/hot100_top_k_frequent.py
"""

from __future__ import annotations

from collections import Counter


class Solution:
    """用按频次分桶的方法找出 Top-k。"""

    def topKFrequent(self, nums: list[int], k: int) -> list[int]:
        """返回出现频次最高的 k 个不同元素。"""
        # 统计每个不同元素的出现次数。
        frequency = Counter(nums)
        # k 必须能从不同元素中选出。
        if k <= 0 or k > len(frequency):
            raise ValueError("k 必须在 1 到不同元素个数之间")
        # 一个元素最高出现 len(nums) 次，因此创建 n+1 个频次桶。
        buckets: list[list[int]] = [[] for _ in range(len(nums) + 1)]
        # 把元素放入以“出现次数”为下标的桶。
        for value, count in frequency.items():
            buckets[count].append(value)

        # 从高频桶向低频桶收集答案。
        answer: list[int] = []
        for count in range(len(buckets) - 1, 0, -1):
            # 同频元素次序不限，全部加入候选。
            for value in buckets[count]:
                answer.append(value)
                # 收集满 k 个立即结束，避免无用遍历。
                if len(answer) == k:
                    return answer
        # 合法输入理论上一定提前返回。
        raise RuntimeError("未能收集足够元素")


def run_tests() -> None:
    """覆盖普通频次、单元素和负数。"""
    solver = Solution()
    assert set(solver.topKFrequent([1, 1, 1, 2, 2, 3], 2)) == {1, 2}
    assert solver.topKFrequent([1], 1) == [1]
    assert set(solver.topKFrequent([-1, -1, 2, 2, 2, 3], 2)) == {-1, 2}
    print("#347 前 K 个高频元素：全部测试通过")


if __name__ == "__main__":
    run_tests()
