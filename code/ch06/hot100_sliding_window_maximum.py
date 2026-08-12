"""LeetCode Hot 100 #239：滑动窗口最大值。

原题：https://leetcode.cn/problems/sliding-window-maximum/
运行：python code/ch06/hot100_sliding_window_maximum.py
"""

from __future__ import annotations

from collections import deque


class Solution:
    """使用单调队列保存仍可能成为窗口最大值的下标。"""

    def maxSlidingWindow(self, nums: list[int], k: int) -> list[int]:
        """返回每个长度为 k 的连续窗口中的最大值。"""
        # 教学脚本主动检查参数；LeetCode 给定的数据本身满足这些约束。
        if not nums or k <= 0 or k > len(nums):
            raise ValueError("nums 必须非空，且 1 <= k <= len(nums)")

        # 队列保存下标，而不是数值：这样才能判断队首是否已经离开窗口。
        candidates: deque[int] = deque()
        # answers 依次保存每个完整窗口的最大值。
        answers: list[int] = []

        # right 是新元素进入窗口的位置，value 是它的数值。
        for right, value in enumerate(nums):
            # 当前窗口左边界；窗口未满时这个值可能为负数。
            left = right - k + 1

            # 队首下标若小于 left，就已经不在当前窗口，必须移除。
            if candidates and candidates[0] < left:
                candidates.popleft()

            # 队尾对应值若不大于新值，以后不可能先于新值成为最大值。
            while candidates and nums[candidates[-1]] <= value:
                # 删除已经被新元素“压住”的旧候选。
                candidates.pop()

            # 新元素至少是未来某个窗口的候选，因此把它的下标放入队尾。
            candidates.append(right)

            # right >= k-1 时，第一个长度为 k 的窗口已经形成。
            if right >= k - 1:
                # 队列从大到小排列，队首值就是当前窗口最大值。
                answers.append(nums[candidates[0]])

        # 返回 n-k+1 个窗口答案。
        return answers


def run_tests() -> None:
    """用典型、重复值和边界窗口做离线自测。"""
    # 创建与 LeetCode 提交相同接口的解题对象。
    solver = Solution()
    # 官方典型结构：最大值会在窗口内停留，也会离开窗口。
    assert solver.maxSlidingWindow([1, 3, -1, -3, 5, 3, 6, 7], 3) == [3, 3, 5, 5, 6, 7]
    # k=1 时每个元素自己就是一个窗口。
    assert solver.maxSlidingWindow([4, 2, 9], 1) == [4, 2, 9]
    # k 等于数组长度时只有一个答案。
    assert solver.maxSlidingWindow([4, 2, 9], 3) == [9]
    # 重复最大值用于验证弹出旧相等值后答案仍正确。
    assert solver.maxSlidingWindow([2, 2, 2, 1], 2) == [2, 2, 2]
    # 所有断言通过后打印明确结果。
    print("#239 滑动窗口最大值：全部测试通过")


if __name__ == "__main__":
    # 直接运行文件时执行自测；提交 LeetCode 时只保留 Solution 类也可以。
    run_tests()
