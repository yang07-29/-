"""LeetCode Hot 100 #215：数组中的第 K 个最大元素。

原题：https://leetcode.cn/problems/kth-largest-element-in-an-array/
运行：python code/ch09/hot100_kth_largest.py
"""

from __future__ import annotations

import random


class Solution:
    """使用三路随机快速选择，只继续搜索包含目标排名的一侧。"""

    def findKthLargest(self, nums: list[int], k: int) -> int:
        """返回按降序排列后的第 k 个元素，不要求元素互不相同。"""
        # LeetCode 输入满足约束；教学版本仍主动拦截无效排名。
        if not nums or k <= 0 or k > len(nums):
            raise ValueError("nums 必须非空，且 1 <= k <= len(nums)")

        # 复制数组，避免为了分区修改调用者传入的原列表。
        values = nums.copy()
        # 降序第 k 大，等价于升序下标 n-k。
        target = len(values) - k
        # 当前只需在闭区间 [low, high] 内继续查找。
        low, high = 0, len(values) - 1
        # 独立固定随机源让示例可复现，同时避免总选固定端点。
        generator = random.Random(2026)

        # 每次分区后至少排除一侧，直到目标落入等于枢轴的区域。
        while low <= high:
            # 从当前区间随机选择枢轴值；随机化带来期望 O(n) 时间。
            pivot = values[generator.randint(low, high)]
            # [low, less) 小于 pivot，[less, scan) 等于 pivot。
            less = low
            scan = low
            # (greater, high] 大于 pivot，未处理区间为 [scan, greater]。
            greater = high

            # 三路分区一次处理重复值，避免大量相等元素时反复搜索。
            while scan <= greater:
                # 当前值小于枢轴，交换到左侧“小于区”。
                if values[scan] < pivot:
                    values[less], values[scan] = values[scan], values[less]
                    # 小于区和已扫描区都向右扩展一格。
                    less += 1
                    scan += 1
                # 当前值大于枢轴，交换到右侧“大于区”。
                elif values[scan] > pivot:
                    values[scan], values[greater] = values[greater], values[scan]
                    # 新换到 scan 的值尚未检查，所以 scan 不前进。
                    greater -= 1
                # 当前值等于枢轴，直接扩展中间“等于区”。
                else:
                    scan += 1

            # 目标下标在小于区，只保留左侧继续选择。
            if target < less:
                high = less - 1
            # 目标下标在大于区，只保留右侧继续选择。
            elif target > greater:
                low = greater + 1
            # 目标落在等于区，答案就是 pivot。
            else:
                return pivot

        # 合法输入理论上一定在循环内返回；此处保护不变量被意外破坏。
        raise RuntimeError("快速选择未找到目标，请检查分区不变量")


def run_tests() -> None:
    """覆盖普通数组、重复值、负数和边界排名。"""
    # 创建 LeetCode 风格解题对象。
    solver = Solution()
    # 降序为 6,5,4,3,2,1，第 2 大是 5。
    assert solver.findKthLargest([3, 2, 1, 5, 6, 4], 2) == 5
    # 重复元素也参与排名，第 4 大是 4。
    assert solver.findKthLargest([3, 2, 3, 1, 2, 4, 5, 5, 6], 4) == 4
    # k=1 返回最大值。
    assert solver.findKthLargest([-4, -2, -9], 1) == -2
    # k=n 返回最小值。
    assert solver.findKthLargest([7, 7, 3], 3) == 3
    # 输出明确的离线验收信息。
    print("#215 数组中的第 K 个最大元素：全部测试通过")


if __name__ == "__main__":
    # 直接运行文件时执行自测。
    run_tests()
