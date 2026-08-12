"""LeetCode Hot 100 #739：每日温度。

原题：https://leetcode.cn/problems/daily-temperatures/
练习：https://codefun2000.com/p/P4081
运行：python code/ch08/hot100_daily_temperatures.py
"""

from __future__ import annotations


class Solution:
    """用单调递减栈保存还没等到更高温度的日期。"""

    def dailyTemperatures(self, temperatures: list[int]) -> list[int]:
        """返回每一天还需等待几天才遇到更高温度。"""
        # 默认答案为 0，表示后面没有更高温度。
        waits = [0] * len(temperatures)
        # 栈保存日期下标；对应温度保持从栈底到栈顶非递增。
        unresolved: list[int] = []

        # current_day 是今天下标，temperature 是今天温度。
        for current_day, temperature in enumerate(temperatures):
            # 今天更热时，栈顶那些更冷的旧日期终于得到答案。
            while unresolved and temperatures[unresolved[-1]] < temperature:
                # 弹出最近一个尚未解决的旧日期。
                previous_day = unresolved.pop()
                # 下标差就是需要等待的天数。
                waits[previous_day] = current_day - previous_day
            # 今天暂时也不知道未来答案，压栈等待。
            unresolved.append(current_day)

        # 栈中剩余日期后面无更高温，答案保持 0。
        return waits


def run_tests() -> None:
    """覆盖升降混合、严格下降、相等温度和空输入。"""
    solver = Solution()
    assert solver.dailyTemperatures([73, 74, 75, 71, 69, 72, 76, 73]) == [1, 1, 4, 2, 1, 1, 0, 0]
    assert solver.dailyTemperatures([3, 2, 1]) == [0, 0, 0]
    assert solver.dailyTemperatures([2, 2, 3]) == [2, 1, 0]
    assert solver.dailyTemperatures([]) == []
    print("#739 每日温度：全部测试通过")


if __name__ == "__main__":
    run_tests()
