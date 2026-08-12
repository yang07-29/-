"""LeetCode Hot 100 #295：数据流的中位数。

原题：https://leetcode.cn/problems/find-median-from-data-stream/
练习入口：https://codefun2000.com/p/P4083
"""

import heapq


class MedianFinder:
    def __init__(self) -> None:
        # lower 用负数模拟大根堆，保存较小的一半数字。
        self.lower: list[int] = []
        # upper 是普通小根堆，保存较大的一半数字。
        self.upper: list[int] = []

    def addNum(self, num: int) -> None:
        # 先按“是否小于较小一半的最大值”决定进入哪个堆。
        if not self.lower or num <= -self.lower[0]:
            heapq.heappush(self.lower, -num)
        else:
            heapq.heappush(self.upper, num)

        # 维护大小不变量：lower 可以比 upper 多一个，但不能多两个。
        if len(self.lower) > len(self.upper) + 1:
            moved = -heapq.heappop(self.lower)
            heapq.heappush(self.upper, moved)
        # upper 也绝不能比 lower 多；若多了就把最小值搬回来。
        elif len(self.upper) > len(self.lower):
            moved = heapq.heappop(self.upper)
            heapq.heappush(self.lower, -moved)

    def findMedian(self) -> float:
        # 数据总数为奇数时，lower 比 upper 多一个，堆顶就是中位数。
        if len(self.lower) > len(self.upper):
            return float(-self.lower[0])
        # 数据总数为偶数时，取两个中间值的平均数。
        return (-self.lower[0] + self.upper[0]) / 2.0


def run_tests() -> None:
    finder = MedianFinder()
    finder.addNum(1)
    assert finder.findMedian() == 1.0
    finder.addNum(2)
    assert finder.findMedian() == 1.5
    finder.addNum(3)
    assert finder.findMedian() == 2.0
    finder.addNum(-10)
    assert finder.findMedian() == 1.5
    print("#295 数据流的中位数：全部测试通过")


if __name__ == "__main__":
    run_tests()
