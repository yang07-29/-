"""LeetCode Hot 100 #121：买卖股票的最佳时机。

原题：https://leetcode.cn/problems/best-time-to-buy-and-sell-stock/
练习：https://codefun2000.com/p/P4029
运行：python code/ch11/hot100_best_time_stock.py
"""

from __future__ import annotations


class Solution:
    """扫描售价时维护此前最低买价和最大利润。"""

    def maxProfit(self, prices: list[int]) -> int:
        """只允许先买一次、后卖一次，返回最大非负利润。"""
        # minimum_price 是扫描到今天前后见过的最低价格。
        minimum_price = float("inf")
        # best_profit 至少为 0，表示可以选择不交易。
        best_profit = 0

        # 把每个价格当成“今天卖出”的候选售价。
        for price in prices:
            # 若今天卖出，最好搭配此前最低买价。
            best_profit = max(best_profit, price - minimum_price)
            # 再把今天价格纳入未来可以买入的最低价。
            minimum_price = min(minimum_price, price)

        # 返回满足买入先于卖出的最大利润。
        return best_profit


def run_tests() -> None:
    """覆盖有利润、持续下降、单元素与空输入。"""
    solver = Solution()
    assert solver.maxProfit([7, 1, 5, 3, 6, 4]) == 5
    assert solver.maxProfit([7, 6, 4, 3, 1]) == 0
    assert solver.maxProfit([2]) == 0
    assert solver.maxProfit([]) == 0
    print("#121 买卖股票的最佳时机：全部测试通过")


if __name__ == "__main__":
    run_tests()
