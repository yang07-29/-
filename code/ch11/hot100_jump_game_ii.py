"""LeetCode Hot 100 #45：跳跃游戏 II。

原题：https://leetcode.cn/problems/jump-game-ii/
练习：https://codefun2000.com/p/P4084
运行：python code/ch11/hot100_jump_game_ii.py
"""

from __future__ import annotations


class Solution:
    """把每次跳跃覆盖范围看成一层，贪心扩展下一层最远边界。"""

    def jump(self, nums: list[int]) -> int:
        """在题目保证可达的前提下，返回到末尾的最少跳跃次数。"""
        # 长度 0 或 1 时已经位于终点，无需跳跃。
        if len(nums) <= 1:
            return 0
        # jumps 是已确定使用的跳跃次数。
        jumps = 0
        # current_end 是当前 jumps 次跳跃最多能覆盖到的边界。
        current_end = 0
        # farthest 是扫描当前覆盖层时发现的下一层最远边界。
        farthest = 0

        # 最后一个位置无需再从它起跳，因此只扫描到倒数第二格。
        for index in range(len(nums) - 1):
            # 用当前位置可达距离扩展下一层边界。
            farthest = max(farthest, index + nums[index])
            # 扫描到当前层末尾，必须使用一次跳跃进入下一层。
            if index == current_end:
                jumps += 1
                current_end = farthest
                # 新边界已覆盖终点，可以提前返回最少次数。
                if current_end >= len(nums) - 1:
                    return jumps

        # 教学版本兼容不可达输入，明确失败而不是给伪答案。
        raise ValueError("末尾位置不可达")


def run_tests() -> None:
    """覆盖典型可达、一步到达、单元素与不可达输入。"""
    solver = Solution()
    assert solver.jump([2, 3, 1, 1, 4]) == 2
    assert solver.jump([2, 3, 0, 1, 4]) == 2
    assert solver.jump([5, 0, 0]) == 1
    assert solver.jump([0]) == 0
    try:
        solver.jump([0, 1])
    except ValueError:
        pass
    else:
        raise AssertionError("不可达输入应触发 ValueError")
    print("#45 跳跃游戏 II：全部测试通过")


if __name__ == "__main__":
    run_tests()
