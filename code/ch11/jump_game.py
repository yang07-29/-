"""LeetCode Hot 100 #55 跳跃游戏：最远可达边界的贪心实现与自测。

运行：python code/ch11/jump_game.py
题目：https://leetcode.cn/problems/jump-game/
"""

from __future__ import annotations


def can_jump(nums: list[int], *, verbose: bool = False) -> bool:
    """判断能否从下标 0 到达最后一个下标，时间 O(n)、额外空间 O(1)。"""
    if not nums:  # 官方约束不会给空数组，但独立程序做防御性检查。
        raise ValueError("nums 不能为空")  # 空数组没有“第一个下标”，应明确拒绝。
    if any(step < 0 for step in nums):  # 题目规定元素必须是非负整数。
        raise ValueError("nums 中的步数必须非负")  # 防止负数破坏“最远边界单调不减”的不变量。

    farthest = 0  # 已处理位置能覆盖到的最远下标；起点 0 默认可达。
    last_index = len(nums) - 1  # 目标是覆盖最后一个下标。

    for index, max_step in enumerate(nums):  # 从左到右只扫描一次数组。
        if index > farthest:  # 当前点已落在覆盖边界之外，后续点也不可能被访问。
            if verbose:  # 仅在教学模式打印失败原因。
                print(f"停在 index={index}：它大于 farthest={farthest}，出现不可跨越的断点")
            return False  # 一旦出现断点即可提前结束。

        candidate = index + max_step  # 从当前可达点出发，最远能覆盖到哪里。
        old_farthest = farthest  # 保存更新前边界，便于打印完整推演。
        farthest = max(farthest, candidate)  # 贪心保留所有已见选择中的最大覆盖边界。

        if verbose:  # 打印“输入 -> 过程 -> 当前输出”的可验证轨迹。
            print(  # 每行展示当前点是否扩张了边界。
                f"index={index}, nums[index]={max_step}, "
                f"candidate={candidate}, farthest: {old_farthest}->{farthest}"
            )

        if farthest >= last_index:  # 覆盖边界已包含终点，无需决定具体跳法。
            return True  # 提前返回成功。

    return True  # 长度为 1 的数组会走到这里，起点本身就是终点。


def run_self_tests() -> None:
    test_cases = [  # 同时覆盖成功、失败、单元素、零和“无需真的跳到每个点”等边界。
        ([2, 3, 1, 1, 4], True),  # 边界会由 2 扩张到 4，能够到达终点。
        ([3, 2, 1, 0, 4], False),  # 最远只能到 3，无法跨过值为 0 的断点。
        ([0], True),  # 起点就是终点。
        ([0, 1], False),  # 起点不能移动。
        ([2, 0, 0], True),  # 起点可直接覆盖终点，中间的 0 不构成问题。
        ([1, 1, 1, 1], True),  # 每步只能前进 1，也能连续到达。
        ([2, 5, 0, 0], True),  # 不必选择“最远一跳”，只维护可达集合边界。
    ]

    for nums, expected in test_cases:  # 逐个执行确定性自测。
        actual = can_jump(nums)  # 调用待测函数。
        assert actual is expected, f"nums={nums} 期望 {expected}，实际 {actual}"  # 失败时给出上下文。

    print(f"{len(test_cases)} 个自测全部通过。")  # 告知脚本可正常运行。


def main() -> None:
    example = [2, 3, 1, 1, 4]  # 选择一个能到达的例子做逐步演示。
    print("示例轨迹:", example)  # 先打印输入。
    result = can_jump(example, verbose=True)  # 打开教学日志，展示 farthest 的每次变化。
    print("能否到达:", result)  # 打印具体输出 True。
    print()  # 用空行分隔演示与自测。
    run_self_tests()  # 验证常见边界情况。


if __name__ == "__main__":  # 直接运行文件才执行演示；被 import 时只提供函数。
    main()
