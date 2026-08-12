"""LeetCode Hot 100 #48：旋转图像。

原题：https://leetcode.cn/problems/rotate-image/
练习：https://codefun2000.com/p/P4040
运行：python code/ch06/hot100_rotate_image.py
"""

from __future__ import annotations


class Solution:
    """先转置、再逐行反转，原地完成顺时针旋转。"""

    def rotate(self, matrix: list[list[int]]) -> None:
        """把 n×n 方阵顺时针旋转 90°，不返回新矩阵。"""
        # n 是方阵边长；空矩阵无需处理。
        n = len(matrix)
        # 教学版本主动检查每行长度，避免“不是方阵”时静默算错。
        if any(len(row) != n for row in matrix):
            raise ValueError("matrix 必须是 n×n 方阵")

        # 第一步沿主对角线转置：只交换上三角与下三角对应位置。
        for row in range(n):
            # column 从 row+1 开始，避免对角线自换和一对元素交换两次。
            for column in range(row + 1, n):
                # (row,column) 与 (column,row) 互换。
                matrix[row][column], matrix[column][row] = (
                    matrix[column][row],
                    matrix[row][column],
                )

        # 第二步把转置后的每一行左右反转。
        for row in matrix:
            # list.reverse 原地修改当前行，不创建另一张矩阵。
            row.reverse()


def run_tests() -> None:
    """覆盖 3×3、2×2、1×1 与非法非方阵。"""
    solver = Solution()
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    solver.rotate(matrix)
    assert matrix == [[7, 4, 1], [8, 5, 2], [9, 6, 3]]
    matrix = [[1, 2], [3, 4]]
    solver.rotate(matrix)
    assert matrix == [[3, 1], [4, 2]]
    matrix = [[5]]
    solver.rotate(matrix)
    assert matrix == [[5]]
    try:
        solver.rotate([[1, 2], [3]])
    except ValueError:
        pass
    else:
        raise AssertionError("非方阵应触发 ValueError")
    print("#48 旋转图像：全部测试通过")


if __name__ == "__main__":
    run_tests()
