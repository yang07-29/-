"""LeetCode Hot 100 #73：矩阵置零。

原题：https://leetcode.cn/problems/set-matrix-zeroes/
练习：https://codefun2000.com/p/P4013
运行：python code/ch06/hot100_set_matrix_zeroes.py
"""

from __future__ import annotations


class Solution:
    """使用首行与首列充当行列标记，做到 O(1) 额外空间。"""

    def setZeroes(self, matrix: list[list[int]]) -> None:
        """若某格原本为 0，把它所在整行整列都原地改为 0。"""
        # 空矩阵没有需要修改的元素。
        if not matrix or not matrix[0]:
            return
        # rows、columns 分别是矩阵高宽。
        rows, columns = len(matrix), len(matrix[0])
        # 防止不规则二维列表破坏列索引。
        if any(len(row) != columns for row in matrix):
            raise ValueError("matrix 的每一行必须等长")

        # 首行自身是否含零，必须单独记录，否则标记过程会覆盖原信息。
        first_row_has_zero = any(matrix[0][column] == 0 for column in range(columns))
        # 首列自身是否含零，也必须单独记录。
        first_column_has_zero = any(matrix[row][0] == 0 for row in range(rows))

        # 扫描内部区域；遇到零就在对应首行、首列位置写标记。
        for row in range(1, rows):
            for column in range(1, columns):
                if matrix[row][column] == 0:
                    matrix[row][0] = 0
                    matrix[0][column] = 0

        # 根据首列标记清理内部各行。
        for row in range(1, rows):
            # 当前行首格为零，说明整行应清零。
            if matrix[row][0] == 0:
                for column in range(1, columns):
                    matrix[row][column] = 0

        # 根据首行标记清理内部各列。
        for column in range(1, columns):
            # 当前列首格为零，说明整列应清零。
            if matrix[0][column] == 0:
                for row in range(1, rows):
                    matrix[row][column] = 0

        # 最后处理首行，避免它过早清零后污染列标记。
        if first_row_has_zero:
            for column in range(columns):
                matrix[0][column] = 0
        # 最后处理首列，避免它过早清零后污染行标记。
        if first_column_has_zero:
            for row in range(rows):
                matrix[row][0] = 0


def run_tests() -> None:
    """覆盖内部零、首行零、首列零与无零矩阵。"""
    solver = Solution()
    matrix = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
    solver.setZeroes(matrix)
    assert matrix == [[1, 0, 1], [0, 0, 0], [1, 0, 1]]
    matrix = [[0, 1], [2, 3]]
    solver.setZeroes(matrix)
    assert matrix == [[0, 0], [0, 3]]
    matrix = [[1, 2], [3, 4]]
    solver.setZeroes(matrix)
    assert matrix == [[1, 2], [3, 4]]
    print("#73 矩阵置零：全部测试通过")


if __name__ == "__main__":
    run_tests()
