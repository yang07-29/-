"""LeetCode Hot 100 #240：搜索二维矩阵 II。

原题：https://leetcode.cn/problems/search-a-2d-matrix-ii/
练习入口：https://codefun2000.com/p/P4041
"""


class Solution:
    def searchMatrix(self, matrix: list[list[int]], target: int) -> bool:
        # 空矩阵没有可搜索元素。
        if not matrix or not matrix[0]:
            return False

        rows, cols = len(matrix), len(matrix[0])
        # 从右上角出发：向左会变小，向下会变大，方向是可判定的。
        row, col = 0, cols - 1

        while row < rows and col >= 0:
            value = matrix[row][col]
            if value == target:
                return True
            if value > target:
                # 当前值太大，这一列当前位置以下更大，只能左移。
                col -= 1
            else:
                # 当前值太小，这一行当前位置左侧更小，只能下移。
                row += 1

        return False


def run_tests() -> None:
    solver = Solution()
    matrix = [
        [1, 4, 7, 11, 15],
        [2, 5, 8, 12, 19],
        [3, 6, 9, 16, 22],
        [10, 13, 14, 17, 24],
        [18, 21, 23, 26, 30],
    ]
    assert solver.searchMatrix(matrix, 5)
    assert not solver.searchMatrix(matrix, 20)
    assert not solver.searchMatrix([], 1)
    print("#240 搜索二维矩阵 II：全部测试通过")


if __name__ == "__main__":
    run_tests()
