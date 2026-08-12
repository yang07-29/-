"""Hot 100 #200：用迭代 DFS 统计二维网格中的岛屿数量。

运行：python code/ch13/number_of_islands.py
"""

from __future__ import annotations


def num_islands(grid: list[list[str]]) -> int:
    """把上下左右相连的 '1' 视为同一岛屿。"""

    # 空网格没有岛屿，也避免访问 grid[0] 越界。
    if not grid or not grid[0]:
        return 0
    # 行数相当于图像高度 H。
    rows = len(grid)
    # 列数相当于图像宽度 W。
    columns = len(grid[0])
    # visited 与输入同空间 Shape，记录像素是否已归入某个连通分量。
    visited = [[False] * columns for _ in range(rows)]
    # 四邻域只允许上下左右，不把对角接触合并。
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    # 每发现一个尚未访问的陆地起点，就新增一个岛。
    islands = 0

    for row in range(rows):
        for column in range(columns):
            # 水面或已归类陆地都不再启动搜索。
            if grid[row][column] != "1" or visited[row][column]:
                continue
            # 当前点是一个新连通分量的第一个像素。
            islands += 1
            # 入栈时立即标记，防止同一邻居被重复压栈。
            stack = [(row, column)]
            visited[row][column] = True

            while stack:
                # 弹出一个待扩展陆地像素。
                current_row, current_column = stack.pop()
                for delta_row, delta_column in directions:
                    # 计算四邻域候选坐标。
                    next_row = current_row + delta_row
                    next_column = current_column + delta_column
                    # 先检查边界，再访问网格，避免负索引悄悄绕到末尾。
                    inside = 0 <= next_row < rows and 0 <= next_column < columns
                    if not inside:
                        continue
                    # 只扩展尚未访问的陆地。
                    if grid[next_row][next_column] == "1" and not visited[next_row][next_column]:
                        visited[next_row][next_column] = True
                        stack.append((next_row, next_column))
    return islands


def main() -> None:
    # 左上大片、中央单点、右下竖条，共 3 个四连通区域。
    grid = [
        ["1", "1", "0", "0", "0"],
        ["1", "1", "0", "1", "0"],
        ["0", "0", "0", "0", "0"],
        ["0", "0", "1", "0", "1"],
        ["0", "0", "1", "0", "1"],
    ]
    # 实际上中央上方单点也是一岛，因此总数为 4。
    result = num_islands(grid)
    assert result == 4
    assert num_islands([]) == 0
    assert num_islands([["0"]]) == 0
    assert num_islands([["1"]]) == 1
    print("岛屿数量自测通过，示例结果:", result)


if __name__ == "__main__":
    main()
