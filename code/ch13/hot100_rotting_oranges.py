"""LeetCode Hot 100 #994：腐烂的橘子。

原题：https://leetcode.cn/problems/rotting-oranges/
练习入口：https://codefun2000.com/p/P4020
"""

from collections import deque


class Solution:
    def orangesRotting(self, grid: list[list[int]]) -> int:
        # 空网格没有新鲜橘子需要等待。
        if not grid or not grid[0]:
            return 0

        rows, cols = len(grid), len(grid[0])
        # 多源 BFS：所有初始腐烂橘子同时在第 0 分钟进入队列。
        queue: deque[tuple[int, int]] = deque()
        fresh = 0

        for row in range(rows):
            for col in range(cols):
                if grid[row][col] == 2:
                    queue.append((row, col))
                elif grid[row][col] == 1:
                    fresh += 1

        minutes = 0
        directions = ((1, 0), (-1, 0), (0, 1), (0, -1))

        # 没有新鲜橘子时不应凭空增加一分钟，所以循环条件包含 fresh > 0。
        while queue and fresh > 0:
            # 当前队列长度就是“这一分钟开始时”所有腐烂源的数量。
            for _ in range(len(queue)):
                row, col = queue.popleft()
                for dr, dc in directions:
                    nr, nc = row + dr, col + dc
                    # 只让新鲜橘子第一次腐烂；空格和已腐烂格都跳过。
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 1:
                        grid[nr][nc] = 2
                        fresh -= 1
                        queue.append((nr, nc))
            # 完整扩散一层，时间才增加一分钟。
            minutes += 1

        # fresh 仍大于 0，说明它们被空格隔开，永远无法腐烂。
        return minutes if fresh == 0 else -1


def run_tests() -> None:
    solver = Solution()
    assert solver.orangesRotting([[2, 1, 1], [1, 1, 0], [0, 1, 1]]) == 4
    assert solver.orangesRotting([[2, 1, 1], [0, 1, 1], [1, 0, 1]]) == -1
    assert solver.orangesRotting([[0, 2]]) == 0
    assert solver.orangesRotting([]) == 0
    print("#994 腐烂的橘子：全部测试通过")


if __name__ == "__main__":
    run_tests()
