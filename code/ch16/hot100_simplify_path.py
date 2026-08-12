"""LeetCode Hot 100 #71：简化路径。

原题：https://leetcode.cn/problems/simplify-path/
练习入口：https://codefun2000.com/p/P4818
"""


class Solution:
    def simplifyPath(self, path: str) -> str:
        # 栈中只保存最终路径里仍然有效的目录名。
        stack: list[str] = []

        # 用 / 切开后，连续斜杠会产生空片段；它们和 . 都无需处理。
        for part in path.split("/"):
            if part == "" or part == ".":
                continue
            if part == "..":
                # 在根目录继续向上仍是根目录，因此空栈时什么也不做。
                if stack:
                    stack.pop()
            else:
                # 其他片段都是普通目录名，包括形如 ... 的合法名字。
                stack.append(part)

        # 规范路径以单斜杠开头，目录之间也只保留一个斜杠。
        return "/" + "/".join(stack)


def run_tests() -> None:
    solver = Solution()
    assert solver.simplifyPath("/home/") == "/home"
    assert solver.simplifyPath("/home//foo/") == "/home/foo"
    assert solver.simplifyPath("/home/user/Documents/../Pictures") == "/home/user/Pictures"
    assert solver.simplifyPath("/../") == "/"
    assert solver.simplifyPath("/.../a/../b") == "/.../b"
    print("#71 简化路径：全部测试通过")


if __name__ == "__main__":
    run_tests()
