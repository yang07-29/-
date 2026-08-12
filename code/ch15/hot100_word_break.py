"""Hot100 #139：单词拆分（动态规划）。

章节迁移：文本模型收到词元前，必须先判断字符串能否被词表切成合法序列。
题目：https://leetcode.cn/problems/word-break/

运行：
    python code/ch15/hot100_word_break.py
"""

from __future__ import annotations


def word_break(text: str, word_dict: list[str]) -> bool:
    """判断 text 能否由词典中的词拼接而成。"""
    # set 让“子串是否在词典”从线性查找降为平均 O(1)。
    words = set(word_dict)
    # 最长词限制回看范围，避免枚举明显不可能的超长子串。
    max_word_length = max((len(word) for word in words), default=0)
    # dp[i] 表示前 i 个字符 text[:i] 能否被合法拆分。
    dp = [False] * (len(text) + 1)
    # 空字符串不需要任何词就能组成，是后续状态的起点。
    dp[0] = True

    for end in range(1, len(text) + 1):
        # 最早只需回看到 end-max_word_length，不能小于 0。
        earliest_start = max(0, end - max_word_length)
        for start in range(earliest_start, end):
            # 左半段必须可拆，右半段 text[start:end] 必须是一个词。
            if dp[start] and text[start:end] in words:
                # 找到任意一种合法最后一刀，dp[end] 就可确定为 True。
                dp[end] = True
                # 本题只问能否拆分，无需继续寻找其他方案。
                break
    # dp[n] 就是整个字符串的答案。
    return dp[len(text)]


def word_break_trace(text: str, word_dict: list[str]) -> tuple[bool, list[str]]:
    """额外保存前驱，返回一种具体拆分，便于观察 DP 不是黑箱。"""
    words = set(word_dict)
    max_word_length = max((len(word) for word in words), default=0)
    reachable = [False] * (len(text) + 1)
    previous = [-1] * (len(text) + 1)
    reachable[0] = True
    for end in range(1, len(text) + 1):
        for start in range(max(0, end - max_word_length), end):
            if reachable[start] and text[start:end] in words:
                reachable[end] = True
                previous[end] = start
                break
    if not reachable[-1]:
        return False, []

    pieces: list[str] = []
    end = len(text)
    while end > 0:
        start = previous[end]
        pieces.append(text[start:end])
        end = start
    # 回溯得到的是从右向左的词，需要反转为阅读顺序。
    pieces.reverse()
    return True, pieces


def self_test() -> None:
    # leetcode = leet + code，应该可拆。
    assert word_break("leetcode", ["leet", "code"]) is True
    # applepenapple 可重复使用 apple。
    ok, pieces = word_break_trace("applepenapple", ["apple", "pen"])
    assert ok is True and pieces == ["apple", "pen", "apple"]
    # catsandog 的最后部分无法与前面合法状态连接。
    assert word_break("catsandog", ["cats", "dog", "sand", "and", "cat"]) is False
    # 空字符串对应 dp[0]，按定义可以由零个词组成。
    assert word_break("", ["a"]) is True
    print("单词拆分自测通过；applepenapple ->", pieces)


if __name__ == "__main__":
    self_test()
