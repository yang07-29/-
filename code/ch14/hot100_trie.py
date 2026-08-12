"""Hot100 #208：实现 Trie（前缀树）。

章节迁移：BPE、子词与词表都会频繁处理“共享前缀”；Trie 把相同前缀只存一次。
题目：https://leetcode.cn/problems/implement-trie-prefix-tree/

运行：
    python code/ch14/hot100_trie.py
"""

from __future__ import annotations


class TrieNode:
    """一个节点代表“从根走到这里形成的前缀”。"""

    def __init__(self) -> None:
        # children 把“下一字符”映射到“下一个前缀节点”。
        self.children: dict[str, TrieNode] = {}
        # is_word 区分“完整单词结尾”和“仅仅是别人的前缀”。
        self.is_word = False


class Trie:
    """支持插入、完整词查询与前缀查询。"""

    def __init__(self) -> None:
        # 根节点对应空前缀，不保存具体字符。
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        # 每个新单词都从空前缀开始走。
        node = self.root
        for character in word:
            # 若边不存在就创建；存在则复用共享前缀。
            if character not in node.children:
                node.children[character] = TrieNode()
            # 沿当前字符对应的边走到下一节点。
            node = node.children[character]
        # 只有最后节点标记为单词结尾；中间节点仍可能只是前缀。
        node.is_word = True

    def _find_node(self, text: str) -> TrieNode | None:
        """沿字符路径查找；路径中断就返回 None。"""
        node = self.root
        for character in text:
            # 少一条边就说明该前缀从未插入。
            if character not in node.children:
                return None
            # 路径存在，继续深入。
            node = node.children[character]
        # 走完 text 后返回对应前缀节点。
        return node

    def search(self, word: str) -> bool:
        # 找到路径还不够，末节点必须确实是某个完整词的结尾。
        node = self._find_node(word)
        return node is not None and node.is_word

    def startsWith(self, prefix: str) -> bool:  # noqa: N802 - 保持 LeetCode 题目接口
        # 前缀查询只关心路径是否存在，不要求 is_word=True。
        return self._find_node(prefix) is not None


def self_test() -> None:
    trie = Trie()
    # 插入 apple 会创建 a->p->p->l->e，并只把 e 节点标成单词。
    trie.insert("apple")
    # 完整路径存在且结尾被标记，因此找到 apple。
    assert trie.search("apple") is True
    # app 路径虽存在，但尚未标成完整词，所以 search 为 False。
    assert trie.search("app") is False
    # startsWith 只检查路径，app 是 apple 的前缀。
    assert trie.startsWith("app") is True
    # 再插入 app，只需复用已有 a->p->p 路径并标记结尾。
    trie.insert("app")
    assert trie.search("app") is True
    # cat 第一条边 c 不存在，前缀查询立即失败。
    assert trie.startsWith("cat") is False
    print("Trie 自测通过：apple/app 的完整词与前缀语义已区分。")


if __name__ == "__main__":
    self_test()
