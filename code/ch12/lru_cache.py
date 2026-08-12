"""Hot 100 #146：哈希表 + 双向链表实现 LRU 缓存。

运行：python code/ch12/lru_cache.py
"""

from __future__ import annotations


class Node:
    """双向链表节点，同时保存 key 才能在淘汰时删除哈希表项。"""

    def __init__(self, key: int = 0, value: int = 0) -> None:
        # key 用于从字典中 O(1) 删除该节点。
        self.key = key
        # value 是缓存真正存放的值。
        self.value = value
        # prev 指向更“新”一侧的节点。
        self.prev: Node | None = None
        # next 指向更“旧”一侧的节点。
        self.next: Node | None = None


class LRUCache:
    """get/put 平均 O(1) 的最近最少使用缓存。"""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity 必须为正整数")
        # capacity 是可存放的真实节点数。
        self.capacity = capacity
        # 哈希表负责从 key O(1) 找到链表节点。
        self.nodes: dict[int, Node] = {}
        # head 是哨兵，不保存业务数据；head.next 永远是最新节点。
        self.head = Node()
        # tail 也是哨兵；tail.prev 永远是最久未使用节点。
        self.tail = Node()
        # 空链表中两个哨兵彼此相连。
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: Node) -> None:
        """从链表摘下已知节点，O(1)。"""

        # 真实节点两侧一定存在哨兵或真实节点。
        assert node.prev is not None and node.next is not None
        # 左邻居跨过 node 指向右邻居。
        node.prev.next = node.next
        # 右邻居跨过 node 指向左邻居。
        node.next.prev = node.prev

    def _add_to_front(self, node: Node) -> None:
        """把节点插到 head 后，标记为最近使用，O(1)。"""

        # 记住原先最新节点。
        first = self.head.next
        assert first is not None
        # 新节点左边接 head。
        node.prev = self.head
        # 新节点右边接原 first。
        node.next = first
        # head 改为指向新节点。
        self.head.next = node
        # 原 first 的左边改为新节点。
        first.prev = node

    def _touch(self, node: Node) -> None:
        """一次访问会刷新“最近使用”顺序。"""

        # 先从旧位置摘除。
        self._remove(node)
        # 再放到最前端。
        self._add_to_front(node)

    def get(self, key: int) -> int:
        """存在则返回值并刷新顺序，否则返回 -1。"""

        # 字典查找平均 O(1)。
        node = self.nodes.get(key)
        if node is None:
            # 未命中不会改变任何节点顺序。
            return -1
        # 命中本身算“使用”，必须移到最新位置。
        self._touch(node)
        return node.value

    def put(self, key: int, value: int) -> None:
        """插入或更新；超容量时淘汰最久未使用项。"""

        node = self.nodes.get(key)
        if node is not None:
            # 已存在时只更新 value，不增加缓存大小。
            node.value = value
            # 写入也算一次使用，因此刷新到最前。
            self._touch(node)
            return

        # 新 key 创建一个真实节点。
        new_node = Node(key, value)
        # 字典登记 key -> 节点。
        self.nodes[key] = new_node
        # 新写入项是最近使用项。
        self._add_to_front(new_node)

        if len(self.nodes) > self.capacity:
            # tail.prev 是当前最久未使用的真实节点。
            oldest = self.tail.prev
            assert oldest is not None and oldest is not self.head
            # 先从链表摘除。
            self._remove(oldest)
            # 再从哈希表删除；两套结构必须保持一致。
            del self.nodes[oldest.key]


def main() -> None:
    # 使用官方示例序列做自测。
    cache = LRUCache(2)
    cache.put(1, 1)
    cache.put(2, 2)
    assert cache.get(1) == 1
    cache.put(3, 3)
    assert cache.get(2) == -1
    cache.put(4, 4)
    result = [cache.get(1), cache.get(3), cache.get(4)]
    assert result == [-1, 3, 4]
    print("LRU 自测通过，最后三次 get:", result)


if __name__ == "__main__":
    main()
