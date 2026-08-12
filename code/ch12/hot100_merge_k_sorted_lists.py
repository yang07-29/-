"""LeetCode Hot 100 #23：合并 K 个升序链表。

原题：https://leetcode.cn/problems/merge-k-sorted-lists/
练习入口：https://codefun2000.com/p/P4051
"""

import heapq
from dataclasses import dataclass
from typing import Optional


@dataclass
class ListNode:
    """最小链表节点定义，与 LeetCode 的节点字段保持一致。"""

    val: int = 0
    next: Optional["ListNode"] = None


class Solution:
    def mergeKLists(self, lists: list[Optional[ListNode]]) -> Optional[ListNode]:
        # 小根堆只保存每条链表“当前尚未合并的最小节点”。
        heap: list[tuple[int, int, ListNode]] = []
        # serial 是稳定的唯一编号，避免节点值相同时让 Python 比较 ListNode。
        serial = 0

        # 先把每条非空链表的头节点放进堆，堆大小最多为 k。
        for head in lists:
            if head is not None:
                heapq.heappush(heap, (head.val, serial, head))
                serial += 1

        # 哑节点让“接上第一个节点”和“接上后续节点”使用同一套逻辑。
        dummy = ListNode()
        # tail 始终指向已经合并结果的最后一个节点。
        tail = dummy

        while heap:
            # 堆顶是所有候选头节点里值最小的那个。
            _, _, node = heapq.heappop(heap)
            # 把它接到结果链表尾部。
            tail.next = node
            # 尾指针随之向后移动。
            tail = node

            # 只有取走某条链表的节点后，它的下一个节点才会成为新候选。
            if node.next is not None:
                heapq.heappush(heap, (node.next.val, serial, node.next))
                serial += 1

        # dummy 本身不属于答案，真正头节点是 dummy.next。
        return dummy.next


def build_linked_list(values: list[int]) -> Optional[ListNode]:
    """把普通列表转成链表，方便本地复制运行和自测。"""
    dummy = ListNode()
    tail = dummy
    for value in values:
        tail.next = ListNode(value)
        tail = tail.next
    return dummy.next


def to_list(head: Optional[ListNode]) -> list[int]:
    """把链表转回普通列表，便于观察结果。"""
    values: list[int] = []
    while head is not None:
        values.append(head.val)
        head = head.next
    return values


def run_tests() -> None:
    solver = Solution()
    case = [build_linked_list(x) for x in [[1, 4, 5], [1, 3, 4], [2, 6]]]
    assert to_list(solver.mergeKLists(case)) == [1, 1, 2, 3, 4, 4, 5, 6]
    assert to_list(solver.mergeKLists([])) == []
    assert to_list(solver.mergeKLists([None])) == []
    print("#23 合并 K 个升序链表：全部测试通过")


if __name__ == "__main__":
    run_tests()
