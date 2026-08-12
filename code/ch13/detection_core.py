"""第 13 章：边界框、IoU、锚框、偏移、NMS 与微型 SSD 核心。

运行：python code/ch13/detection_core.py
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def box_corner_to_center(boxes: torch.Tensor) -> torch.Tensor:
    """(xmin,ymin,xmax,ymax) -> (cx,cy,w,h)。"""

    xmin, ymin, xmax, ymax = boxes.unbind(dim=-1)
    # 中心坐标是两条边的平均。
    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2
    # 宽高必须非负；数据清洗时应检查反向框。
    width = xmax - xmin
    height = ymax - ymin
    return torch.stack((center_x, center_y, width, height), dim=-1)


def box_center_to_corner(boxes: torch.Tensor) -> torch.Tensor:
    """(cx,cy,w,h) -> (xmin,ymin,xmax,ymax)。"""

    center_x, center_y, width, height = boxes.unbind(dim=-1)
    # 左上角等于中心减半宽/半高。
    xmin = center_x - width / 2
    ymin = center_y - height / 2
    # 右下角等于中心加半宽/半高。
    xmax = center_x + width / 2
    ymax = center_y + height / 2
    return torch.stack((xmin, ymin, xmax, ymax), dim=-1)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """返回两组角点框的两两 IoU，Shape 为 (N,M)。"""

    # 每个框面积 Shape 分别为 (N,) 和 (M,)。
    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp_min(0) *
             (boxes1[:, 3] - boxes1[:, 1]).clamp_min(0))
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp_min(0) *
             (boxes2[:, 3] - boxes2[:, 1]).clamp_min(0))
    # 广播后左上角 Shape 为 (N,M,2)，取较靠右下者。
    upper_left = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    # 右下角取较靠左上者。
    lower_right = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    # 不相交时宽高截为 0，避免负面积。
    intersection_wh = (lower_right - upper_left).clamp_min(0)
    intersection = intersection_wh[..., 0] * intersection_wh[..., 1]
    # 并集 = 面积1 + 面积2 - 交集，减交集是避免重复计数。
    union = area1[:, None] + area2[None, :] - intersection
    return intersection / union.clamp_min(1e-12)


def multibox_prior(feature_map: torch.Tensor, sizes: list[float], ratios: list[float]) -> torch.Tensor:
    """为特征图每个像素生成 len(sizes)+len(ratios)-1 个归一化锚框。"""

    _, _, height, width = feature_map.shape
    device = feature_map.device
    boxes_per_pixel = len(sizes) + len(ratios) - 1
    # 像素中心位于 (i+0.5)/尺寸，而不是格点边界。
    center_y = (torch.arange(height, device=device) + 0.5) / height
    center_x = (torch.arange(width, device=device) + 0.5) / width
    shift_y, shift_x = torch.meshgrid(center_y, center_x, indexing="ij")
    # 组合规则：(s_i,r_0) 加上 (s_0,r_j)，避免笛卡尔积过多。
    size_tensor = torch.tensor(sizes, device=device)
    ratio_tensor = torch.tensor(ratios, device=device)
    widths = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
                        size_tensor[0] * torch.sqrt(ratio_tensor[1:])))
    heights = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),
                         size_tensor[0] / torch.sqrt(ratio_tensor[1:])))
    # 图像非正方时，以高宽归一化会让显示宽度需乘 H/W 修正。
    widths = widths * height / width
    offsets = torch.stack((-widths, -heights, widths, heights), dim=1) / 2
    # 所有 H*W 个中心复制 A 次。
    centers = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=-1)
    centers = centers.reshape(-1, 1, 4).repeat(1, boxes_per_pixel, 1)
    # 广播相加得到 (H*W,A,4)，再摊平为 (1,H*W*A,4)。
    anchors = centers + offsets.reshape(1, boxes_per_pixel, 4)
    return anchors.reshape(1, -1, 4)


def assign_anchors(anchors: torch.Tensor, ground_truth: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """把每个锚框映射到 GT 索引；-1 表示背景。"""

    ious = box_iou(anchors, ground_truth)
    # 先让超过阈值的锚框匹配其最高 IoU 真值框。
    max_iou, assigned = ious.max(dim=1)
    assigned[max_iou < threshold] = -1
    # 再贪心保证每个 GT 至少拥有一个锚框，即使 IoU 未过阈值。
    work = ious.clone()
    for _ in range(ground_truth.shape[0]):
        flat_index = work.argmax()
        anchor_index = flat_index // ground_truth.shape[0]
        gt_index = flat_index % ground_truth.shape[0]
        assigned[anchor_index] = gt_index
        work[anchor_index, :] = -1
        work[:, gt_index] = -1
    return assigned


def offset_boxes(anchors: torch.Tensor, assigned_boxes: torch.Tensor) -> torch.Tensor:
    """把真值框编码为相对锚框的中心/尺度偏移。"""

    anchor_center = box_corner_to_center(anchors)
    box_center = box_corner_to_center(assigned_boxes)
    # 中心差除以锚框宽高，10 是常用的数值缩放系数。
    offset_xy = 10 * (box_center[:, :2] - anchor_center[:, :2]) / anchor_center[:, 2:]
    # 宽高比用对数把乘法尺度变成加法，5 同样用于调整数值范围。
    offset_wh = 5 * torch.log(box_center[:, 2:] / anchor_center[:, 2:].clamp_min(1e-12))
    return torch.cat((offset_xy, offset_wh), dim=1)


def offset_inverse(anchors: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """把网络预测偏移解码回角点框。"""

    anchor_center = box_corner_to_center(anchors)
    # 解码中心：撤销乘 10 与按锚框宽高归一化。
    predicted_xy = offsets[:, :2] * anchor_center[:, 2:] / 10 + anchor_center[:, :2]
    # 解码宽高：exp 撤销 log，随后乘回锚框尺度。
    predicted_wh = torch.exp(offsets[:, 2:] / 5) * anchor_center[:, 2:]
    return box_center_to_corner(torch.cat((predicted_xy, predicted_wh), dim=1))


def nms(boxes: torch.Tensor, scores: torch.Tensor, threshold: float) -> torch.Tensor:
    """按置信度贪心抑制同类重叠框，返回保留索引。"""

    # 先按分数从高到低排序。
    order = scores.argsort(descending=True)
    keep: list[torch.Tensor] = []
    while order.numel() > 0:
        # 当前最高分框必定保留。
        best = order[0]
        keep.append(best)
        if order.numel() == 1:
            break
        # 只计算最佳框与剩余框的 IoU。
        overlap = box_iou(boxes[best].reshape(1, 4), boxes[order[1:]]).reshape(-1)
        # 保留重叠未超过阈值者；NMS 应按类别分别执行。
        order = order[1:][overlap <= threshold]
    return torch.stack(keep)


def cls_predictor(in_channels: int, anchors_per_pixel: int, classes: int) -> nn.Conv2d:
    # 每个像素的每个锚框输出“背景 + C 个目标类”分数。
    return nn.Conv2d(in_channels, anchors_per_pixel * (classes + 1), kernel_size=3, padding=1)


def box_predictor(in_channels: int, anchors_per_pixel: int) -> nn.Conv2d:
    # 每个锚框输出 4 个偏移量。
    return nn.Conv2d(in_channels, anchors_per_pixel * 4, kernel_size=3, padding=1)


def flatten_predictions(prediction: torch.Tensor) -> torch.Tensor:
    # (B,A*K,H,W) -> (B,H,W,A*K) -> (B,H*W*A*K)。
    return prediction.permute(0, 2, 3, 1).flatten(start_dim=1)


class TinySSDCore(nn.Module):
    """两个尺度的 SSD 核心，只展示预测头与 Shape，不追求真实精度。"""

    def __init__(self, classes: int = 2) -> None:
        super().__init__()
        self.classes = classes
        self.sizes = [[0.2, 0.3], [0.5, 0.7]]
        self.ratios = [[1.0, 2.0, 0.5], [1.0, 2.0, 0.5]]
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU()),
            nn.Sequential(nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU()),
        ])
        anchors_per_pixel = len(self.sizes[0]) + len(self.ratios[0]) - 1
        self.class_heads = nn.ModuleList([
            cls_predictor(16, anchors_per_pixel, classes),
            cls_predictor(32, anchors_per_pixel, classes),
        ])
        self.box_heads = nn.ModuleList([
            box_predictor(16, anchors_per_pixel),
            box_predictor(32, anchors_per_pixel),
        ])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchors, class_parts, box_parts = [], [], []
        for index, block in enumerate(self.blocks):
            # 每个块缩小空间尺寸，从而让深层特征负责更大感受野。
            x = block(x)
            # 锚框只依赖特征图空间 Shape，不参与梯度学习。
            anchors.append(multibox_prior(x, self.sizes[index], self.ratios[index]))
            # 分类头与框头都建立参数计算图。
            class_parts.append(flatten_predictions(self.class_heads[index](x)))
            box_parts.append(flatten_predictions(self.box_heads[index](x)))
        # 两尺度锚框沿候选维拼接为 (1,N,4)。
        all_anchors = torch.cat(anchors, dim=1)
        # 分类摊平结果还原为 (B,N,C+1)。
        class_predictions = torch.cat(class_parts, dim=1).reshape(x.shape[0], -1, self.classes + 1)
        # 框偏移还原为 (B,N,4)。
        box_predictions = torch.cat(box_parts, dim=1).reshape(x.shape[0], -1, 4)
        return all_anchors, class_predictions, box_predictions


def main() -> None:
    torch.manual_seed(13)
    # 三个归一化角点框，用于基础几何验证。
    boxes = torch.tensor([[0.10, 0.10, 0.50, 0.50],
                          [0.12, 0.12, 0.48, 0.48],
                          [0.60, 0.60, 0.90, 0.90]])
    scores = torch.tensor([0.90, 0.80, 0.75])
    print("两两 IoU:\n", box_iou(boxes, boxes))
    print("NMS 保留索引:", nms(boxes, scores, threshold=0.5).tolist())

    # 编码后再解码，验证框参数化是一一对应的数值变换。
    anchors = boxes[[0, 2]]
    targets = torch.tensor([[0.15, 0.12, 0.52, 0.55], [0.58, 0.62, 0.88, 0.92]])
    recovered = offset_inverse(anchors, offset_boxes(anchors, targets))
    error = (recovered - targets).abs().max().item()
    print(f"偏移编码/解码最大误差: {error:.3e}")
    assert error < 1e-6

    # 微型 SSD 对合成 batch 做一次前向和反向。
    model = TinySSDCore(classes=2)
    images = torch.randn(2, 3, 32, 32)
    all_anchors, class_preds, box_preds = model(images)
    # 假目标只用于验证两个预测头都收到梯度。
    class_targets = torch.zeros(class_preds.shape[:2], dtype=torch.long)
    box_targets = torch.zeros_like(box_preds)
    class_loss = F.cross_entropy(class_preds.reshape(-1, 3), class_targets.reshape(-1))
    box_loss = F.smooth_l1_loss(box_preds, box_targets)
    loss = class_loss + box_loss
    loss.backward()
    print(f"SSD: anchors={tuple(all_anchors.shape)}, cls={tuple(class_preds.shape)}, box={tuple(box_preds.shape)}")
    print(f"一次联合损失={loss.item():.4f}；分类头和框回归头均已反向传播")


if __name__ == "__main__":
    main()
