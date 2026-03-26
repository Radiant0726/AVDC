from typing import List, Tuple
import json

def interval_iou(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Compute Intersection over Union for 1D intervals [start, end]."""
    start_a, end_a = a
    start_b, end_b = b
    if end_a <= start_a or end_b <= start_b:
        return 0.0
    inter_start = max(start_a, start_b)
    inter_end = min(end_a, end_b)
    inter = max(0.0, inter_end - inter_start)
    union = (end_a - start_a) + (end_b - start_b) - inter
    if union <= 0:
        return 0.0
    return inter / union

def compute_map(
    gt_intervals: List[Tuple[float, float]],
    pred_intervals: List[Tuple[float, float]],
    iou_thresh: float = 0.5
) -> Tuple[float, float, float]:
    """
    Compute precision, recall, mAP for interval predictions vs ground truth.
    Matching is many-to-many: each pred can match multiple gt and vice versa,
    as long as IoU >= iou_thresh.

    Returns: precision, recall, mAP
    """
    # Count how many predictions are matched (i.e., those pred intervals that
    # have IoU >= thresh with at least one gt)
    matched_preds = 0
    for p in pred_intervals:
        for g in gt_intervals:
            if interval_iou(p, g) >= iou_thresh:
                matched_preds += 1
                break  # this pred counted once

    # Similarly, count how many gt intervals are matched by at least one pred
    matched_gts = 0
    for g in gt_intervals:
        for p in pred_intervals:
            if interval_iou(p, g) >= iou_thresh:
                matched_gts += 1
                break  # this gt counted once

    num_preds = len(pred_intervals)
    num_gts = len(gt_intervals)

    precision = matched_preds / num_preds if num_preds > 0 else 0.0
    recall = matched_gts / num_gts if num_gts > 0 else 0.0

    mAP = precision 

    return precision, recall, mAP

from itertools import product
def compute_mean_iou(
    gt_intervals: List[Tuple[float, float]],
    pred_intervals: List[Tuple[float, float]],
    iou_thresh: float = 0.5
) -> float:

    ious = [
        interval_iou(p, g)
        for p, g in product(pred_intervals, gt_intervals)
        if interval_iou(p, g) >= iou_thresh
    ]
    if not ious:
        return 0.0
    return sum(ious) / len(ious)

def calculate_metrics(ref_intervals, pred_intervals, iou_thresh):

    precision, recall, mAP = compute_map(ref_intervals, pred_intervals, iou_thresh)
    print("precision, recall, mAP:", precision, recall, mAP)

    miou = compute_mean_iou(ref_intervals, pred_intervals, iou_thresh)
    print("miou:", miou)

    return precision, recall, mAP, miou
