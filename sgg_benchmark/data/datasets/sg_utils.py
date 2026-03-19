"""
Shared utility functions for scene-graph dataset processing.

Previously duplicated verbatim in visual_genome.py, data.py, and psg.py.
"""
import numpy as np


def box_filter(boxes, must_overlap=False):
    """Return pairs of box indices that can form relations.

    If ``must_overlap`` is ``True`` only overlapping pairs are returned; if no
    such pairs exist, all non-self pairs are returned as fallback.
    """
    overlaps = bbox_overlaps(boxes.astype(float), boxes.astype(float), to_move=0) > 0
    np.fill_diagonal(overlaps, 0)

    all_possib = np.ones_like(overlaps, dtype=bool)
    np.fill_diagonal(all_possib, 0)

    if must_overlap:
        possible_boxes = np.column_stack(np.where(overlaps))
        if possible_boxes.size == 0:
            possible_boxes = np.column_stack(np.where(all_possib))
    else:
        possible_boxes = np.column_stack(np.where(all_possib))
    return possible_boxes


def bbox_overlaps(boxes1, boxes2, to_move=1):
    """Compute pairwise intersection areas between two sets of boxes.

    Parameters
    ----------
    boxes1, boxes2 : numpy.ndarray, shape [N, 4] / [M, 4]  (x1, y1, x2, y2)
    to_move : int
        Pixel offset added to ``(rb - lt)`` before clipping (legacy behaviour).

    Returns
    -------
    numpy.ndarray, shape [N, M]
    """
    num_box1 = boxes1.shape[0]
    num_box2 = boxes2.shape[0]
    lt = np.maximum(
        boxes1.reshape([num_box1, 1, -1])[:, :, :2],
        boxes2.reshape([1, num_box2, -1])[:, :, :2],
    )
    rb = np.minimum(
        boxes1.reshape([num_box1, 1, -1])[:, :, 2:],
        boxes2.reshape([1, num_box2, -1])[:, :, 2:],
    )
    wh = (rb - lt + to_move).clip(min=0)
    return wh[:, :, 0] * wh[:, :, 1]
