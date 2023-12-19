import random
import cv2
import numpy as np


def random_rectangular_occlusion(org_mask):
    mask = org_mask.copy()
    height, width = mask.shape[:2]
    bbox = cv2.boundingRect(mask.astype(np.uint8))  # x, y, w, h

    # get occluding range
    y_thickness = height // 10
    y1 = random.sample(list(range(bbox[1], bbox[1] + bbox[3] - y_thickness)), 1)[0]
    y2 = y1 + y_thickness

    x_thickness = height // 10
    x1 = random.sample(list(range(bbox[0], bbox[0] + bbox[2] - x_thickness)), 1)[0]
    x2 = x1 + x_thickness

    # occlude
    mask[y1:y2] = 0
    mask[:, x1:x2] = 0

    return mask
