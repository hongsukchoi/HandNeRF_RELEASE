import numpy as np
import cv2
import random
import math

from config import cfg

def load_img(path, order='RGB'):
    img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if not isinstance(img, np.ndarray):
        raise IOError("Fail to read %s" % path)

    if order=='RGB':
        img = img[:,:,::-1].copy()
    
    img = img.astype(np.float32)
    return img


def process_bbox(bbox, img_width, img_height, out_shape, expand_ratio):
    # sanitize bboxes
    x, y, w, h = bbox
    x1 = np.max((0, x))
    y1 = np.max((0, y))
    x2 = np.min((img_width - 1, x1 + np.max((0, w - 1))))
    y2 = np.min((img_height - 1, y1 + np.max((0, h - 1))))
    if w * h > 0 and x2 > x1 and y2 > y1:
        bbox = np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)
    else:
        # just return the image shape
        return np.array([0, 0, img_width, img_height], dtype=np.float32)

    # aspect ratio preserving bbox
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w / 2.
    c_y = bbox[1] + h / 2.
    aspect_ratio = out_shape[1] / out_shape[0]
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w * expand_ratio
    bbox[3] = h * expand_ratio
    bbox[0] = c_x - bbox[2] / 2.
    bbox[1] = c_y - bbox[3] / 2.

    bbox = bbox.astype(np.float32)
    return bbox

def get_affine_trans_mat(bbox, out_shape):
    src_center = np.array([bbox[0] + 0.5 * bbox[2], bbox[1] + 0.5 * bbox[3]], dtype=np.float32)
    src_downdir = np.array([0, bbox[3] * 0.5], dtype=np.float32)
    src_rightdir = np.array([bbox[2] * 0.5, 0], dtype=np.float32)
    
    dst_w = out_shape[1]
    dst_h = out_shape[0]
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    trans = trans.astype(np.float32)
    return trans


def get_intersection(bbox_a, bbox_b):  # returns intersection area if bboxes intersect
    # x, y, w, h
    a_xmin, a_xmax = bbox_a[0], bbox_a[0] + bbox_a[2]
    a_ymin, a_ymax = bbox_a[1], bbox_a[1] + bbox_a[3]
    b_xmin, b_xmax = bbox_b[0], bbox_b[0] + bbox_b[2]
    b_ymin, b_ymax = bbox_b[1], bbox_b[1] + bbox_b[3]

    dx = min(a_xmax, b_xmax) - max(a_xmin, b_xmin)
    dy = min(a_ymax, b_ymax) - max(a_ymin, b_ymin)
    if (dx >= 0) and (dy >= 0):
        return dx*dy
    else:
        return -1


def rigid_transform_3D(A, B):
    n, dim = A.shape
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    H = np.dot(np.transpose(A - centroid_A), B - centroid_B) / n
    U, s, V = np.linalg.svd(H)
    R = np.dot(np.transpose(V), np.transpose(U))
    if np.linalg.det(R) < 0:
        s[-1] = -s[-1]
        V[2] = -V[2]
        R = np.dot(np.transpose(V), np.transpose(U))

    varP = np.var(A, axis=0).sum()
    c = 1/varP * np.sum(s)

    t = -np.dot(c*R, np.transpose(centroid_A)) + np.transpose(centroid_B)
    return c, R, t


def rigid_align(A, B):
    c, R, t = rigid_transform_3D(A, B)
    A2 = np.transpose(np.dot(c*R, np.transpose(A))) + t
    return A2
