import random
from datetime import datetime
import cv2
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
from plate_generator import (
    parse_label,
    label_yolo2voc,
    label_voc2yolo,
    draw_bbox_on_img,
)

random.seed(datetime.now().timestamp())


def warp_point(x: int, y: int, M: np.ndarray) -> Tuple[int, int]:
    d = M[2, 0] * x + M[2, 1] * y + M[2, 2]

    return (
        int((M[0, 0] * x + M[0, 1] * y + M[0, 2]) / d),  # x
        int((M[1, 0] * x + M[1, 1] * y + M[1, 2]) / d),  # y
    )


def random_perspective(
    img: np.ndarray,
    labels: np.ndarray,
    mode: str,
    max_pad_order: Tuple[int, int] = (4, 8),
) -> Tuple[np.ndarray, np.ndarray]:
    H, W = img.shape[:2]
    max_pad_h = H // max_pad_order[0]
    max_pad_w = W // max_pad_order[1]

    labels = label_yolo2voc(labels, H, W)

    if mode in ["top", "bottom"]:
        pad_l, pad_r = (random.randint(1, max_pad_w), random.randint(1, max_pad_w))

        img_padded = np.zeros([H, W + pad_l + pad_r, 3], dtype=np.uint8)
        img_padded[:, :, :] = 255
        img_padded[:, pad_l:-pad_r, :] = img

        if mode == "top":
            point_before = np.float32(
                [
                    [2 * pad_l, 0],
                    [W + pad_l - pad_r, 0],
                    [2 * pad_l, H],
                    [W + pad_l - pad_r, H],
                ]
            )
            point_after = np.float32(
                [[pad_l, 0], [W + pad_l, 0], [2 * pad_l, H], [W + pad_l - pad_r, H]]
            )

        elif mode == "bottom":
            point_before = np.float32(
                [
                    [2 * pad_l, 0],
                    [W + pad_l - pad_r, 0],
                    [2 * pad_l, H],
                    [W + pad_l - pad_r, H],
                ]
            )
            point_after = np.float32(
                [[2 * pad_l, 0], [W + pad_l - pad_r, 0], [pad_l, H], [W + pad_l, H]]
            )

    elif mode in ["left", "right"]:
        pad_top, pad_bottom = (
            random.randint(1, max_pad_h),
            random.randint(1, max_pad_h),
        )
        img_padded = np.zeros([H + pad_top + pad_bottom, W, 3], dtype=np.uint8)
        img_padded[:, :, :] = 255
        img_padded[pad_top:-pad_bottom, :, :] = img

        if mode == "left":
            point_before = np.float32(
                [
                    [0, 2 * pad_top],
                    [W, pad_top],
                    [0, H + pad_top - pad_bottom],
                    [W, H + pad_top],
                ]
            )
            point_after = np.float32(
                [[0, pad_top], [W, pad_top], [0, H + pad_top], [W, H + pad_top]]
            )

        elif mode == "right":
            point_before = np.float32(
                [
                    [0, pad_top],
                    [W, 2 * pad_top],
                    [0, H + pad_top],
                    [W, H + pad_top - pad_bottom],
                ]
            )
            point_after = np.float32(
                [[0, pad_top], [W, pad_top], [0, H + pad_top], [W, H + pad_top]]
            )

    mtrx = cv2.getPerspectiveTransform(point_before, point_after)
    result = cv2.warpPerspective(img_padded, mtrx, img_padded.shape[:2][::-1])

    for i, label in enumerate(labels):
        xtl, ytl, xbr, ybr = label.tolist()[1:]

        if mode in ["top", "bottom"]:
            xtl += pad_l
            xbr += pad_l

        elif mode in ["left", "right"]:
            ytl += pad_top
            ybr += pad_top

        xtl_new, ytl_new = warp_point(xtl, ytl, mtrx)
        xtr_new, ytr_new = warp_point(xbr, ytl, mtrx)
        xbl_new, ybl_new = warp_point(xtl, ybr, mtrx)
        xbr_new, ybr_new = warp_point(xbr, ybr, mtrx)

        labels[i, 1:] = np.uint32(
            [
                (xtl_new + xbl_new) // 2,
                (ytl_new + ytr_new) // 2,
                (xtr_new + xbr_new) // 2,
                (ybl_new + ybr_new) // 2,
            ]
        )

    labels = label_voc2yolo(labels, *result.shape[:2])

    return result, labels


img = cv2.imread("Z01sk4261X.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
labels = parse_label("Z01sk4261X.txt")

plt.subplot(221)
result1, labels1 = random_perspective(img, labels, mode="left")
result1 = draw_bbox_on_img(result1, labels1, is_voc=False)
plt.imshow(result1)

plt.subplot(222)
result2, labels2 = random_perspective(img, labels, mode="right")
result2 = draw_bbox_on_img(result2, labels2, is_voc=False)
plt.imshow(result2)

plt.subplot(223)
result3, labels3 = random_perspective(img, labels, mode="top")
result3, labels3 = random_perspective(result3, labels3, mode="left")
result3 = draw_bbox_on_img(result3, labels3, is_voc=False)
plt.imshow(result3)

plt.subplot(224)
result4, labels4 = random_perspective(img, labels, mode="bottom")
result4, labels4 = random_perspective(result4, labels4, mode="right")
result4 = draw_bbox_on_img(result4, labels4, is_voc=False)
plt.imshow(result4)

plt.show()
