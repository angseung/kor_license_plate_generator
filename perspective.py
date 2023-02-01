import cv2
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
from plate_generator import parse_label, label_yolo2voc, label_voc2yolo, draw_bbox_on_img


def warp_point(x: int, y: int, M: np.ndarray) -> Tuple[int, int]:
    d = M[2, 0] * x + M[2, 1] * y + M[2, 2]

    return (
        int((M[0, 0] * x + M[0, 1] * y + M[0, 2]) / d),  # x
        int((M[1, 0] * x + M[1, 1] * y + M[1, 2]) / d),  # y
    )


img = cv2.imread("Z01sk4261X.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
H, W = img.shape[:2]

labels = parse_label("Z01sk4261X.txt")
img = draw_bbox_on_img(img, labels, is_voc=False)
labels_voc = label_yolo2voc(labels, H, W)

pad = 50
img_padded = np.zeros([H, W + 2 * pad, 3], dtype=np.uint8)
img_padded[:, :, :] = 255
img_padded[:, pad : -pad, :] = img

point_before = np.float32([
    [2 * pad, 0],
    [W, 0],
    [pad, H],
    [W + pad, H]
])
point_after = np.float32([
    [pad, 0],
    [W + pad, 0],
    [pad, H],
    [W + pad, H]
])
mtrx = cv2.getPerspectiveTransform(point_before, point_after)
result = cv2.warpPerspective(img_padded, mtrx, img_padded.shape[:2][::-1])

for i, label in enumerate(labels_voc):
    xtl, ytl, xbr, ybr = label.tolist()[1:]
    xtl += pad
    xbr += pad
    # xtl_new, ytl_new , _ = np.matmul(mtrx, np.array([xtl, ytl, 1])).tolist()
    # xtr_new, _ , _ = np.matmul(mtrx, np.array([xbr, ytl, 1])).tolist()
    # xbl_new, _ , _ = np.matmul(mtrx, np.array([xtl, ybr, 1])).tolist()
    # xbr_new, ybr_new , _ = np.matmul(mtrx, np.array([xbr, ybr, 1])).tolist()

    # transformed_points = cv2.warpPerspective(np.float32([xtl, ytl]), mtrx, (2, 1), cv2.WARP_INVERSE_MAP)
    # transformed_points = cv2.warpPerspective(np.float32([xtl, ytl]), mtrx, (2, 1))
    xtl_new, ytl_new = warp_point(xtl, ytl, mtrx)
    xtr_new, ytr_new = warp_point(xbr, ytl, mtrx)
    xbl_new, ybl_new = warp_point(xtl, ybr, mtrx)
    xbr_new, ybr_new = warp_point(xbr, ybr, mtrx)

    # w_new, h_new = xtr_new - xtl_new, ybr_new - ytl_new
    # w_new, h_new = xbr_new - xbl_new, ybr_new - ytl_new
    # xbr_new -= pad
    # labels_voc[i, 1:] = np.uint32([xtl_new, ytl_new, xtl_new + w_new, ytl_new + h_new])
    # labels_voc[i, 1:] = np.uint32([xbr_new - w_new, ybr_new - h_new, xbr_new, ybr_new])

# labels_yolo = label_voc2yolo(labels_voc, *result.shape[:2])
# result = draw_bbox_on_img(result, labels_yolo, is_voc=False)
# result = draw_bbox_on_img(result, labels, is_voc=False)

plt.subplot(211)
plt.imshow(img_padded)
plt.plot(xtl, ytl, "og", markersize=5)  # og:shorthand for green circle
plt.plot(xbr, ybr, "og", markersize=5)  # og:shorthand for green circle

plt.subplot(212)
plt.imshow(result)
plt.plot(xtl_new, ytl_new, "og", markersize=5)  # og:shorthand for green circle
plt.plot(xbr_new, ybr_new, "og", markersize=5)  # og:shorthand for green circle
plt.show()
