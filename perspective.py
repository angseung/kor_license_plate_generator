import cv2
import numpy as np
import matplotlib.pyplot as plt
from plate_generator import parse_label, label_yolo2voc, label_voc2yolo, draw_bbox_on_img

img = cv2.imread("Z01sk4261X.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
H, W = img.shape[:2]

labels = parse_label("Z01sk4261X.txt")
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
    xtl_new, ytl_new , _ = np.matmul(mtrx, np.array([xtl, ytl, 1])).tolist()
    xtr_new, _ , _ = np.matmul(mtrx, np.array([xbr, ytl, 1])).tolist()
    xbl_new, _ , _ = np.matmul(mtrx, np.array([xtl, ybr, 1])).tolist()
    xbr_new, ybr_new , _ = np.matmul(mtrx, np.array([xbr, ybr, 1])).tolist()

    # w_new, h_new = xtr_new - xtl_new, ybr_new - ytl_new
    w_new, h_new = xbr_new - xbl_new, ybr_new - ytl_new
    xbr_new -= pad
    # labels_voc[i, 1:] = np.uint32([xtl_new, ytl_new, xtl_new + w_new, ytl_new + h_new])
    labels_voc[i, 1:] = np.uint32([xbr_new - w_new, ybr_new - h_new, xbr_new, ybr_new])

labels_yolo = label_voc2yolo(labels_voc, *result.shape[:2])
result = draw_bbox_on_img(result, labels_yolo, is_voc=False)
# result = draw_bbox_on_img(result, labels, is_voc=False)

plt.imshow(result)
plt.show()
