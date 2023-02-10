import random
from datetime import datetime
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import (
    parse_label,
    draw_bbox_on_img,
    blend_bgra_on_bgr,
    random_perspective,
    remove_bg_from_img,
    get_angle_from_warp,
)

random.seed(datetime.now().timestamp())

if __name__ == "__main__":
    img = cv2.imread("Z01sk4261X.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    labels = parse_label("Z01sk4261X.txt")
    H, W = img.shape[:2]

    plt.subplot(221)
    result1, labels1, mtrx = random_perspective(
        img, labels, mode="left", return_mat=True, pads=(H // 4, H // 4)
    )
    result1 = draw_bbox_on_img(result1, labels1)
    plt.imshow(result1)
    # a_1, a_2, a_3 = get_angle_from_warp(mtrx)

    plt.subplot(222)
    result2, labels2, mtrx = random_perspective(
        img, labels, mode="right", return_mat=True, pads=(H // 4, H // 4)
    )
    result2 = draw_bbox_on_img(result2, labels2)
    plt.imshow(result2)

    plt.subplot(223)
    result3, labels3, mtrx = random_perspective(
        img, labels, mode="top", return_mat=True, pads=(W // 8, W // 8)
    )
    # result3, labels3 = random_perspective(result3, labels3, mode="left")
    result3 = draw_bbox_on_img(result3, labels3)
    a_1, a_2, a_3 = get_angle_from_warp(mtrx)
    plt.imshow(result3)
    a_1, a_2, a_3 = get_angle_from_warp(mtrx)

    plt.subplot(224)
    result4, labels4, mtrx = random_perspective(
        img, labels, mode="bottom", return_mat=True, pads=(W // 8, W // 8)
    )
    # result4, labels4 = random_perspective(result4, labels4, mode="right")
    result4 = draw_bbox_on_img(result4, labels4)
    plt.imshow(result4)
    plt.show()

    tmp = np.zeros((3000, 3000, 3), dtype=np.uint8)
    tmp[:, :, :] = 0
    dst = remove_bg_from_img(result4)
    img_tr = blend_bgra_on_bgr(fg=dst, bg=tmp, row=0, col=1)

    plt.imshow(img_tr)
    plt.show()

    # Writing and saving to a new image
    img_tr = cv2.cvtColor(img_tr, cv2.COLOR_BGR2RGB)
    cv2.imwrite("this.png", img_tr)
