import random
from datetime import datetime
import cv2
import numpy as np
import matplotlib.pyplot as plt
from plate_generator import (
    parse_label,
    draw_bbox_on_img,
    blend_argb_with_rgb,
    random_perspective,
    remove_white_bg,
)

random.seed(datetime.now().timestamp())

if __name__ == "__main__":
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
    # result3, labels3 = random_perspective(result3, labels3, mode="left")
    result3 = draw_bbox_on_img(result3, labels3, is_voc=False)
    plt.imshow(result3)

    plt.subplot(224)
    result4, labels4 = random_perspective(img, labels, mode="bottom")
    # result4, labels4 = random_perspective(result4, labels4, mode="right")
    result4 = draw_bbox_on_img(result4, labels4, is_voc=False)
    plt.imshow(result4)
    plt.show()

    tmp = np.zeros((3000, 3000, 3), dtype=np.uint8)
    tmp[:, :, :] = 0
    dst = remove_white_bg(result4)
    img_tr = blend_argb_with_rgb(fg=dst, bg=tmp, row=0, col=1)

    plt.imshow(img_tr)
    plt.show()

    # Writing and saving to a new image
    img_tr = cv2.cvtColor(img_tr, cv2.COLOR_BGR2RGB)
    cv2.imwrite("this.png", img_tr)
