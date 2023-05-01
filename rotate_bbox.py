import cv2
from matplotlib import pyplot as plt
from utils import (
    parse_label,
    rotate_img_and_bboxes,
    draw_bbox_on_img,
    label_voc2yolo,
    label_yolo2voc,
)

img_ori = cv2.imread("test.png")
height, width = img_ori.shape[:2]
labels = parse_label("test.txt")
labels_yolo = label_voc2yolo(labels, w=width, h=height)

# angle in degree, labels MUST BE YOLO FORMATTED
img_rot, labels_rot = rotate_img_and_bboxes(
    img_ori, labels_yolo, angle=30, bg_color="black"
)
width_new, height_new = img_rot.shape[:2]  # get a new shape of the rotated image
labels_voc = label_yolo2voc(labels_rot, w=width_new, h=height_new)

# labels for draw_bbox_on_img can be both VOC and YOLO format
img_ori_bbox = draw_bbox_on_img(img_ori, labels_yolo)
img_rot_bbox = draw_bbox_on_img(img_rot, labels_voc)

plt.subplot(121)
plt.imshow(img_ori_bbox)
plt.subplot(122)
plt.imshow(img_rot_bbox)
plt.show()
