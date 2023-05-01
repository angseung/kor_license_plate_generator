import cv2
from matplotlib import pyplot as plt
from utils import parse_label, rotate_img_and_bboxes, draw_bbox_on_img, label_voc2yolo, label_yolo2voc

img_ori = cv2.imread("test.png")
height, width = img_ori.shape[:2]
labels = parse_label("test.txt")
labels_yolo = label_voc2yolo(labels, w=width, h=height)

img, labels_rot = rotate_img_and_bboxes(img_ori, labels_yolo, angle=30, bg_color="black")
labels_voc = label_yolo2voc(labels_rot, w=width, h=height)
img_ori_bbox = draw_bbox_on_img(img_ori, labels_yolo)
img_rot_bbox = draw_bbox_on_img(img, labels_voc)

plt.subplot(211)
plt.imshow(img_ori_bbox)
plt.subplot(212)
plt.imshow(img_rot_bbox)
plt.show()
