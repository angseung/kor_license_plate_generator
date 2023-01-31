import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("Z01sk4261X.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
H, W = img.shape[:2]

pad = 50
img_padded = np.ones([H, W + 2 * pad, 3], dtype=np.uint8)
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

plt.imshow(result)
plt.show()
