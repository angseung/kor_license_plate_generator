import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("Z01sk4261X.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
H, W = img.shape[:2]

pad = 80

point_before = np.float32([
    [pad, 0],
    [W - pad, 0],
    [0, H],
    [W, H]
])
point_after = np.float32([
    [0, 0],
    [W, 0],
    [0, H],
    [W, H]
])
mtrx = cv2.getPerspectiveTransform(point_before, point_after)
result = cv2.warpPerspective(img, mtrx, (W, H))

plt.imshow(result)
plt.show()
