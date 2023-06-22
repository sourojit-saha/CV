import numpy as np
import cv2


img_left = np.asarray(cv2.imread('../data/left_img.jpg'))
img_right = np.asarray(cv2.imread('../data/right_img.jpg'))
print(img_left.shape)
scale = 0.5
height = 300
width = 600
dim = (width, height)
img_left = cv2.resize(img_left, dim)
img_right = cv2.resize(img_right, dim)
stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
error, result = stitcher.stitch((img_right, img_left))

cv2.imshow("left", img_left)
cv2.imshow("right", img_right)
cv2.imshow("result", result)

cv2.waitKey(0)