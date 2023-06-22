from configparser import Interpolation
import numpy as np
import cv2

# Import necessary functions

# Q4.1
# import the necessary packages
img = np.asarray(cv2.imread("../data/cv_desk.png"))
height, width, channels = img.shape
scale = 0.5

# print(height, width, channels)
img_g = cv2.GaussianBlur(img,(5,5),0)
img_l = img-img_g
height_new = int(img_g.shape[0] * scale)
width_new = int(img_g.shape[1] * scale)
dim = (width_new, height_new)
img_g_sub = cv2.resize(img_g, dim, interpolation = cv2.INTER_AREA)

dim = (width, height)
img_g_up = cv2.resize(img_g_sub, dim, interpolation = cv2.INTER_AREA)
img_up_re = img_g_up + img_l

cv2.imshow("img", img)
cv2.imshow("img_g", img_g)
cv2.imshow("img_l", img_l)
cv2.imshow("sub", img_g_sub)
cv2.imshow("up", img_g_up)
cv2.imshow("up", img_up_re)
cv2.waitKey(0)
# img_sub = img[np.asarray(range(0,img.shape[0],2)), np.asarray(range(0,img.shape[1],2))]
# print(np.asarray(range(0,10,2)))
# img_g = cv2.GaussianBlur(img,(11,11),0)
# img_g = img
# # img_l = img - img_g
# lvl =6
# gaus=[]
# lapl=[]
# gaus.append(img)
# for i in range(lvl):
#     print(i, len(gaus), len(lapl))
#     img_g = cv2.GaussianBlur(gaus[i],(11,11),0)
#     img_l = gaus[i]-img_g
#     lapl.append(img_l)
#     img_sub = img_g[range(0,img_g.shape[0],2), range(0,img_g.shape[1],2)]
#     gaus.append(img_sub)

#     cv2.imshow("g", img_sub)
#     cv2.imshow("l", img_l[i])
#     cv2.waitKey(0)
# # cv2.imshow("original", img)
# # cv2.imshow("gaussian", img_g)
# # cv2.imshow("laplacian", img_l)
# cv2.waitKey(0)