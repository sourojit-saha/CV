from configparser import Interpolation
import numpy as np
import cv2
# from hw2.python.planarH import compute_x1x2
from planarH import compute_x1x2
from planarH import generate_x1_x2
import skimage.io 
import skimage.color
from matchPics import matchPics
from planarH import computeH_ransac
from opts import get_opts
from planarH import compositeH

# Import necessary functions

# Q2.2.4

def warpImage(opts):

    img1 = cv2.imread('../data/cv_desk.png')
    img2 = cv2.imread('../data/cv_cover.jpg')
    img3 = cv2.imread('../data/hp_cover.jpg')
    img3 = cv2.resize(img3, (img2.shape[1], img2.shape[0]), interpolation=cv2.INTER_AREA)

    matches, loc1, loc2 = matchPics(img1, img2, opts)

    x1, x2 = generate_x1_x2(loc1, loc2, matches)
    H, inliers = computeH_ransac(x1, x2, opts)

    print('H: ',H)
    print('Inliers: ', inliers)

    img_out = cv2.warpPerspective(img3, H, (img1.shape[1], img1.shape[0]))
    composite_img = compositeH(H, img_out, img1)

    cv2.imshow("img_out",composite_img)
    cv2.waitKey(0)

    return

if __name__ == "__main__":

    opts = get_opts()
    warpImage(opts)


