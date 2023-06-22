import numpy as np
import cv2

# Import necessary functions
from helper import loadVid
from planarH import computeH_ransac
from planarH import matchPics
from planarH import get_opts
from planarH import generate_x1_x2
from planarH import compositeH
import os
# Q3.1
src_frame = loadVid("../data/ar_source.mov")# Panda-shorter
tar_frame = loadVid("../data/book.mov")# Book-longer
cv_cover = cv2.imread('../data/cv_cover.jpg')
src_len = len(src_frame)
tar_len = len(tar_frame)
opts = get_opts()
src_shape = src_frame[0].shape
tar_shape = tar_frame[0].shape
cv_cover_shape = cv_cover.shape


i = 0
while i < src_len:
    panda_frame = src_frame[i]
    book_frame = tar_frame[i]
    matches, loc1, loc2 = matchPics(book_frame, cv_cover, opts)
    x1, x2 = generate_x1_x2(loc1, loc2, matches)
    H, inliers = computeH_ransac(x1, x2, opts)


    panda_frame_midpoint = int(np.asarray(panda_frame.shape[1])/2)
    cv_cover_limit = int(np.asarray(cv_cover_shape[1])/2)

    x_low = panda_frame_midpoint-cv_cover_limit
    x_high = panda_frame_midpoint+cv_cover_limit
    panda_crop = panda_frame[:, x_low:x_high]
    y_nonzero, x_nonzero, _ = np.nonzero(panda_frame)

    panda_frame_crop = panda_frame[50:300,:]
    panda_frame_resize = cv2.resize(panda_frame_crop, (cv_cover.shape[1], cv_cover.shape[0]), interpolation=cv2.INTER_LINEAR)
    panda_frame_resize = np.where(panda_frame_resize<1,1,panda_frame_resize )

    img_out = cv2.warpPerspective(panda_frame_resize, H, (book_frame.shape[1], book_frame.shape[0]))
    composite_img = compositeH(H, img_out, book_frame)

    file_name = "../data/results/" + str(i) + ".jpg"
    cv2.imwrite(file_name, composite_img)
    print(file_name)

    i=i+1
print("Done!!")
