import matplotlib
import numpy as np
import cv2
from matchPics import matchPics
from opts import get_opts
import os
from scipy import ndimage, misc
import helper
import matplotlib.pyplot as plt
from displayMatch  import displayMatched
# Q2.1.6


def rotTest(opts):

    path = os.path.join("../data/","cv_cover.jpg")
    img = cv2.imread(path)

    opts = get_opts()
 
    range_i = 36
    hist = np.zeros(range_i)
    for i in range(range_i):#36
        i=9
        # Rotate Image
        print(i)
        angle = i*10
        img_rot = ndimage.rotate(img, angle, reshape=True)
        displayMatched(opts, img, img_rot )


        matches, locs1, locs2 = matchPics(img, img_rot, opts)

        hist[i] = len(matches)

        

    # Display histogram
    plt.bar(np.arange(range_i)*10, hist, color='blue', alpha=0.75)
    plt.show()

if __name__ == "__main__":

    opts = get_opts()
    rotTest(opts)