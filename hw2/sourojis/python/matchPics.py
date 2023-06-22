import numpy as np
import cv2
import skimage.color
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection
from helper import plotMatches
from opts import get_opts
import os

# Q2.1.4


def matchPics(I1, I2, opts):
    """
    Match features across images

    Input
    -----
    I1, I2: Source images
    opts: Command line args

    Returns
    -------
    matches: List of indices of matched features across I1, I2 [p x 2]
    locs1, locs2: Pixel coordinates of matches [N x 2]
    """

    ratio = opts.ratio  # 'ratio for BRIEF feature descriptor'
    sigma = opts.sigma  # 'threshold for corner detection using FAST feature detector'

    matches, locs1, locs2 = None, None, None


    img1 = I1
    img2 = I2
    # TODO: Convert Images to GrayScale
    img1_grey = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_grey = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # TODO: Detect Features in Both Images
    locs1 = corner_detection(img1_grey, sigma)
    locs2 = corner_detection(img2_grey, sigma)
    # TODO: Obtain descriptors for the computed feature locations
    desc1, loc1 = computeBrief(img1_grey, locs1)
    desc2, loc2 = computeBrief(img2_grey, locs2)
    # TODO: Match features using the descriptors
    matches = briefMatch(desc1, desc2, ratio)

    return matches, loc1, loc2

# if __name__ == "__main__":
    # opts = get_opts()
    # matchPics("cv_cover.jpg","cv_desk.png", opts)