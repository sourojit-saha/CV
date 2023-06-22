from os.path import join
from turtle import width

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import util
import visual_words
import visual_recog
from opts import get_opts

import os, multiprocessing
from os.path import join, isfile

from PIL import Image 
import scipy.ndimage
import skimage.color
import sklearn.cluster

import cv2 as cv

import numpy.matlib


def main():
    opts = get_opts()

    out_dir = opts.out_dir
    data_dir = opts.data_dir

    # laundromat/sun_afrmdtnsnxzodzwq.jpg
    # laundromat/sun_ahkztsdqkecbpuoe.jpg
    # laundromat/sun_afrrjykuhhlwiwun.jpg

    # trained_system = np.load(join(out_dir, 'trained_system.npz'))
    # print(trained_system.files)

    # test_files = open(join(data_dir, 'test_files.txt')).read().splitlines()

    # for i in range(len(test_files)):
    #     # if i==204:
    #     #     continue
    #     img_path = join(opts.data_dir, test_files[i])
    #     img = Image.open(img_path)
    #     img = np.array(img).astype(np.float32)/255
    #     print("before idx: ", i, "--",img.shape)
    #     if len(img.shape)==2:
    #         img = np.matlib.repmat(img, 3, 1)
    #     print("after idx: ", i, "--",img.shape)
    
    # l=["laundromat/sun_afrmdtnsnxzodzwq.jpg", "laundromat/sun_ahkztsdqkecbpuoe.jpg", "laundromat/sun_afrrjykuhhlwiwun.jpg"]
    # for i in range(len(l)):
    #     img_path = join(opts.data_dir, l[i])
    #     img = Image.open(img_path)
    #     img = np.array(img).astype(np.float32)/255
    #     print("idx: ", i, "--",img.shape)

    conf = np.loadtxt(join(opts.out_dir, 'confmat.csv'), delimiter=",")
    print(conf)



if __name__ == '__main__':
    main()
