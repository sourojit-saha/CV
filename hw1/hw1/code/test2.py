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

    arr1=np.random.randint(100, size=(4,4))
    rows, cols = arr1.shape
    print(arr1)
    a=np.empty((rows, cols, 3))
    print(a.shape)
    a[:,:,0]=arr1
    a[:,:,1]=arr1
    a[:,:,2]=arr1
    print(a[:,:,0])
    print(a[:,:,1])
    print(a[:,:,2])
    # # arr2 = np.matlib.repmat(arr1, 1, 1, 3)
    # arr2 = np.array([arr1, arr1, arr1])
    # print(arr1)
    # # arr1=np.reshape(arr1, (4,4,1))
    # print(arr1.shape)
    # # arr2 = np.repeat(arr1, 3, axis=2)
    # print(arr1)
    # print("\n")
    # print(arr2)
    # print(arr2.shape)



if __name__ == '__main__':
    main()
