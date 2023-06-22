# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# April 27, 2022
# ##################################################################### #

import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgb2xyz
from utils import plotSurface, integrateFrankot
import os as os
from skimage import io

from PIL import Image
import skimage.color
import cv2 as cv2

def renderNDotLSphere(center, rad, light, pxSize, res):

    """
    Question 1 (b)

    Render a hemispherical bowl with a given center and radius. Assume that
    the hollow end of the bowl faces in the positive z direction, and the
    camera looks towards the hollow end in the negative z direction. The
    camera's sensor axes are aligned with the x- and y-axes.

    Parameters
    ----------
    center : numpy.ndarray
        The center of the hemispherical bowl in an array of size (3,)

    rad : float
        The radius of the bowl

    light : numpy.ndarray
        The direction of incoming light

    pxSize : float
        Pixel size

    res : numpy.ndarray
        The resolution of the camera frame

    Returns
    -------
    image : numpy.ndarray
        The rendered image of the hemispherical bowl
    """

    [X, Y] = np.meshgrid(np.arange(res[0]), np.arange(res[1]))
    X = (X - res[0]/2) * pxSize*1.e-4
    Y = (Y - res[1]/2) * pxSize*1.e-4
    Z = np.sqrt(rad**2+0j-X**2-Y**2)
    X[np.real(Z) == 0] = 0
    Y[np.real(Z) == 0] = 0
    Z = np.real(Z)
    image = None
    # Your code here
    n_vec = np.dstack((X,Y,Z))
    n_norm_factor = np.linalg.norm(n_vec, axis = 2)

    n_vec_norm = np.dstack((X/n_norm_factor, Y/n_norm_factor, Z/n_norm_factor))
    n_vec_norm[np.isnan(n_vec_norm)]=0


    dot_prod = np.sum((n_vec_norm * light), axis = 2) 

    image = dot_prod
    print("renderNDotLSphere done")
    return image


def loadData(path = "../data/"):

    """
    Question 1 (c)

    Load data from the path given. The images are stored as input_n.tif
    for n = {1...7}. The source lighting directions are stored in
    sources.mat.

    Parameters
    ---------
    path: str
        Path of the data directory

    Returns
    -------
    I : numpy.ndarray
        The 7 x P matrix of vectorized images

    L : numpy.ndarray
        The 3 x 7 matrix of lighting directions

    s: tuple
        Image shape

    """

    I = None
    L = None
    s = None
    img_path = path + 'input_1.tif'
    temp = Image.open(img_path)
    temp_array = np.array(temp)
    I = np.zeros((0, temp_array.shape[0]*temp_array.shape[1]))
    print ("Shape of I ", I.shape)
    for i in range(1, 8):
        img_path = path + 'input_' + str(i) + '.tif'
        dummy_img_array = np.asarray(Image.open(img_path), dtype = np.uint32)
        dummy_img_colorsp = rgb2xyz(dummy_img_array)
        dummy_vector = dummy_img_colorsp[:, :, 1]
        dummy_vector = dummy_vector.flatten()
        I = np.vstack((I, dummy_vector))
        if i==7:
            r, c, _ = dummy_img_array.shape
            s = [r, c]
    
    L = np.load(path + 'sources.npy')
    L = L.T

    return I, L, s


def estimatePseudonormalsCalibrated(I, L):

    """
    Question 1 (e)

    In calibrated photometric stereo, estimate pseudonormals from the
    light direction and image matrices

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P array of vectorized images

    L : numpy.ndarray
        The 3 x 7 array of lighting directions

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals
    """

    B = None
    # Your code here
    #Ax=B
    print(L.shape, I.shape)
    A = L@L.T
    y = L@I
    x = np.linalg.pinv(A)@y
    B = x
    print("estimatePseudonormalsCalibrated done")

    return B


def estimateAlbedosNormals(B):

    '''
    Question 1 (e)

    From the estimated pseudonormals, estimate the albedos and normals

    Parameters
    ----------
    B : numpy.ndarray
        The 3 x P matrix of estimated pseudonormals

    Returns
    -------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals
    '''

    albedos = None
    normals = None
    # Your code here

    # print("S: ",s)
    print("B: ", B.shape)
    albedos = np.linalg.norm(B, axis = 0)
    normals = B/albedos



    return albedos, normals


def displayAlbedosNormals(albedos, normals, s):

    """
    Question 1 (f, g)

    From the estimated pseudonormals, display the albedo and normal maps

    Please make sure to use the `coolwarm` colormap for the albedo image
    and the `rainbow` colormap for the normals.

    Parameters
    ----------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    -------
    albedoIm : numpy.ndarray
        Albedo image of shape s

    normalIm : numpy.ndarray
        Normals reshaped as an s x 3 image

    """

    albedoIm = None
    normalIm = None
    # Your code here

    albedoIm = albedos.reshape(s)

    channel1 = normals[0,:].reshape(s)
    channel2 = normals[1,:].reshape(s)
    channel3 = normals[2,:].reshape(s)

    print("--->>max-min: ", np.max(normalIm), np.min(normalIm))

    # normalIm = (np.dstack((channel1, channel2, channel3))+1)/2
    
    normalIm = np.dstack((channel1, channel2, channel3))
    normalIm = (normalIm - np.min(normalIm))/(np.max(normalIm)-np.min(normalIm))

    print("--->>max-min: ", np.max(normalIm), np.min(normalIm))

    # print("albedosIm: ", albedoIm.shape)
    # print("normalsIm: ", normalIm.shape)

    plt.imshow(albedoIm,cmap='gray')
    plt.imshow(normalIm,cmap='rainbow')
    # plt.show()

    # z = np.linalg.norm(normalIm, axis = 2)
    # plt.imshow(z)
    # plt.show()


    return albedoIm, normalIm


def estimateShape(normals, s):

    """
    Question 1 (j)

    Integrate the estimated normals to get an estimate of the depth map
    of the surface.

    Parameters
    ----------
    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    ----------
    surface: numpy.ndarray
        The image, of size s, of estimated depths at each point

    """

    surface = None
    # Your code here
    x_deriv = -normals[0,:]/normals[2,:]
    y_deriv = -normals[1,:]/normals[2,:]

    x_deriv = x_deriv.reshape(s)
    y_deriv = y_deriv.reshape(s)

    surface = integrateFrankot(x_deriv, y_deriv)
    # print("estimateShape done")
    return surface


if __name__ == '__main__':
    # Part 1(b)
    radius = 0.75 # cm
    center = np.asarray([0, 0, 0]) # cm
    pxSize = 7 # um
    res = (3840, 2160)

    light = np.asarray([1, 1, 1])/np.sqrt(3)
    image = renderNDotLSphere(center, radius, light, pxSize, res)
    plt.figure()
    plt.imshow(image, cmap = 'gray')
    plt.imsave('1b-a.png', image, cmap = 'gray')

    light = np.asarray([1, -1, 1])/np.sqrt(3)
    image = renderNDotLSphere(center, radius, light, pxSize, res)
    plt.figure()
    plt.imshow(image, cmap = 'gray')
    plt.imsave('1b-b.png', image, cmap = 'gray')

    light = np.asarray([-1, -1, 1])/np.sqrt(3)
    image = renderNDotLSphere(center, radius, light, pxSize, res)
    plt.figure()
    plt.imshow(image, cmap = 'gray')
    plt.imsave('1b-c.png', image, cmap = 'gray')

    # Part 1(c)
    I, L, s = loadData('../data/')

    # Part 1(d)
    # Your code here
    print("I: ", I.shape)
    print("L: ", L.shape)

    u, sig, vt = np.linalg.svd(I, full_matrices=False)
    print("sigma: ",sig)
    rank = np.linalg.matrix_rank(I)
    print("rank: ", rank)
    sig[3:] = 0
    sig = np.diag(sig)
    # print(sig)
    I = u@sig@vt
    print("I_new rank: ",np.linalg.matrix_rank(I))
    # Part 1(e)
    B = estimatePseudonormalsCalibrated(I, L)

    # Part 1(f)
    albedos, normals = estimateAlbedosNormals(B)
    albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)
    plt.imsave('1f-a.png', albedoIm*1000, cmap = 'gray')
    plt.imsave('1f-b.png', normalIm, cmap = 'rainbow')

    # Part 1(i)
    surface = estimateShape(normals, s)
    plotSurface(surface)
