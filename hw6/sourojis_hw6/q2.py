# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# April 27, 2022
# ##################################################################### #

import numpy as np
import matplotlib.pyplot as plt
from q1 import loadData, estimateAlbedosNormals, displayAlbedosNormals, estimateShape, plotSurface 
from q1 import estimateShape
from utils import enforceIntegrability, plotSurface 

def estimatePseudonormalsUncalibrated(I):

    """
    Question 2 (b)

    Estimate pseudonormals without the help of light source directions. 

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P matrix of loaded images

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pseudonormals
    
    L : numpy.ndarray
        The 3 x 7 array of lighting directions

    """

    B = None
    L = None
    # Your code here
    u, sigma, vt = np.linalg.svd(I, full_matrices=False)
    sigma = sigma[:3]
    sigma = np.diag(sigma)
    L = u[:,:3]@sigma
    L = L.T
    B = vt[:3,:]
    return B, L

def plotBasRelief(B, mu, nu, lam):

    """
    Question 2 (f)

    Make a 3D plot of of a bas-relief transformation with the given parameters.

    Parameters
    ----------
    B : numpy.ndarray
        The 3 x P matrix of pseudonormals

    mu : float
        bas-relief parameter

    nu : float
        bas-relief parameter
    
    lambda : float
        bas-relief parameter

    Returns
    -------
        None

    """

    # Your code here
    pass

if __name__ == "__main__":

    # Part 2 (b)
    # Your code here
    I, L, s = loadData('../data/')
    B,L_new = estimatePseudonormalsUncalibrated(I)
    albedos, normals = estimateAlbedosNormals(B)
    albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)

    print("L:\n", L)
    print("L_new:\n", L_new)

    plt.imsave('2b-albedos.png', albedoIm, cmap = 'gray')
    plt.imsave('2b-normals.png', normalIm, cmap = 'rainbow')
    plt.imshow(albedoIm, cmap = 'gray')
    plt.show()
    plt.imshow(normalIm, cmap = 'rainbow')
    plt.show()

    # Part 2 (d)
    # Your code here
    # attempt 1
    surface = estimateShape(normals, s)
    # plotSurface(surface)

    # Part 2 (e)
    # Your code here
    # attempt 2
    print("attempt 2")
    normals_new = enforceIntegrability(normals, s)
    surface_new = estimateShape(normals_new, s)
    # plotSurface(surface_new)
    # Part 2 (f)
    # Your code here
    #u,v,lamba
    # ref = np.array([[0,0,1],
    #                 [15,0,1],
    #                 [0,15,1],
    #                 [0,0,15],
    #                 [15,15,15],
    #                 [0,0,0.0000001]])

    ref = np.array([[100,100,100]])
    G = np.eye(3)
    for i in range(ref.shape[0]):
        print(ref[i,:])
        G[2,:] = ref[i,:]
        albedos, normals = estimateAlbedosNormals(B)
        normals = np.linalg.inv(G.T)@normals
        normals_new = enforceIntegrability(normals, s)
        normals_new = normals_new/np.linalg.norm(normals_new,axis=0)
        surface = estimateShape(normals_new, s)
        plotSurface(surface)
