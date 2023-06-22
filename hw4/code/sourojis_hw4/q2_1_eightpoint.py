import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv2
from scipy.linalg import block_diag
from helper import displayEpipolarF, calc_epi_error, toHomogenous, refineF, _singularize

# Insert your package here



'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix

    HINTS:
    (1) Normalize the input pts1 and pts2 using the matrix T.
    (2) Setup the eight point algorithm's equation.
    (3) Solve for the least square solution using SVD. 
    (4) Use the function `_singularize` (provided) to enforce the singularity condition. 
    (5) Use the function `refineF` (provided) to refine the computed fundamental matrix. 
        (Remember to usethe normalized points instead of the original points)
    (6) Unscale the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    # Replace pass by your implementation

    # T1 = np.asarray([[2/M[0,1],0,-1],
    #                 [0,2/M[0,0], -1],
    #                 [0,0,1]]) 
    # T2 = np.asarray([[2/M[1,1],0,-1],
    #             [0,2/M[1,0], -1],
    #             [0,0,1]]) 
    T1 = np.asarray([[1/M,0,0],
                [0,1/M, 0],
                [0,0,1]]) 
    # T2 = T1
    # In our case T1 = T2.
    pts1 = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
    pts2 = np.hstack((pts2, np.ones((pts2.shape[0], 1))))

    pts1_normalised = (T1@pts1.T).T
    pts2_normalised = (T1@pts2.T).T
    
    # pts1_normalised = pts1/M
    # pts2_normalised = pts2/M

    #the size of pts1 and pts2 should be same

    A = np.zeros((0,9))
    for i in range(pts1_normalised.shape[0]):
        x1, y1, dummy = pts1_normalised[i,:]
        x2, y2, dummy = pts2_normalised[i,:]
        # x1, y1 = pts1_normalised[i,:]
        # x2, y2 = pts2_normalised[i,:]
        a = np.asarray([[x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1]])
        # a = np.asarray([[x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]])
        # print(a)
        A = np.vstack((A, a))

    U, sigma, Vt = np.linalg.svd(A)
    # print(sigma)
    F = Vt[-1,:].reshape(3,3).T
    # print("F: \n", F)

    # u, sigma, vt = np.linalg.svd(F, full_matrices=True)
    # # print(sigma.shape)
    # Sigma = np.zeros((len(sigma), len(sigma)))
    # for i in range(Sigma.shape[0]):
    #     for j in range(Sigma.shape[0]):
    #         if i == j:
    #             Sigma[i,j] = sigma[i]
    # # print(Sigma)
    # Sigma[2,2] = 0
    # F = u@Sigma@vt
    # print(np.linalg.matrix_rank(F))

    # print(pts1_normalised)
    F = _singularize(F)
    F = refineF(F, pts1_normalised[:,0:2], pts2_normalised[:,0:2])

    # F = _singularize(F)
    # F = refineF(F, pts1_normalised, pts2_normalised)
    # print(F)

    F = T1.T@F@T1
    F = F/F[2,2]

    # print(F)

    return F




if __name__ == "__main__":
        
    correspondence = np.load('../data/some_corresp.npz') # Loading correspondences
    intrinsics = np.load('../data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    im1 = plt.imread('../data/im1.png')
    im2 = plt.imread('../data/im2.png')

    # cv2.imshow('im1', im1)
    # cv2.imshow('im2', im2)
    # cv2.waitKey(0)

    # height1, width1, channel1 = im1.shape
    # height2, width2, channel2 = im2.shape
    # M = np.asarray([np.max([height1, height2]), np.max([width1, width2])])
    # M = np.asarray([im1.shape, im2.shape])
    # print("M:\n", M)
    M=np.max([*im1.shape, *im2.shape])
    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))
    # F = eightpoint(pts1, pts2, M)
    # Q2.1
    # Write your code here



    # Simple Tests to verify your implementation:
    pts1_homogenous, pts2_homogenous = toHomogenous(pts1), toHomogenous(pts2)

    assert(F.shape == (3, 3))
    assert(F[2, 2] == 1)
    assert(np.linalg.matrix_rank(F) == 2)
    print("------------------------------")
    print(np.mean(calc_epi_error(pts1_homogenous, pts2_homogenous, F)))
    assert(np.mean(calc_epi_error(pts1_homogenous, pts2_homogenous, F)) < 1)

    np.savez("q2_1.npz", F=F, M=M) 

    displayEpipolarF(im1, im2, F)