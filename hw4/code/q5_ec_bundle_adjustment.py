from random import triangular
import numpy as np
import matplotlib.pyplot as plt
from q3_1_essential_matrix import essentialMatrix

from helper import camera2, displayEpipolarF, calc_epi_error, toHomogenous
from q2_1_eightpoint import eightpoint
# from q2_2_sevenpoint import sevenpoint
from q3_2_triangulate import findM2, triangulate

import scipy

# Insert your package here


# Helper functions for this assignment. DO NOT MODIFY!!!
"""
Helper functions.

Written by Chen Kong, 2018.
Modified by Zhengyi (Zen) Luo, 2021
"""
def plot_3D_dual(P_before, P_after):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Blue: before; red: after")
    ax.scatter(P_before[:,0], P_before[:,1], P_before[:,2], c = 'blue')
    ax.scatter(P_after[:,0], P_after[:,1], P_after[:,2], c='red')
    while True:
        x, y = plt.ginput(1, mouse_stop=2)[0]
        plt.draw()


'''
Q5.1: RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
            nIters, Number of iterations of the Ransac
            tol, tolerence for inliers
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers

    Hints:
    (1) You can use the calc_epi_error from q1 with threshold to calcualte inliers. Tune the threshold based on 
        the results/expected number of inliners. You can also define your own metric. 
    (2) Use the seven point alogrithm to estimate the fundamental matrix as done in q1
    (3) Choose the resulting F that has the most number of inliers
    (4) You can increase the nIters to bigger/smaller values
 
'''
def ransacF(pts1, pts2, M, nIters=10, tol=2000):
    N = pts1.shape[0] #pts1 and pts2 are same shape
    pts1_homogenous = np.hstack((pts1, np.ones((N, 1))))
    pts2_homogenous = np.hstack((pts2, np.ones((N, 1))))
    prev_inliers = 0
    best_inliers = None
    best_F = None
    for i in range(nIters):
        print("~~",i,"~~")
        pts_idx = np.random.choice(N, size = 5, replace = False)
        pts11 = pts1[pts_idx, :]
        pts22 = pts2[pts_idx, :]
        F = eightpoint(pts11, pts22, M)
        err = calc_epi_error(pts1_homogenous, pts2_homogenous, F)
        inliers = (err < tol)
        num_inliers = np.sum(inliers)
        if (num_inliers>prev_inliers):
            best_inliers = inliers
            best_F = F
            prev_inliers = num_inliers
    print("------Inliers With RANSAC: ",prev_inliers)
    # print("------With RANSAC: ",(prev_inliers/N))
    F_orig = eightpoint(pts1, pts2, M)
    err_orig = calc_epi_error(pts1_homogenous, pts2_homogenous, F_orig)
    inliers_orig = (err_orig < tol)
    num_inliers_orig = np.sum(inliers_orig)
    # print("------Without RANSAC: ", (num_inliers_orig/N))
    print("------Inliers Without RANSAC: ",num_inliers_orig)

    
    return best_F, best_inliers        

'''
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    # Replace pass by your implementation
    r_norm = np.linalg.norm(r)
    theta = r_norm # radians
    if theta == 0:
        R = np.eye(3)
        return R
    u = r/r_norm
    R = np.eye(3)*np.cos(theta) + (1-np.cos(theta))*(u@u.T) + np.array([[0,-u[2],u[1]],
                                                                        [u[2], 0, -u[0]],
                                                                        [-u[1], u[0], 0]])*np.sin(theta)
    return R


'''
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''

def S_half(r):
    tolerance = 0.001
    r1 = r[0]
    r2 = r[1]
    r3 = r[2]
    if abs(np.linalg.norm(r)-np.pi)<0.001 and ((r1==0 and r2==0 and r3<0) or (r1==0 and r2<0) or (r1<0)):
        result = -r
        return result
    else: 
        result = r
        return result


def invRodrigues(R):
    # Replace pass by your implementation
    A = (R-R.T)*0.5
    rho = np.asarray([[A[2,1], A[0,2], A[1,0]]]).T
    s = np.linalg.norm(rho)
    c = (np.sum(np.diag(R))-1)/2
    if s==0 and c==1:
        r = np.zeros((3))
    elif s==0 and c==-1:
        RI = R + np.eye(3)
        non_zero_indices = np.transpose(np.nonzero(RI))
        # non_zero_col = non_zero_indices[0,1]
        non_zero_col = non_zero_indices[0,1]
        column_index = -1
        for i in range(3):
            count = 0
            col = RI[:, i]
            count = np.count_nonzero(col)
            if count==3:
                column_index = i
                break
        print ("Column index: ", column_index)
        v = RI[:, column_index]
        v = RI[:,non_zero_col]
        u = v/np.linalg.norm(v)
        u_pi = u*np.pi
        r = S_half(u_pi)
        theta = np.arctan2(s,c)
    elif np.sin(np.arctan2(s,c))!=0:
        theta = np.arctan2(s,c)
        u = rho/s
        r = u*theta
    return r


'''
Q5.3: Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # Replace pass by your implementation
    N = p1.shape[0]
    P = x[0:len(x)-6].reshape((N,3))
    P = np.hstack((P, np.ones((N,1))))
    R2 = rodrigues(x[-6:-3].reshape((3,)))
    T2 = x[-3:].reshape((3,1))
    M2 = np.hstack((R2, T2))

    C1 = K1@M1
    C2 = K2@M2

    p1_projected = C1@P.T
    p2_projected = C2@P.T
    p1_projected = p1_projected.T
    p2_projected = p2_projected.T
    print(p1_projected.shape)
    print(p2_projected.shape)
    print(p1_projected[:,2].shape)
    print(p2_projected[:,2].shape)
    for i in range(p1_projected.shape[0]):
        p1_projected[i,:] = p1_projected[i,:]/p1_projected[i,-1]
    for i in range(p2_projected.shape[0]):
        p2_projected[i,:] = p2_projected[i,:]/p2_projected[i,-1]
    residuals = np.concatenate([(p1-p1_projected[:,0:2]).reshape([-1]),(p2-p2_projected[:,0:2]).reshape([-1])])

    return residuals


'''
Q5.3 Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
            o1, the starting objective function value with the initial input
            o2, the ending objective function value after bundle adjustment

    Hints:
    (1) Use the scipy.optimize.minimize function to minimize the objective function, rodriguesResidual. 
        You can try different (method='..') in scipy.optimize.minimize for best results. 
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    # Replace pass by your implementation

    obj_start = obj_end = 0
    # ----- TODO -----
    # YOUR CODE HERE
    R2 = M2_init[:, 0:3]
    T2 = M2_init[:,-1]

    r2 = invRodrigues(R2).flatten()
    P_init = P_init.flatten()
    T2 = T2.flatten()
    x_init = np.concatenate((P_init, r2, T2))
    residuals = rodriguesResidual(K1, M1, p1, K2, p2, x_init)
    def blah(x):
        output = np.linalg.norm(x)
        return output
    obj_start = blah(residuals)
    x_update = scipy.optimize.minimize(blah, x_init, method = "CG")
    x_update = x_update.x
    obj_end = blah(x_update)
    N = p1.shape[0]
    P = x_update[0: len(x_update)-6].reshape((N,3))
    R2 = rodrigues(x_update[-6:-3].reshape((3,)))
    T2 = x_update[-3:].reshape((3,1))
    M2 = np.hstack((R2, T2))


    return M2, P, obj_start, obj_end



if __name__ == "__main__":
              
    np.random.seed(1) #Added for testing, can be commented out

    some_corresp_noisy = np.load('../data/some_corresp_noisy.npz') # Loading correspondences
    intrinsics = np.load('../data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    noisy_pts1, noisy_pts2 = some_corresp_noisy['pts1'], some_corresp_noisy['pts2']
    im1 = plt.imread('../data/im1.png')
    im2 = plt.imread('../data/im2.png')

    F, inliers = ransacF(noisy_pts1, noisy_pts2, M=np.max([*im1.shape, *im2.shape]))
    print("F--->\n", F)

    # # YOUR CODE HERE


    # # Simple Tests to verify your implementation:
    # pts1_homogenous, pts2_homogenous = toHomogenous(noisy_pts1), toHomogenous(noisy_pts2)

    # assert(F.shape == (3, 3))
    # assert(F[2, 2] == 1)
    # assert(np.linalg.matrix_rank(F) == 2)
    

    # YOUR CODE HERE
    p1 = noisy_pts1[inliers.astype(bool), :] 
    p2 = noisy_pts2[inliers.astype(bool), :] 
    F1 = eightpoint(p1, p2, M=np.max([*im1.shape, *im2.shape]))
    print("F1-->\n", F1)

    E = essentialMatrix(F, K1, K2)

    M1 = np.hstack((np.eye(3), np.zeros((3,1))))
    C1 = K1@M1
    # M2s = camera2(E)
    M2, C2, best_P = findM2(F, p1, p2, intrinsics)
    # C2 = K2@M2

    # P_init, err = triangulate(C1, p1, C2, p2)
    # print(K1.shape, M1.shape, p1.shape, K2.shape, M2.shape, p2.shape, P_init.shape)
    R2 = M2[:, 0:3]
    r2 = invRodrigues(R2).flatten()
    t2 = M2[:,-1].flatten()
    p_flatten = best_P[:,0:3].flatten()
    # P_new = np.hstack((p_flatten, r2, t2))
    P_new = np.concatenate((p_flatten, r2, t2))

    print(p1.shape, p2.shape, best_P[:,0:3].shape)

    M2, P, obj_start, obj_end = bundleAdjustment(K1, M1, p1, K2, M2, p2, best_P[:,0:3])


    print("obj_start: ", obj_start)
    print("obj_end: ", obj_end)
    
    plot_3D_dual(P, P)




    # Simple Tests to verify your implementation:
    from scipy.spatial.transform import Rotation as sRot
    rotVec = sRot.random()
    mat = rodrigues(rotVec.as_rotvec())

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(np.linalg.norm(rotVec.as_rotvec() - invRodrigues(mat)))
    assert(np.linalg.norm(rotVec.as_rotvec() - invRodrigues(mat)) < 1e-3)
    assert(np.linalg.norm(rotVec.as_matrix() - mat) < 1e-3)



    # YOUR CODE HERE
    M2_init = findM2(F, noisy_pts1, noisy_pts2, intrinsics)