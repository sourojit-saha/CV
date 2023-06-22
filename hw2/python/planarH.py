import numpy as np
from matchPics import matchPics
from opts import get_opts
import cv2

def compute_x1x2(img1, img2, opts):
    matches, locs1, locs2 = matchPics(img1, img2, opts)
    x1 = locs1[matches[:,0],:]
    x2 = locs2[matches[:,1],:]
    
    # print(x1.shape, x2.shape)

    return x1, x2

def generate_x1_x2(locs1,locs2,matches):

    x1 = np.zeros((matches.shape[0],2))
    x2 = np.zeros((matches.shape[0],2))

    for i in range(matches.shape[0]):
        x1[i][1] = locs1[matches[i][0]][0]
        x1[i][0] = locs1[matches[i][0]][1]
        x2[i][1] = locs2[matches[i][1]][0]
        x2[i][0] = locs2[matches[i][1]][1]
    
    return x1, x2

def computeH(x1, x2):
    # Q2.2.1
    # Compute the homography between two sets of points
    rows = x1.shape[0]
    a = np.zeros((2*rows, 9))
    for i in range(rows):
        row1 = np.asarray([-x2[i,0], -x2[i,1], -1, 0, 0, 0, x2[i,0]*x1[i,0], x2[i,1]*x1[i,0], x1[i,0]])
        row2 = np.asarray([0, 0, 0, -x2[i,0], -x2[i,1], -1, x2[i,0]*x1[i,1], x2[i,1]*x1[i,1], x1[i,1]])

        a[2*i] = row1
        a[(2*i)+1] = row2

    u,s,vt = np.linalg.svd(a)

    eig_val = s[-1]
    eig_vect = vt[-1,:]
    H2to1 = eig_vect.reshape(3,3)
    H2to1 = H2to1/H2to1[-1,-1]

    return H2to1


def computeH_norm(x1, x2):
    mean1 = np.mean(x1, axis=0)
    mean2 = np.mean(x2, axis=0)

    x1_trans = x1-mean1
    x2_trans = x2-mean2

    x1_trans_dist = ((x1_trans[:,0]**2) + (x1_trans[:,1])**2)**0.5
    x2_trans_dist = ((x2_trans[:,0]**2) + (x2_trans[:,1])**2)**0.5
    
    scale1 = (2**0.5)/np.max(x1_trans_dist)
    scale2 = (2**0.5)/np.max(x2_trans_dist)

    t1 = np.array([[scale1, 0, 0],
                    [0, scale1, 0],
                    [0, 0, 1]]) @ np.array([[1, 0, -mean1[0]],
                                            [0, 1, -mean1[1]],
                                            [0, 0, 1]])
            

    x1_homo = np.hstack((x1, np.ones((x1.shape[0], 1))))
    x1t = t1@x1_homo.T

    t2 = np.array([[scale2, 0, 0],
                [0, scale2, 0],
                [0, 0, 1]]) @ np.array([[1, 0, -mean2[0]],
                                        [0, 1, -mean2[1]],
                                        [0, 0, 1]])

    x2_homo = np.hstack((x2, np.ones((x2.shape[0], 1))))
    x2t = t2@x2_homo.T

    H2to1 = computeH(x1t.T, x2t.T) 
 
    H2to1 = np.linalg.inv(t1)@H2to1@t2

    return H2to1


def computeH_ransac(locs1, locs2, opts):
    # Q2.2.3
    # Compute the best fitting homography given a list of matching points
    max_iters = opts.max_iters  # the number of iterations to run RANSAC for
    rows = locs1.shape[0]
    if rows<4:
        print("Less than 4 points.")
        return -1, -1
    else:
        score_prev = 0
        bestH2to1 = np.zeros((3,3))
        for i in range(max_iters):
            idx = np.random.choice(range(rows), 4, replace=False)
            x1 = locs1[idx,:]
            x2 = locs2[idx,:]

            H = computeH_norm(x1,x2)

            idx_subset = np.setdiff1d(range(rows), idx)

            locs1_subset = locs1[idx_subset,:]
            locs2_subset = locs2[idx_subset,:]
            
            # locs1_subset_homo = np.hstack((locs1_subset, np.zeros((locs1_subset.shape[0], 1))))
            locs2_subset_homo = np.hstack((locs2_subset, np.ones((locs2_subset.shape[0], 1))))

            locs1_pred = H@locs2_subset_homo.T
            locs1_pred = locs1_pred.T
            locs1_pred[:,0] = locs1_pred[:,0]/locs1_pred[:,2]
            locs1_pred[:,1] = locs1_pred[:,1]/locs1_pred[:,2]
            locs1_pred = locs1_pred[:,0:2]
            # print(locs1_subset_homo.shape, locs1_subset_homo.shape, locs1_pred.shape)
            diff = locs1_pred-locs1_subset
            dist = (diff[:,0]**2 + diff[:,1]**2)**0.5

            # the tolerance value for considering a point to be an inlier
            inlier_tol = opts.inlier_tol
            inliers= np.zeros(diff.shape[0])
            for m in range(len(inliers)):
                # print("here: ", m, "dist: ", dist[m], "diff: ", diff[m])
                if dist[m]<inlier_tol:
                    # print("dist: ", dist)
                    inliers[m] = 1

            score = np.sum(inliers)
            # print("score: ", score)
            if score > score_prev:
                score_prev = score
                # print("score: ", score, "iteration: ", i)
                bestH2to1 = H
                             

    return bestH2to1, inliers

    


def compositeH(H2to1, template, img):


    invert_mask = cv2.inRange(template, np.array([0,0,0]),np.array([0,0,0]))
    mask = cv2.bitwise_not(invert_mask)
    fg = cv2.bitwise_or(template, template, mask=mask)
    bk = cv2.bitwise_or(img, img, mask=invert_mask)
    
    composite_img = cv2.bitwise_or(fg, bk)
    

    return composite_img

