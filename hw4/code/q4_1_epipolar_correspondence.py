import numpy as np
import matplotlib.pyplot as plt

from helper import _epipoles

from q2_1_eightpoint import eightpoint

# Insert your package here
from scipy.interpolate import RectBivariateSpline
import cv2 as cv2


# Helper functions for this assignment. DO NOT MODIFY!!!
def epipolarMatchGUI(I1, I2, F):
    e1, e2 = _epipoles(F)

    sy, sx, _ = I2.shape

    f, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 9))
    ax1.imshow(I1)
    ax1.set_title('Select a point in this image')
    ax1.set_axis_off()
    ax2.imshow(I2)
    ax2.set_title('Verify that the corresponding point \n is on the epipolar line in this image')
    ax2.set_axis_off()

    while True:
        plt.sca(ax1)
        # x, y = plt.ginput(1, mouse_stop=2)[0]

        out = plt.ginput(1, timeout=3600, mouse_stop=2)

        if len(out) == 0:
            print(f"Closing GUI")
            break
        
        x, y = out[0]

        xc = int(x)
        yc = int(y)
        v = np.array([xc, yc, 1])
        l = F.dot(v)
        s = np.sqrt(l[0]**2+l[1]**2)

        if s==0:
            print('Zero line vector in displayEpipolar')

        l = l/s

        if l[0] != 0:
            ye = sy-1
            ys = 0
            xe = -(l[1] * ye + l[2])/l[0]
            xs = -(l[1] * ys + l[2])/l[0]
        else:
            xe = sx-1
            xs = 0
            ye = -(l[0] * xe + l[2])/l[1]
            ys = -(l[0] * xs + l[2])/l[1]

        # plt.plot(x,y, '*', 'MarkerSize', 6, 'LineWidth', 2);
        ax1.plot(x, y, '*', markersize=6, linewidth=2)
        ax2.plot([xs, xe], [ys, ye], linewidth=2)

        # draw points
        x2, y2 = epipolarCorrespondence(I1, I2, F, xc, yc)
        # x2, y2 = epipolarCorrespondence(I1, I2, F, 119,217)

        ax2.plot(x2, y2, 'ro', markersize=8, linewidth=2)
        plt.draw()


'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2
            
    Hints:
    (1) Given input [x1, x2], use the fundamental matrix to recover the corresponding epipolar line on image2
    (2) Search along this line to check nearby pixel intensity (you can define a search window) to 
        find the best matches
    (3) Use guassian weighting to weight the pixel simlairty

'''
def epipolarCorrespondence(im1, im2, F, x1, y1):
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    line = F@(np.array([x1,y1,1]).T)
    window_size = 20

    im1x = np.arange(0, im1.shape[0])
    im1y = np.arange(0, im1.shape[1])
    spline_im1 = RectBivariateSpline(im1x, im1y, im1)

    imx2 = np.arange(0, im2.shape[0])
    imy2 = np.arange(0, im2.shape[1])
    spline_im2 = RectBivariateSpline(imx2, imy2, im2)

    im1x1 =  x1-(window_size//2)
    im1y1 =  y1-(window_size//2)
    im1x2 =  x1+(window_size//2)
    im1y2 =  y1+(window_size//2)

    x11 = np.arange(im1x1,im1x2)
    y11 = np.arange(im1y1,im1y2)
    X,Y = np.meshgrid(x11,y11)
    im1_template = spline_im1.ev(Y, X)

    # im1_template = im1[int(y1-(window_size//2)): int(y1+(window_size//2)), int(x1-(window_size//2)):int(x1+(window_size//2))]
    coordinates = np.zeros((0,2))
    error = np.zeros((0,1))
    for i in range(window_size//2, im2.shape[0]-window_size//2):
        # print(i)
        y = i + (window_size*0.5)
        x = -(line[1] * y + line[2])/line[0]
        coord = np.array([[x,y]])
        # print(coord)
        im2x1 =  x-(window_size//2)
        im2y1 =  y-(window_size//2)
        im2x2 =  x+(window_size//2)
        im2y2 =  y+(window_size//2)

        x22 = np.arange(im2x1,im2x2)
        y22 = np.arange(im2y1,im2y2)
        X,Y = np.meshgrid(x22,y22)
        im2_template = spline_im2.ev(Y, X)



        # im2_patch = im1[int(y-(window_size//2)): int(y+(window_size//2)), int(x-(window_size//2)):int(x+(window_size//2))]
        # err = np.sum(((im1_template - im2_patch)**2)/(window_size**2))
        min_row = np.min([im1_template.shape[0], im2_template.shape[0]])
        min_col = np.min([im1_template.shape[1], im2_template.shape[1]])
        im1_template_min = im1_template[0:min_row, 0:min_col]
        im2_template_min = im2_template[0:min_row, 0:min_col]
        # print(min_row, min_col)
        # print(im1_template.shape, im2_template.shape)
        err = np.sum((im1_template_min - im2_template_min)**2)
        coordinates = np.vstack((coordinates, coord))
        error = np.vstack((error, err))
    best_match = np.argmin(error)
    best_coord = coordinates[best_match,:]
    x2 = best_coord[0]
    y2 = best_coord[1]

    return x2, y2







if __name__ == "__main__":

    correspondence = np.load('../data/some_corresp.npz') # Loading correspondences
    intrinsics = np.load('../data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    im1 = plt.imread('../data/im1.png')
    im2 = plt.imread('../data/im2.png')


    # ----- TODO -----
    # YOUR CODE HERE
    
    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))
    np.savez("q4_1.npz", F=F, pts1=pts1, pts2=pts2)
    
    # Simple Tests to verify your implementation:
    x2, y2 = epipolarCorrespondence(im1, im2, F, 119, 217)
    x2, y2 = epipolarMatchGUI(im1, im2, F)
    assert(np.linalg.norm(np.array([x2, y2]) - np.array([118, 181])) < 10)