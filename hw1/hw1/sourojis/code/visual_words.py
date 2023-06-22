import os, multiprocessing
from os.path import join, isfile
from os import listdir

import numpy as np
from PIL import Image 
import scipy.ndimage
import skimage.color
import sklearn.cluster
import util

# My changes
from timeit import default_timer as timer
import numpy.matlib

def extract_filter_responses(opts, img):
    '''
    Extracts the filter responses for the given image.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)

    [hint]
    * To produce the expected collage, loop first over scales, then filter types, then color channel.
    * Note the order argument when using scipy.ndimage.gaussian_filter. 
    '''
    # Checking for gray-scale image. If image is gray-scale, then duplicating it to a m x n x 3 array. 
    if len(img.shape)==2:
        # img = np.matlib.repmat(img, 3, 1)
        rows, cols = img.shape
        arr1 = np.empty((rows, cols, 3))
        arr1[:,:,0]=img
        arr1[:,:,1]=img
        arr1[:,:,2]=img
        img=arr1
    # Converting to LAB colorspace as mentioned in question.
    img = (skimage.color.rgb2lab(img, illuminant='D65', observer='2'))/225
    # Getting the dimensions of the image.
    img_shape = img.shape
    height=img_shape[0]
    width=img_shape[1]
    channel=img_shape[2]
    
    num_filter=4 # Constant, as given in question.
    # Creating an array where result will be stored.
    result = np.zeros((height, width, channel*num_filter*len(opts.filter_scales)))
    # Getting filter scales from opts.py file
    filter_scales = opts.filter_scales
    # Starting position for a given scale. For a new filter-scale, new array is input at the start + k posiiton 
    start=0
    for i in filter_scales: # Looping over 
        k=0 # Gives the position where the result of convolution over each channel will be stored in the 'result' array.
        for j in range(channel): # Looping over each channel.
            # Using filters and storing the result.
            gauss_filter = scipy.ndimage.gaussian_filter(img[:,:,j], sigma=i)
            result[:,:,start+k] = gauss_filter
            laplacian_of_gauss = scipy.ndimage.gaussian_laplace(img[:,:,j], sigma=i)
            result[:,:,start+k+3] = laplacian_of_gauss
            gauss_filter_x = scipy.ndimage.gaussian_filter1d(img[:,:,j], sigma=i, axis= 1, order = 1)
            result[:,:,start+k+6] = gauss_filter_x
            gauss_filter_y = scipy.ndimage.gaussian_filter1d(img[:,:,j], sigma=i, axis= 0, order = 1)
            result[:,:,start+k+9] = gauss_filter_y
            k=k+1 # Updating the relative position to store the result of each individual channel.
        start=start+12 # Updating the start position for a new filter-scale.
        
    filter_responses=result
    return filter_responses

    # # ----- TODO -----
    # pass

def compute_dictionary_one_image(args):
    '''
    Extracts a random subset of filter responses of an image and save it to disk
    This is a worker function called by compute_dictionary

    Your are free to make your own interface based on how you implement compute_dictionary
    '''
    image_path, alpha, opts, idx = args
    # print(args)
    img = Image.open(image_path)
    img = np.array(img).astype(np.float32)/255

    filter_response = extract_filter_responses(opts, img)
    x = np.random.choice(filter_response.shape[0], alpha, replace=False)
    y = np.random.choice(filter_response.shape[1], alpha, replace=False)
    pixel_response = filter_response[x,y,:]
    print("(~ ",idx," ~)")
    tmp_dir = '../tmp'
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    np.savetxt(os.path.join(tmp_dir, str(idx) + ".csv"), pixel_response, delimiter="," )

    # ----- TODO -----
    pass

def compute_dictionary(opts, n_worker):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * opts         : options
    * n_worker     : number of workers to process in parallel
    
    [saved]
    * dictionary : numpy.ndarray of shape (K,3F)
    '''

    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    K = opts.K
    alpha = opts.alpha
    
    # For testing purpose, you can create a train_files_small.txt to only load a few images.
    # train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    image_path = ["../data/" + x for x in train_files]
    # num_images=len(image_path)
    
    pool = multiprocessing.Pool(processes=n_worker)
    args = zip(image_path, [alpha]*len(image_path), [opts]*len(image_path), range(len(image_path)))
    pool.map(compute_dictionary_one_image, args)

    # Starting k-means

    tmp_files = listdir("../tmp")

    pixel_response_path = ["../tmp/" + x for x in tmp_files]
    file0=pixel_response_path[0]
    arr1=np.loadtxt(file0, delimiter=",")
    channel = arr1.shape[1]

    
    pixel_response_master = np.empty([0,channel])
    for file in pixel_response_path:
        pixel_response = np.loadtxt(file, delimiter=",")
        pixel_response_master=np.concatenate((pixel_response_master, pixel_response), axis=0)
    # print("pixel_respose_master: ",pixel_response_master.shape)


    start = timer()
    kmeans = sklearn.cluster.KMeans(n_clusters=opts.K).fit(pixel_response_master)
    centroids = kmeans.cluster_centers_

    np.save("dictionary.npy", centroids)
    # ----- TODO -----
    pass

    ## example code snippet to save the dictionary
    # np.save(join(out_dir, 'dictionary.npy'), dictionary)

def get_visual_words(opts, img, dictionary):
    '''
    Compute visual words mapping for the given img using the dictionary of visual words.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    
    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''
    filter_response = extract_filter_responses(opts, img)
    height, width, _ = filter_response.shape
    filter_response_2d=filter_response.reshape(height*width, -1)
    dist = scipy.spatial.distance.cdist(filter_response_2d, dictionary, metric='euclidean')

    wordmap=np.argmin(dist, axis=1)
    wordmap=wordmap.reshape(height, width)

    return wordmap

    # ----- TODO -----
    pass

