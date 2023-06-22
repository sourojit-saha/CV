from importlib.util import LazyLoader
import os, math, multiprocessing
from os.path import join
from copy import copy

import numpy as np
from PIL import Image

import visual_words

from sklearn.metrics import confusion_matrix
import numpy.matlib



def get_feature_from_wordmap(opts, wordmap):
    '''
    Compute histogram of visual words.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist: numpy.ndarray of shape (K)
    '''

    K = opts.K    

    wordmap_1d = wordmap.reshape(-1)    
    histogram = np.bincount(wordmap_1d, minlength = K)
    histogram = histogram/np.sum(histogram)

    return histogram

    
    # ----- TODO -----
    pass

def get_feature_from_wordmap_SPM(opts, wordmap):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^L-1)/3)
    '''
        
    K = opts.K
    L = opts.L

  

    
    hist_all = []
    for l in range(L):
        num_tile=2**l
        cols = np.array_split(wordmap, num_tile, axis=1)
        for i in range(len(cols)):
            rows=np.array_split(cols[i], num_tile, axis=0)
            for j in range(len(cols)):
                tile = rows[j]
                hist = get_feature_from_wordmap(opts, tile)
                if l==0 | l==1:
                    weight =1
                else:
                    weight = 2**(-l)
                hist_weighted = hist*weight
    
                hist_all = np.append(hist_all, hist_weighted, axis=0)
    
    hist_all = hist_all/np.sum(hist_all)

    return hist_all

    # ----- TODO -----
    pass
    
def get_image_feature(opts, img_path, dictionary):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * opts      : options
    * img_path  : path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)


    [output]
    * feature: numpy.ndarray of shape (K*(4^L-1)/3)
    '''

    img = Image.open(img_path)
    img = np.array(img).astype(np.float32)/255
    dictionary = np.load(join(opts.out_dir, 'dictionary.npy'))
    wordmap = visual_words.get_visual_words(opts, img, dictionary)  
    feature_SPM = get_feature_from_wordmap_SPM(opts,wordmap)
    feature = feature_SPM

    return feature

    # # ----- TODO -----
    # pass

def build_recognition_system(opts, n_worker=1):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir
    SPM_layer_num = opts.L
    K=opts.K
    L=opts.L

    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    train_labels = np.loadtxt(join(data_dir, 'train_labels.txt'), np.int32)
    dictionary = np.load(join(out_dir, 'dictionary.npy'))

    features=np.empty([0,int(K*((4**L)-1)/3)])
    for i in range(len(train_files)):
        img_path=join(opts.data_dir, train_files[i])
        img_features = get_image_feature(opts, img_path, dictionary)
        img_features = np.asarray(img_features)
        img_features = np.reshape(img_features, (1, int(K*((4**L)-1)/3)))


        features = np.concatenate((features, img_features), axis=0)
        print("Train File: ", i)




    # # ----- TODO -----
    # pass

    ## example code snippet to save the learned system
    np.savez_compressed(join(out_dir, 'trained_system.npz'),
        features=features,
        labels=train_labels,    
        dictionary=dictionary,
        SPM_layer_num=SPM_layer_num,
    )

def similarity_to_set(word_hist, histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * sim: numpy.ndarray of shape (N)
    '''

    similarity_matrix=np.minimum(word_hist, histograms)
    similarity_score=np.sum(similarity_matrix, axis=1)
    sim=similarity_score

    return sim

    # # ----- TODO -----
    # pass    
    
def evaluate_recognition_system(opts, n_worker=1):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir

    trained_system = np.load(join(out_dir, 'trained_system.npz'))
    dictionary = trained_system['dictionary']

    # using the stored options in the trained system instead of opts.py
    test_opts = copy(opts)
    test_opts.K = dictionary.shape[0]
    test_opts.L = trained_system['SPM_layer_num']

    test_files = open(join(data_dir, 'test_files.txt')).read().splitlines()
    test_labels = np.loadtxt(join(data_dir, 'test_labels.txt'), np.int32)

    # trained_system = np.load(join(out_dir, 'trained_system.npz'))
    labels=trained_system['labels']
    features = trained_system['features']



    prediction = []
    actual = []
    for i in range(len(test_files)):

        img_path = join(opts.data_dir, test_files[i])
        word_histogram = get_image_feature(opts,img_path, dictionary)

        idx = np.argmax(similarity_to_set(word_histogram, features))
        prediction = np.append(prediction, labels[idx])
        actual = np.append(actual, test_labels[i])
        print("(- ",i," -)")

    conf_mat = confusion_matrix(actual, prediction)
    accuracy = np.trace(conf_mat)/np.sum(conf_mat)

    return conf_mat, accuracy

    # # ----- TODO -----
    # pass

