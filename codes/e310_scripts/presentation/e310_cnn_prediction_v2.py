#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 13:38:46 2019

@author: rcetin
"""
#from __future__ import print_function, division
import numpy as np
np.random.seed(42)
import os
import glob
import matplotlib.pyplot as plt

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def load_image_mat(path):
    data = []
    data_id = []

    combined_path = os.path.join(path, '*png')
    print('Loading images from {}'.format(combined_path))
    files = glob.glob(combined_path)  
    for fl in files:   
        flbase = os.path.basename(fl)
        
        d = plt.imread(fl)
        d = d[:,:,0:3]
        
        data.append(d)
        data_id.append(flbase)
        
    data = np.array(data, dtype=np.float32)
    data = (data - data.mean()) / data.std()    # Normalization
    return data, data_id

#%%
def convolution(image, filt, bias, s=1):
    '''
    Confolves `filt` over `image` using stride `s`
    '''
    (n_f, n_c_f, f, _) = filt.shape # filter dimensions 5x5x3x32
    #print("nf: {}, ncf: {}, f: {}, filter: {}".format(n_f, n_c_f, f, filt.shape))
    
    n_c, in_dim, _ = image.shape # image dimensions 128x128x3
    #print("nc: {}, in_dim: {}, im: {}".format(n_c, in_dim, image.shape))
    
    out_dim = int((in_dim - f)/s)+1 # calculate output dimensions
    
    # ensure that the filter dimensions match the dimensions of the input image
    assert n_c == n_c_f, "Dimensions of filter must match dimensions of input image"
    
    out = np.zeros((n_f,out_dim,out_dim)) # create the matrix to hold the values of the convolution operation
    
    # convolve each filter over the image
    for curr_f in range(n_f):
        curr_y = out_y = 0
        # move filter vertically across the image
        while curr_y + f <= in_dim:
            curr_x = out_x = 0
            # move filter horizontally across the image 
            while curr_x + f <= in_dim:
                # perform the convolution operation and add the bias
                out[curr_f, out_y, out_x] = np.sum(filt[curr_f] * image[:,curr_y:curr_y+f, curr_x:curr_x+f]) + bias[curr_f]
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1
        
    return out

#%%%
# BATCH NORMALIZATION

def batch_norm(input_data, mean, variance, gamma, beta):
    x_hat = input_data - mean
    x1_hat = x_hat / ((variance + 0.001)**0.5)
    batch_out = gamma * x1_hat + beta
    return batch_out

#%%

def relu(data):
    data[data<=0] = 0 # pass through ReLU non-linearity
    return data

#%%

def maxpool(image, f=2, s=2):
    '''
    Downsample input `image` using a kernel size of `f` and a stride of `s`
    '''
    n_c, h_prev, w_prev = image.shape
    
    # calculate output dimensions after the maxpooling operation.
    h = int((h_prev - f)/s)+1 
    w = int((w_prev - f)/s)+1
    
    # create a matrix to hold the values of the maxpooling operation.
    downsampled = np.zeros((n_c, h, w)) 
    
    # slide the window over every part of the image using stride s. Take the maximum value at each step.
    for i in range(n_c):
        curr_y = out_y = 0
        # slide the max pooling window vertically across the image
        while curr_y + f <= h_prev:
            curr_x = out_x = 0
            # slide the max pooling window horizontally across the image
            while curr_x + f <= w_prev:
                # choose the maximum value within the window at each step and store it to the output matrix
                downsampled[i, out_y, out_x] = np.max(image[i, curr_y:curr_y+f, curr_x:curr_x+f])
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1
    return downsampled
#%%
def softmax(raw_preds):
    '''
    pass raw predictions through softmax activation function
    '''
    out = np.exp(raw_preds) # exponentiate vector of raw predictions
    return out/np.sum(out) # divide the exponentiated vector by its sum. All values in the output sum to 1.

#%%

#weight get funx from model
import h5py

def traverse_datasets(hdf_file):

    def h5py_dataset_iterator(g, prefix=''):
        for key in g.keys():
            item = g[key]
            #path = f"{prefix}/{key}"
            path = "{}/{}".format(prefix, key)  # FOR E310!! to python2
            #print("path formatter: ", path)
            if isinstance(item, h5py.Dataset): # test for dataset
                yield (path, item)
            elif isinstance(item, h5py.Group): # test for group (go down)
                #yield from h5py_dataset_iterator(item, path)
                for bar in h5py_dataset_iterator(item, path):   # FOR E310!! to python2
                    yield bar # FOR E310!! to python2

    with h5py.File(hdf_file, 'r') as f:
        for path, _ in h5py_dataset_iterator(f):
            yield path
#%%

#   print dataset features

weights = {}
layers = []
filename = '/home/rcetin/workspace/BOUN/Thesis/signals/cnn/cnn_weights_class3_extended_rgb_v1v2v3_afteriir_90.0%.h5'

with h5py.File(filename, 'r') as f:
    layers = list(f.keys())
    
    
    for dset in traverse_datasets(filename):
        #if dset.split('/')[1] == 'batch_normalization_1':
        weights["{}/{}".format(dset.split('/')[1], dset.split('/')[3])] = f[dset][()]

layer_names =['conv2d_1',
 'batch_normalization_1',
 'activation_1',
 'max_pooling2d_1',
 'conv2d_2',
 'batch_normalization_2',
 'activation_2',
 'max_pooling2d_2',
 'conv2d_3',
 'batch_normalization_3',
 'activation_3',
 'max_pooling2d_3',
 'conv2d_4',
 'batch_normalization_4',
 'activation_4',
 'max_pooling2d_4',
 'flatten_1',
 'dense_1',
 'batch_normalization_5',
 'activation_5',
 'dropout_1',
 'dense_2',
 'batch_normalization_6',
 'activation_6']
#%%

def conv_layer(layer_name, input_):
    filters = weights["{}/kernel:0".format(layer_name)]  
    biases = weights["{}/bias:0".format(layer_name)]
    
    
    filters1 = np.transpose(filters, (3,2,0,1))  # convert from 5x5x3x32 to 32x3x5x5
    #print("filter: ", filters1.shape, " input_: ", input_.shape)

    convResult = convolution(input_, filters1, biases)
    return convResult

#%%
def batch_norm_layer(layer_name, input_, reshape):    
    mean = weights["{}/moving_mean:0".format(layer_name)]
    var = weights["{}/moving_variance:0".format(layer_name)]
    beta = weights["{}/beta:0".format(layer_name)]
    gamma = weights["{}/gamma:0".format(layer_name)]
    
    if reshape == True:
        mean = mean.reshape(mean.shape[0],1,1)
        var = var.reshape(var.shape[0],1,1)
        beta = beta.reshape(beta.shape[0],1,1)
        gamma = gamma.reshape(gamma.shape[0],1,1)
    
    # normally use numConv0
    batchResult = batch_norm(input_data=input_, mean=mean, variance=var, gamma=gamma, beta=beta) # for now!
    return batchResult

#%%
def dense_layer(layer_name, input_, output_shape):    
    kernel = weights["{}/kernel:0".format(layer_name)]  
    biases = weights["{}/bias:0".format(layer_name)]
    
    kernel = np.transpose(kernel, (1,0))
    
    denseResult = np.dot(kernel, input_)
    denseResult = denseResult.reshape(output_shape) 
    denseResult = denseResult + biases
    return denseResult

#%%
def flatten_layer(input_):
    input_ = input_.transpose(1,2,0)
    (_, dim2, nf2) = input_.shape
    return input_.reshape((nf2 * dim2 * dim2, 1))

#%%
    
def get_result_class(soft_out):
    return classNames[soft_out.argmax()]

#%%

#Hyper Parameters
classNames = ["b", "g", "n"] 
testPath = '/tmp/data2/test/g'
#testPath = '/home/root/e310_test_images' # for E310

imgWidth, imgHeight = 128, 128 
numChannels = 3
#%%

#Load images
print("Preparing Test Images...")
testImages, testImageNames = load_image_mat(path=testPath)

#testImages = normalize_images(data=testImages)
#testIm2, testIm2_id = load_image_mat(testPath)
#t_cropped=testIm2[:,:,0:3]

#%%
class bcolors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
#%%    
# ALL LAYERS
import time
from joblib import Parallel, delayed
import multiprocessing

num_cores = multiprocessing.cpu_count()
totalImageCount = len(testImages)
print("Starting Tests... Total test images: {}\n".format(totalImageCount))

#import threading
#lock = threading.Lock()
#predicted = np.zeros(3) # 3 class result vector

def predict_spectrogram(data, name):
#for data, name in zip(testImages, testImageNames):
    start = time.time()
    image = np.transpose(data, (2,0,1))    # convert from 128x128x3 to 3x128x128
    
    
    result = conv_layer(layer_names[0], image)
    result = batch_norm_layer(layer_names[1], result, True)
    result = relu(result)
    result = maxpool(result, f=2, s=2)
    
    result = conv_layer(layer_names[4], result)
    result = batch_norm_layer(layer_names[5], result, True)
    result = relu(result)
    result = maxpool(result, f=2, s=2)
    
    result = conv_layer(layer_names[8], result)
    result = batch_norm_layer(layer_names[9], result, True)
    result = relu(result)
    result = maxpool(result, f=2, s=2)

    result = conv_layer(layer_names[12], result)
    result = batch_norm_layer(layer_names[13], result, True)
    result = relu(result)
    result = maxpool(result, f=2, s=2)

    # Flatten Layer
    result = flatten_layer(result)
    
    result = dense_layer(layer_names[17], result, 16) # 16 output
    result = batch_norm_layer(layer_names[18], result, False) # dont reshape in dense layer
    result = relu(result)
    
    #Pass dropout
    result = dense_layer(layer_names[21], result, 3) # 3 output
    result = batch_norm_layer(layer_names[22], result, False) # dont reshape in dense layer
    result = softmax(result)
    stop = time.time()
    print(bcolors.BOLD+bcolors.YELLOW+"----------------------------------------"+bcolors.ENDC)
    print("Image Name: {}\nResult Matrix: {}\nResult: {}\nDuration: {} seconds".format(name, result, get_result_class(result), stop - start))
    print(bcolors.BOLD+bcolors.YELLOW+"----------------------------------------\n"+bcolors.ENDC)
    os.remove("{}/{}".format(testPath, name))
    
    return result.argmax()

ret = Parallel(n_jobs=num_cores)(delayed(predict_spectrogram)(data=data, name=name) for data, name in zip(testImages, testImageNames))
b=0.0
g=0.0
n=0.0
for i in ret:
    if i == 0:
        b = b + 1
    elif i == 1:
        g = g + 1
    else:
        n = n + 1
#print("b:",b, "g",g, "n",n)
print("\n\n\n----------------------------------------\n")
print(bcolors.BOLD+bcolors.GREEN+"TOTAL PREDICTION RESULTS:"+bcolors.ENDC)
print(bcolors.BOLD+bcolors.GREEN+"802.11b:"+bcolors.ENDC+ "{}%".format((b/totalImageCount)*100))
print(bcolors.BOLD+bcolors.GREEN+"802.11g:"+bcolors.ENDC+ "{}%".format((g/totalImageCount)*100))
print(bcolors.BOLD+bcolors.GREEN+"802.11n:"+bcolors.ENDC+ "{}%".format((n/totalImageCount)*100))

