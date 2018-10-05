# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 16:03:19 2017

@author: Weiyu_Lee
"""

"""
Scipy version > 0.18 is needed, due to 'mode' option from scipy.misc.imread function
"""

import h5py
import numpy as np
import random
import scipy.misc as misc
from sklearn.metrics.cluster import entropy
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

def read_data(path):  
    """
    Read h5 format data file
      
    Args:
        path: file path of desired file
        data: '.h5' file format that contains train data values
        label: '.h5' file format that contains train label values
    """
    
    with h5py.File(path, 'r') as hf:
        data = np.array(hf.get('input'))
        label = np.array(hf.get('label'))

    return data, label

def batch_shuffle(data, label, batch_size):
    """
    Shuffle the batch data
    """
    # Shuffle the batch data
#    shuffled_data = list(zip(data, *label))
#    random.shuffle(shuffled_data)
#    tmp = list(zip(*shuffled_data))
#    
#    data_shuffled = tmp[0]
#    label_shuffled = tmp[1:]

    shuffled_data = list(zip(data, label))
    random.shuffle(shuffled_data)
    data_shuffled, label_shuffled = zip(*shuffled_data)
    
    return data_shuffled, label_shuffled


def batch_shuffle_rndc(data, label, scale, subimage_size, index, batch_size):

    crop_data = []
    crop_label = []
    i = index

#    hard_entropy_thred = 3.5
#    soft_entropy_thred = 3.5
    hard_entropy_thred = 0
    soft_entropy_thred = 0

    retry_cnt = 0
    fail_cnt = 0
    fail_idx = []
        
    while i < index+batch_size:
            
        lridx = random_crop(scale, data[i].shape, subimage_size)
        crop_lr = data[i][lridx[0]:lridx[0]+subimage_size,
                              lridx[1]:lridx[1]+subimage_size,
                              :]
        crop_hr = label[i][lridx[0]*scale:lridx[0]*scale+scale*subimage_size,
                              lridx[1]*scale:lridx[1]*scale+scale*subimage_size,
                              :]
        #Reject small spatial gradient patch                  
        #patch_grad = np.sum(np.gradient(crop_hr))
        #crop_hr_sobel = np.array(scipy.ndimage.filters.sobel(crop_hr), dtype=np.int32)
        patch_entropy = entropy(crop_hr)            
        #patch_grad = np.sum(np.absolute(crop_hr_sobel - 128))

        if len(crop_label) < 13:
            entropy_thred = hard_entropy_thred
        else:
            entropy_thred = soft_entropy_thred
            
        if patch_entropy > entropy_thred:
            crop_data.append(crop_lr)
            crop_label.append(crop_hr) 

            i += 1                      
            fail_cnt = 0
            #scipy.misc.imsave("./crop/crop_{}_{}.png".format(patch_grad, i), crop_hr)
        else:
            fail_cnt += 1
                
        # If the ROI cannot be found:                
        if fail_cnt > 25:
            fail_cnt = 0
            retry_cnt += 1
            i += 1
            fail_idx.append(i)
                
       # Crop new ROI from the data been used to meet the batch_size
    while retry_cnt > 0:
            
        idx = random.randint(index, index+batch_size-1)
        if idx in fail_idx:
            continue
            
        lridx = random_crop(scale, data[idx].shape, subimage_size)
        crop_lr = data[idx][lridx[0]:lridx[0]+subimage_size,
                              lridx[1]:lridx[1]+subimage_size,
                              :]
        crop_hr = label[idx][lridx[0]*scale:lridx[0]*scale+scale*subimage_size,
                              lridx[1]*scale:lridx[1]*scale+scale*subimage_size,
                              :]
            #Reject small spatial gradient patch                  
        patch_entropy = entropy(crop_hr)            
           
        if patch_entropy > entropy_thred:
                crop_data.append(crop_lr)
                crop_label.append(crop_hr)  
                retry_cnt -= 1

    return crop_data, crop_label




def random_crop(scale, LR_image_size, subimage_size):

    x_list = list(range(0,LR_image_size[0]-subimage_size-1))
    y_list = list(range(0,LR_image_size[1]-subimage_size-1))
    random.shuffle(x_list)
    random.shuffle(y_list)

    return (x_list[0], y_list[0])

def log10(x):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator

def laplacian_filter(image_target):

    laplacian_kernel = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]], dtype=np.float32)
    laplacian_kernel = np.reshape(laplacian_kernel,[3,3,1,1])

       
    gt = tf.split(image_target, 3, axis=3)  

    for i in range(3):

        gt[i] = tf.nn.conv2d(gt[i], laplacian_kernel, [1,1,1,1], padding='SAME')
        
    gt = tf.abs(tf.concat(gt, axis=3))    
    mean_lap = tf.reduce_mean(1/tf.reduce_mean(gt, axis=0))

    return mean_lap