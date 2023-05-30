# -*- coding: utf-8 -*-
"""
Created on Sun May 15 12:54:05 2022

@author: joaomoura
"""

import sys

sys.path.append('UDSEP/code')

import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage import segmentation
import json
import os
from unet_model import *
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from scipy import ndimage
from skimage.measure import label
from skimage.measure import regionprops
from keras.models import load_model
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization


def binaryMaskIOU(mask1, mask2, smooth=1):   # From the question.
    mask1_area = np.count_nonzero(mask1 == 1)
    mask2_area = np.count_nonzero(mask2 == 1)
    intersection = np.count_nonzero(np.logical_and( mask1==1,  mask2==1 ))
    iou = (smooth+intersection)/(mask1_area+mask2_area-intersection+smooth)
    return iou

flag = 0
model_name = sys.argv[2]



if sys.argv[1] == "unet":
    flag = 0
    model = unet_model(n_classes=2, class_weights=[1 ,1] ) #atenção aos pesos de cada classe ATENCAO!!!
    model.load_weights('UDSEP/models/'+model_name)
    
elif sys.argv[1] == "cyclegan":
    flag = 1
    cust = {'InstanceNormalization': InstanceNormalization}
    model = load_model('Cyclegan/models/'+model_name,cust)
        
else:
    exit()


     
i=0


path = 'data/test/images_test'
path_seg = 'data/test/segmentation_gt_test'


def pre_process(image): #fill image and remove small objects

    image = image/255
    image[image<0.5]=0
    image[image>=0.5]=1
    
    image = ndimage.binary_fill_holes(image)

    labels = label(image)
    props = regionprops(labels)
    for prop in props:
        if(prop.area < 30):

            L = np.isin(labels,prop.label, invert = True)
            labels = L*labels    
            
    labels[labels > 0] = 1
    image = labels
    return image


for ff in os.listdir(path):
    
    img_train = load_img(os.path.join(path,ff), color_mode = "grayscale", target_size=(256, 256))
    img_array = img_to_array(img_train)

    img_seg = load_img(os.path.join(path_seg,ff), color_mode = "grayscale", target_size=(256, 256))
    img_seg_array = img_to_array(img_seg)

    im = img_array
    im = np.reshape(im,(256,256,1)) 
    
    
    if(flag == 1):
        img= np.expand_dims(im,axis=0)
        img = np.reshape(img,(1,256,256,1))
        img = (img - 127.5) / 127.5 #[-1,1]
        
        B_generated  = model.predict(img)
        ypred = np.reshape(B_generated,(256,256))
    
        ypred[ypred<0.5]=0
        ypred[ypred>=0.5]=1
        image__ = 255*np.uint8(ypred)
        
        
    elif(flag == 0):
        
        img = np.expand_dims(im,axis=0)/255
        ypred=np.squeeze(model.predict(img))
        
        ypred[ypred<0.5]=0
        ypred[ypred>=0.5]=1
    
        image__ = 255*np.uint8(ypred[:,:,0])
        ypred = np.argmax(ypred, axis=2)
        
    im11 = Image.fromarray(image__,'L')
    im11.save('data/results/segmentation_results_binary/'+ff,"JPEG")
        
    aux = np.reshape(img_seg_array,(256,256))/255
    
    aux[aux<0.5]=0
    aux[aux>=0.5]=1
    
    mask = aux

    
    result_image = segmentation.mark_boundaries(np.reshape(im,(256,256))/255, ypred, mode='inner')
    result_image = segmentation.mark_boundaries(result_image, mask, mode='inner', color=(1, 0, 0))
    
    
    #ypred = ypred^(ypred&1==ypred) #changes bits 0 and 1 to correctly comput in accuracy

    plt.figure()
    plt.imshow(result_image)
    plt.savefig("data/results/figs_results/"+ff)

                
    i=i+1

    