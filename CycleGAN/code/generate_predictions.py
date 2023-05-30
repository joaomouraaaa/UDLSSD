#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 15 12:54:05 2022

@author: joaomoura
"""

import numpy as np
from PIL import Image
from os.path import dirname, abspath
import os
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization



def binaryMaskIOU(mask1, mask2, smooth=1):   # From the question.
    mask1_area = np.count_nonzero(mask1 == 1)
    mask2_area = np.count_nonzero(mask2 == 1)
    intersection = np.count_nonzero(np.logical_and( mask1==1,  mask2==1 ))
    iou = (smooth+intersection)/(mask1_area+mask2_area-intersection+smooth)
    return iou



model_name = 'g_model_AtoB_016894.h5'

# load the models
cust = {'InstanceNormalization': InstanceNormalization}
model = load_model('CycleGAN/models/'+model_name,cust)

path = 'UDSEP/data/images'


for ff in os.listdir(path):
    # print(ff)
    img_train = load_img(os.path.join(path,ff), color_mode = "grayscale")# target_size=(im_size, im_size))
    img_array = img_to_array(img_train)
    
    # for i, ID in enumerate(list_IDs_test):
    # im=np.array(Image.open('images_test/' + ID))
    im = img_array
    im = np.reshape(im,(256,256,1)) #adicionei isto
    img= np.expand_dims(im,axis=0)
    
    img = np.reshape(img,(1,256,256,1))
    img = (img - 127.5) / 127.5 #[-1,1]
    
    B_generated  = model.predict(img)
    
    ypred = np.reshape(B_generated,(256,256))
    ypred[ypred<0.5]=0
    ypred[ypred>=0.5]=1
    
    
    image = 255*np.uint8(ypred)
    im = Image.fromarray(image,'L')
    im.save('UDSEP/data/segmentation_from_cycle/'+ff,"JPEG")
    
   
    


   

   
