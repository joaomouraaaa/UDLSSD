#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 17:00:44 2022

@author: joaomoura
"""


import random
import numpy as np
import os
from os.path import dirname, abspath
import cv2
import glob
from skimage.measure import label
from skimage.measure import regionprops
from scipy import ndimage
from PIL import Image, ImageEnhance


def get_number_boat():
    aux = random.random()
     
    if aux<=0.7:
        nr_boat = 1
    elif (aux>0.7 and aux<=0.9):
        nr_boat = 2
    elif (aux>0.9):
        nr_boat = 3
        
    return nr_boat
    

def pre_process(image): # fill image and remove small objects
    
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
    


def compute_saliency(img, filter_size):
    c = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
    mag = np.sqrt(c[:,:,0]**2 + c[:,:,1]**2)
    spectralResidual = np.exp(np.log(mag) - cv2.boxFilter(np.log(mag), -1, (3,3)))
    c[:,:,0] = c[:,:,0] * spectralResidual / mag
    c[:,:,1] = c[:,:,1] * spectralResidual / mag
    c = cv2.dft(c, flags = (cv2.DFT_INVERSE | cv2.DFT_SCALE))
    mag = c[:,:,0]**2 + c[:,:,1]**2
    cv2.normalize(cv2.GaussianBlur(mag,(filter_size,filter_size),filter_size,filter_size), mag, 0., 1., cv2.NORM_MINMAX)
    mag = mag*255
    threshMap = cv2.threshold(mag.astype("uint8"), 0, 255,
     	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return mag, threshMap


def clean_image(img, filter_size):
    saliencyMap, threshMap = compute_saliency(img,11)
    image_inverse = 1 - threshMap/255
    image_clean = img*image_inverse

    return image_clean, threshMap


def cover_non_black(img, img_clean, labels_cover, nr_boat,ii):
    props_cover = regionprops(labels_cover)
    # print(len(props_cover))
    props_cover = props_cover[nr_boat:len(props_cover)]
    # print(len(props_cover))
    
    index = 0
    img_new_background = img_clean
    for prop in props_cover:
        binary_boat_location = labels_cover == index+1+nr_boat   # binary map with the location of the boat in original image
        
        binary_boat_location = 1*binary_boat_location        
        binary_boat_location = ndimage.binary_dilation(binary_boat_location)
        binary_boat_location = ndimage.binary_dilation(binary_boat_location)
        binary_boat_location = ndimage.binary_dilation(binary_boat_location)
        binary_boat_location = ndimage.binary_dilation(binary_boat_location)
        binary_boat_location = ndimage.binary_dilation(binary_boat_location)
               
        counter = 0
        
        print(ii)

        while True:

            delta_yy = random.randint(-128, 128)
            delta_xx = random.randint(-128, 128)
            
            binary_boat_location_rolled = np.roll(binary_boat_location, delta_yy, axis = 0)
            patch_paste = np.roll(binary_boat_location_rolled, delta_xx, axis = 1)        
        
            if np.sum(patch_paste * labels_cover) == 0:
                img_with_covered = patch_paste * img #image with the background in its original location
                background_location_rolled = np.roll(img_with_covered, -delta_yy, axis = 0)
                background_location_rolled = np.roll(background_location_rolled, -delta_xx, axis = 1) 
             
                
                img_new_background = background_location_rolled + img_new_background
                # fig, ax = plt.subplots()
                # plt.imshow(img_new_background, cmap="gray")
                # plt.show()
                
                
                # print(counter)
                break
            if counter>30:
                print("Did not find non boat with this size, covering with black..")
                break
            counter+=1
        index+=1
    return img_new_background 
    

    # return

def save_mask(mask, f):
    
    im = Image.fromarray(255*np.uint8(mask),'L')
    im.save('data/train/UDSEP/DSEP_results/segmentation/'+f,"JPEG", vmin = 0, vmax = 255)

    
def save_image(image, f):
    image = Image.fromarray(np.uint8(image),'L')
    image.save('data/train/UDSEP/DSEP_results/images/'+f, "JPEG", vmin = 0, vmax = 255)
   

def random_noise(image):
    rng = np.random.default_rng(1)
    noise = rng.normal(0.2, 0.1 ** 0.5, image.shape)
    out = image + image * noise
    
    return out

def adjust_gamma(image, gamma):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)
    

# --------- main --------

files = glob.glob('data/train/UDSEP/DSEP_results/images/*')
for f in files: #cleans folders
    os.remove(f)
files = glob.glob('data/train/UDSEP/DSEP_results/segmentation/*')
for f in files: #cleans folders
    os.remove(f)

patch_save = []
X_total =[]

ii = 0

parent_dir = dirname(dirname(abspath(os.getcwd())))
path = 'UDSEP/data/images'
path_cycle = 'UDSEP/data/segmentation_from_cycle'

ff1 = os.listdir(path_cycle)

ii = 0
for ff in ff1:
    img_array = cv2.imread(path+'/'+ff,0)
    img_array = np.reshape(img_array,(256,256))
    seg_cycle = cv2.imread(path_cycle+'/'+ff,0)
    seg_cycle = np.reshape(seg_cycle,(256,256))
    threshMap = seg_cycle/255 
    
    threshMap[threshMap<0.5]=0
    threshMap[threshMap>=0.5]=1
    threshMap = ndimage.binary_fill_holes(threshMap)
    labels = label(threshMap)
    props = regionprops(labels)
        
    props_save = props
    new_image_clean = img_array
    
        
    # removes small boats from image
    for prop in props:
        if(prop.area < 60 or prop.area > 6000):
            
            L = np.isin(labels,prop.label, invert = True)
            labels = L*labels    

    props = regionprops(labels)

    if len(props) == 0:
        for prop in props_save:
            if(prop.area < 30 or prop.area > 10000):
                
                L = np.isin(labels,prop.label, invert = True)
                labels = L*labels 
        props = regionprops(labels)

             
    image = labels  
     
    
    nr_boat = get_number_boat()
    index = 0

    labels_save = label(labels)
    
    mask = np.zeros((256,256))
    
    if (len(props)>0 and len(props) <= 4):
        nr_boat = len(props)
    if (len(props) > 4):
        nr_boat = 4
        
    props_to_paste = props[0:nr_boat]
    props_to_erase = props[nr_boat:len(props)]
    
    # -------- cleaning image (only non selected boats) -----------
    img_clean = img_array
    cnr = 0
    for prop in props_to_erase:
    
        patch = 1*prop.image
        patch = Image.fromarray(np.uint8(patch),'L')

        
        bbox = prop.bbox
    
        binary_boat_location = labels_save == nr_boat + cnr + 1   # binary map with the location of the boat in original image
        binary_boat_location = 1*binary_boat_location
        
        binary_boat_location = ndimage.binary_dilation(binary_boat_location)
        binary_boat_location = ndimage.binary_dilation(binary_boat_location)
        binary_boat_location = ndimage.binary_dilation(binary_boat_location)
        binary_boat_location = ndimage.binary_dilation(binary_boat_location)
        binary_boat_location = ndimage.binary_dilation(binary_boat_location)
        binary_boat_location = 1 - binary_boat_location

        img_clean = img_clean * binary_boat_location
        
        cnr+=1
        
    # -------------------

    
    new_image_clean = cover_non_black(img_array, img_clean, labels_save, nr_boat,ff)    
    
    for counter in range(0,nr_boat):
        props = props_to_paste
        image_final = Image.fromarray(np.uint8(np.zeros((256,256))),'L')
        mask_aux = Image.fromarray(np.uint8(np.zeros((256,256))),'L')
        
        repeat = random.random()
        
        
        if len(props) == 0: #did not find any boat
            break
        
        if counter > len(props)-1:
            index = random.randint(0, len(props)-1)
        patch = 1*props[index].image
        patch = Image.fromarray(np.uint8(patch),'L')

        
        bbox = props[index].bbox
    
        binary_boat_location = labels_save == index+1   # binary map with the location of the boat in original image
        patch_boat_original = binary_boat_location*img_array # boat and all zero
        
        small_patch_boat = patch_boat_original[bbox[0]:bbox[2],bbox[1]:bbox[3]]
        small_patch_boat = Image.fromarray(np.uint8(small_patch_boat),'L')
        

        paste_left = bbox[1]
        paste_top = bbox[0]
        
        mask_aux.paste(patch, ((paste_left , paste_top)), mask=None)
        
        
        index_yy_0 = paste_top
        index_yy_end_1 = paste_top+np.shape(small_patch_boat)[1]
        index_yy_end = 255 - index_yy_end_1
        index_xx_0 = paste_left
        index_xx_end_1 = paste_left+np.shape(small_patch_boat)[0]
        index_xx_end = 255 - index_xx_end_1
        
  
        if index_yy_end>-index_yy_0 and  index_xx_end>-index_xx_0:
            delta_yy = random.randint(-index_yy_0, index_yy_end)
            delta_xx = random.randint(-index_xx_0, index_xx_end)
        else:
            delta_yy = 10
            delta_xx = 10
        

        patch_paste = image_final
        patch_binary_paste_location = mask_aux

        mask = patch_binary_paste_location + mask
        
        patch_to_remove_boat = patch_binary_paste_location*img_array
        patch_to_remove_boat = patch_to_remove_boat == 0
        patch_to_remove_boat = 1*patch_to_remove_boat
        
        
        img_array_aux = new_image_clean - patch_binary_paste_location*new_image_clean # image with black space where is to add the boat
        
        
        img_array_aux = new_image_clean + patch_paste
        
        new_image_clean = img_array_aux
        
        
        if repeat > 0.8:
            
            image_final = Image.fromarray(np.uint8(np.zeros((256,256))),'L')
            mask_aux = Image.fromarray(np.uint8(np.zeros((256,256))),'L')
            
            
            if len(props) == 0: #did not find any boat
                break
            
            if counter > len(props)-1:
                index = random.randint(0, len(props)-1)
            patch = 1*props[index].image
            patch = Image.fromarray(np.uint8(patch),'L')

            
            bbox = props[index].bbox
        

            binary_boat_location = labels_save == index+1   # binary map with the location of the boat in original image
            patch_boat_original = binary_boat_location*img_array # boat and all zero
            
            
            rot_deg = random.uniform(*[-180,180])
            patch = patch.rotate(rot_deg, expand=True)
            mask_rot = patch.split()[-1]
            
            small_patch_boat = patch_boat_original[bbox[0]:bbox[2],bbox[1]:bbox[3]]
            small_patch_boat = Image.fromarray(np.uint8(small_patch_boat),'L')
            # fig, ax = plt.subplots()
            # plt.imshow(small_patch_boat, cmap="gray")
            # plt.show()
            small_patch_boat = small_patch_boat.rotate(rot_deg, expand=True)
            
            
            # corrects bug of boats being near border and bugged after rotation
            
            paste_top = bbox[0]
            paste_left = bbox[1]
            if bbox[0]+np.shape(small_patch_boat)[1] > 255:
                paste_top = bbox[1]-np.shape(small_patch_boat)[1]
            if bbox[1]+np.shape(small_patch_boat)[0] > 255:
                paste_left = bbox[0]-np.shape(small_patch_boat)[0]
            

            paste_left = bbox[1]
            paste_top = bbox[0]
            
            mask_aux.paste(patch, ((paste_left , paste_top)), mask=None)
                    
            
            image_final.paste(small_patch_boat, ((paste_left), paste_top), mask=None)

            index_yy_0 = paste_top
            index_yy_end_1 = paste_top+np.shape(small_patch_boat)[1]
            index_yy_end = 255 - index_yy_end_1
            index_xx_0 = paste_left
            index_xx_end_1 = paste_left+np.shape(small_patch_boat)[0]
            index_xx_end = 255 - index_xx_end_1
            
            

            if index_yy_end>-index_yy_0 and  index_xx_end>-index_xx_0:
                delta_yy = random.randint(-index_yy_0, index_yy_end)
                delta_xx = random.randint(-index_xx_0, index_xx_end)
            else:
                delta_yy = 10
                delta_xx = 10
            

            patch_paste = np.roll(image_final, delta_yy, axis = 0)
            patch_paste = np.roll(patch_paste, delta_xx, axis = 1)
            patch_binary_paste_location = np.roll(mask_aux, delta_yy, axis = 0)
            patch_binary_paste_location = np.roll(patch_binary_paste_location, delta_xx, axis = 1)



            mask = patch_binary_paste_location + mask
            
            patch_to_remove_boat = patch_binary_paste_location*img_array
            patch_to_remove_boat = patch_to_remove_boat == 0
            patch_to_remove_boat = 1*patch_to_remove_boat
            
            
            img_array_aux = new_image_clean - patch_binary_paste_location*new_image_clean # image with black space where is to add the boat
            
            
            
            if random.choice([0, 1]): #50% chance of adding noise
                patch_paste = random_noise(np.array(patch_paste))
                

            if random.choice([0, 1]): #50% chance of adding noise
                patch_paste = Image.fromarray(np.uint8(patch_paste),'L')
                enhancer_bri = ImageEnhance.Brightness(patch_paste)
                patch_paste = enhancer_bri.enhance(1)
                enhancer_con = ImageEnhance.Contrast(patch_paste)
                patch_paste = enhancer_con.enhance(1)
                patch_paste = adjust_gamma(np.array(patch_paste), gamma=2)
                patch_paste = np.array(patch_paste)
            
            
            img_array_aux = img_array_aux + patch_paste
    
            new_image_clean = img_array_aux
           
        
        index+=1
        
    
    save_image(new_image_clean,str(ii))
    
    mask[mask>0.5] = 1
    save_mask(mask,str(ii))
    
    
    ii+=1
    


