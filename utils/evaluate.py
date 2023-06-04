
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 14:19:27 2022

@author: joaomoura
"""

import os
import numpy as np
from skimage.measure import label
from skimage.measure import regionprops
import cv2
from scipy import ndimage


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
    

def intersection_over_union(ground_truth, prediction):
    # Count objects
    true_objects = len(np.unique(ground_truth))
    pred_objects = len(np.unique(prediction))
    
    # Compute intersection
    
    h = np.histogram2d(ground_truth.flatten(), prediction.flatten(), bins=(true_objects,pred_objects))
    intersection = h[0]
    
    # Area of objects
    area_true = np.histogram(ground_truth, bins=true_objects)[0]
    area_pred = np.histogram(prediction, bins=pred_objects)[0]
    
    # Calculate union
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)
    union = area_true + area_pred - intersection
    
    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    
    # Compute Intersection over Union
    union[union == 0] = 1e-9
    IOU = intersection/union
    
    return IOU


def measures_at(threshold, IOU):
    
    matches = IOU > threshold
    
    true_positives  = np.sum(matches, axis=1) == 1   # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Extra objects
    false_negatives = np.sum(matches, axis=1) == 0  # Missed objects

    assert np.all(np.less_equal(true_positives, 1))
    assert np.all(np.less_equal(false_positives, 1))
    assert np.all(np.less_equal(false_negatives, 1))
    
    TP, FP, FN = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
    
    precision = (TP) / (TP + FP) 

    recall = (TP) / (TP + FN)

    f1 = 2*TP / (2*TP + FP + FN + 1e-9)
    
    return f1, float(TP), float(FP), float(FN), precision, recall

def compute_results(ground_truth, prediction, image_name, counter, TP_total, FP_total, FN_total, TP_simple, FP_simple, FN_simple, TP_complex, FP_complex, FN_complex, complexity):

    # Compute IoU
    IOU = intersection_over_union(ground_truth, prediction)
    if IOU.shape[0] > 0:
        jaccard = np.max(IOU, axis=0).mean()
    else:
        jaccard = 0.0
    
    #get splits and merges
    current_index = 0
    # Calculate F1 score at all thresholds
    for t in np.arange(0.1, 1.0, 0.1):
        
        
        f1, tp, fp, fn, precision, recall = measures_at(t, IOU)
        
        TP_total[current_index] += tp
        FP_total[current_index] += fp
        FN_total[current_index] += fn
        if complexity == 1:
            TP_simple[current_index]  += tp
            FP_simple[current_index]  += fp
            FN_simple[current_index]  += fn

        elif complexity == 0:
            TP_complex[current_index]  += tp
            FP_complex[current_index]  += fp
            FN_complex[current_index]  += fn


        current_index+=1
        
    return TP_total, FP_total, FN_total

def evaluate(ground_truth_dir, predictions_dir, complexity_list):
    
    counter = 0
    TP_total = np.zeros((9,1))
    FP_total = np.zeros((9,1))
    FN_total = np.zeros((9,1))
    TP_simple = np.zeros((9,1))
    FP_simple = np.zeros((9,1))
    FN_simple = np.zeros((9,1))
    TP_complex = np.zeros((9,1))
    FP_complex = np.zeros((9,1))
    FN_complex = np.zeros((9,1))
    for gt, pred in zip(sorted(os.listdir(ground_truth_dir)), sorted(os.listdir(predictions_dir))):
        
        index = sorted_int_list.index(int(gt))
        complexity = complexity_list[index]
        
        #read each file (ground truth segmentation and predicted segmentation)
        ground_truth = cv2.imread(os.path.join(ground_truth_dir, gt), cv2.IMREAD_GRAYSCALE)
        prediction = cv2.imread(os.path.join(predictions_dir, pred), cv2.IMREAD_GRAYSCALE)
        
        print(gt)
        print(pred)
        
    
        ground_truth = pre_process(ground_truth) 
        prediction = pre_process(prediction) 

        ground_truth = label(ground_truth)
        prediction = label(prediction)
    
        #compute evaluation metrics for the pair ground truth and predicted mask
        TP_total, FP_total, FN_total = compute_results(
                ground_truth,
                prediction,
                gt,counter, TP_total, FP_total, FN_total,
                TP_simple, FP_simple, FN_simple,
                TP_complex, FP_complex, FN_complex, complexity)
        
        counter+=1
        
    return TP_total, FP_total, FN_total, TP_simple, FP_simple, FN_simple, TP_complex, FP_complex, FN_complex




path = 'data/test/images_test'

ff = os.listdir(path)
ff_sorted = sorted(ff)
int_list = list(map(int, ff_sorted))
sorted_int_list = sorted(int_list)

complexity_list = np.load("utils/lista_final.npy")
complexity_list = complexity_list - 1 #offset to 0-1: 0 complex, 1 simple

     
ground_truth_dir = 'data/test/segmentation_gt_test' #directory with the ground truth segmentation masks
predictions_dir = 'experiments/results/segmentation_results_binary'


model_number = '01'


 
TP_total, FP_total, FN_total, TP_simple, FP_simple, FN_simple, TP_complex, FP_complex, FN_complex = evaluate(ground_truth_dir, predictions_dir, complexity_list)

precision = (TP_total) / (TP_total + FP_total) 
recall = (TP_total) / (TP_total + FN_total)
f1 = 2*TP_total / (2*TP_total + FP_total + FN_total)
precision_simple = (TP_simple) / (TP_simple + FP_simple) 
recall_simple = (TP_simple) / (TP_simple + FN_simple)
f1_simple = 2*TP_simple / (2*TP_simple + FP_simple + FN_simple)
precision_complex = (TP_complex) / (TP_complex + FP_complex) 
recall_complex = (TP_complex) / (TP_complex + FN_complex)
f1_complex = 2*TP_complex / (2*TP_complex + FP_complex + FN_complex)

results = [f1,precision,recall,f1_simple,precision_simple,recall_simple,f1_complex,precision_complex,recall_complex]
np.save("experiments/results/Metrics/"+model_number,results)





