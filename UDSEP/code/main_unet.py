#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 15:14:16 2022

@author: joaomoura
"""


from unet_model import *
from data2 import *
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from skimage import segmentation
import random
import os

from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


#data_gen_args = dict(rotation_range=0.2,
#                    width_shift_range=0.05,
#                    height_shift_range=0.05,
#                    shear_range=0.05,
#                    zoom_range=0.05,
#                    horizontal_flip=True,
#                    fill_mode='nearest')


directory='UDSEP/data/images'
names = os.listdir(directory)

#data_gen_args = dict()
val_samples=5 
train_samples=len(names)-val_samples

random.shuffle(names, random.random)
partition= {} 
a = [''  for x in range(train_samples)]
for j in range(train_samples):
    a[j]=names[j]
partition ["train"] = a	
a = [''  for x in range(val_samples)]
for j in range(val_samples):
    a[j]=names[train_samples+j]
partition ["validation"] = a	

import json
with open('UDSEP/models/unet_partition.json', 'w+') as json_file:
  	json.dump(partition,json_file)

train_samples=len(partition['train'])
val_samples=len(partition['validation'])

num_class=2
batch_size=5
steps_per_epoch=train_samples//batch_size
epochs=150
params = {'dim': (256,256),          
           'batch_size': batch_size,
          'n_classes':num_class,
          'n_channels': 1, #RGB images (num_channels=1 for grayscale)
          'shuffle': True}


training_generator = TrainDataGenerator(partition['train'], **params)
validation_generator = DataGenerator(partition['validation'], **params)

model = unet_model(n_classes=num_class, class_weights=[1 ,1]) 
model_name='unet_2classes_weighted.hdf5'

model_checkpoint = ModelCheckpoint(model_name, monitor='val_loss',verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights='true')

history = model.fit(
 	training_generator,
 	steps_per_epoch=steps_per_epoch,
 	epochs=epochs,
 	validation_data=validation_generator,
    validation_steps = val_samples//batch_size,
    callbacks=[early_stopping])

model.save_weights('UDSEP/models/01')
np.save('UDSEP/models/history_model_01.npy',history.history)




