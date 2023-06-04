from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
#import skimage.transform as trans
from PIL import Image



def transform(img, mask):
    random_transformation = np.random.randint(1,4)
    if random_transformation == 1:  # reverse first dimension
        img = img[::-1,:,:]
        mask = mask[::-1,:,:]
    elif random_transformation == 2:    # reverse second dimension
        img = img[:,::-1,:]
        mask = mask[:,::-1,:]
    elif random_transformation == 3:    # transpose(interchange) first and second dimensions
        img = img.transpose([1,0,2])
        mask = mask.transpose([1,0,2])
    # elif random_transformation == 4:
    #     img = np.rot90(img, 1)
    #     mask = np.rot90(mask, 1)
    # elif random_transformation == 5:
    #     img = np.rot90(img, 2)
    #     mask = np.rot90(mask, 2)        
    # elif random_transformation == 6:
    #     img = np.rot90(img, 3)
    #     mask = np.rot90(mask, 3)
    else:
        pass
    
    return (img,mask)

def adjustData(img,mask,flag_multi_class,num_class):
    if(flag_multi_class):
        img = img / 255
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            #for one pixel in the image, find the class in mask and convert it into one-hot vector
            #index = np.where(mask == i)
            #index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
            #new_mask[index_mask] = 1
            new_mask[mask == i,i] = 1
        new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
        mask = new_mask
    elif(np.max(img) > 1):
        img = img / 255
        mask = mask /255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img,mask)



import keras as keras
from keras.preprocessing import sequence
from keras.preprocessing import image
    

class TrainDataGenerator(keras.utils.all_utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32, dim=(64,64), n_channels=8,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        img = np.empty((self.batch_size, *self.dim, 1))# alterar aqui se as imagens não forem RBG
        mask = np.empty((self.batch_size, *self.dim, self.n_classes))
       
        msk = np.empty((*self.dim,self.n_classes))
#        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            im=Image.open('data/train/UDSEP/DSEP_results/images/' + ID)
            # print(np.shape(im))

            im = np.reshape(im,(256,256,1)) #adicionei isto
            im=np.array(im)/255

            a=Image.open('data/train/UDSEP/DSEP_results/segmentation/' + ID)
            a=np.array(a)
            msk[:,:,0]=a/255
            msk[:,:,1]=np.logical_not(msk[:,:,0])
            
            #im,msk=transform(im, msk)
            
            # print(np.shape(img))
           
            img[i,] = np.expand_dims(im,axis=0)
            mask[i,] = np.expand_dims(msk,axis=0)
          

        return img, mask

class DataGenerator(keras.utils.all_utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32, dim=(64,64), n_channels=8,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        img = np.empty((self.batch_size, *self.dim, 1))
        mask = np.empty((self.batch_size, *self.dim, self.n_classes))
        msk = np.empty((*self.dim, self.n_classes))
       
        # Generate data sem data augmentation
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            im=Image.open('data/train/UDSEP/DSEP_results/images/' + ID)
            im = np.reshape(im,(256,256,1)) #adicionei isto
            # im = im.reshape(im,(*self.dim,n_channels))
            im=np.array(im)/255

            a=Image.open('data/train/UDSEP/DSEP_results/segmentation/' + ID)
            a=np.array(a)
            msk[:,:,0]=a/255
            msk[:,:,1]=np.logical_not(msk[:,:,0])            
                        
            img[i,] = np.expand_dims(im,axis=0)
            mask[i,] = np.expand_dims(msk,axis=0)

        return img, mask

def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out


def saveResult(save_path,npyfile,flag_multi_class = False,num_class = 2):
    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        #img=item
        io.imsave(os.path.join(save_path,"%d_predict.png"%i),img,check_contrast=False)
        
