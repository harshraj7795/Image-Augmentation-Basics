# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 20:26:24 2021

@author: HSingh
"""

#importing the libraries
from keras.preprocessing.image import load_img, ImageDataGenerator, img_to_array


#Creating image data generator

datagen=ImageDataGenerator(horizontal_flip=0.2, vertical_flip=0.2, fill_mode='nearest', zoom_range=0.2,
                           height_shift_range=0.2, rotation_range=40)

#loading image
img=load_img('woof_meow.jpg')

#converting the image into array
arr_img=img_to_array(img)
arr_img=arr_img.reshape((1,)+arr_img.shape)

#generating batches of randomly transformed images using datagen.flow()

i=0
for batches in datagen.flow(arr_img,batch_size=1,save_to_dir='preview',save_prefix='cat',save_format='jpg'):
   i+=1
   if i>20:
       break