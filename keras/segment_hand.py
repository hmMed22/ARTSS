'''

pip install -U segmentation-models

## If we hav  a large image we can patch it 
from patchify import patchify

all_img_patches=[]
for img in range(images.shape[0]):
  large_image=images[img]
  patches_img=patchify(large_image,(244,244),step=244)

for i in range(patches_img.shape[0]):
   for j in range (patches_img.shape[1]):
     single_patch_img=patches_img[i,j,:,:]
     single_patch_img=(single_patch_img.astype('flot32'))/255
     all_img_patches.append(single_patch_img)
images_all=np.array(all_img_patches)
images_all=np.stack((images_all,)*3,axis=-1)

all_mask_patches=[]
for img in range(masks.shape[0]):
  large_mask=masks[img]
  patches_mask=patchify(large_mask,(244,244),step=244)

for i in range(patches_mask.shape[0]):
   for j in range (patches_mask.shape[1]):
     single_patch_mask=patches_mask[i,j,:,:]
     single_patch_mask=single_patch_mask/255
     all_mask_patches.append(single_patch_mask)
masks_all=np.array(all_img_patches)
masks_all=np.expand_dims(masks_all,-1)

*Expand** the dimention for image data generator
masks=np.array(input_mask)
images=np.array(input_image)
images_all=np.stack((images,)*3,axis=-1)
masks_all=np.expand_dims(masks,-1)



import tensorflow as tf
from tensorflow import keras
from keras import layers
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import numpy as np

import os
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
import scipy.stats as stats
from os import listdir
from os.path import isfile, join
import cv2 as cv
from google.colab.patches import cv2_imshow
import tensorflow as tf
import keras  
import glob 
from skimage import io 
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import apply_affine_transform
import math
##Convert gray image to 3 channels by copying channel 3 times
## we do this as our unet  model expect 3 channel input 
## images= np.stack((images,)*3,axis=-1)
img_size=256
input_image_name=[img.name for img in Path(path_image).iterdir() if img.suffix==".jpg"]
mask_name= [img.name for img in Path(path_mask).iterdir() if img.suffix==".jpg"]

#get missing name
set_in=set(input_image_name)
set_mask=set(mask_name)
match_name=list(set_in&set_mask)
missing_name=list(set_in-set_mask)
input_image=[]
input_mask=[]
test_image=[]
for index in range(len(match_name)):
  img=cv.imread(os.path.join('/content/drive/MyDrive/Segmentation/Train', match_name[index]))
  resized_img=cv.resize(img,(img_size,img_size))

  norm_image = cv.normalize(resized_img, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX,dtype=cv.CV_32F)

  #cv.imwrite(os.path.join('/content/drive/MyDrive/Segmentation/Mask_Image_resizeCorrect', match_name[index]),resized_img)
  input_image.append(norm_image)
  mask=cv.imread(os.path.join('/content/drive/MyDrive/Segmentation/Label_Train',match_name[index]))
  resized_mask=cv.resize(mask,(img_size,img_size))

 # cv.imwrite(os.path.join('/content/drive/MyDrive/Segmentation/Label_resizeTraincorrect', match_name[index]),resized_mask)

  input_mask.append(resized_mask/255)
seed=24
import segmentation_models as sm
##Backbone is the model to be used for the encoder part of the UNet. 
## This let us benefit from transfer learning by using pretrained weights such as "imagenet"
#BACKBONE='resnet50'
BACKBONE='efficientnetb0'

#BACKBONE='densenet121'
#BACKBONE='inceptionv3'

preprocess_input1=sm.get_preprocessing(BACKBONE)
##preprocess input
images1=preprocess_input1(images)
masks1=preprocess_input1(masks)
print(images1.shape)
print(masks1.shape)

from keras.preprocessing.image import ImageDataGenerator
img_data_gen_args=dict(rotation_range=10,
                       width_shift_range=0.2,
                       height_shift_range=0.2,
                       zoom_range=0.2,
                       horizontal_flip=True,
                       vertical_flip=True)

mask_data_gen_args=dict(rotation_range=10,
                       width_shift_range=0.2,
                       height_shift_range=0.2,
                       zoom_range=0.2,
                       horizontal_flip=True,
                       vertical_flip=True)
                      # preprocessing_function=lambda x:np.where(x>0,1,0).astype(x.dtype))##binarize the mask when rotate it again
image_data_generator=ImageDataGenerator(**img_data_gen_args)
image_data_generator.fit(X_train,augment=True,seed=seed)
image_generator=image_data_generator.flow(X_train,seed=seed)
valid_img_generator=image_data_generator.flow(X_test,seed=seed)
mask_data_generator=ImageDataGenerator(**mask_data_gen_args)
mask_data_generator.fit(y_train,augment=True,seed=seed)
mask_generator=mask_data_generator.flow(y_train,seed=seed)
valid_mask_generator=mask_data_generator.flow(y_test,seed=seed)
def my_image_mask_generator(image_generator,mask_generator):
    train_genratot= zip(image_generator,mask_generator)
    for (img,mask) in train_genratot:
      yield(img,mask)

my_generator=my_image_mask_generator(image_generator,mask_generator)
validation_datagen=my_image_mask_generator(valid_img_generator,valid_mask_generator)
'''
import os
import numpy as np
from patchify import patchify
from pathlib import Path
import cv2 as cv
from keras.preprocessing.image import ImageDataGenerator
import segmentation_models as sm

class SegmentationPipeline:
    def __init__(self, image_path, mask_path, img_size=256, patch_size=244, backbone='efficientnetb0', seed=24):
        self.image_path = image_path
        self.mask_path = mask_path
        self.img_size = img_size
        self.patch_size = patch_size
        self.backbone = backbone
        self.seed = seed

        # Initialize lists for images and masks
        self.input_images = []
        self.input_masks = []

        # Initialize processed data
        self.images_all = None
        self.masks_all = None

    def load_and_preprocess_data(self):
        # Gather filenames
        input_image_names = [img.name for img in Path(self.image_path).iterdir() if img.suffix == ".jpg"]
        mask_names = [img.name for img in Path(self.mask_path).iterdir() if img.suffix == ".jpg"]

        # Match images and masks
        matched_names = list(set(input_image_names) & set(mask_names))

        for name in matched_names:
            # Load and normalize images
            img = cv.imread(os.path.join(self.image_path, name))
            resized_img = cv.resize(img, (self.img_size, self.img_size))
            norm_image = cv.normalize(resized_img, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
            self.input_images.append(norm_image)

            # Load and preprocess masks
            mask = cv.imread(os.path.join(self.mask_path, name))
            resized_mask = cv.resize(mask, (self.img_size, self.img_size))
            self.input_masks.append(resized_mask / 255.0)

        self.input_images = np.array(self.input_images)
        self.input_masks = np.array(self.input_masks)

    def patchify_data(self):
        # Patchify images
        all_img_patches = []
        for img in self.input_images:
            patches_img = patchify(img, (self.patch_size, self.patch_size, 3), step=self.patch_size)
            for i in range(patches_img.shape[0]):
                for j in range(patches_img.shape[1]):
                    patch = patches_img[i, j, 0]
                    all_img_patches.append(patch.astype('float32') / 255.0)

        self.images_all = np.array(all_img_patches)
        self.images_all = np.stack((self.images_all,), axis=-1)

        # Patchify masks
        all_mask_patches = []
        for mask in self.input_masks:
            patches_mask = patchify(mask, (self.patch_size, self.patch_size, 1), step=self.patch_size)
            for i in range(patches_mask.shape[0]):
                for j in range(patches_mask.shape[1]):
                    patch = patches_mask[i, j, 0]
                    all_mask_patches.append(patch.astype('float32'))

        self.masks_all = np.array(all_mask_patches)
        self.masks_all = np.expand_dims(self.masks_all, axis=-1)

    def preprocess_input(self):
        # Apply preprocessing for the selected backbone
        preprocess_input_fn = sm.get_preprocessing(self.backbone)
        self.images_all = preprocess_input_fn(self.images_all)
        self.masks_all = preprocess_input_fn(self.masks_all)

    def create_data_generators(self, X_train, X_test, y_train, y_test):
        # Define data augmentation parameters
        img_data_gen_args = dict(
            rotation_range=10,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True
        )

        mask_data_gen_args = dict(
            rotation_range=10,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True
        )

        # Create generators
        image_data_generator = ImageDataGenerator(**img_data_gen_args)
        image_data_generator.fit(X_train, augment=True, seed=self.seed)
        image_generator = image_data_generator.flow(X_train, seed=self.seed)
        valid_img_generator = image_data_generator.flow(X_test, seed=self.seed)

        mask_data_generator = ImageDataGenerator(**mask_data_gen_args)
        mask_data_generator.fit(y_train, augment=True, seed=self.seed)
        mask_generator = mask_data_generator.flow(y_train, seed=self.seed)
        valid_mask_generator = mask_data_generator.flow(y_test, seed=self.seed)

        # Combine image and mask generators
        def my_image_mask_generator(image_gen, mask_gen):
            for img, mask in zip(image_gen, mask_gen):
                yield img, mask

        train_generator = my_image_mask_generator(image_generator, mask_generator)
        validation_generator = my_image_mask_generator(valid_img_generator, valid_mask_generator)

        return train_generator, validation_generator

# Example 
if __name__ == "__main__":
    pipeline = SegmentationPipeline(
        image_path="/path/to/images",
        mask_path="/path/to/masks",
        img_size=256,
        patch_size=244,
        backbone='efficientnetb0'
    )

    pipeline.load_and_preprocess_data()
    pipeline.patchify_data()
    pipeline.preprocess_input()
    
    # Train/test split (example split, replace with actual data)
    X_train, X_test, y_train, y_test = train_test_split(pipeline.images_all, pipeline.masks_all, test_size=0.2, random_state=42)
    
    train_gen, val_gen = pipeline.create_data_generators(X_train, X_test, y_train, y_test)
    print("Data pipeline successfully prepared.")
