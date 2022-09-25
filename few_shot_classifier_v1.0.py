#!/usr/bin/env python
# coding: utf-8

# In[21]:


#Importing libraries

import pydicom
import numpy
import numpy as np
import cv2
import os
import math
import pylab
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from scipy import ndimage
from skimage import morphology

import torch
from torch import nn, optim
from torchvision.models import resnet18


# In[22]:


def transform_to_hu(medical_image, image):
    intercept = medical_image.RescaleIntercept
    slope = medical_image.RescaleSlope
    hu_image = image * slope + intercept

    return hu_image


def window_image(image, window_center, window_width):
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    window_image = image.copy()
    window_image[window_image < img_min] = img_min
    window_image[window_image > img_max] = img_max
    
    return window_image


def remove_noise(medical_image, image,display=False):
    
    
    hu_image = transform_to_hu(medical_image, image)
    lung_image = window_image(hu_image, 40, 80)
    
    segmentation = morphology.dilation(lung_image, np.ones((1, 1)))
    labels, label_nb = ndimage.label(segmentation)
    
    label_count = np.bincount(labels.ravel().astype(np.int))
    label_count[0] = 0

    mask = labels == label_count.argmax()
    
    # Improve the brain mask
    mask = morphology.dilation(mask, np.ones((1, 1)))
    mask = ndimage.morphology.binary_fill_holes(mask)
    mask = morphology.dilation(mask, np.ones((3, 3)))
    
    masked_image = mask * lung_image
    

#     if display:
#         plt.figure(figsize=(15, 2.5))
#         plt.subplot(141)
#         plt.imshow(lung_image)
#         plt.title('Original Image')
#         plt.axis('off')
        
        
    return lung_image
        


# In[23]:


#Directory path for images


root_path_support ='C:/Nimesha/MSC_UOM/Research/Lung_cancer_stage_detection/Data/Support'
root_path_query = 'C:/Nimesha/MSC_UOM/Research/Lung_cancer_stage_detection/Data/Query' 

support_images=[]
support_labels=[]

#Iterate through the path 
for s in os.listdir(root_path_support):

     medical_image = pydicom.read_file(root_path_support + '/' + s)
     label= ord( (((medical_image['PatientID']).value).split('-')[1])[0])
   
     image =medical_image.pixel_array
     processedImg = remove_noise(medical_image,image, display=True)
     processedImg=numpy.uint8(processedImg)
     
     #Image with 3 dimensions 
     Image_BGR = cv2.cvtColor(processedImg, cv2.COLOR_GRAY2BGR)
        
    
     support_images.append(Image_BGR)
     support_labels.append(label)



# In[24]:


#Iterate through the path 
query_images=[]
query_labels=[]

for s in os.listdir(root_path_query):

     medical_image = pydicom.read_file(root_path_query + '/' + s)
     label= ord( (((medical_image['PatientID']).value).split('-')[1])[0])
   
     image =medical_image.pixel_array
     processedImg = remove_noise(medical_image,image, display=True)
     processedImg=numpy.uint8(processedImg)
     
     #Image with 3 dimensions 
     Image_BGR = cv2.cvtColor(processedImg, cv2.COLOR_GRAY2BGR)
        
    
     query_images.append(Image_BGR)
     query_labels.append(label)


# In[25]:


support_images=np.array(support_images)
support_labels=np.array(support_labels)


query_images=np.array(query_images)
query_labels=np.array(query_labels)


# In[26]:


support_images= torch.from_numpy(support_images)
support_labels=  torch.Tensor(support_labels)

query_images= torch.from_numpy(query_images)
query_labels=  torch.Tensor(query_labels)

print(type(support_images))


# In[29]:


support_images = support_images.float()


# In[30]:


z_proto = torch.cat(
            [
                support_images[torch.nonzero(support_labels == label)].mean(0)
                for label in range(4)
            ]
        )


# In[ ]:




