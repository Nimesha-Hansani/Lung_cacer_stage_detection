#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
import tensorflow_io as tfio
import matplotlib.pyplot as plt
import numpy as np


# In[8]:


image_bytes = tf.io.read_file('C:/Nimesha/MSC_UOM/Research/Lung_cacer_stage_detection/Data/Img/A_01.dcm')
print(image_bytes)


# In[9]:


image_eagertensor = tfio.image.decode_dicom_image(image_bytes, dtype=tf.uint16)
print(image_eagertensor)


# In[11]:


image_tensor=tf.convert_to_tensor(image_eagertensor)
print(type(image_tensor))


# # Convert DCM 

# In[20]:


import pydicom
import torch


# In[21]:


ds = pydicom.dcmread('C:/Nimesha/MSC_UOM/Research/Lung_cacer_stage_detection/Data/Train/A_01.dcm')
#Extracting pixel array from dicom image
new_image = ds.pixel_array.astype(float)
image = torch.from_numpy(image)


# In[ ]:


final_image.show()

