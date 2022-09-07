#!/usr/bin/env python
# coding: utf-8

# In[9]:


#Importing Libraries

import os
import pydicom
import torch
import numpy as np
import matplotlib.pyplot as plt
from easyfsl.utils import plot_images, sliding_average
from torchvision.models import resnet18
from torch import nn, optim


# In[2]:


root_path_support ='C:/Nimesha/MSC_UOM/Research/Lung_cancer_stage_detection/Data/Support'
root_path_query = 'C:/Nimesha/MSC_UOM/Research/Lung_cancer_stage_detection/Data/Query' 

def load_scan(path):
    
    slices = [pydicom.dcmread(path + '/' + s) for s in os.listdir(path)]
    
    slices = [s for s in slices if 'SliceLocation' in s]
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    
    for s in slices:
        s.SliceThickness = slice_thickness

    labels = [  (((x['PatientID']).value).split('-')[1])[0] for x in slices if 'PatientID' in x]
    
    new_labels = []
    for character in labels:
        new_labels.append(ord(character))

    return slices,new_labels


# In[3]:


def get_pixels_hu(scans):
    
    image = np.stack([s.pixel_array for s in scans])
    image = image.astype(np.int16)
    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)


# In[4]:


patient_dicom_s=load_scan(root_path_support)

patient_pixels_s = get_pixels_hu(patient_dicom_s[0])
patient_tensor_s= torch.from_numpy(patient_pixels_s)

example_support_images=  patient_tensor_s
example_support_labels = patient_dicom_s[1]
example_support_labels=  torch.Tensor(example_support_labels)


# In[6]:


patient_dicom_q=load_scan(root_path_query)
patient_pixels_q = get_pixels_hu(patient_dicom_q[0])
patient_tensor_q= torch.from_numpy(patient_pixels_q)

example_query_images=  patient_tensor_q
print(len(example_query_images))
example_query_labels = patient_dicom_q[1]

example_query_labels=  torch.Tensor(example_query_labels)

print(example_query_labels)


# In[10]:


class PrototypicalNetworks(nn.Module):
    def __init__(self, backbone: nn.Module):
        super(PrototypicalNetworks, self).__init__()
        self.backbone = backbone
        
    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
    ) -> torch.Tensor:
        
        
        """
        Predict query labels using labeled support images.
        """
     # Extract the features of support and query images
     z_support = self.backbone.forward(support_images)
     z_query = self.backbone.forward(query_images)


# In[11]:


convolutional_network = resnet18(pretrained=True)
convolutional_network.fc = nn.Flatten()

model = PrototypicalNetworks(convolutional_network)


# In[ ]:




