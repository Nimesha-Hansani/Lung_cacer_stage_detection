#!/usr/bin/env python
# coding: utf-8

# In[8]:


import os
import pydicom


# In[11]:


class LungDataset():
    
    def __init__(self, root, image_dir):
          
          self.image_dir = os.path.join(root, image_dir) 
          
            
    def __getitem__(self, index):
        
        image_file = pydicom.dcmread(os.path.join(self.image_dir, self.data[index][1], self.data[index][0]))
        print(image_file)
        
        
          


# In[12]:


ROOT_PATH='C:/Nimesha/MSC_UOM/Research/Lung_cacer_stage_detection/Data/'
train_set = LungDataset(ROOT_PATH, 'Train')


# In[13]:


train_set.__getitem__(1)


# In[ ]:




