
from preprocessing import transform_to_hu,get_mask,preprocess_images
from torch.utils.data import Dataset
import os
import pydicom
import numpy as np
import torch
import cv2

class DICOMDataset(Dataset):

    def __init__(self, root_dir, transform=None):

        self.img_labels = []
        self.root_dir = root_dir
        self.dcm_files = os.listdir(root_dir)

        for filename in os.listdir(root_dir):
            
            image_name = filename
            category = image_name[0]

            if category =='A' :
               label=1
            elif category =='B':
               label=2
            elif category =='G':  
               label=3 
            elif category =='E':
               label=4 
            else : label=5        
          
            self.img_labels.append((image_name,label))

    def __len__(self):
        # print(len(self.img_labels))
        return len(self.img_labels)

    def __getitem__(self, idx):

        dcm_file = self.dcm_files[idx]
   
        label_ch =dcm_file[0]
        if label_ch =='A' :
            label=1
        elif label_ch =='B':
            label=2
        elif label_ch =='G':  
            label=3 
        elif label_ch =='E':
            label=4  
        else : label=5
        dcm_path = os.path.join(self.root_dir, dcm_file)
        
        dicom_image= pydicom.dcmread(dcm_path)
        image = np.array(dicom_image.pixel_array)
        print(image.shape)

        cleaned_image = preprocess_images(image,dicom_image)
        masked_img=get_mask(dcm_path,plot_mask=True,return_val=True)
    
        mask_on_orginal = cleaned_image * masked_img
        mask_on_orginal = cv2.resize(mask_on_orginal, (224, 224))
       
        image = mask_on_orginal.astype('float32')
        image = np.expand_dims(image, axis=0)
       
        image = torch.from_numpy(image)

        return image, label