# %%
import pandas as pd
import os
import pydicom
import numpy as np
import cv2
from CNN import CNN
import torch
from preprocessing import transform_to_hu,get_mask,preprocess_images
import torch.nn as nn

# %%

def  model_run():
    support_images = []
    support_labels = []

    dicom_folder = 'C:/Users/Nimesha/Documents/MSC_RESEARCH/IMAGES/Support_Set'
    dicom_query = 'C:/Users/Nimesha/Documents/MSC_RESEARCH/Implementation/Prediction/data'
    # Load and preprocess the DICOM images
    for root, dirs, files in os.walk(dicom_folder):
        for file in files:
            file_path = os.path.join(root, file)
            dicom_image = pydicom.dcmread(file_path)
            image = np.array(dicom_image.pixel_array)

            cleaned_image = preprocess_images(image,dicom_image)
            masked_img=get_mask(file_path,plot_mask=True,return_val=True)
            
            mask_on_orginal = cleaned_image * masked_img
            mask_on_orginal = cv2.resize(mask_on_orginal, (224, 224))
            
            image = mask_on_orginal.astype('float32')
            image = np.expand_dims(image, axis=0)
            
            image = torch.from_numpy(image)

            print(type(image))
    
            label_ch = file[0]
            print(label_ch)
            print(image)
            if label_ch =='A' :
                label=0
            elif label_ch =='B':
                label=1
            elif label_ch =='G':  
                label=2
            elif label_ch =='E':
                label=3 
            else : label=4
            
            label = torch.tensor(label, dtype=torch.int64)
            
            support_images.append(image)
            support_labels .append(label)
            
    support_images = torch.stack(support_images)
    support_labels = torch.stack(support_labels)


    # %%
    query_images = []

    for root, dirs, files in os.walk(dicom_query):
        for file in files:
            file_path = os.path.join(root, file)
            dicom_image = pydicom.dcmread(file_path)
            image = np.array(dicom_image.pixel_array)

            cleaned_image = preprocess_images(image,dicom_image)
            masked_img=get_mask(file_path,plot_mask=True,return_val=True)
        
            mask_on_orginal = cleaned_image * masked_img
            mask_on_orginal = cv2.resize(mask_on_orginal, (224, 224))
        
            image = mask_on_orginal.astype('float32')
            image = np.expand_dims(image, axis=0)
        
            image = torch.from_numpy(image)

            query_images.append(image)


    query_images = torch.stack(query_images)
    print(type(query_images))


    # %%
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


            
            
            # Infer the number of different classes from the labels of the support set
            n_way = len(torch.unique(support_labels))
            
        
            # Prototype i is the mean of all instances of features corresponding to labels == i
            z_proto = torch.cat(
                [
                    
                    (z_support[torch.nonzero(support_labels == label)].mean(0)).round()
                    for label in range(n_way) 
                ]
            )
            
            print(z_proto)


        
            # Compute the euclidean distance from queries to prototypes
            dists = torch.cdist(z_query, z_proto)
    

            # And here is the super complicated operation to transform those distances into classification scores!
            scores = -dists
            print(type(scores))
            return scores

    # %%
    convolutional_network = CNN()
    model = PrototypicalNetworks(convolutional_network)
    model.load_state_dict(torch.load('model.pth'))

    model.eval()

    # %%


    example_scores = model(
            support_images,
            support_labels,
            query_images,
        ).detach()

    # _, example_predicted_labels = torch.max(example_scores.data, 1)

    # Use torch.topk to get indices of the top 2 maximum values
    topk_values, topk_indices = torch.topk(example_scores, k=2,dim=1)

    
    return topk_indices

        


