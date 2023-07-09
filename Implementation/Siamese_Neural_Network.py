# %%

import os
import os
import random
import torch
import cv2
import numpy as np
from lungmask import mask
import SimpleITK as sitk
import torchvision.transforms as transforms
from scipy.ndimage.filters import median_filter
import pydicom
from torch.utils.data import Dataset

import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch import optim
import torch.nn.functional as F
import random
from itertools import combinations, product


# %%
def transform_to_hu(medical_image, image):
    intercept = medical_image.RescaleIntercept
    slope = medical_image.RescaleSlope
    hu_image = image * slope + intercept
    return hu_image

def get_mask(filename, plot_mask=False, return_val=False):

    input_image = sitk.ReadImage(filename)
    mask_out = mask.apply(input_image)[0]  #default model is U-net(R231)

    if return_val:
        return mask_out

def preprocess_images(img,dicom_image):

    hu_image = transform_to_hu(dicom_image, img)
    filtered_image = median_filter(hu_image, size=(3, 3))
    return filtered_image

# %%
class DICOMDataset(Dataset):

   def __init__(self, root_dir ,transform=None):

      self.img_labels = []
      self.root_dir = root_dir
      self.dcm_files = os.listdir(root_dir)

      # Group images by category
      categories = {'A': 1, 'B': 2, 'G': 3, 'E': 4}
      category_images = {category: [] for category in categories}

      for filename in os.listdir(root_dir):
            image_name = filename
            category = image_name[0]
            if category in categories:
                category_images[category].append(image_name)

      
      # Create pairs of images from the same category or different categories
      for category, images in category_images.items():
         # Same category pairs
         same_category_pairs = list(combinations(images, 2))
         same_category_labels = [1] * len(same_category_pairs)
         self.img_labels.extend(zip(same_category_pairs, same_category_labels))

         # Different category pairs
         for other_category, other_images in category_images.items():
            if other_category != category:
                  different_category_pairs = list(product(images, other_images))
                  different_category_labels = [0] * len(different_category_pairs)
                  self.img_labels.extend(zip(different_category_pairs, different_category_labels))
            
      # Shuffle the image labels
      random.shuffle(self.img_labels)
   

   def __getitem__(self, index):
        img_pair, label = self.img_labels[index]

        img1_path, img2_path = img_pair

        img1 = os.path.join(self.root_dir, img1_path)
        img2 = os.path.join(self.root_dir, img2_path)

        dicom_image1= pydicom.dcmread(img1)
        image1 = np.array(dicom_image1.pixel_array)

        dicom_image2= pydicom.dcmread(img2)
        image2 = np.array(dicom_image2.pixel_array)

        cleaned_image1 = preprocess_images(image1,dicom_image1)
        masked_img1=get_mask(img1,plot_mask=True,return_val=True)

        cleaned_image2 = preprocess_images(image2,dicom_image2)
        masked_img2=get_mask(img2,plot_mask=True,return_val=True)

        mask_on_orginal1 = cleaned_image1 * masked_img1
        mask_on_orginal1 = cv2.resize(mask_on_orginal1, (224, 224))


        mask_on_orginal2 = cleaned_image2 * masked_img2
        mask_on_orginal2 = cv2.resize(mask_on_orginal2, (224, 224))

        image1 = mask_on_orginal1.astype('float32')
        image1 = np.expand_dims(image1, axis=0)

        image2 = mask_on_orginal2.astype('float32')
        image2 = np.expand_dims(image2, axis=0)
       
        image1 = torch.from_numpy(image1)
        image2 = torch.from_numpy(image2)
        print(label)
        return image1,image2, label
   
   def __len__(self):
        # print(len(self.img_labels))
        return len(self.img_labels)
       


# %%
#create the Siamese Neural Network
class SiameseNetwork(nn.Module):

    def __init__(self):
        
        super(SiameseNetwork, self).__init__()

        # Setting up the Sequential of CNN Layers
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3,stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
          
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3,stride=1),
            nn.ReLU(inplace=True)
        )

        # Calculate output shape of the convolutional layers
        conv_output_shape = self.cnn1(torch.zeros(1, *(1, 224, 224))).shape

        # Setting up the Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.Linear(conv_output_shape[1] * conv_output_shape[2] * conv_output_shape[3], 1024),
            nn.ReLU(inplace=True),
            
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            
            nn.Linear(256,2)
        )
        
    def forward_once(self, x):
        
       
        #  This function will be called for both images
        # Its output is used to determine the similiarity
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

       

    def forward(self, input1, input2):
        # In this function we pass in both images and obtain both vectors
        # which are returned
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1, output2

# %%
# Define the Contrastive Loss Function
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
      # Calculate the euclidean distance and calculate the contrastive loss
      euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)

      loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                    (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


      return loss_contrastive

# %%
siamese_dataset = DICOMDataset(root_dir='../IMAGES/TRAIN_SET/', transform=None)

# Create a simple dataloader just for simple visualization
train_dataloader = DataLoader(siamese_dataset,
                        shuffle=True,
                        num_workers=0)
net = SiameseNetwork()
criterion = ContrastiveLoss()
optimizer = optim.Adam(net.parameters(), lr = 0.0001 )

# %%
counter = []
loss_history = [] 
iteration_number= 0

# Iterate throught the epochs
for epoch in range(5):
    print(epoch)

    #  Iterate over batches
    for i, (img0, img1, label) in enumerate(train_dataloader, 0):
        
        print(i)
        
        # Zero the gradients
        optimizer.zero_grad()
   
        # Pass in the two images into the network and obtain two outputs
        output1, output2 = net(img0, img1)

        # Pass the outputs of the networks and label into the loss function
        loss_contrastive = criterion(output1, output2, label)


         # Calculate the backpropagation
        loss_contrastive.backward()

        # Optimize
        optimizer.step()

        # Every 5  batches print out the loss
        if i % 10 == 0 :
            print (i)
            print(f"Epoch number {epoch}\n Current loss {loss_contrastive.item()}\n")
            iteration_number += 10

            counter.append(iteration_number)
            loss_history.append(loss_contrastive.item())



# %%
import matplotlib.pyplot as plt
plt.plot(counter, loss_history)
plt.show()

# %%
test_dataset = DICOMDataset(root_dir='../IMAGES/TEST_SET/', transform=None)

# Create a simple dataloader just for simple visualization
test_dataloader = DataLoader(test_dataset,
                        shuffle=False,
                        num_workers=0)


# %%
import torchvision



# # Grab one image that we are going to test
dataiter = iter(test_dataloader)
x0, _, _ = next(dataiter)



for i in range(30):
    # Iterate over 5 images and test them with the first image (x0)
    x0, x1, label2 = next(dataiter)
    
    print(label2)

    

     # Concatenate the two images together
    concatenated = torch.cat((x0, x1), dim=3)

    # Convert the concatenated tensor to a numpy array
    concatenated_np = concatenated.squeeze().numpy()


    output1, output2 = net(x0, x1)
    euclidean_distance = F.pairwise_distance(output1, output2)

    if label2==torch.FloatTensor([[1]]):
       status ="DICOM pair comes from same lung cancer type"
    else :
       status ="DICOM  pair comes from different lung cancer types"
       

    print(status)
    plt.imshow(concatenated_np,cmap="ocean")
    plt.title(f'{status}\nDissimilarity: {euclidean_distance.item():.2f}')

    plt.axis('off')
    plt.show()

    if label2==torch.FloatTensor([[0]])  and  euclidean_distance >= 1.5 :
       break
      




   
    
    
    
    

# %% [markdown]
# 


