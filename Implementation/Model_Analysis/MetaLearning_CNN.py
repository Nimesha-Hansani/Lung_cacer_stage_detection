# %%
import os
import random
import torch
import numpy as np
import torchvision.transforms as transforms
import pydicom
from scipy.ndimage.filters import median_filter
from lungmask import mask
import SimpleITK as sitk
import cv2
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from easyfsl.samplers import TaskSampler
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision.models as models

from PIL import Image
from torch.utils.data import Dataset
import tensorflow as tf
from torch import nn, optim
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from CustomTaskSampler import TaskSampler
import pickle

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


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x


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
                z_support[torch.nonzero(support_labels == label)].mean(0)
                for label in range(n_way)
            ]
        )

    
        # Compute the euclidean distance from queries to prototypes
        dists = torch.cdist(z_query, z_proto)
   

        # And here is the super complicated operation to transform those distances into classification scores!
        scores = -dists
        
        return scores

convolutional_network = CNN()
model = PrototypicalNetworks(convolutional_network)

test_set = DICOMDataset(root_dir='../IMAGES/TEST_SET/', transform=None)
train_set = DICOMDataset(root_dir='../IMAGES/TRAIN_SET/', transform=None)

N_WAY = 4  # Number of classes in a task
N_SHOT = 4 # Number of images per class in the support set
N_QUERY = 3 # Number of images per class in the query set
N_EVALUATION_TASKS = 1

# The sampler needs a dataset with a "get_labels" method. Check the code if you have any doubt!
test_set.get_labels = lambda: [
    instance[1] for instance in test_set.img_labels
]

test_sampler = TaskSampler(
    test_set, n_way=N_WAY , n_shot=N_SHOT, n_query=N_QUERY, n_tasks=N_EVALUATION_TASKS
)

test_loader = DataLoader(
    test_set,
    batch_sampler=test_sampler,
    num_workers=0,
    pin_memory=True,
    collate_fn=test_sampler.episodic_collate_fn,
)

(
    example_support_images,
    example_support_labels,
    example_query_images,
    example_query_labels,
    example_class_ids,
) = next(iter(test_loader))

example_scores = model(
    example_support_images,
    example_support_labels,
    example_query_images,
).detach()

_, example_predicted_labels = torch.max(example_scores.data, 1)


# Training a meta-learning algorithm
#Meta Training
N_TRAINING_EPISODES = 1

train_set.get_labels = lambda: [ instance[1] for instance in train_set.img_labels]

train_sampler = TaskSampler(
    train_set, n_way=N_WAY, n_shot=N_SHOT, n_query=N_QUERY, n_tasks=N_TRAINING_EPISODES
)
train_loader = DataLoader(
    train_set,
    batch_sampler=train_sampler,
    num_workers=0,
    pin_memory=True,
    collate_fn=train_sampler.episodic_collate_fn,
)

#Validation Set

val_set = DICOMDataset(root_dir='../IMAGES/VALIDATION_SET/', transform=None)
val_set.get_labels = lambda: [ instance[1] for instance in train_set.img_labels]

val_sampler = TaskSampler(
    train_set, n_way=N_WAY, n_shot=N_SHOT, n_query=N_QUERY, n_tasks=N_TRAINING_EPISODES
)
val_loader = DataLoader(
    train_set,
    batch_sampler=train_sampler,
    num_workers=0,
    pin_memory=True,
    collate_fn=train_sampler.episodic_collate_fn,
)

def sliding_average(lst, window_size):
    if window_size == 0:
        return 0.0
    return sum(lst[-window_size:]) / min(len(lst), window_size)

from tqdm import tqdm
import matplotlib.pyplot as plt

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def fit(
    support_images: torch.Tensor,
    support_labels: torch.Tensor,
    query_images: torch.Tensor,
    query_labels: torch.Tensor,
) -> float:
    optimizer.zero_grad()
    classification_scores = model(
        support_images, support_labels, query_images
    )

    loss = criterion(classification_scores, query_labels)
    loss.backward()
    optimizer.step()

    return loss.item()

def evaluate_on_one_task(
    support_images: torch.Tensor,
    support_labels: torch.Tensor,
    query_images: torch.Tensor,
    query_labels: torch.Tensor,
) -> [int, int]:
    return (
         torch.max(  
            model(support_images, support_labels, query_images).detach().data,1,)[1]
         )

log_update_frequency = 1

val_frequency = 1

all_loss = []
all_predictions = []
all_labels = []
# model.train()

episode_indices = []
training_losses = []
val_losses =[]
train_accuracies = []
val_accuracies=[]
best_accuracy = 0.0

all_loss = []
# model.train()

with tqdm(enumerate(train_loader), total=len(train_loader)) as tqdm_train:
    
    for episode_index, (
        support_images,
        support_labels,
        query_images,
        query_labels,
        _,
    ) in tqdm_train:
        
        loss_value = fit(support_images, support_labels, query_images, query_labels)
        all_loss.append(loss_value)
      
        if episode_index % log_update_frequency == 0:
            tqdm_train.set_postfix(loss=sliding_average(all_loss, log_update_frequency))

        print("Episode Index")
        print(episode_index)


        if episode_index % val_frequency == 0:
           
            model.eval()

            with torch.no_grad():
                total_correct = 0
                total_examples = 0
                for (
                    support_images,
                    support_labels,
                    query_images,
                    query_labels,
                    _,
                ) in train_loader:
                    classification_scores = model(
                        support_images, support_labels, query_images
                    )
                    predicted_labels = classification_scores.argmax(dim=-1)
                    total_correct += (predicted_labels == query_labels).sum().item()
                    total_examples += predicted_labels.shape[0]

                accuracy = total_correct / total_examples

                # Append episode index and accuracy to lists
                episode_indices.append(episode_index)
                train_accuracies.append(accuracy)
                
                print("Training")
                print(episode_indices)
                print(train_accuracies)
                print(all_loss)

             # Perform validation using the validation dataloader
            with torch.no_grad():
                total_correct = 0
                total_examples = 0
                val_loss = 0.0  # Initialize validation loss
                for (
                    val_support_images,
                    val_support_labels,
                    val_query_images,
                    val_query_labels,
                    _,
                ) in val_loader:  # Use the validation dataloader
                    val_classification_scores = model(
                        val_support_images, val_support_labels, val_query_images
                    )
                    val_loss += criterion(val_classification_scores, val_query_labels).item()
                    val_predicted_labels = val_classification_scores.argmax(dim=-1)
                    total_correct += (val_predicted_labels == val_query_labels).sum().item()
                    total_examples += val_predicted_labels.shape[0]

                val_accuracy = total_correct / total_examples
                val_losses.append(val_loss / len(val_loader))  # Average validation loss
                val_accuracies.append(val_accuracy)
                
                print("Validation")
                print(episode_indices)
                print(val_accuracies)
                print(val_loss)

            model.train()

# Save the model
torch.save(model.state_dict(), 'model.pth')


# Create a figure with two subplots
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

# Plot training accuracy in the first subplot
# ax1.plot(episode_indices,train_accuracies, label='Training Accuracy', marker='o', color='blue')
# ax1.plot(episode_indices, val_accuracies, label='Validation Accuracy', marker='o', color='orange')
# ax1.set_xlabel('Episode Index')
# ax1.set_ylabel('Accuracy')
# ax1.legend()

# Plot training loss in the second subplot
# ax2.plot(episode_indices, all_loss, label='Training Loss', marker='o', color='blue')
# ax2.plot(episode_indices, val_losses, label='Validation Loss', marker='o', color='orange')
# ax2.set_xlabel('Episode Index')
# ax2.set_ylabel('Loss')
# ax2.legend()

# ... (set labels, ticks, legend, etc. for ax2)

# Adjust spacing between subplots
# plt.tight_layout()

# Show the plot
# plt.show()


# def pe_evaluate(data_loader: DataLoader):

#     model.eval()
#     with torch.no_grad():
#         for episode_index, (
#             support_images,
#             support_labels,
#             query_images,
#             query_labels,
#             class_ids,
#         ) in tqdm(enumerate(data_loader), total=len(data_loader)):

#             predicted_labels =evaluate_on_one_task(support_images, support_labels, query_images, query_labels)
#             actual_labels  = query_labels

#             actual_labels_np = actual_labels.cpu().numpy()
#             predicted_labels_np = predicted_labels.cpu().numpy()

  
#             precision = precision_score(actual_labels_np, predicted_labels_np, average='macro')
#             recall = recall_score(actual_labels, predicted_labels, average='macro')
#             f1_score_macro = f1_score(actual_labels, predicted_labels, average='macro')
            
#             # Calculate accuracy
#             accuracy = accuracy_score(actual_labels, predicted_labels)

      
#             print("Precision (Macro):", precision)
#             print("Recall (Macro):", recall)
#             print("F1 Score (Macro):", f1_score_macro)
#             print("Accuracy:", accuracy)



# %%

   
       

             










# %%



