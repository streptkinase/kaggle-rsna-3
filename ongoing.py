#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().system('sudo mount -o discard,defaults /dev/sdb /mnt/disks/mount_point')


# In[2]:


#get_ipython().system('sudo chmod a+w /mnt/disks/mount_point/')


# In[3]:


import os
import sys

sys.path.append('/home/streptkinase/.local/lib/python3.5/site-packages')


# In[4]:


dir_csv = '/mnt/disks/mount_point'
dir_train_img = '/mnt/disks/mount_point/stage_1_train_images_jpg'
dir_test_img = '/mnt/disks/mount_point/stage_1_test_images_jpg'

n_classes = 6
n_epochs = 1
batch_size = 11


import os
import cv2
import glob
#import pydicom
import numpy as np
import pandas as pd
from efficientnet_pytorch import EfficientNet
import torch
import torch.optim as optim
from albumentations import Compose, ShiftScaleRotate, Resize, CenterCrop, ToFloat, PadIfNeeded
from albumentations.pytorch import ToTensor
from torch.utils.data import Dataset
from tqdm import tqdm_notebook as tqdm
#from torch_lr_finder import LRFinder
from sklearn.model_selection import train_test_split


# In[5]:


class IntracranialDataset(Dataset):

    def __init__(self, csv_file, path, labels, transform=None):

        self.path = path
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.labels = labels

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.path, self.data.loc[idx, 'Image'] + '.jpg')
        img = cv2.imread(img_name)

        if self.transform:

            augmented = self.transform(image=img)
            img = augmented['image']

        if self.labels:

            labels = torch.tensor(
                self.data.loc[idx, ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']])
            #return {'image': img, 'labels': labels}
            return (img, labels)

        else:

            #return {'image': img}
            return img


# In[6]:


transform_train = Compose([
    PadIfNeeded(min_height=456, min_width=456, always_apply=True),
    CenterCrop(456, 456, always_apply=True),
    ToFloat(max_value=127, always_apply=True),
    ShiftScaleRotate(),
    ToTensor()
])

transform_valid = Compose([
    Resize(456, 456),
    ToFloat(max_value=127, always_apply=True),
    ToTensor()
])

transform_test= Compose([
    Resize(456, 456),
    ToFloat(max_value=127, always_apply=True),
    ToTensor()
])

train_dataset = IntracranialDataset(
    csv_file='/mnt/disks/mount_point/train.csv', path='/mnt/disks/mount_point/stage_1_train_images_jpg', transform=transform_train, labels=True)

valid_dataset = IntracranialDataset(
    csv_file='/mnt/disks/mount_point/valid.csv', path='/mnt/disks/mount_point/stage_1_train_images_jpg', transform=transform_valid, labels=True)

test_dataset = IntracranialDataset(
    csv_file='/mnt/disks/mount_point/test.csv', path='/mnt/disks/mount_point/stage_1_test_images_jpg', transform=transform_test, labels=False)

data_loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
data_loader_valid = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
data_loader_test = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


# In[7]:


device = torch.device("cuda:0")
model = EfficientNet.from_pretrained('efficientnet-b5') 


# In[8]:


model._fc = torch.nn.Linear(2048, n_classes)


# In[9]:


model.to(device)
criterion = torch.nn.BCEWithLogitsLoss()
plist = [{'params': model.parameters(), 'lr': 8e-2}]
optimizer = optim.Adam(plist, lr=8e-2)


# In[10]:


import time
import os
#i = 0
#train_csv_file = pd.read_csv(f'/mnt/disks/mount_point/train_splitted/train{i}.csv')


# In[12]:


def training_model(i):
    
    train_dataset = IntracranialDataset(
        csv_file=f'/mnt/disks/mount_point/train_splitted/train{i}.csv', path='/mnt/disks/mount_point/stage_1_train_images_jpg', transform=transform_train, labels=True)
    print(f'/mnt/disks/mount_point/train_splitted/train{i}.csv')
    data_loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    since = time.time()
    
    if i > 0:
        model.load_state_dict(torch.load(f'/mnt/disks/mount_point/train{i-1}.pth'))
        print(f'/mnt/disks/mount_point/train{i-1}.pth')
    
    for epoch in range(n_epochs):
        print('Epoch {}/{}'.format(epoch, n_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            if phase == 'train':
                dataloader = data_loader_train
            else:
                dataloader = data_loader_valid
            for inputs, labels in dataloader:
                inputs = inputs.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.float)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    #_, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                #running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                dataset_size = len(train_dataset)
            else:
                dataset_size = len(valid_dataset)
            epoch_loss = running_loss / dataset_size
            #epoch_acc = running_corrects.double() / dataset_size

            #print('{} Loss: {:.4f} Acc: {:.4f}'.format(
             #   phase, epoch_loss, epoch_acc))
            print(f'Loss: {epoch_loss}')

            # deep copy the model
            #if phase == 'val' :
             #   val_acc = epoch_acc
              #  print(f'val_acc: {val_acc}')
                

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    torch.save(model.state_dict(), f'/mnt/disks/mount_point/train{i}.pth')
    return
    


# In[13]:


for i in range(0, 27):
    training_model(i)


# In[ ]:




