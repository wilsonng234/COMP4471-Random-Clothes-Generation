import torch
from torchvision import datasets, transforms

import pandas as pd
import os
import shutil

def get_dataloader(batch_size, img_size):
    transform = transforms.Compose([
        transforms.Resize((img_size,img_size)), 
        transforms.ToTensor(),
        transforms.Lambda(lambda img: img*2-1),
    ])
    dataset = datasets.ImageFolder('datasets', transform=transform)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader

def split_data_folders():  
    def folders_create(id, label):
        for i in set(label):
            mkdir(str(i))

    def read_csv():
        img = pd.read_csv('datasets/images.csv')

        return img

    def mkdir(path):
        path1 = os.path.join('datasets/classes', path)
        folder1 = os.path.exists(path1)
        if not folder1:                   
            os.makedirs(path1)
            
    def copy_files(id, label):
        for i in range(len(id)):
            name = str(id[i])
            folder = str(label[i])
            file_name = name+'.jpg'

            img_source = os.path.join('datasets/images', file_name)
            img_target = os.path.join('datasets/classes', folder, file_name)
            
            try:
                if not os.path.exists(img_target):
                    shutil.copy(img_source,img_target)
            except FileNotFoundError:   # some image files are corrupted and be deleted
                pass

    img = read_csv()
    id = img['image']
    label = img['label']
    
    folders_create(id, label)
    copy_files(id, label)
