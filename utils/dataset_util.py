import torch
from torchvision import datasets, transforms
import pandas as pd
import numpy as np
import os
import glob
import shutil
from sys import exit

def get_dataloader(batch_size, img_size):
    transform = transforms.Compose([
        transforms.Resize((img_size,img_size)), 
        transforms.ToTensor(),
        transforms.Lambda(lambda img: img*2-1),
    ])
    dataset = datasets.ImageFolder('datasets', transform=transform)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader


def transform_data():
    img = read_csv()
    id = img['image']
    label = img['label']
    if not os.path.exists('compressed'):
        os.makedirs('compressed')
    folder_create(id,label)
    copy_file(id,label)


def folder_create(id,label):
    
    for i in label:
        mkdir(str(i))
    
def read_csv():
    img = pd.read_csv('images.csv')
    print(img.head(10))
    return img


def mkdir(path):
    path1 = os.path.join('compressed',path)
    path2 = os.path.join('original',path)
    folder1 = os.path.exists(path1)
    folder2 = os.path.exists(path2)
    if not folder1:                   
	    os.makedirs(path1)
    if not folder2:
        os.makedirs(path2)


def copy_file(id,label):
    for i in range(len(id)):
        name = str(id[i])
        folder = str(label[i])
        file_name = name+'.jpg'
        # print(source_file)
        compressed_source = os.path.join('./images_compressed',file_name)
        compressed_target = os.path.join('./compressed',folder,file_name)
        # print(compressed_target)
        # print(compressed_source)
        
        if not os.path.exists(compressed_target):
            shutil.copy(compressed_source,compressed_target)
        if not os.path.exists(original_target):
            shutil.copy(original__source,original_target)