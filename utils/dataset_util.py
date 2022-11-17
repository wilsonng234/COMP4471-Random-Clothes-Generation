import torch
from torchvision import datasets, transforms

import rembg
from PIL import Image
import pandas as pd
import os
from os import listdir
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

def remove_background():
    images = listdir('datasets/images')
    if not os.path.exists('datasets/images_without_bg'):
        os.makedirs('datasets/images_without_bg')
    
    for image in images:
        img_path = os.path.join('datasets/images', image)
        output_path = os.path.join('datasets/images_without_bg', image)
        
        if os.path.exists(output_path):
            continue
        
        input = Image.open(img_path)
        output = rembg.remove(input)
        output = output.convert('RGB')
        output.save(output_path)

def split_data_folders():  
    def read_csv():
        img = pd.read_csv('datasets/images.csv')
        return img
        
    def mkdir(path):
        path1 = os.path.join('datasets/classes', path)
        folder1 = os.path.exists(path1)
        if not folder1:
            os.makedirs(path1)

    def folders_create(label):
        for i in set(label):
            mkdir(str(i))

    def copy_files(id, label):
        assert os.path.exists('datasets/images_without_bg')
        assert os.path.exists('datasets/classes')

        for i in range(len(id)):
            name = str(id[i])
            folder = str(label[i])
            file_name = name+'.jpg'

            img_source = os.path.join('datasets/images_without_bg', file_name)
            img_target = os.path.join('datasets/classes', folder, file_name)
            
            try:
                if not os.path.exists(img_target):
                    shutil.copy(img_source, img_target)
            except FileNotFoundError:   # some image files are corrupted and be deleted
                pass

    img = read_csv()
    id = img['image']
    label = img['label']
    
    folders_create(label)
    copy_files(id, label)
