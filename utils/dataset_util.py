import torch
from torchvision import datasets, transforms

import rembg
from PIL import Image
import pandas as pd
import os
from os import listdir
import shutil
import cv2
import numpy as np

def get_dataloader(batch_size, img_size):
    transform = transforms.Compose([
        transforms.Resize((img_size,img_size)), 
        transforms.ToTensor(),
        transforms.Lambda(lambda img: img*2-1),
    ])
    dataset = datasets.ImageFolder('datasets/classes', transform=transform)

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

def combine_edges(dataloader, img_channel=3, img_size=240):
    def merge(edges, original_img):
        blank_space = 12
        combined_img = np.zeros((edges.shape[0], edges.shape[1]*2 + blank_space, img_channel))
        combined_img[:, edges.shape[1]:edges.shape[1]+blank_space, :] = 255

        combined_img[:, 0:edges.shape[1], :] = edges
        combined_img[:, edges.shape[1]+blank_space : (edges.shape[1]*2)+blank_space, :] = original_img

        return combined_img

    output_path = 'datasets/combined_images'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    cnt = 1
    for _, (images, _) in enumerate(dataloader):
        for _, img in enumerate(images):
            img = img.numpy()
            img = img.transpose(1, 2, 0)    # 3, 255, 255 -> 255, 255, 3

            img_diff = np.ones(img.shape) * 255 - img
            img_diff = np.uint8(img_diff)

            threshold = 200
            img_diff[img_diff < threshold] = 0
            img_diff[img_diff >= threshold] = 255
            
            edges = cv2.Canny(image=img_diff, threshold1=90, threshold2=200)  # Canny Edge Detection
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3),np.uint8))

            edges = np.ones(edges.shape) * 255 - edges
            edges = np.uint8(edges)
            edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            
            combined = merge(edges, (img+1)/2*255)
            cv2.imwrite(os.path.join(output_path, str(cnt) + "_combined.jpg"), combined)
            cnt += 1

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
