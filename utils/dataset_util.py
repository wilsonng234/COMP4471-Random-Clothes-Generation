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
    dataset = datasets.ImageFolder('datasets/images_without_bg', transform=transform)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader

def remove_background():
    images = listdir('datasets/images')
    if not os.path.exists('datasets/images_without_bg/images'):
        os.makedirs('datasets/images_without_bg/images')
    
    for image in images:
        img_path = os.path.join('datasets/images', image)
        output_path = os.path.join('datasets/images_without_bg/images', image)
        
        if os.path.exists(output_path):
            continue
        
        input = Image.open(img_path)
        output = rembg.remove(input)
        output = output.convert('RGB')
        output.save(output_path)

def combine_edges(images_dir, combined_dir, img_channel=3, img_size=256):
    def merge(edges, original_img):
        blank_space = 12
        combined_img = np.zeros((edges.shape[0], edges.shape[1]*2 + blank_space, img_channel))
        combined_img[:, edges.shape[1]:edges.shape[1]+blank_space, :] = 255

        combined_img[:, 0:edges.shape[1], :] = edges
        combined_img[:, edges.shape[1]+blank_space : (edges.shape[1]*2)+blank_space, :] = original_img

        return combined_img

    images_names = listdir(images_dir)
    if not os.path.exists(combined_dir):
        os.makedirs(combined_dir)

    cnt =0
    for img_name in images_names:
        img_path = os.path.join(images_dir, img_name)
        output_path2 = os.path.join(combined_dir, str(cnt) + "_combined.jpg")

        input = Image.open(img_path)
        new_image =input.resize((img_size,img_size))

        img = np.array(new_image)
        img_diff = np.ones(img.shape) * 255 - img
        img_diff = np.uint8(img_diff)

        new_im = img_diff.copy()
        # threshold = 30
        # new_im[new_im < threshold] = 0
        # new_im[new_im >= threshold] = 255
        
        edges = cv2.Canny(image=new_im, threshold1=90, threshold2=230)  # Canny Edge Detection
        # edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((1, 1),np.uint8))
        edges = np.ones(edges.shape) * 255 - edges
        edges = np.uint8(edges)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        rgb = img[...,::-1].copy()


        combined = merge(edges, rgb)
        cv2.imwrite(output_path2, combined)
        cnt = cnt +1


def train_valid_test_split(images_dir, train=0.8, valid=0.1, test=0.1):
    images_names = os.listdir(images_dir)
    num_images = len(images_names)

    num_train = num_images*train
    num_valid = num_images*valid
    num_test = num_images-num_train-num_valid

    train_dir = os.path.join(images_dir, "..", "train")
    valid_dir = os.path.join(images_dir, "..", "valid")
    test_dir = os.path.join(images_dir, "..", "test")

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(valid_dir):
        os.makedirs(valid_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    i = 0
    while i < num_train:
        img_name = images_names[i]
        src = os.path.join(images_dir, img_name)
        dst = os.path.join(train_dir, img_name)
        shutil.copy2(src, dst)
        i += 1

    while i < num_train+num_valid:
        img_name = images_names[i]
        src = os.path.join(images_dir, img_name)
        dst = os.path.join(valid_dir, img_name)
        shutil.copy2(src, dst)
        i += 1

    while i < num_train+num_valid+num_test:
        img_name = images_names[i]
        src = os.path.join(images_dir, img_name)
        dst = os.path.join(test_dir, img_name)
        shutil.copy2(src, dst)
        i += 1

# def split_data_folders():  # it is not used in pix2pix
#     def read_csv():
#         img = pd.read_csv('datasets/images.csv')
#         return img
        
#     def mkdir(path):
#         path1 = os.path.join('datasets/classes', path)
#         folder1 = os.path.exists(path1)
#         if not folder1:
#             os.makedirs(path1)

#     def folders_create(label):
#         for i in set(label):
#             mkdir(str(i))

#     def copy_files(id, label):
#         assert os.path.exists('datasets/images_without_bg')
#         assert os.path.exists('datasets/classes')

#         for i in range(len(id)):
#             name = str(id[i])
#             folder = str(label[i])
#             file_name = name+'.jpg'

#             img_source = os.path.join('datasets/images_without_bg', file_name)
#             img_target = os.path.join('datasets/classes', folder, file_name)
            
#             try:
#                 if not os.path.exists(img_target):
#                     shutil.copy(img_source, img_target)
#             except FileNotFoundError:   # some image files are corrupted and be deleted
#                 pass

#     img = read_csv()
#     id = img['image']
#     label = img['label']
    
#     folders_create(label)
#     copy_files(id, label)
