from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

import os
from . import config
import random
import numpy as np
from PIL import Image

class ClothingDataset(Dataset): 
    def __init__(self, img_size, blank_space, images_dir, augmentation=True):
        self.img_size = img_size
        self.blank_space = blank_space
        self.images_dir = images_dir
        self.images_name = os.listdir(images_dir)
        self.augmentation = augmentation

    def __len__(self):
        return len(self.images_name)

    def __getitem__(self, index):
        img_name = self.images_name[index]
        img_path = os.path.join(self.images_dir, img_name)
        img = Image.open(img_path)
        img = np.array(img)
        
        edge_img = img[:, :self.img_size, :]
        original_img = img[:, (self.img_size+self.blank_space):, :]

        original_img = transforms.ToTensor()(original_img)
        edge_img = transforms.ToTensor()(edge_img)

        if (edge_img.min() < 0 or edge_img.max() > 1):
            raise ValueError("before augmentation edge_img is not normalized")
        if (original_img.min() < 0 or original_img.max() > 1):
            raise ValueError("before augmentation original_img is not normalized")
        
        if self.augmentation:
            colorJitter = transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0.5)
            horizontalFlip = transforms.RandomHorizontalFlip(p=1)
            
            threshold = 1/2
            if random.random() <= threshold:
                original_img = colorJitter(original_img)

            if random.random() <= threshold:
                original_img = horizontalFlip(original_img)
                edge_img = horizontalFlip(edge_img)

        if (edge_img.min() < 0 or edge_img.max() > 1):
            raise ValueError("after aug edge_img is not normalized")
        if (original_img.min() < 0 or original_img.max() > 1):
            raise ValueError("after aug original_img is not normalized")
        
        normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        edge_img = normalize(edge_img)
        original_img = normalize(original_img)
        
        if (edge_img.min() < -1 or edge_img.max() > 1):
            raise ValueError("invalid range")
        if (original_img.min() < -1 or original_img.max() > 1):
            raise ValueError("invalid range")
        
        return edge_img, original_img

    def get_dataloader(self, batch_size, shuffle=True):
        loader = DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=config.PIN_MEMORY,
            num_workers=config.NUM_WORKERS
        )

        return loader
