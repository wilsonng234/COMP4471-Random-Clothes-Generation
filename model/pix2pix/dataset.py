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

        if self.augmentation:
            original_img = transforms.ToPILImage()(original_img)
            edge_img = transforms.ToPILImage()(edge_img)
            
            colorJitter = transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0.5)
            horizontalFlip = transforms.RandomHorizontalFlip(p=1)
            
            threshold = 1/2
            if random.random() <= threshold:
                original_img = colorJitter(original_img)

            if random.random() <= threshold:
                original_img = horizontalFlip(original_img)
                edge_img = horizontalFlip(edge_img)

            if random.random() <= threshold:
                shift = [random.randint(-50, 50), random.randint(-50, 50)]
                original_img = transforms.functional.affine(original_img, angle=0, translate=shift, scale=1, shear=0, fill=255)
                edge_img = transforms.functional.affine(edge_img, angle=0, translate=shift, scale=1, shear=0, fill=255)

        both_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        edge_img = both_transform(edge_img)
        original_img = both_transform(original_img)

        return edge_img, original_img

    def get_dataloader(self, batch_size, shuffle=True):
        loader = DataLoader(
            self,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=False,
            num_workers=config.NUM_WORKERS
        )

        return loader
