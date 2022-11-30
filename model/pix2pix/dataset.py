from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import os
import numpy as np
from PIL import Image

class ClothingDataset(Dataset): 
    def __init__(self, img_size, blank_space, images_dir):
        self.img_size = img_size
        self.blank_space = blank_space
        self.images_dir = images_dir
        self.images_name = os.listdir(images_dir)

    def __len__(self):
        return len(self.images_name)

    def __getitem__(self, index):
        img_name = self.images_name[index]
        img_path = os.path.join(self.images_dir, img_name)
        img = Image.open(img_path)
        img = np.array(img)
        img = np.float32(img)
        
        edge_img = img[:, :self.img_size, :]
        original_img = img[:, (self.img_size+self.blank_space):, :]

        edge_img = edge_img.transpose(2, 0, 1)
        original_img = original_img.transpose(2, 0, 1)
        
        edge_img = (edge_img/125.0) -1
        original_img = (original_img/125.0) -1

        return edge_img, original_img

    def get_dataloader(self, batch_size, shuffle=True):
        loader = DataLoader(
            self,
            batch_size=batch_size,
            shuffle=True
        )

        return loader
        