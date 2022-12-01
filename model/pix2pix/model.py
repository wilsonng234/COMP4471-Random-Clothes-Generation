import os
import sys 
import random
from . import config
from tqdm import tqdm
cwd = os.getcwd()
sys.path.append(cwd)
sys.path.append(os.path.join(cwd, "..", "..", "utils"))

import torch
import torch.nn as nn
from torchvision import transforms
from .generator import Generator
from .discriminator import Discriminator
from .dataset import ClothingDataset

from utils.optimizer_util import get_adam_optimizer
from utils.initalizer_util import he_initialization

class Pix2Pix():
    def __init__(self):
        self.train_loader = ClothingDataset(config.IMG_SIZE, config.BLANK_SPACE, config.TRAIN_DIR).get_dataloader(config.BATCH_SIZE, shuffle=True)
        self.val_loader = ClothingDataset(config.IMG_SIZE, config.BLANK_SPACE, config.VAL_DIR).get_dataloader(config.BATCH_SIZE, shuffle=False)
        self.test_loader = ClothingDataset(config.IMG_SIZE, config.BLANK_SPACE, config.TEST_DIR).get_dataloader(config.BATCH_SIZE, shuffle=False)

        self.generator = None
        self.discriminator = None

        if config.LOAD_MODEL:
            pass
        else:
            self.generator = Generator(config.IMG_CHANNELS).to(config.DEVICE) 
            self.discriminator = Discriminator(in_channels =3).to(config.DEVICE)
            he_initialization(self.generator)
            he_initialization(self.discriminator)

        self.img_channels = config.IMG_CHANNELS
        self.img_size = config.IMG_SIZE

    def G(self, x):
        return self.generator(x)

    def D(self, x, y):
        return self.discriminator(x, y)
    
    def train(self, num_epochs=100):
        D = self.discriminator
        G = self.generator

        D_solver = get_adam_optimizer(self.discriminator)
        G_solver = get_adam_optimizer(self.generator)

        bce = nn.BCEWithLogitsLoss()
        l1 = nn.L1Loss()
        
        for epoch in range(num_epochs):
            for edge_images, original_images in tqdm(self.train_loader):
                edge_images = edge_images.to(config.DEVICE)
                original_images = original_images.to(config.DEVICE)

                fake_images = G(edge_images)
                fake_logits = D(edge_images, fake_images.detach())
                real_logits = D(edge_images, original_images)

                fake_loss = bce(fake_logits, torch.zeros(fake_logits.shape))
                real_loss = bce(real_logits, torch.ones(real_logits.shape))
                discriminator_loss = fake_loss + real_loss

                discriminator_loss.backward()
                D_solver.step()
                D_solver.zero_grad()

                fake_logits = D(edge_images, fake_images)
                fake_loss = bce(fake_logits, torch.ones(fake_logits.shape))
                l1_loss = 100*l1(fake_images, original_images)
                generator_loss = fake_loss + l1_loss

                generator_loss.backward()
                G_solver.step()
                G_solver.zero_grad()

        evaluation_dir = config.EVALUATION_DIR
        if not os.path.exists(evaluation_dir):
            os.makedirs(evaluation_dir)

        # validation evaluation output
        output_dir = os.path.join(evaluation_dir, "valid")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        idx = random.randint(0, self.val_loader.batch_size-1)

        edges, images = self.val_loader.__iter__().__next__()
        edges = (edges[idx] + 0.5).to(config.DEVICE)
        images = (images[idx] + 0.5).to(config.DEVICE)
        fake_images = G(edges).to(config.DEVICE)

        edge = transforms.ToPILImage()(edges[idx])
        image = transforms.ToPILImage()(images[idx])
        fake_image = transforms.ToPILImage()(fake_images[idx])
        edge.save(os.path.join(output_dir, f"edge_{epoch}"))
        image.save(os.path.join(output_dir, f"image{epoch}"))
        fake_image.save(os.path.join(output_dir, f"fake_image{epoch}"))

        # test evaluation output
        output_dir = os.path.join(evaluation_dir, "test")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        idx = random.randint(0, self.val_loader.batch_size-1)

        edges, images = self.val_loader.__iter__().__next__()
        edges = (edges[idx] + 0.5).to(config.DEVICE)
        images = (images[idx] + 0.5).to(config.DEVICE)
        fake_images = G(edges).to(config.DEVICE)

        edge = transforms.ToPILImage()(edges[idx])
        image = transforms.ToPILImage()(images[idx])
        fake_image = transforms.ToPILImage()(fake_images[idx])
        edge.save(os.path.join(output_dir, f"edge_{epoch}"))
        image.save(os.path.join(output_dir, f"image{epoch}"))
        fake_image.save(os.path.join(output_dir, f"fake_image{epoch}"))
