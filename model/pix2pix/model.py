import os
import sys 
from . import config
cwd = os.getcwd()
sys.path.append(cwd)
sys.path.append(os.path.join(cwd, "..", "..", "utils"))

import torch.nn as nn
from .generator import Generator
from .discriminator import Discriminator
from .dataset import ClothingDataset

from utils.optimizer_util import get_adam_optimizer
from utils.initalizer_util import he_initialization
from utils.model_utils import load_model
from utils.train_utils import train
from torch.utils import tensorboard

class Pix2Pix():
    def __init__(self):
        self.train_loader = ClothingDataset(config.IMG_SIZE, config.BLANK_SPACE, config.TRAIN_DIR).get_dataloader(config.BATCH_SIZE, shuffle=True)
        self.val_loader = ClothingDataset(config.IMG_SIZE, config.BLANK_SPACE, config.VAL_DIR).get_dataloader(config.BATCH_SIZE, shuffle=True)
        self.test_loader = ClothingDataset(config.IMG_SIZE, config.BLANK_SPACE, config.TEST_DIR).get_dataloader(config.BATCH_SIZE, shuffle=True)

        self.generator = Generator(config.IMG_CHANNELS).to(config.DEVICE) 
        self.discriminator = Discriminator(in_channels=3).to(config.DEVICE)

        if config.LOAD_MODEL:
            load_model(self.generator, config.MODEL_PATH, "generator")
            load_model(self.discriminator, config.MODEL_PATH, "discriminator")
        else:
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
        
        summary_writer = tensorboard.SummaryWriter(log_dir=config.TENSORBOARD_DIR)

        train(D, G, self.train_loader, self.val_loader, D_solver, G_solver, bce, l1, config.DEVICE, config.MODEL_PATH, config.EVALUATION_DIR, 
                cur_epoch=config.CURRENT_EPOCH, num_epochs=num_epochs, summary_writer=summary_writer)
