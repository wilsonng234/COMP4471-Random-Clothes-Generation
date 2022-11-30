import config
from generator import Generator
from discriminator import Discriminator
from dataset import ClothingDataset
from utils.train_util import train
from utils.losses import discriminator_loss, generator_loss

import torch


def get_optimizer(model,learning_rate,beta1,beta2):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=[beta1, beta2])
    return optimizer

class Pix2Pix():
    def __init__(self):
        self.train_loader = ClothingDataset(config.IMG_SIZE, config.BLANK_SPACE, config.TRAIN_DIR).get_dataloader(config.BATCH_SIZE)
        self.val_loader = ClothingDataset(config.IMG_SIZE, config.BLANK_SPACE, config.VAL_DIR).get_dataloader(config.BATCH_SIZE)
        self.test_loader = ClothingDataset(config.IMG_SIZE, config.BLANK_SPACE, config.TEST_DIR).get_dataloader(config.BATCH_SIZE)

        self.generator = Generator(config.IMG_CHANNELS).to(config.DEVICE) 
        self.discriminator = Discriminator(in_channels =3)
        self.img_channels = config.IMG_CHANNELS
        self.img_size = config.IMG_SIZE
        
        # TODO: initalizer
        # hehe_initialization(self.generator)
        # hehe_initialization(self.discriminator)

    def G(self, x):
        return self.generator(x)

    def D(self, x, y):
        return self.discriminator(x, y)
        pass
    
    def train(self, num_epochs=100):
        D = self.discriminator
        G = self.generator
        #  init the optimizer
        D_solver = get_optimizer(D)
        G_solver = get_optimizer(G)
        loader_train = self.train_loader
        
        train(loader_train, D, G, D_solver, G_solver, discriminator_loss, generator_loss, show_every=250, 
          batch_size=128, noise_size=96, num_epochs=10)
        
        
        