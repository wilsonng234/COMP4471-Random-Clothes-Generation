import time
import datasets
import os
import numpy as np
from model.cgan.cgan import CGan
from losses.squared_loss import discriminator_loss, generator_loss
from options.train_options import BaseOptions 
from torch.utils.tensorboard import SummaryWriter


def train():
    opt = BaseOptions().parse()
    print("options loaded successfully")
    model = CGan()
    
    

if __name__ == '__main__':
    train()