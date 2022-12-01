import torch.nn as nn
import os

def mkdirs(paths):
    # path builder
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    # make new directions if not exits
    if not os.path.exists(path):
        os.makedirs(path)

def he_initialization(model):
    def init_weights(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    model.apply(init_weights)
    