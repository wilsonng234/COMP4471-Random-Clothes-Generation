import torch.nn as nn

def he_initialization(model):
    def init_weights(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    model.apply(init_weights)
    