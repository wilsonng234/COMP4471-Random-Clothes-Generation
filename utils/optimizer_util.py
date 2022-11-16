import torch.optim as optim
import torch.nn as nn

def get_adam_optimizer(model, lr=1e-3, beta1=0.9, beta2=0.999):
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=[beta1, beta2])

    return optimizer
