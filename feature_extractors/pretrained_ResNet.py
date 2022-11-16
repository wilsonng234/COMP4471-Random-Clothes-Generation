import torch.nn as nn
import torchvision.models as models

def get_pretrained_resnet152():
    resnet152 = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V2)

    modules = list(resnet152.children())[:-2]
    resnet152 = nn.Sequential(*modules)

    for param in resnet152.parameters():
        param.requires_grad = False

    return resnet152
