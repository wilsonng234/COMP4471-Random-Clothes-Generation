import torch
import torch.nn as nn
from torchvision import transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
IMG_CHANNELS = 3
IMG_SIZE = 256
BLANK_SPACE = 12

TRAIN_DIR = None
VAL_DIR = None
TEST_DIR = None

LOAD_MODEL = False
TRAIN_MODEL = False

augmentation_transform = transforms.RandomApply(
    nn.ModuleList(
        [
            transforms.ColorJitter(brightness=0, contrast=0, saturation=0.2, hue=0.2),
            transforms.RandomRotation(degrees=(45, 45))
        ]
    ), p=1/3
)
