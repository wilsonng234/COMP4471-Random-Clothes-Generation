import torch
import torch.nn as nn
from torchvision import transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
IMG_CHANNELS = 3
IMG_SIZE = 256
BLANK_SPACE = 12
CURRENT_EPOCH = 0
NUM_WORKERS = 8

TRAIN_DIR = "datasets/combined_images/train"
VAL_DIR = "datasets/combined_images/valid"
TEST_DIR = "datasets/combined_images/test"
EVALUATION_DIR = "datasets/evaluation"
TENSORBOARD_DIR = "tensorboard"

LOAD_MODEL = False
MODEL_PATH = "checkpoints"

augmentation_transform = transforms.RandomApply(
    nn.ModuleList(
        [   
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomRotation(degrees=(10, 10))
        ]
    ), p=1/3
)
