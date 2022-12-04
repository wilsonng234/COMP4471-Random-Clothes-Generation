import torch
import torch.nn as nn
from torchvision import transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
IMG_CHANNELS = 3
IMG_SIZE = 256
BLANK_SPACE = 12
L1_LAMBDA = 100
CURRENT_EPOCH = 0
NUM_WORKERS = 8

TRAIN_DIR = "datasets/combined_images/train"
VAL_DIR = "datasets/combined_images/valid"
TEST_DIR = "datasets/combined_images/test"
EVALUATION_DIR = "datasets/evaluation"
TENSORBOARD_DIR = "tensorboard"

LOAD_MODEL = False
MODEL_PATH = "checkpoints"
