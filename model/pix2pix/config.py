import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
IMG_CHANNELS = 3
IMG_SIZE = 256
BLANK_SPACE = 12

TRAIN_DIR = None
VAL_DIR = None
TEST_DIR = None

LOAD_MODEL = False
