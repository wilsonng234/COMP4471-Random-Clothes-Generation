import torch
from os import walk
import re

def get_current_epoch():
    current_epoch = 0
    
    for (_, _, filenames) in walk("checkpoints"):
        for filename in filenames:
            try:
                epoch = int(re.findall(r'\d+', filename)[0])
                current_epoch = max(current_epoch, epoch)
            except:
                print('invalid file name in checkpoints')
            
        break

    return current_epoch + 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
IMG_CHANNELS = 3
IMG_SIZE = 256
BLANK_SPACE = 12
L1_LAMBDA = 100
# set CURRENT_EPOCH to multiple of five for checkpoints and tensorboard
CURRENT_EPOCH = get_current_epoch()
PIN_MEMORY = True
NUM_WORKERS = 8

TRAIN_DIR = "datasets/combined_images/train"
VAL_DIR = "datasets/combined_images/valid"
TEST_DIR = "datasets/combined_images/test"
EVALUATION_DIR = "datasets/evaluation"
TENSORBOARD_DIR = "tensorboard"

# set to True if want to load model at checkpoint, CURRENT_EPOCH - 1
LOAD_MODEL = False
MODEL_PATH = "checkpoints"
