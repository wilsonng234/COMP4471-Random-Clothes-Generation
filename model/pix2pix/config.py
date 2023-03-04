import torch
from os import walk
import re

def get_current_epoch(LOAD_MODEL, MODEL_PATH):
    if (not LOAD_MODEL):
        return 0
    
    current_epoch = 0
    for (_, _, filenames) in walk(MODEL_PATH):
        for filename in filenames:
            try:
                epoch = int(re.findall(r'\d+', filename)[0])
                current_epoch = max(current_epoch, epoch)
            except:
                if (filename != ".gitignore"):
                    print(f'invalid file name in {MODEL_PATH}')
            
        break

    return current_epoch + 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1
IMG_CHANNELS = 3
IMG_SIZE = 256
BLANK_SPACE = 12
L1_LAMBDA = 100

TRAIN_DIR = "datasets/combined_images/train"
VAL_DIR = "datasets/combined_images/valid"
TEST_DIR = "datasets/combined_images/test"
EVALUATION_DIR = "datasets/evaluation"
TENSORBOARD_DIR = "tensorboard"

# set to True if want to load model at checkpoint, CURRENT_EPOCH - 1
LOAD_MODEL = False
MODEL_PATH = "checkpoints"
# set CURRENT_EPOCH to multiple of five for checkpoints and tensorboard
CURRENT_EPOCH = get_current_epoch(LOAD_MODEL, MODEL_PATH)

PIN_MEMORY = True
NUM_WORKERS = 8
