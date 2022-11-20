import config
from generator import Generator
from dataset import ClothingDataset

class Pix2Pix():
    def __init__(self):
        self.train_loader = ClothingDataset(config.IMG_SIZE, config.BLANK_SPACE, config.TRAIN_DIR).get_dataloader(config.BATCH_SIZE)
        self.val_loader = ClothingDataset(config.IMG_SIZE, config.BLANK_SPACE, config.VAL_DIR).get_dataloader(config.BATCH_SIZE)
        self.test_loader = ClothingDataset(config.IMG_SIZE, config.BLANK_SPACE, config.TEST_DIR).get_dataloader(config.BATCH_SIZE)

        self.generator = Generator(config.IMG_CHANNELS).to(config.DEVICE) 
        self.discriminator = None   #.to(device)
        self.img_channels = config.IMG_CHANNELS
        self.img_size = config.IMG_SIZE
        
        # TODO: initalizer
        # he_initialization(self.generator)
        # he_initialization(self.discriminator)

    def G(self, x):
        return self.generator(x)

    def D(self, x):
        pass
    
    def train(self, num_epochs=100):
        pass
    