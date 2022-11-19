import config
from generator import Generator

class Pix2Pix():
    def __init__(self, batch_size=config.BATCH_SIZE, img_channels=config.IMG_CHANNELS, 
                img_size=config.IMG_SIZE, device=config.DEVICE):
        
        self.generator = Generator(img_channels).to(device) 
        self.discriminator = None   #.to(device)
        self.img_channels = img_channels
        self.img_size = img_size
        
        # TODO: initalizer
        # he_initialization(self.generator)
        # he_initialization(self.discriminator)

    def G(self, x):
        return self.generator(x)

    def D(self, x):
        pass
    
    def train(self, num_epochs=100):
        pass