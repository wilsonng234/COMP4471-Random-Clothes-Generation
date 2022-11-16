import torch
import torch.nn as nn
from generator import generator
from discriminator import discriminator
from utils.noise_util import sample_noise
from utils.optimizer_util import get_adam_optimizer
from losses.squared_loss import discriminator_loss
from losses.squared_loss import generator_loss

class CGan():
    def __init__(self, noise_dim, num_classes, img_channels, img_size, embed_size):
        super(CGan, self).__init__()

        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.img_size = img_size
        self.embed_size = embed_size

        self.generator_embedding = nn.Embedding(num_classes, embed_size)
        self.discriminator_embedding = nn.Embedding(num_classes, img_size*img_size)

        self.generator = generator(noise_dim, noise_dim+embed_size)
        self.discriminator = discriminator(img_channels+1)

    def train(self):
        G_solver = get_adam_optimizer(self.generator)
        D_solver = get_adam_optimizer(self.discriminator)

        pass

    def G(self, labels):
        assert(labels < self.num_classes)
        
        batch_size = labels.shape[0]
        noise = sample_noise(shape=(batch_size, self.noise_dim)).view(batch_size, self.noise_dim, 1, 1)

        embedding = self.embedding(labels).view(batch_size, self.embed_size, 1, 1)
        embedded_noise = torch.cat([noise, embedding], dim=1)

        return self.generator(embedded_noise)

    def D(self, input_images, labels):
        assert(labels < self.num_classes)

        embedding = self.discriminator_embedding(labels).view(labels.shape[0], 1, self.img_size, self.img_size)
        embedded_images = torch.cat([input_images, embedding], dim=1)

        return self.discriminator(embedded_images)
