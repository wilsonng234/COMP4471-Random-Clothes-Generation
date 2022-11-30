import torch
import torch.nn as nn
from torch.autograd import Variable
from model.cgan.generator import generator
from model.cgan.discriminator import discriminator
from utils.noise_util import sample_noise
from utils.optimizer_util import get_adam_optimizer
from losses.squared_loss import discriminator_loss
from losses.squared_loss import generator_loss
from utils.dataset_util import get_dataloader 
from utils.initalizer_util import he_initialization
import matplotlib.pyplot as plt
import numpy as np

class CGan():
    def __init__(self, num_classes, batch_size=64, img_channels=3, img_size=128, noise_dim=10, embed_size=10):
        super(CGan, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.batch_size = batch_size
        self.dataloader = get_dataloader(batch_size=batch_size, img_size=img_size)

        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.img_channels = img_channels
        self.img_size = img_size
        self.embed_size = embed_size

        self.generator_embedding = nn.Embedding(num_classes, embed_size)
        self.discriminator_embedding = nn.Embedding(num_classes, img_size*img_size)
        
        self.generator = generator(noise_dim+embed_size, img_size).to(self.device)
        self.discriminator = discriminator(img_channels+1, img_size).to(self.device)
        he_initialization(self.generator)
        he_initialization(self.discriminator)

    def G(self, labels):
        # assert(labels < self.num_classes)
        
        noise = sample_noise(shape=(self.batch_size, self.noise_dim)).view(self.batch_size, self.noise_dim, 1, 1).to(self.device).long()
        
        embedding = self.generator_embedding(labels).to(self.device)
        embedding = embedding.view(self.batch_size, self.embed_size, 1, 1).to(self.device)
        embedded_noise = torch.cat([noise, embedding], dim=1).to(self.device)

        return self.generator(embedded_noise).view(self.batch_size, self.img_channels, self.img_size, self.img_size)

    def D(self, input_images, labels):
        # assert(labels < self.num_classes)
        
        embedding = self.discriminator_embedding(labels).to(self.device)
        embedding = embedding.view(labels.shape[0], 1, self.img_size, self.img_size).to(self.device)
        embedded_images = torch.cat([input_images, embedding], dim=1).to(self.device)

        return self.discriminator(embedded_images)

    def train(self, num_epochs=10, show_every=100):
        G_solver = get_adam_optimizer(self.generator, lr=5e-7, beta1=0.5)
        D_solver = get_adam_optimizer(self.discriminator, lr=5e-7, beta1=0.5)

        iter_count = 0

        for epoch in range(num_epochs):
            for images, labels in self.dataloader:
                # print(images[0].min(), images[0].max())
                if len(images) != self.batch_size:
                    continue

                D_solver.zero_grad()
                real_data = Variable(images).to(self.device)
                logits_real = self.D(2* (real_data - 0.5), labels).to(self.device)
                
                # noise = Variable(sample_noise(shape=(self.batch_size, self.noise_dim)))
                # detach() separate a tensor from the computational graph by returning a new tensor that doesn’t require a gradient
                fake_images = self.G(labels).to(self.device) # .detach() 
                # print(fake_images.shape)
                logits_fake = self.D(fake_images, labels).to(self.device)

                d_total_error = discriminator_loss(logits_real, logits_fake).to(self.device)
                d_total_error.backward()        
                D_solver.step()

                G_solver.zero_grad()
                # noise = Variable(sample_noise(shape=(self.batch_size, self.noise_dim)))
                fake_images = self.G(labels).to(self.device)

                gen_logits_fake = self.D(fake_images, labels).to(self.device)
                g_error = generator_loss(gen_logits_fake).to(self.device)

                g_error.backward()
                G_solver.step()
                
                # with torch.no_grad():
                if (iter_count % show_every == 0):
                    print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count,d_total_error,g_error))
                    # labels = torch.zeros((self.batch_size, 1)).to(self.device).long()
                    img = self.G(labels).to(self.device)[0]
                    img = img.detach().cpu().numpy()
                    
                    img = img.transpose((1, 2, 0))
                    img = (img + 1)/2
                    
                    print(img.shape)
                    plt.imshow(img)
                    plt.show()
                    
                    print()
                iter_count += 1
                
