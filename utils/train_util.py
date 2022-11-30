import torch
from torch.autograd import Variable
from noise_util import sample_noise
from tqdm import tqdm
from torch.utils.tensorboard import Summarywriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(loader_train, D, G, D_solver, G_solver, discriminator_loss, generator_loss, show_every=250, 
          batch_size=128, noise_size=96, num_epochs=10):
    """
    Train a GAN!
    
    Inputs:
    - D, G: PyTorch models for the discriminator and generator
    - D_solver, G_solver: torch.optim Optimizers to use for training the
      discriminator and generator.
    - discriminator_loss, generator_loss: Functions to use for computing the generator and
      discriminator loss, respectively.
    - show_every: Show samples after every show_every iterations.
    - batch_size: Batch size to use for training.
    - noise_size: Dimension of the noise to use as input to the generator.
    - num_epochs: Number of epochs over the training dataset to use for training.
    """
    iter_count = 0
    writer = Summarywriter.writer()
    
    for epoch in range(num_epochs):
        d_total_error = None
        g_error = None
        for img, images in loader_train:
            assert len(images) == batch_size

            D_solver.zero_grad()
            real_data = Variable(images).to(device)
            logits_real = D(2* (real_data - 0.5)).to(device)

            g_fake_seed = Variable(img).to(device)
            # detach() separate a tensor from the computational graph by returning a new tensor that doesn’t require a gradient
            fake_images = G(g_fake_seed).detach() 
            logits_fake = D(fake_images.view(batch_size, 1, 28, 28))

            d_total_error = discriminator_loss(logits_real, logits_fake)
            d_total_error.backward()        
            D_solver.step()

            G_solver.zero_grad()
            g_fake_seed = Variable(img).to(device)
            fake_images = G(g_fake_seed)

            gen_logits_fake = D(fake_images.view(batch_size, 1, 28, 28))
            g_error = generator_loss(gen_logits_fake)
            g_error.backward()
            G_solver.step()

            # if (iter_count % show_every == 0):
            #     print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count,d_total_error,g_error))
            #     imgs_numpy = fake_images.data.cpu().numpy()
            #     show_images(imgs_numpy[0:16])
            #     plt.show()
            #     print()
            iter_count += 1
        
        writer.add_scalar('d_total_error', d_total_error, epoch)
        writer.add_scalar('g_error', g_error, epoch)
        
        if epoch%5 == 0:
          torch.save(D,'./logs/epoch'+str(epoch)+'_D.pth')
          torch.save(G,'./logs/epoch'+str(epoch)+'_G.pth')

    writer.close()
