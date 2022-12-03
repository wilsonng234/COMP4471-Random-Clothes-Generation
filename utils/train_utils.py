import os
from tqdm import tqdm
import torch
from torchvision import transforms
import random

from .model_utils import save_model, write_history
from .metric_utils import get_D_loss_batch, get_G_loss_batch, get_losses_dataset
from .dataset_util import denormalization

def save_image(dataloader, G, output_dir, epoch, device):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    edges, images = dataloader.__iter__().__next__()
    idx = random.randint(0, images.shape[0]-1)
    
    edges = edges.to(device)
    images = images.to(device)

    fake_images = denormalization(G(edges)).to(device)
    edges = denormalization(edges)
    images = denormalization(edges)
    
    to_pil = transforms.ToPILImage()
    fake_image = to_pil(fake_images[idx])
    edge = to_pil(edges[idx])
    image = to_pil(images[idx])
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    edge.save(os.path.join(output_dir, f"edge_{epoch}.jpg"))
    image.save(os.path.join(output_dir, f"image{epoch}.jpg"))
    fake_image.save(os.path.join(output_dir, f"fake_image{epoch}.jpg"))

def train_one_epoch(D, G, train_dataloader, valid_dataloader, D_solver, G_solver, bce, l1, device, evaluation_dir, epoch):
    for edge_images, original_images in tqdm(train_dataloader):
        edge_images = edge_images.to(device)
        original_images = original_images.to(device)
        fake_images = G(edge_images)
        
        discriminator_loss = get_D_loss_batch(D, G, edge_images, original_images, fake_images, bce, device)
        if not torch.any(torch.isnan(discriminator_loss)):
            discriminator_loss.backward()
            D_solver.step()
            D_solver.zero_grad()

        generator_loss = get_G_loss_batch(D, G, edge_images, original_images, fake_images, bce, l1, device)
        if not torch.any(torch.isnan(generator_loss)):
            generator_loss.backward()
            G_solver.step()
            G_solver.zero_grad()

    with torch.no_grad():
        discriminator_train_loss, generator_train_loss = get_losses_dataset(D, G, train_dataloader, bce, l1, device)
        discriminator_valid_loss, generator_valid_loss = get_losses_dataset(D, G, valid_dataloader, bce, l1, device)

        save_image(train_dataloader, G, os.path.join(evaluation_dir, "train"), epoch, device)
        save_image(valid_dataloader, G, os.path.join(evaluation_dir, "valid"), epoch, device)

    return discriminator_train_loss, generator_train_loss, discriminator_valid_loss, generator_valid_loss

def train(D, G, train_dataloader, valid_dataloader, D_solver, G_solver, bce, l1, device, model_path, evaluation_dir, cur_epoch=0, num_epochs=100, summary_writer=None):
    D.train()
    G.train()

    discriminator_train_loss_history = []
    generator_train_loss_history = [] 
    discriminator_valid_loss_history = [] 
    generator_valid_loss_history = []
    
    for epoch in range(cur_epoch, cur_epoch+num_epochs):
        discriminator_train_loss, generator_train_loss, discriminator_valid_loss, generator_valid_loss = \
            train_one_epoch(D, G, train_dataloader, valid_dataloader, D_solver, G_solver, bce, l1, device, evaluation_dir, epoch)
        
        discriminator_train_loss_history.append(discriminator_train_loss)
        generator_train_loss_history.append(generator_train_loss)
        discriminator_valid_loss_history.append(discriminator_valid_loss)
        generator_valid_loss_history.append(generator_valid_loss)

        if epoch%5 == 4:
            save_model(D, model_path, f"discriminator_{epoch}")
            save_model(G, model_path, f"generator_{epoch}")
            
            write_history(summary_writer, "Discriminator Loss/train", discriminator_train_loss_history, epoch-4)
            write_history(summary_writer, "Generator Loss/train", generator_train_loss_history, epoch-4)
            write_history(summary_writer, "Discriminator Loss/valid", discriminator_valid_loss_history, epoch-4)
            write_history(summary_writer, "Generator Loss/valid", generator_valid_loss_history, epoch-4)
            
            discriminator_train_loss_history.clear()
            generator_train_loss_history.clear()
            discriminator_valid_loss_history.clear()
            generator_valid_loss_history.clear()

    D.eval()
    G.eval()
