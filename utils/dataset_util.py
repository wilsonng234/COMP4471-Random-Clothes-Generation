import torch
from torchvision import datasets, transforms

def get_dataloader(batch_size, img_size):
    transform = transforms.Compose([transforms.Resize((img_size,img_size)), transforms.ToTensor()])
    dataset = datasets.ImageFolder('datasets', transform=transform)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader
