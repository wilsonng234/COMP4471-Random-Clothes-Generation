import torch.nn as nn

def generator(input_channels, img_size):
    """
    Build and return a PyTorch model implementing the architecture.
    """
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_channels, 1024),
        nn.LeakyReLU(),
        nn.Dropout(),
        nn.Linear(1024, 512),
        nn.LeakyReLU(),
        nn.Dropout(),
        nn.Linear(512, 18*30*30),
        nn.LeakyReLU(),
        nn.Dropout(),
        nn.Unflatten(1, unflattened_size=(18, 30, 30)),
        nn.Conv2d(18, 9, 3, 1, "same"),
        nn.LeakyReLU(),
        nn.Dropout(),
        nn.Conv2d(9, 3, 3, 1, "same"),
        nn.LeakyReLU(),
        nn.Dropout(),
        nn.Conv2d(3, 3, 3, 1, "same"),
        nn.LeakyReLU(),
        nn.Dropout(),
        nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(3),
        nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(3),
        nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=4, stride=2, padding=1),
        nn.Tanh(),
        # nn.Flatten(),



        # nn.Linear(64, 3*img_size*img_size),  # channels*img_size*img_size
        # nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=3, stride=2, padding=1),
    )
    return model
