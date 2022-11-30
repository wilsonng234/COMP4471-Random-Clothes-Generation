import torch.nn as nn

def discriminator(input_channels, input_size):
    """
    Build and return a PyTorch model implementing the architecture.
    """
    model = nn.Sequential(
        # nn.Unflatten((input_channels, input_size, input_size), input_channels*input_size*input_size),
        # nn.Conv2d(1, 32, 5, 1),
        # nn.LeakyReLU(negative_slope=0.01, inplace=True),
        # nn.MaxPool2d(2, 2),
        # nn.Conv2d(32, 64, 5, 1),
        # nn.LeakyReLU(negative_slope=0.01, inplace=True),
        # nn.MaxPool2d(2, 2),
        # nn.Flatten(),
        # nn.Linear(4*4*64, 4*4*64),
        # nn.LeakyReLU(negative_slope=0.01, inplace=True),
        # nn.Linear(4*4*64, 1)

        nn.Flatten(),
        nn.Linear(input_channels*input_size*input_size, 512),
        nn.LeakyReLU(),
        nn.Dropout(),
        nn.Linear(512, 256),
        nn.LeakyReLU(),
        nn.Dropout(),

        # nn.Conv2d(input_channels, 128, 3, 1, "same"),
        # nn.LeakyReLU(),
        # nn.Conv2d(128, 128, 5, 1, "same"),
        # nn.LeakyReLU(),
        # nn.Conv2d(128, 128, 7, 1, "same"),
        # nn.LeakyReLU(),
        # nn.Conv2d(128, 64, 3, 1, "same"),
        # nn.LeakyReLU(),
        # nn.Conv2d(64, 64, 5, 1, "same"),
        # nn.LeakyReLU(),
        # nn.Conv2d(64, 64, 7, 1, "same"),
        # nn.LeakyReLU(),
        # nn.Flatten(),

        nn.Linear(256, 1),
    )
    return model
