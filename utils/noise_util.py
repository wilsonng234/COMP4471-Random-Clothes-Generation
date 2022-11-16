import torch

def sample_noise(shape):
    """
    Generate a PyTorch Tensor of uniform random noise.

    Input:
    - shape: Integer giving the batch size of noise to generate.
    
    Output:
    - A PyTorch Tensor of shape (shape) containing gaussian 
      random noise in the range (-1, 1).
    """
    gaussianTensor = torch.randn(shape)

    return gaussianTensor.mul_(2).add_(-1)   # mul_ and add_ are inplace operations
