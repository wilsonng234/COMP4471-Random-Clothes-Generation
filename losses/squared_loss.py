import torch

def squared_loss(input, label):
    """
    Inputs: 
    - input: PyTorch Variable of shape (N,) giving scores
    - label: Scalar value
    """

    loss = ((input - label)**2) / 2
    return loss.mean()

def discriminator_loss(scores_real, scores_fake):
    """
    Compute the Least-Squares GAN loss for the discriminator.
    
    Inputs:
    - scores_real: PyTorch Variable of shape (N,) giving scores for the real data.
    - scores_fake: PyTorch Variable of shape (N,) giving scores for the fake data.
    
    Outputs:
    - loss: A PyTorch Variable containing the loss.
    """
    loss = squared_loss(scores_real, 1) + squared_loss(scores_fake, 0)
    return loss

def generator_loss(scores_fake):
    """
    Computes the Least-Squares GAN loss for the generator.
    
    Inputs:
    - scores_fake: PyTorch Variable of shape (N,) giving scores for the fake data.
    
    Outputs:
    - loss: A PyTorch Variable containing the loss.
    """
    loss = squared_loss(scores_fake, 1)
    return loss
