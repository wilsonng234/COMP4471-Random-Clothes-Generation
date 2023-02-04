# COMP4471-Random-Clothes-Generation
PyTorch reimplementation of [Pix2Pix](https://arxiv.org/abs/1611.07004) and apply for random clothes generation based on edge information

## Dataset
Dataset available: https://www.kaggle.com/datasets/agrigorev/clothing-dataset-full

## Installation 
Make sure conda is installed
```
1. conda install pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.7 -c pytorch -c nvidia
2. pip install -r requirements.txt
```

## Getting started
```
1. Install images from https://www.kaggle.com/datasets/agrigorev/clothing-dataset-full
2. Put images under datasets/images directory
3. Run main.ipynb
```

## Training
Hyperparameters can be tuned in `model/pix2pix/config.py`

## Results
Results can be found under `datasets/evaluation` for each 5 epochs of training <br>
Loss and accuracy history is recorded in `tensorboard` directory
