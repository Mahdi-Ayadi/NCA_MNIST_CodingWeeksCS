from animate_nca import *
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
#importing an exemple image numpy array or tensor of shape(width,height)
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.squeeze(0))
    ])
batch_size = 1
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
example = next(iter(test_loader))
input_image = example[0][0]
#animating the image 
color_map = {
    0: [255, 0, 0],        # rouge
    1: [0, 255, 0],        # vert
    2: [0, 0, 0],          # noir
    3: [255, 165, 0],      # orange
    4: [255, 192, 203],    # Rose
    5: [0, 0, 139],        # Bleu
    6: [255, 255, 0],      # jaune
    7: [169, 169, 169],    # Gris
    8: [238, 130, 238],    # violet
    9: [139, 69, 19]       # Marron
}

animate_nca(input_image,color_map)

