from animation_matplotlib_with_MOCK_NCA import *
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
#importing an exemple image of shape(width,height)
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
    0: [224, 176, 255],    # Mauve
    1: [255, 0, 255],      # Magenta
    2: [0, 255, 255],      # Cyan
    3: [0, 255, 0],        # Vert
    4: [255, 192, 203],    # Rose
    5: [0, 0, 139],        # Bleu foncé
    6: [144, 238, 144],    # Vert clair
    7: [169, 169, 169],    # Gris
    8: [64, 224, 208],     # Turquoise
    9: [139, 69, 19]       # Marron
}

animate_nca(input_image,color_map)

