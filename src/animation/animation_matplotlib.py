import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from affichage import *
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def animate_nca(initial_grid,color_map, steps=100, interval=50):
    """
    Crée une animation montrant l'évolution de la grille avec le modèle NCA.
    
    :param initial_grid: Grille de départ au format (n, n, channels).
    :param steps: Nombre d'étapes à visualiser.
    :param interval: Temps entre chaque frame (ms).
    """
    nca_display = Affiche_NCA(initial_grid, color_map)

    fig, ax = plt.subplots()
    img = ax.imshow(nca_display.next())

    def update(frame):
        nonlocal nca_display
        # Mise à jour de l'image
        img.set_array(nca_display.next())
        return [img]

    ani = animation.FuncAnimation(
        fig, update, frames=steps, interval=interval, blit=True
    )
    plt.show()

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

