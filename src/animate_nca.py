import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from affichage import *


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

