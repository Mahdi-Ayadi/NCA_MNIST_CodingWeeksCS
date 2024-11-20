import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from affichage import *

    
    
#####################################EXPLICATION#########################################################
#Là, je code une classe MockNCA parce que pour voir si le code de la fonction animate_nca de  
#l'animation matplotlib fonctionne bien, il me faut la foction update (cf ligne 75) qu'on n'a   
#pas encore codée, du coup là elle est faite avec un random, le but étant juste de tester animate_nca
#########################################################################################################


class MockNCA:
    def __init__(self, channels=3):
        self.channels = channels  # Nombre de canaux dans la grille (ex: RGB)

    def update(self, grid):
        """
        Simule une mise à jour de la grille en ajoutant un bruit aléatoire.
        """
        noise = np.random.uniform(-0.1, 0.1, grid.shape)  # Génère du bruit aléatoire
        grid = np.clip(grid + noise, 0, 1)  # Ajoute le bruit et garde les valeurs entre 0 et 1
        return grid

    def convert_to_rgb(self, grid):
        """
        Simule une conversion en RGB : utilise les 3 premiers canaux pour créer une image colorée.
        """
        return np.clip(grid[..., :3], 0, 1)  # Retourne les 3 premiers canaux pour RGB


# Crée une grille initiale de taille 28x28 avec 3 canaux (par exemple pour RGB)
initial_grid = np.random.uniform(0, 1, (28, 28, 3))


#####################################EXPLICATION#########################################################
#La fonction ci-dessous "animate_nca" est celle qui nous intéresse
#########################################################################################################



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

