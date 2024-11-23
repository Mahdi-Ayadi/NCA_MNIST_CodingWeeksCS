"""
    Ce fichier est le fichier main. Il contient le code qui permet de lancer l'application.
    Il contient aussi des snippets de code provenant de différents fichiers du projet.
    Pour lancer l'application, il suffit d'exécuter le code.
"""

from src.animation.animation_pygame import main
from src.model import *
from src.train import *

retrain = False
if retrain:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    # Define the model
    model = NCA(width=28,
                height=28,
                n_channels=20,
                n_filters=64,
                n_dense=128 * 4,
                tmin=50,
                tmax=75).to(DEVICE) # Change the values of parameters if needed
    train(model=model)
    
main()
