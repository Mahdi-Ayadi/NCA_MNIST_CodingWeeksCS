# MNIST_NCA

## Description

 
#    Sprint 0:
F1: installation des modules nécessaitres en local: Pytorch, Matplotlib, tkinter 

F2: création du repo git et version control

F3: conception globale

#   Sprint 1: Network Training
F4: récupération & encodage du Dataset

F5: Définition du modèle NCA (POO)

F6: initialisation et entrainement de NCA

F7: validation du modèle

F8: sauvegarde des poids du modèle

#    Sprint 2: Visualisation et intégration du modèle 
F9: implémentation avec matplotlib.animation

F10: création de la fonction main (input ==> identification par code de couleurs)

#    Sprint 3: développement de l'UI:
F11: Préparation de l'interface TKINTER

F12: Visualisation des grilles en évolution sur TKINTER

F13: création de l'outil du dessin autonome

#  Conception

 Class NCA: -width  -height  -N channels -N filters [3,3,20]

Method= NCA(channel_n=CHANNEL_N, fire_rate=CELL_FIRE_RATE,add_noise=ADD_NOISE)

Model(input)

Methodes: 

Affiche(n,n,1) --> n,n,3

Forward(grid0) --> gridTR

Update{gridn) --> gridn+1

Convertir(grid: n, n ,20 --> n,n,3)
==> TKINTER

Class Affichage _NCA :

__init__(input, color_map)

Next_img() --> n,n,3
 



## Installation
Instructions pour installer le projet :

git clone https://gitlab-cw2.centralesupelec.fr/aymen.awainia/mnist_nca.git
