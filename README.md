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

F14: Amélioration de la solution en utilisant Pygame au lieu de Tkinter 

#  Sprint 4: Améliorations du projet

F15: Ajout des courbes d'évolution de la précision (précision générale, précision sur chaque chiffre) et des courbes de loss lors du training


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


# Modules à avoir pour faire tourner le code 

Il faudra avoir les modules : - numpy
                              - matplotlib (matplotlib.pyplot, matplotlib.animation)
                              - pytorch (torch, torch.nn, torchvision, torch.utils.data, )
                              - pygame
                              - sys
                              - math
                              - random
                              - os
                              - PIL (Image, ImageDraw)
                              - tkinter
                              - tqdm
                              

# Comment avoir la démo ?

Il faut :  - faire tourner le code du fchier better_animation

           - dessiner un chiffre

           - appuyer sur la lettre 'a' du clavier

           - Voir si le résultat correspond à la palette suivante : 
           
                - 0 : rouge
                - 1 : vert
                - 2 : noir
                - 3 : orange
                - 4 : rose
                - 5 : bleu 
                - 6 : jaune 
                - 7 : gris
                - 8 : violet
                - 9 : marron

           - Pour refaire un nouveau test :

           Apuyer sur 'a' puis 'c', et dessiner le chiffre souhaité

# MVP

            1) fichier animation_with_model.py :
            En input, cette fonction prend une image du database,
            elle la colore grâce à matplotlib.animation puis elle 
            s'affiche grâce à affichage.py

            2) fichier partie_tkinter.py :
            Grâce au module tkinter, on a pu créer une interface 
            utilisateur qui permet de dessiner un chiffre à la 
            main et de le colorier d'une maniere random, on a 
            remarqué que l'interface tkinter était trop lente par 
            rapport à pygame, on s'est plutôt intéressé à pygame

            3) fichier better_animation.py :
            Grâce au module pygame, on a réussi à créer une 
            interface sur laquelle on dessine un chiffre qui sera 
            reconnu selon le code couleur énoncé plutôt




## Installation
Instructions pour installer le projet :

git clone https://gitlab-cw2.centralesupelec.fr/aymen.awainia/mnist_nca.git

