import pygame
import sys
import math
import random
import numpy as np
# import sys
# sys.path.append('/src/animation')
from .affichage import Affiche_NCA

# Initialisation de Pygame
pygame.init()

# Définition d'une carte des couleurs associant des indices à des couleurs spécifiques
color_map = {
    0: [255, 0, 0],        # rouge
    1: [0, 255, 0],        # vert
    2: [0, 0, 0],          # noir
    3: [255, 165, 0],      # orange
    4: [255, 192, 203],    # rose
    5: [0, 0, 139],        # bleu
    6: [255, 255, 0],      # jaune
    7: [169, 169, 169],    # gris
    8: [238, 130, 238],    # violet
    9: [139, 69, 19]       # marron
}

# Définition des constantes pour l'application
WIDTH, HEIGHT = 600, 600         # Dimensions de la fenêtre
MIN_CELL_SIZE = 5                # Taille minimale d'une cellule
MAX_CELL_SIZE = 50               # Taille maximale d'une cellule
SLIDER_WIDTH = 200               # Largeur des curseurs (sliders)
SLIDER_HEIGHT = 20               # Hauteur des curseurs
PADDING = 10                     # Espacement entre les curseurs et les bords
BRUSH_MAX_SIZE = 10              # Taille maximale du pinceau

# Couleurs utilisées dans l'application
BACKGROUND_COLOR = (255, 255, 255)  # Blanc pour l'arrière-plan
GRID_COLOR = (200, 200, 200)        # Gris pour les lignes de la grille
DEFAULT_DRAW_COLOR = (0, 0, 0)      # Noir pour le pinceau par défaut

# Chargement d'une police pour afficher du texte
font = pygame.font.Font(None, 24)  # Police par défaut avec une taille de 24

# Contrôle de la fréquence d'images (FPS)
FPS = 60  # 60 images par seconde

# Classe Canvas : encapsule la logique de la grille
class Canvas:
    def __init__(self, width, height, cell_size):
        self.cell_size = cell_size  # Taille de chaque cellule
        self.width = width          # Largeur du canvas
        self.height = height        # Hauteur du canvas
        
        # Initialisation de la grille
        self.grid, self.grid_width, self.grid_height = self.create_grid(cell_size)
        self.animated_pixels = set()  # Pixels animés (pour changer de couleur)
        self.set_new_input()  # Préparation pour la nouvelle animation

    def set_new_input(self):
        """Initialise l'entrée pour l'animation."""
        self.gird_animator = Affiche_NCA(self.adapt_grid_for_input(), color_map)

    def create_grid(self, cell_size):
        """Crée une grille vide basée sur la taille des cellules."""
        grid_width = self.width // cell_size
        grid_height = self.height // cell_size
        # Chaque cellule est initialisée avec la couleur de fond
        return [[BACKGROUND_COLOR for _ in range(grid_width)] for _ in range(grid_height)], grid_width, grid_height

    def adapt_grid_for_input(self):
        """
        Convertit la grille actuelle en une matrice d'entrée binaire (1 si occupé, 0 si vide)
        pour le simulateur (gird_animator).
        """
        input_mask = np.sum(np.array(self.grid), axis=2) != 255 * 3  # Vérifie les cellules non blanches
        return input_mask.astype(int)  # Convertit le masque booléen en entiers

    def adapt_grid_for_output(self, output):
        """
        Convertit les données de sortie du simulateur en un format utilisable pour le canvas.
        """
        return list(output)

    def apply_brush(self, pos, brush_size, color):
        """
        Applique un pinceau de la taille et couleur spécifiées à une position donnée sur la grille.
        """
        x, y = pos
        col = x // self.cell_size
        row = y // self.cell_size

        # Parcourt les cellules affectées par le pinceau
        for r in range(row - brush_size, row + brush_size + 1):
            for c in range(col - brush_size, col + brush_size + 1):
                # Vérifie si la cellule est dans la grille et dans le rayon du pinceau circulaire
                if 0 <= r < len(self.grid) and 0 <= c < len(self.grid[0]):
                    distance = math.sqrt((r - row) ** 2 + (c - col) ** 2)
                    if distance <= brush_size:
                        self.grid[r][c] = color
                        self.animated_pixels.add((r, c))  # Ajoute à la liste des pixels animés

    def draw_grid(self, surface):
        """Dessine la grille sur la surface de l'écran."""
        for row in range(len(self.grid)):
            for col in range(len(self.grid[0])):
                rect = pygame.Rect(col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(surface, self.grid[row][col], rect)  # Remplit chaque cellule
                pygame.draw.rect(surface, GRID_COLOR, rect, 1)  # Trace les lignes de la grille

    def clear_grid(self):
        """Réinitialise la grille et l'état d'animation."""
        self.grid, self.grid_width, self.grid_height = self.create_grid(self.cell_size)
        self.animated_pixels.clear()

    def update_animated_pixels(self):
        """Met à jour les couleurs des pixels animés avec les nouvelles données."""
        self.grid = self.adapt_grid_for_output(self.gird_animator.next())

# Fonctions pour dessiner et interagir avec les curseurs (sliders)
def draw_slider(surface, x, y, value, max_value, color):
    """
    Dessine un curseur (slider) à une position donnée avec une valeur actuelle.
    """
    slider_rect = pygame.Rect(x, y, SLIDER_WIDTH, SLIDER_HEIGHT)
    pygame.draw.rect(surface, (200, 200, 200), slider_rect)  # Fond du curseur

    # Partie remplie du curseur
    fill_width = int((value / max_value) * SLIDER_WIDTH)
    pygame.draw.rect(surface, color, (x, y, fill_width, SLIDER_HEIGHT))

    # Bordure
    pygame.draw.rect(surface, (0, 0, 0), slider_rect, 2)
    return slider_rect

def handle_slider(event, slider_rect, max_value, min_value=0):
    """
    Gère l'interaction avec un curseur pour ajuster une valeur.
    """
    if event.type == pygame.MOUSEBUTTONDOWN or event.type == pygame.MOUSEMOTION:
        if pygame.mouse.get_pressed()[0]:  # Si le bouton gauche de la souris est enfoncé
            mouse_x, mouse_y = pygame.mouse.get_pos()
            if slider_rect.collidepoint(mouse_x, mouse_y):  # Vérifie si le curseur est cliqué
                # Calcule la nouvelle valeur en fonction de la position de la souris
                relative_x = mouse_x - slider_rect.x
                value = max(min_value, min(max_value, int((relative_x / SLIDER_WIDTH) * (max_value - min_value) + min_value)))
                return value
    return None

def render_text(surface, text, pos, color):
    """
    Affiche du texte à plusieurs lignes sur la surface donnée.
    """
    lines = text.split("\n")  # Divise le texte en lignes
    x, y = pos  # Position de départ
    for line in lines:
        text_surface = font.render(line, True, color)  # Rend chaque ligne
        text_rect = text_surface.get_rect(midleft=(x, y))
        surface.blit(text_surface, text_rect)
        y += text_surface.get_height() + 5  # Ajoute un espacement entre les lignes

# Fonction principale contenant la boucle du programme
def main():
    # Initialisation de la fenêtre et de l'horloge pour contrôler le FPS
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Pixel Grid Animation")
    clock = pygame.time.Clock()  # Pour contrôler la fréquence d'images

    # Création d'une instance de la classe Canvas
    canvas = Canvas(WIDTH, HEIGHT, cell_size=20)

    # Positions des curseurs (sliders)
    size_slider_pos = (PADDING, HEIGHT - 2 * (SLIDER_HEIGHT + PADDING))  # Curseur pour la taille du pinceau
    grid_slider_pos = (PADDING, HEIGHT - (SLIDER_HEIGHT + PADDING))     # Curseur pour la taille des cellules

    # États pour les interactions utilisateur
    drawing = False                  # État de dessin
    interacting_with_slider = False  # État d'interaction avec les curseurs
    brush_size = 1                   # Taille initiale du pinceau
    animating = False                # État de l'animation

    # Boucle principale du programme
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # Si l'utilisateur ferme la fenêtre
                pygame.quit()
                sys.exit()

            # Vérification des interactions
            interacting_with_slider = False

            # Gestion du curseur de taille de pinceau
            size_value = handle_slider(event, draw_slider(screen, *size_slider_pos, brush_size, BRUSH_MAX_SIZE, (100, 100, 100)), BRUSH_MAX_SIZE)
            if size_value is not None:
                brush_size = size_value
                interacting_with_slider = True

            # Gestion du curseur de taille des cellules
            grid_value = handle_slider(event, draw_slider(screen, *grid_slider_pos, canvas.cell_size, MAX_CELL_SIZE, (150, 150, 150)), MAX_CELL_SIZE, MIN_CELL_SIZE)
            if grid_value is not None and grid_value != canvas.cell_size:
                canvas.cell_size = grid_value
                canvas.clear_grid()  # Réinitialise la grille avec la nouvelle taille des cellules
                interacting_with_slider = True

            # Début/fin du dessin
            if event.type == pygame.MOUSEBUTTONDOWN and not interacting_with_slider:
                drawing = True
            if event.type == pygame.MOUSEBUTTONUP:
                drawing = False
                canvas.set_new_input()  # Met à jour l'entrée pour le simulateur après le dessin

            # Effacement de la grille
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:  # Appuie sur 'C' pour effacer
                    canvas.clear_grid()
                    canvas.set_new_input()
        # Désactive l'animation si l'utilisateur dessine
        animating = not drawing

        # Applique le pinceau si l'utilisateur dessine
        if drawing and not interacting_with_slider:
            mouse_pos = pygame.mouse.get_pos()
            canvas.apply_brush(mouse_pos, brush_size, DEFAULT_DRAW_COLOR)
            animating = False
            canvas.set_new_input()

        # Met à jour les pixels animés si l'animation est activée
        if animating:
            canvas.update_animated_pixels()

        # Dessine la grille mise à jour
        screen.fill(BACKGROUND_COLOR)  # Efface l'écran
        canvas.draw_grid(screen)      # Dessine la grille

        # Dessine les curseurs
        draw_slider(screen, *size_slider_pos, brush_size, BRUSH_MAX_SIZE, (100, 100, 100))
        draw_slider(screen, *grid_slider_pos, canvas.cell_size, MAX_CELL_SIZE, (150, 150, 150))

        # Ajout des étiquettes pour les curseurs
        text_size_slider = "brush size"
        text_size_slider_surface = font.render(text_size_slider, True, (0, 0, 0))  # Texte pour la taille
        text_size_slider_rect = text_size_slider_surface.get_rect(midleft=(2 * PADDING + SLIDER_WIDTH, HEIGHT - 2 * (SLIDER_HEIGHT + PADDING) + SLIDER_HEIGHT // 2))
        
        text_grid_slider = "grid size"
        text_grid_slider_surface = font.render(text_grid_slider, True, (0, 0, 0))  # Texte pour la grille
        text_grid_slider_rect = text_grid_slider_surface.get_rect(midleft=(2 * PADDING + SLIDER_WIDTH, HEIGHT - (SLIDER_HEIGHT + PADDING) + SLIDER_HEIGHT // 2))
        
        screen.blit(text_grid_slider_surface, text_grid_slider_rect)
        screen.blit(text_size_slider_surface, text_size_slider_rect)

        # Instructions pour l'utilisateur
        instructions = "Left click to draw, Release to animate. Press 'C' to clear the grid."
        instructions_text = font.render(instructions, True, (0, 0, 0))
        screen.blit(instructions_text, (PADDING, PADDING))

        # Affiche la légende des couleurs
        text_color_map = (
            "Color Map: \n"
            "0: rouge\n"
            "1: vert\n"
            "2: noir\n"
            "3: orange\n"
            "4: Rose\n"
            "5: Bleu \n"
            "6: jaune\n"
            "7: Gris\n"
            "8: violet\n"
            "9: Marron"
        )
        render_text(screen, text_color_map, (WIDTH - 100, HEIGHT // 2), (0, 0, 0))

        # Met à jour l'affichage et limite la fréquence d'images
        pygame.display.flip()
        clock.tick(FPS)  # Limite à 60 FPS

# Point d'entrée principal
if __name__ == "__main__":
    main()  # Lance la fonction principale

