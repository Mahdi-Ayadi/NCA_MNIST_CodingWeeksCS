"""
Pixel Grid Animation with Pygame and Neural Cellular Automata (NCA).
"""

import math
import sys
import numpy as np
import pygame
from .affichage import AfficheNCA

# Pygame Initialization
pygame.init()

# Constants
WIDTH, HEIGHT = 600, 600
MIN_CELL_SIZE = 5
MAX_CELL_SIZE = 50
SLIDER_WIDTH = 200
SLIDER_HEIGHT = 20
PADDING = 10
BRUSH_MAX_SIZE = 10

BACKGROUND_COLOR = (255, 255, 255)
GRID_COLOR = (200, 200, 200)
DEFAULT_DRAW_COLOR = (0, 0, 0)

FPS = 60

# Font for rendering text
FONT = pygame.font.Font(None, 24)

# Color Map
COLOR_MAP = {
    0: [255, 0, 0],        # Red
    1: [0, 255, 0],        # Green
    2: [0, 0, 0],          # Black
    3: [255, 165, 0],      # Orange
    4: [255, 192, 203],    # Pink
    5: [0, 0, 139],        # Blue
    6: [255, 255, 0],      # Yellow
    7: [169, 169, 169],    # Gray
    8: [238, 130, 238],    # Violet
    9: [139, 69, 19]       # Brown
}


class Canvas:
    """Canvas class for managing grid-based drawing and animations."""

    def __init__(self, width, height, cell_size):
        self.cell_size = cell_size
        self.width = width
        self.height = height
        self.grid, self.grid_width, self.grid_height = self.create_grid(cell_size)
        self.animated_pixels = set()
        self.set_new_input()

    def set_new_input(self):
        """Initialize input for animation."""
        self.gird_animator = AfficheNCA(self.adapt_grid_for_input(), COLOR_MAP)

    def create_grid(self, cell_size):
        """Create an empty grid based on cell size."""
        grid_width = self.width // cell_size
        grid_height = self.height // cell_size
        grid = [[BACKGROUND_COLOR for _ in range(grid_width)] for _ in range(grid_height)]
        return grid, grid_width, grid_height

    def adapt_grid_for_input(self):
        """Convert the grid into a binary input matrix for the animator."""
        input_mask = np.sum(np.array(self.grid), axis=2) != 255 * 3
        return input_mask.astype(int)

    def adapt_grid_for_output(self, output):
        """Convert animator output into a usable grid format."""
        return list(output)

    def apply_brush(self, pos, brush_size, color):
        """Apply a circular brush at a given position."""
        x, y = pos
        col = x // self.cell_size
        row = y // self.cell_size

        for r in range(row - brush_size, row + brush_size + 1):
            for c in range(col - brush_size, col + brush_size + 1):
                if 0 <= r < len(self.grid) and 0 <= c < len(self.grid[0]):
                    distance = math.sqrt((r - row) ** 2 + (c - col) ** 2)
                    if distance <= brush_size:
                        self.grid[r][c] = color
                        self.animated_pixels.add((r, c))

    def draw_grid(self, surface):
        """Draw the grid on the given surface."""
        for row in range(len(self.grid)):
            for col in range(len(self.grid[0])):
                rect = pygame.Rect(col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(surface, self.grid[row][col], rect)
                pygame.draw.rect(surface, GRID_COLOR, rect, 1)

    def clear_grid(self):
        """Clear the grid and reset animation state."""
        self.grid, self.grid_width, self.grid_height = self.create_grid(self.cell_size)
        self.animated_pixels.clear()

    def update_animated_pixels(self):
        """Update animated pixels with new data."""
        self.grid = self.adapt_grid_for_output(self.gird_animator.next())


def draw_slider(surface, x, y, value, max_value, color):
    """Draw a slider on the given surface."""
    slider_rect = pygame.Rect(x, y, SLIDER_WIDTH, SLIDER_HEIGHT)
    pygame.draw.rect(surface, (200, 200, 200), slider_rect)

    fill_width = int((value / max_value) * SLIDER_WIDTH)
    pygame.draw.rect(surface, color, (x, y, fill_width, SLIDER_HEIGHT))
    pygame.draw.rect(surface, (0, 0, 0), slider_rect, 2)
    return slider_rect


def handle_slider(event, slider_rect, max_value, min_value=0):
    """
    Handle slider interactions, ensuring the interaction is strictly within the slider bounds.

    Args:
        event: Pygame event object.
        slider_rect: Rectangle representing the slider's clickable area.
        max_value: Maximum value for the slider.
        min_value: Minimum value for the slider (default is 0).

    Returns:
        New slider value if interacted, otherwise None.
    """
    if event.type in {pygame.MOUSEBUTTONDOWN, pygame.MOUSEMOTION} and pygame.mouse.get_pressed()[0]:
        mouse_x, mouse_y = pygame.mouse.get_pos()
        if slider_rect.collidepoint(mouse_x, mouse_y):  # Ensure interaction is within the full slider rect
            relative_x = mouse_x - slider_rect.x
            return max(min_value, min(max_value, int((relative_x / SLIDER_WIDTH) * (max_value - min_value) + min_value)))
    return None



def render_text(surface, text, pos, color):
    """Render multi-line text on the given surface."""
    lines = text.split("\n")
    x, y = pos
    for line in lines:
        text_surface = FONT.render(line, True, color)
        text_rect = text_surface.get_rect(midleft=(x, y))
        surface.blit(text_surface, text_rect)
        y += text_surface.get_height() + 5


def main():
    """Main function containing the program loop."""
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Pixel Grid Animation")
    clock = pygame.time.Clock()

    canvas = Canvas(WIDTH, HEIGHT, cell_size=20)

    size_slider_pos = (PADDING, HEIGHT - 2 * (SLIDER_HEIGHT + PADDING))
    grid_slider_pos = (PADDING, HEIGHT - (SLIDER_HEIGHT + PADDING))

    drawing = False
    brush_size = 1
    animating = False
    interacting_with_slider = False  # Flag to track slider interaction

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # Reset the slider interaction flag
            interacting_with_slider = False

            # Handle brush size slider
            size_value = handle_slider(
                event,
                pygame.Rect(*size_slider_pos, SLIDER_WIDTH, SLIDER_HEIGHT),  # Slider region for brush size
                BRUSH_MAX_SIZE,
            )
            if size_value is not None:
                brush_size = size_value
                interacting_with_slider = True

            # Handle grid size slider
            grid_value = handle_slider(
                event,
                pygame.Rect(*grid_slider_pos, SLIDER_WIDTH, SLIDER_HEIGHT),  # Slider region for grid size
                MAX_CELL_SIZE,
                MIN_CELL_SIZE,
            )
            if grid_value is not None and grid_value != canvas.cell_size:
                canvas.cell_size = grid_value
                canvas.clear_grid()
                interacting_with_slider = True

            # Handle mouse events for drawing
            if not interacting_with_slider:  # Prevent drawing during slider interactions
                if event.type == pygame.MOUSEBUTTONDOWN:
                    drawing = True
                if event.type == pygame.MOUSEBUTTONUP:
                    drawing = False
                    canvas.set_new_input()

            # Handle 'C' key for clearing the grid
            if event.type == pygame.KEYDOWN and event.key == pygame.K_c:
                canvas.clear_grid()
                canvas.set_new_input()

        # Toggle animation based on drawing state
        animating = not drawing and not interacting_with_slider

        if drawing and not interacting_with_slider:
            mouse_pos = pygame.mouse.get_pos()
            canvas.apply_brush(mouse_pos, brush_size, DEFAULT_DRAW_COLOR)

        if animating:
            canvas.update_animated_pixels()

        # Clear the screen and redraw everything
        screen.fill(BACKGROUND_COLOR)
        canvas.draw_grid(screen)

        # Draw sliders
        draw_slider(screen, *size_slider_pos, brush_size, BRUSH_MAX_SIZE, (100, 100, 100))
        draw_slider(screen, *grid_slider_pos, canvas.cell_size, MAX_CELL_SIZE, (150, 150, 150))

        # Render instructions
        instructions = "Left click to draw, Release to animate. Press 'C' to clear the grid."
        render_text(screen, instructions, (PADDING, PADDING), (0, 0, 0))

        pygame.display.flip()
        clock.tick(FPS)




if __name__ == "__main__":
    main()
