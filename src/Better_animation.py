import pygame
import sys
import math
import random
from affichage import *
# Initialize Pygame
pygame.init()
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
# Constants
WIDTH, HEIGHT = 600, 600
MIN_CELL_SIZE = 5  # Minimum size of each cell
MAX_CELL_SIZE = 50  # Maximum size of each cell
SLIDER_WIDTH = 200
SLIDER_HEIGHT = 20
PADDING = 10
BRUSH_MAX_SIZE = 10

# Colors
BACKGROUND_COLOR = (255, 255, 255)
GRID_COLOR = (200, 200, 200)
DEFAULT_DRAW_COLOR = (0, 0, 0)

# Frame rate control
FPS = 60  # Set frame rate to 60 frames per second

# Canvas Class to encapsulate grid logic
class Canvas:
    def __init__(self, width, height, cell_size):
        self.cell_size = cell_size
        self.width = width
        self.height = height
        # Initialize the grid
        self.grid, self.grid_width, self.grid_height = self.create_grid(cell_size)
        self.animated_pixels = set()  # Track animated pixels for color change
        self.set_new_input()
        self.index =0

    def set_new_input(self):
        self.gird_animator = Affiche_NCA(self.adapt_grid_for_input(),color_map)
    def create_grid(self, cell_size):
        """Create a blank grid based on the cell size."""
        grid_width = self.width // cell_size
        grid_height = self.height // cell_size
        return [[BACKGROUND_COLOR for _ in range(grid_width)] for _ in range(grid_height)], grid_width, grid_height

    def adapt_grid_for_input(self):
        # Create a mask of booleans to determine if the cells are empty
        input_mask = np.sum(np.array(self.grid), axis=2) != 255*3 
        
        # Convert the boolean mask to an integer array (1 for True, 0 for False)
        int_mask = input_mask.astype(int)
    
        return int_mask
    def adapt_grid_for_output(self,output):
        return list(output)
        
    def apply_brush(self, pos, brush_size, color):
        """Apply the brush to the grid based on mouse position and brush size."""
        x, y = pos
        col = x // self.cell_size
        row = y // self.cell_size

        # Iterate over cells in the brush's area
        for r in range(row - brush_size, row + brush_size + 1):
            for c in range(col - brush_size, col + brush_size + 1):
                # Check if the cell is within the grid and within the circular brush
                if 0 <= r < len(self.grid) and 0 <= c < len(self.grid[0]):
                    distance = math.sqrt((r - row) ** 2 + (c - col) ** 2)
                    if distance <= brush_size:
                        self.grid[r][c] = color
                        self.animated_pixels.add((r, c))  # Add to animated pixels

    def draw_grid(self, surface):
        """Draw the pixel grid on the screen."""
        for row in range(len(self.grid)):
            for col in range(len(self.grid[0])):
                rect = pygame.Rect(col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(surface, self.grid[row][col], rect)
                pygame.draw.rect(surface, GRID_COLOR, rect, 1)  # Draw grid lines

    def clear_grid(self):
        """Clear the grid and reset the animation state."""
        self.grid, self.grid_width, self.grid_height = self.create_grid(self.cell_size)
        self.animated_pixels.clear()

    def update_animated_pixels(self):
        """Update the colors of animated pixels to random non-white colors."""
        self.grid = self.adapt_grid_for_output(self.gird_animator.next())
    def print_exemple(self):
        L=list(self.animated_pixels)
        x,y = L[self.index]
        print(self.gird_animator.grid[x,y,:])
        
        


# Slider drawing and interaction functions
def draw_slider(surface, x, y, value, max_value, color):
    """Draw a slider and return the updated value."""
    slider_rect = pygame.Rect(x, y, SLIDER_WIDTH, SLIDER_HEIGHT)
    pygame.draw.rect(surface, (200, 200, 200), slider_rect)  # Slider background

    # Filled part of the slider
    fill_width = int((value / max_value) * SLIDER_WIDTH)
    pygame.draw.rect(surface, color, (x, y, fill_width, SLIDER_HEIGHT))

    # Border
    pygame.draw.rect(surface, (0, 0, 0), slider_rect, 2)
    return slider_rect

def handle_slider(event, slider_rect, max_value, min_value=0):
    """Handle slider interaction and return the updated value."""
    if event.type == pygame.MOUSEBUTTONDOWN or event.type == pygame.MOUSEMOTION:
        if pygame.mouse.get_pressed()[0]:  # Left mouse button
            mouse_x, mouse_y = pygame.mouse.get_pos()
            if slider_rect.collidepoint(mouse_x, mouse_y):
                # Calculate new value based on mouse position
                relative_x = mouse_x - slider_rect.x
                value = max(min_value, min(max_value, int((relative_x / SLIDER_WIDTH) * (max_value - min_value) + min_value)))
                return value
    return None

# Random color generation
def random_color():
    """Generate a random non-white color."""
    return random.choice([
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255)
    ])  # Avoid pure white

# Main loop
def main():
    # Initialize the screen and clock for FPS control
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Pixel Grid Animation")
    clock = pygame.time.Clock()  # Control the frame rate

    # Instantiate the Canvas object
    canvas = Canvas(WIDTH, HEIGHT, cell_size=20)

    # Slider positions
    size_slider_pos = (PADDING, HEIGHT - 2 * (SLIDER_HEIGHT + PADDING))
    grid_slider_pos = (PADDING, HEIGHT - (SLIDER_HEIGHT + PADDING))

    # Initial drawing state and animation control
    drawing = False
    interacting_with_slider = False
    brush_size = 1
    animating = False

    # Main game loop
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # Check for interactions
            interacting_with_slider = False

            # Handle Brush Size slider
            size_value = handle_slider(event, draw_slider(screen, *size_slider_pos, brush_size, BRUSH_MAX_SIZE, (100, 100, 100)), BRUSH_MAX_SIZE)
            if size_value is not None:
                brush_size = size_value
                interacting_with_slider = True

            # Handle Grid Size slider
            grid_value = handle_slider(event, draw_slider(screen, *grid_slider_pos, canvas.cell_size, MAX_CELL_SIZE, (150, 150, 150)), MAX_CELL_SIZE, MIN_CELL_SIZE)
            if grid_value is not None and grid_value != canvas.cell_size:
                canvas.cell_size = grid_value
                canvas.clear_grid()  # Reset grid with new cell size
                interacting_with_slider = True

            # Start/stop drawing
            if event.type == pygame.MOUSEBUTTONDOWN and not interacting_with_slider:
                drawing = True
            if event.type == pygame.MOUSEBUTTONUP:
                drawing = False
                canvas.set_new_input()

            # Clear the board
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:  # Press 'C' to clear
                    canvas.clear_grid()

                if event.key == pygame.K_a:  # Press 'A' to toggle animation
                    animating = not animating
                if event.key == pygame.K_p:
                    canvas.print_exemple()

        # Apply brush if drawing and not interacting with sliders
        if drawing and not interacting_with_slider:
            mouse_pos = pygame.mouse.get_pos()
            canvas.apply_brush(mouse_pos, brush_size, DEFAULT_DRAW_COLOR)

        # Update animated pixels if animation is active
        if animating:
            canvas.update_animated_pixels()

        # Draw the updated grid
        screen.fill(BACKGROUND_COLOR)
        canvas.draw_grid(screen)

        # Draw sliders
        draw_slider(screen, *size_slider_pos, brush_size, BRUSH_MAX_SIZE, (100, 100, 100))
        draw_slider(screen, *grid_slider_pos, canvas.cell_size, MAX_CELL_SIZE, (150, 150, 150))

        # Update the display and control the frame rate
        pygame.display.flip()
        clock.tick(FPS)  # Limit to 60 FPS

# Run the main function
if __name__ == "__main__":
    main()
