import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
import random

# Dictionnaire des classes et couleurs correspondantes
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

class Affiche_NCA():
    def __init__(self, input, color_map):
        self.input = input
        self.color_map = color_map
        self.transform()

    def transform(self):
        """
        Transforms the input image to a color image based on color_map.
        """
        return self.input

    def next(self):
        """
        Returns a new colorized image with random colors.
        """
        n, p = self.input.shape
        return np.random.rand(n, p, 3)  # Generates random RGB values


class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Drawing App")

        # Canvas Size and Grid Size
        self.canvas_size = 1000
        self.grid_size = 8
        self.canvas = tk.Canvas(root, bg="white", width=self.canvas_size, height=self.canvas_size)
        self.canvas.pack(side=tk.RIGHT)

        # Control panel for colors, brush size, and reset
        self.palette_frame = tk.Frame(root)
        self.palette_frame.pack(side=tk.LEFT, fill=tk.Y)

        self.color_palette = tk.Canvas(self.palette_frame, width=200, height=200)
        self.color_palette.pack(pady=10)
        self.color_palette.bind("<Button-1>", self.select_color)
        self.fill_color_palette()

        # Label to display selected color
        self.color_display = tk.Label(self.palette_frame, bg="black", width=20, height=2)
        self.color_display.pack(pady=5)

        # Brush size control with scale and icons
        self.brush_size_frame = tk.Frame(self.palette_frame)
        self.brush_size_frame.pack(pady=10)

        self.small_brush_icon = tk.Label(self.brush_size_frame, text="🔴", font=("Arial", 12))
        self.small_brush_icon.pack(side=tk.LEFT)

        self.brush_size_scale = tk.Scale(self.brush_size_frame, from_=1, to=self.grid_size, orient=tk.HORIZONTAL)
        self.brush_size_scale.set(6)
        self.brush_size_scale.pack(side=tk.LEFT)

        self.large_brush_icon = tk.Label(self.brush_size_frame, text="🔴", font=("Arial", 24))
        self.large_brush_icon.pack(side=tk.LEFT)

        # Reset button to clear the canvas
        self.reset_button = tk.Button(self.palette_frame, text="Reset Grid", command=self.reset_grid)
        self.reset_button.pack(pady=10)

        # Bind mouse events to canvas for drawing
        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)

        # Brush size update while moving the mouse
        self.canvas.bind("<Motion>", self.update_brush_indicator)

        # Default brush color is black
        self.color = "black"
        self.last_x, self.last_y = None, None

        # Initialize the image for Affiche_NCA with a black and white input image
        self.input_image = np.zeros((self.canvas_size // self.grid_size, self.canvas_size // self.grid_size), dtype=np.uint8)
        self.affiche_nca = Affiche_NCA(self.input_image, color_map)

        # Store drawn items to update colors later
        self.drawn_items = []

        # Flag to track drawing state
        self.is_drawing = False

        # Start the periodic update of the image (this should not block the drawing)
        self.update_colorize()

    def update_colorize(self):
        # Update the image only if the user is not drawing
        if not self.is_drawing:
            self.colorize_with_next()  # Perform one update now
        # Schedule the next update in 100 ms (10 FPS)
        self.root.after(33, self.update_colorize)

    def fill_color_palette(self):
        # Fill the color palette with a gradient of colors
        for i in range(200):
            color = self.hsv_to_rgb(i / 200.0, 1, 1)
            self.color_palette.create_line(i, 0, i, 200, fill=color, width=2)

    def hsv_to_rgb(self, h, s, v):
        # Convert HSV to RGB color space
        if s == 0.0:
            return f'#{int(v * 255):02x}{int(v * 255):02x}{int(v * 255):02x}'
        i = int(h * 6)
        f = (h * 6) - i
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)
        i %= 6
        if i == 0: return f'#{int(v * 255):02x}{int(t * 255):02x}{int(p * 255):02x}'
        if i == 1: return f'#{int(q * 255):02x}{int(v * 255):02x}{int(p * 255):02x}'
        if i == 2: return f'#{int(p * 255):02x}{int(v * 255):02x}{int(t * 255):02x}'
        if i == 3: return f'#{int(p * 255):02x}{int(q * 255):02x}{int(v * 255):02x}'
        if i == 4: return f'#{int(t * 255):02x}{int(p * 255):02x}{int(v * 255):02x}'
        if i == 5: return f'#{int(v * 255):02x}{int(p * 255):02x}{int(q * 255):02x}'

    def reset_grid(self):
        # Clear all drawings on the canvas and redraw the grid
        self.canvas.delete("all")
        self.draw_grid()
        self.drawn_items.clear()  # Clear the list of drawn items

    def select_color(self, event):
        # Select color from the palette based on click position
        x = event.x
        color = self.color_palette.itemcget(self.color_palette.find_closest(x, 0), "fill")
        self.color = color
        self.color_display.config(bg=self.color)

    def start_drawing(self, event):
        # Start drawing when mouse button is pressed
        self.is_drawing = True  # Set drawing flag to True
        self.paint_pixel(event.x, event.y)

    def stop_drawing(self, event):
        # Stop drawing when mouse button is released
        self.is_drawing = False  # Set drawing flag to False

    def paint_pixel(self, x, y):
        # Calculate the pixel position based on grid size
        pixel_x = (x // self.grid_size) * self.grid_size
        pixel_y = (y // self.grid_size) * self.grid_size

        # Get the brush size
        brush_size = self.brush_size_scale.get()
        half_brush = brush_size // 2

        # Color the pixels around the touch position
        for i in range(-half_brush, half_brush + 1):
            for j in range(-half_brush, half_brush + 1):
                rect = self.canvas.create_rectangle(
                    pixel_x + i * self.grid_size, pixel_y + j * self.grid_size,
                    pixel_x + (i + 1) * self.grid_size, pixel_y + (j + 1) * self.grid_size,
                    fill=self.color, outline=self.color)
                self.drawn_items.append(rect)  # Track the drawn items

    def paint(self, event):
        # Paint the pixels touched by the mouse
        self.paint_pixel(event.x, event.y)

    def update_brush_indicator(self, event):
        # Update the brush indicator circle
        brush_size = self.brush_size_scale.get() * self.grid_size
        self.canvas.delete("brush_indicator")  # Remove previous indicator
        self.canvas.create_oval(event.x - brush_size // 2, event.y - brush_size // 2,
                                event.x + brush_size // 2, event.y + brush_size // 2,
                                outline=self.color, width=2, tags="brush_indicator")

    def colorize_with_next(self):
        # Get the new random image from next() method
        new_image = self.affiche_nca.next()

        # Update only the drawn pixels' colors
        for rect in self.drawn_items:
            coords = self.canvas.coords(rect)
            if len(coords) == 4:  # Rectangle item
                pixel_x = int(coords[0] // self.grid_size)
                pixel_y = int(coords[1] // self.grid_size)
                color = new_image[pixel_x, pixel_y]  # RGB value from the new image
                color_hex = f'#{int(color[0] * 255):02x}{int(color[1] * 255):02x}{int(color[2] * 255):02x}'

                # Update the color of the pixel
                self.canvas.itemconfig(rect, fill=color_hex)

if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()
