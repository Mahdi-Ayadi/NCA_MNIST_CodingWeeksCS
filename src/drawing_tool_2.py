import os
from PIL import Image, ImageDraw
import tkinter as tk

class DrawingApp:
    def __init__(self, root):
        self.root = root  # Store the main window reference
        self.root.title("Drawing App")  # Set the window title 
        
        # Canvas configuration
        self.canvas_size = 1000  # Canvas size (1000x1000)
        self.grid_size = 40  # Size of each grid cell (40x40 pixels)
        self.canvas = tk.Canvas(root, bg="white", width=self.canvas_size, height=self.canvas_size)
        self.canvas.pack(side=tk.RIGHT)  # Add the canvas to the right side of the window

        # Color palette and controls
        self.palette_frame = tk.Frame(root)
        self.palette_frame.pack(side=tk.LEFT, fill=tk.Y)

        # Color palette setup
        self.color_palette = tk.Canvas(self.palette_frame, width=200, height=200)
        self.color_palette.pack(pady=10)
        self.color_palette.bind("<Button-1>", self.select_color)
        self.fill_color_palette()

        # Color display
        self.color_display = tk.Label(self.palette_frame, bg="black", width=20, height=2)
        self.color_display.pack(pady=5)

        # Brush size controls
        self.brush_size_frame = tk.Frame(self.palette_frame)
        self.brush_size_frame.pack(pady=10)
        self.small_brush_icon = tk.Label(self.brush_size_frame, text="🔴", font=("Arial", 12))  
        self.small_brush_icon.pack(side=tk.LEFT)
        
        self.brush_size_scale = tk.Scale(self.brush_size_frame, from_=1, to=6, orient=tk.HORIZONTAL)
        self.brush_size_scale.set(3)  # Default brush size (3x3 grid)
        self.brush_size_scale.pack(side=tk.LEFT)
        
        self.large_brush_icon = tk.Label(self.brush_size_frame, text="🔴", font=("Arial", 24))  
        self.large_brush_icon.pack(side=tk.LEFT)

        # Reset button to clear the grid
        self.reset_button = tk.Button(self.palette_frame, text="Reset Grid", command=self.reset_grid)
        self.reset_button.pack(pady=10)

        # Mouse bindings for drawing
        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)

        # Brush update on motion
        self.canvas.bind("<Motion>", self.update_brush_indicator)

        # Initialization of variables
        self.color = "black"
        self.last_x, self.last_y = None, None

        # Initial grid drawing
        self.draw_grid()

        # Image setup for saving
        self.image = Image.new("RGB", (self.canvas_size, self.canvas_size), "white")
        self.draw = ImageDraw.Draw(self.image)

        # Track whether the image needs to be saved
        self.image_needs_save = False

        # Automatically start saving the image (for now, call save periodically)
        self.save_image()

    def fill_color_palette(self):
        # Create a gradient of colors in the color palette
        for i in range(200):
            color = self.hsv_to_rgb(i / 200.0, 1, 1)
            self.color_palette.create_line(i, 0, i, 200, fill=color, width=2)

    def hsv_to_rgb(self, h, s, v):
        # Convert HSV to RGB
        if s == 0.0:  # Achromatic (gray)
            return f'#{int(v * 255):02x}{int(v * 255):02x}{int(v * 255):02x}'
        i = int(h * 6)  # Sector 0 to 5
        f = (h * 6) - i  # Factorial part
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

    def draw_grid(self):
        # Draw a grid on the canvas
        for i in range(0, self.canvas_size, self.grid_size):
            self.canvas.create_line(i, 0, i, self.canvas_size, fill="lightgrey")
            self.canvas.create_line(0, i, self.canvas_size, i, fill="lightgrey")

    def reset_grid(self):
        # Clear canvas and reset grid
        self.canvas.delete("all")
        self.draw_grid()

        # Reset the image to a blank white canvas and redraw the grid
        self.image = Image.new("RGB", (self.canvas_size, self.canvas_size), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.image_needs_save = True  # Mark that the image needs to be saved

        # DEBUGGING: Check if reset works
        print("Grid Reset. Image needs to be saved.")

    def select_color(self, event):
        # Select color from the palette
        x = event.x
        color = self.color_palette.itemcget(self.color_palette.find_closest(x, 0), "fill")
        self.color = color
        self.color_display.config(bg=self.color)

    def start_drawing(self, event):
        # Start drawing with a selected color
        self.paint_pixel(event.x, event.y)

    def stop_drawing(self, event):
        # Stop drawing on mouse release (no specific action needed here)
        
        pass

    def paint_pixel(self, x, y):
        # Paint the pixel grid based on mouse position and brush size
        pixel_x = (x // self.grid_size) * self.grid_size
        pixel_y = (y // self.grid_size) * self.grid_size

        brush_size = self.brush_size_scale.get()
        half_brush = brush_size // 2

        # Paint pixels with the current brush size
        for i in range(-half_brush, half_brush + 1):
            for j in range(-half_brush, half_brush + 1):
                self.canvas.create_rectangle(
                    pixel_x + i * self.grid_size, pixel_y + j * self.grid_size,
                    pixel_x + (i + 1) * self.grid_size, pixel_y + (j + 1) * self.grid_size,
                    fill=self.color, outline=self.color
                )

                # Update the image in real-time
                self.draw.rectangle(
                    [pixel_x + i * self.grid_size, pixel_y + j * self.grid_size,
                     pixel_x + (i + 1) * self.grid_size, pixel_y + (j + 1) * self.grid_size],
                    fill=self.color
                )

        # Mark that the image needs to be saved
        self.image_needs_save = True
        self.save_image()


    def paint(self, event):
        # Paint while dragging the mouse
        self.paint_pixel(event.x, event.y)

    def update_brush_indicator(self, event):
        # Update the visual brush size indicator
        brush_size = self.brush_size_scale.get() * self.grid_size
        self.canvas.delete("brush_indicator")
        self.canvas.create_oval(
            event.x - brush_size // 2, event.y - brush_size // 2,
            event.x + brush_size // 2, event.y + brush_size // 2,
            outline=self.color, width=2, tags="brush_indicator"
        )

    def save_image(self):
        # Automatically save the image to a file when it's updated
        if self.image_needs_save:
            file_path = "drawing_image.png"
            print(f"Saving image to {file_path}...")  # Debugging message
            self.image.save(file_path)
            self.image_needs_save = False  # Reset the flag after saving
            print("Image saved.")

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()
