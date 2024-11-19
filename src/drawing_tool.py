import tkinter as tk  # Import the Tkinter library for GUI
import math  # Import math for calculations

class DrawingApp:
    def __init__(self, root):
        self.root = root  # Store the main window reference
        self.root.title("Drawing App")  # Set the window title
        
        # Create a larger canvas for drawing
        self.canvas_size = 1000  # Size of the canvas (800x800)
        self.grid_size = 8  # Size of each grid cell (40x40 pixels)
        self.canvas = tk.Canvas(root, bg="white", width=self.canvas_size, height=self.canvas_size)
        self.canvas.pack(side=tk.RIGHT)  # Add the canvas to the right side of the window

        # Create a frame for the color palette and controls
        self.palette_frame = tk.Frame(root)
        self.palette_frame.pack(side=tk.LEFT, fill=tk.Y)  # Add the palette frame to the left side

        # Create a rectangle for the color palette
        self.color_palette = tk.Canvas(self.palette_frame, width=200, height=200)
        self.color_palette.pack(pady=10)  # Add the color palette to the palette frame
        self.color_palette.bind("<Button-1>", self.select_color)  # Bind click event to select color

        # Fill the color palette with a gradient of colors
        self.fill_color_palette()

        # Create a label to display the selected color
        self.color_display = tk.Label(self.palette_frame, bg="black", width=20, height=2)
        self.color_display.pack(pady=5)  # Add the color display label

        # Create a refined scale for brush size with icons
        self.brush_size_frame = tk.Frame(self.palette_frame)
        self.brush_size_frame.pack(pady=10)

        self.small_brush_icon = tk.Label(self.brush_size_frame, text="🔴", font=("Arial", 12))  # Small brush icon
        self.small_brush_icon.pack(side=tk.LEFT)

        self.brush_size_scale = tk.Scale(self.brush_size_frame, from_=1, to=self.grid_size, orient=tk.HORIZONTAL)
        self.brush_size_scale.set(6)  # Set default brush size to 6 pixels in diameter
        self.brush_size_scale.pack(side=tk.LEFT)

        self.large_brush_icon = tk.Label(self.brush_size_frame, text="🔴", font=("Arial", 24))  # Large brush icon
        self.large_brush_icon.pack(side=tk.LEFT)

        # Create a reset button to clear the grid
        self.reset_button = tk.Button(self.palette_frame, text="Reset Grid", command=self.reset_grid)
        self.reset_button.pack(pady=10)

        # Bind mouse events to the canvas
        self.canvas.bind("<Button-1>", self.start_drawing)  # Start drawing on mouse button press
        self.canvas.bind("<B1-Motion>", self.paint)  # Draw while dragging the mouse with the left button
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)  # Stop drawing on mouse release

        # Bind mouse motion event to update brush indicator
        self.canvas.bind("<Motion>", self.update_brush_indicator)

        self.color = "black"  # Set the default brush color to black
        self.last_x, self.last_y = None, None  # Initialize last mouse position

        self.draw_grid()  # Draw the grid on startup

    def fill_color_palette(self):
        # Fill the color palette with a gradient of colors
        for i in range(200):
            color = self.hsv_to_rgb(i / 200.0, 1, 1)  # Convert HSV to RGB
            self.color_palette.create_line(i, 0, i, 200, fill=color, width=2)  # Draw color lines

    def hsv_to_rgb(self, h, s, v):
        # Convert HSV to RGB color space
        if s == 0.0:  # Achromatic (grey)
            return f'#{int(v * 255):02x}{int(v * 255):02x}{int(v * 255):02x}'
        i = int(h * 6)  # Sector 0 to 5
        f = (h * 6) - i  # Factorial part
        p = v * (1 - s)  # Value at the lower end
        q = v * (1 - f * s)  # Value at the upper end
        t = v * (1 - (1 - f) * s)  # Value at the lower end
        i %= 6
        if i == 0: return f'#{int(v * 255):02x}{int(t * 255):02x}{int(p * 255):02x}'
        if i == 1: return f'#{int(q * 255):02x}{int(v * 255):02x}{int(p * 255):02x}'
        if i == 2: return f'#{int(p * 255):02x}{int(v * 255):02x}{int(t * 255):02x}'
        if i == 3: return f'#{int(p * 255):02x}{int(q * 255):02x}{int(v * 255):02x}'
        if i == 4: return f'#{int(t * 255):02x}{int(p * 255):02x}{int(v * 255):02x}'
        if i == 5: return f'#{int(v * 255):02x}{int(p * 255):02x}{int(q * 255):02x}'

    def draw_grid(self):
        # Draw grid lines on the canvas
        for i in range(0, self.canvas_size, self.grid_size):
            self.canvas.create_line(i, 0, i, self.canvas_size, fill="lightgrey")
            self.canvas.create_line(0, i, self.canvas_size, i, fill="lightgrey")

    def reset_grid(self):
        # Clear all drawings on the canvas and redraw the grid
        self.canvas.delete("all")  # Clear all drawings
        self.draw_grid()  # Redraw the grid lines

    def select_color(self, event):
        # Get the color from the color palette based on the click position
        x = event.x
        color = self.color_palette.itemcget(self.color_palette.find_closest(x, 0), "fill")
        self.color = color  # Set the selected color
        self.color_display.config(bg=self.color)  # Update color display

    def start_drawing(self, event):
        # Start drawing when the mouse button is pressed
        self.paint_pixel(event.x, event.y)

    def stop_drawing(self, event):
        # Stop drawing when the mouse button is released
        pass

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
                self.canvas.create_rectangle(pixel_x + i * self.grid_size, pixel_y + j * self.grid_size,
                                              pixel_x + (i + 1) * self.grid_size, pixel_y + (j + 1) * self.grid_size,
                                              fill=self.color, outline=self.color)

    def paint(self, event):
        # Paint the pixels touched by the mouse
        self.paint_pixel(event.x, event.y)

    def update_brush_indicator(self, event):
        # Update the brush indicator circle
        brush_size = self.brush_size_scale.get() * self.grid_size  # Scale according to grid size
        self.canvas.delete("brush_indicator")  # Remove previous indicator
        self.canvas.create_oval(event.x - brush_size // 2, event.y - brush_size // 2,
                                event.x + brush_size // 2, event.y + brush_size // 2,
                                outline=self.color, width=2, tags="brush_indicator")

if __name__ == "__main__":
    root = tk.Tk()  # Create the main window
    app = DrawingApp(root)  # Initialize the drawing application
    root.mainloop()  # Start the Tkinter event loop