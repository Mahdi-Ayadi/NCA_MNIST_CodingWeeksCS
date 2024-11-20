import numpy as np
import matplotlib.animation as animate
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from model import *

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Affiche_NCA:
    def __init__(self, input, color_map, model_path="model_full.pth"):
        """
        Initialize the model with the input grid, color map, and model path.

        Args:
            input (numpy.ndarray or torch.Tensor): The input grid to initialize.
            color_map: A color map for visualization.
            model_path (str): Path to the saved model file.
        """
        # If input is a NumPy array, convert it to a PyTorch tensor

        if isinstance(input, np.ndarray):
            input = torch.from_numpy(input).float()  # Convert to float32 tensor

        # Import the model
        self.import_model(model_path)

        # Initialize the grid
        self.initialise_grid(input)

        # Define attributes
        self.color_map = color_map

    def generate_RGB_grid(self):
        # Extract dimensions
        height, width, _ = self.grid.shape

        # Extract the probability grid and compute the number grid
        proba_grid = self.grid[:, :, 1:11]  # Extract probability grid
        number_grid = proba_grid.argmax(dim=-1) + 1  # Compute number grid (1 to 10)
        mask = self.grid[:, :, 0] > 0.1  # Mask for active cells
        number_grid = number_grid * mask - 1  # Apply mask and shift to range (-1 to 9)

        # Initialize RGB_grid with ones
        RGB_grid = torch.ones((height, width, 3), dtype=torch.int32) * 255  # Initialize to white

        # Create a default RGB grid in numpy for efficient mapping
        RGB_grid_np = RGB_grid.cpu().numpy()
        number_grid_np = number_grid.cpu().numpy()  # Convert number grid to numpy

        # Map colors for active cells
        for num, color in self.color_map.items():
            # Apply the color only where number_grid == num
            mask = number_grid_np == num
            RGB_grid_np[mask] = color  # Set RGB values for matching cells

        return RGB_grid

    def next(self):
        # Update grid using the model
        self.grid = self.model.update_grid(self.grid)
        return self.generate_RGB_grid()

    def import_model(self, path="model_full.pth"):
        # Load model and map it to the device (GPU or CPU)
        self.model = torch.load(path, map_location=device)
        self.model = self.model.to(device)  # Ensure the model is on the correct device
        self.model.eval()

    def initialise_grid(self, input):
        # Initialize the grid on the correct device
        width, height = input.shape
        self.grid = torch.zeros(height, width, self.model.n_channels, device=device)
        self.grid[:, :, 0] = input.clone().detach().to(device)
