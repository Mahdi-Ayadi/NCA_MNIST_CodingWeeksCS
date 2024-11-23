"""
Module for visualizing Neural Cellular Automata (NCA) behavior using RGB grids.
"""

import numpy as np
import torch
import torch.nn as nn

# Set device (GPU if available, otherwise CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AfficheNCA:
    """
    Class to handle visualization and grid updates for an NCA model.
    """

    def __init__(self, input_grid, color_map, model_path="model/model_full_4.pth"):
        """
        Initialize the model with the input grid, color map, and model path.

        Args:
            input_grid (numpy.ndarray or torch.Tensor): The input grid to initialize.
            color_map (dict): A dictionary mapping numbers to RGB colors.
            model_path (str): Path to the saved model file.
        """
        # Convert input to a PyTorch tensor if it's a NumPy array
        if isinstance(input_grid, np.ndarray):
            input_grid = torch.from_numpy(input_grid).float()

        # Import the model
        self.import_model(model_path)

        # Initialize the grid
        self.initialize_grid(input_grid)

        # Define attributes
        self.color_map = color_map

    def generate_rgb_grid(self):
        """
        Generate an RGB grid for visualization.

        Returns:
            numpy.ndarray: RGB grid of shape (height, width, 3) with values in [0, 255].
        """
        # Extract dimensions
        height, width, _ = self.grid.shape

        # Extract the probability grid and compute the number grid
        proba_grid = self.grid[:, :, 1:11]  # Extract probability grid
        number_grid = proba_grid.argmax(dim=-1) + 1  # Compute number grid (1 to 10)
        mask = self.grid[:, :, 0] > 0.1  # Mask for active cells
        number_grid = number_grid * mask - 1  # Apply mask and shift to range (-1 to 9)

        # Initialize RGB grid with white color
        rgb_grid = torch.ones((height, width, 3), dtype=torch.int32) * 255

        # Convert grids to NumPy for efficient mapping
        rgb_grid_np = rgb_grid.cpu().numpy()
        number_grid_np = number_grid.cpu().numpy()

        # Map colors for active cells
        for num, color in self.color_map.items():
            mask = number_grid_np == num
            rgb_grid_np[mask] = color  # Set RGB values for matching cells

        return rgb_grid_np

    def next(self):
        """
        Perform a single update on the grid and generate the next RGB grid.

        Returns:
            numpy.ndarray: Updated RGB grid.
        """
        self.grid = self.model.update_grid(self.grid)
        return self.generate_rgb_grid()

    def import_model(self, model_path):
        """
        Import the NCA model from the specified path.

        Args:
            model_path (str): Path to the saved model file.
        """
        self.model = torch.load(model_path, map_location=DEVICE)
        self.model = self.model.to(DEVICE)  # Ensure the model is on the correct device
        self.model.eval()

    def initialize_grid(self, input_grid):
        """
        Initialize the grid with the input values.

        Args:
            input_grid (torch.Tensor): Input tensor of shape (height, width).
        """
        width, height = input_grid.shape
        self.grid = torch.zeros(height, width, self.model.n_channels, device=DEVICE)
        self.grid[:, :, 0] = input_grid.clone().detach().to(DEVICE)
