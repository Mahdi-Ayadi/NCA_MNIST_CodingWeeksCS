"""
Implementation of a Neural Cellular Automaton (NCA) for MNIST classification.
"""

import torch
from torch import nn  # Use explicit imports
import numpy as np

import os
import sys

# Add the directory containing train.py to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from .train import *

# Define device (use GPU if available, otherwise CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


class NCA(nn.Module):
    """Neural Cellular Automaton for grid-based learning."""
    def __init__(self, width, height, n_channels, n_filters, n_dense, tmin=50, tmax=75):
        """
        Initializes the NCA model.

        Args:
            width (int): Width of the grid.
            height (int): Height of the grid.
            n_channels (int): Number of input channels.
            n_filters (int): Number of convolutional filters.
            n_dense (int): Number of neurons in dense layers.
            tmin (int, optional): Minimum time steps for updates. Defaults to 50.
            tmax (int, optional): Maximum time steps for updates. Defaults to 75.
        """
        super().__init__()
        self.width = width
        self.height = height
        self.n_channels = n_channels
        self.n_filters = n_filters
        self.n_dense = n_dense
        self.tmin = tmin
        self.tmax = tmax

        # Defining layers
        self.conv = nn.Conv2d(in_channels=n_channels, out_channels=n_filters, kernel_size=3, stride=1, padding=1)
        self.dense1 = nn.Linear(n_filters, n_dense)
        self.dense2 = nn.Linear(n_dense, n_channels)

    def update_time(self):
        """Returns the number of periods before giving the output."""
        return np.random.randint(self.tmin, self.tmax)

    def forward(self, input_tensor):
        """
        Forward pass of the NCA model.

        Args:
            input_tensor (torch.Tensor): Input tensor of size (batch_size, height, width, channels).

        Returns:
            torch.Tensor: Output logits of size (batch_size * height * width, 10).
        """
        batch_size = input_tensor.shape[0]
        grid = torch.zeros((batch_size, self.height, self.width, self.n_channels),
                           dtype=torch.float32, device=input_tensor.device)
        grid[:, :, :, 0] = (input_tensor > 0.1).float()
        for _ in range(self.update_time()):
            convolved_grid = self.conv(grid.permute(0, 3, 1, 2))
            convolved_grid = torch.relu(convolved_grid)

            reshaped_grid = convolved_grid.permute(0, 2, 3, 1).reshape(-1, self.n_filters)
            dense_output = torch.relu(self.dense1(reshaped_grid))
            dense_output = self.dense2(dense_output)
            delta_grid = dense_output.view(-1, self.height, self.width, self.n_channels)

            delta_grid_c = delta_grid.clone()
            delta_grid_c[:, :, :, 0] = 0

            mask = grid[:, :, :, 0] > 0.1
            mask = mask.unsqueeze(-1).expand_as(delta_grid)
            delta_grid_c *= mask

            last_channel = grid[:, :, :, -1].unsqueeze(-1)
            delta_grid_c *= torch.sigmoid(last_channel)

            grid = grid + delta_grid_c

        logits = grid[:, :, :, 1:11].view(-1, 10)
        return logits

    def update_grid(self, grid_tensor):
        """
        Perform a single update on the grid.

        Args:
            grid_tensor (torch.Tensor): Input tensor of size (height, width, channels).

        Returns:
            torch.Tensor: Updated tensor of the same size.
        """
        with torch.no_grad():
            height, width, _ = grid_tensor.shape
            grid_tensor = grid_tensor.unsqueeze(0)
            convolved_grid = self.conv(grid_tensor.permute(0, 3, 1, 2))
            convolved_grid = torch.relu(convolved_grid)

            reshaped_grid = convolved_grid.permute(0, 2, 3, 1).reshape(-1, self.n_filters)
            dense_output = torch.relu(self.dense1(reshaped_grid))
            dense_output = self.dense2(dense_output)

            delta_grid = dense_output.view(-1, height, width, self.n_channels)
            delta_grid_c = delta_grid.clone()
            delta_grid_c[:, :, :, 0] = 0

            mask = grid_tensor[:, :, :, 0] > 0.1
            mask = mask.unsqueeze(-1).expand_as(delta_grid)
            delta_grid_c *= mask

            last_channel = grid_tensor[:, :, :, -1].unsqueeze(-1)
            delta_grid_c *= torch.sigmoid(last_channel)

            grid_tensor = grid_tensor + delta_grid_c
            grid_tensor = grid_tensor.squeeze(0)
        return grid_tensor


def transform_labels_to_probagrid(inputs, labels):
    """
    Transform labels into probabilistic grid.

    Args:
        inputs (torch.Tensor): Input tensor of size (batch_size, height, width).
        labels (torch.Tensor): Labels tensor of size (batch_size).

    Returns:
        torch.Tensor: Probabilistic grid of size (batch_size, height, width, 10).
    """
    batch_size, height, width = inputs.shape
    transformed_labels = torch.zeros((batch_size, height, width, 10), device=inputs.device)
    for i in range(batch_size):
        transformed_labels[i, :, :, labels[i]] = (inputs[i, :, :] > 0.1).float()
    return transformed_labels

if __name__ == "__main__":
    # Define the model
    model = NCA(width=width,
                height=height,
                n_channels=n_channels,
                n_filters=n_filters,
                n_dense=n_dense,
                tmin=tmin,
                tmax=tmax).to(DEVICE)
    train(model=model)
