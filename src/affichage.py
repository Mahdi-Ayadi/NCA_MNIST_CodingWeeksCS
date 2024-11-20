import numpy as np
import matplotlib.animation as animate
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
from model import *

class Affiche_NCA():
    def __init__(self, input, color_map,model_path= "model_full.pth"):
        #import the model
        self.import_model(model_path)
        #initialise the grid
        self.initialise_grid(input)
        
        #define attributes
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
        self.grid=self.model.update_grid(self.grid)
        return self.generate_RGB_grid()
    
    #import the model
    def import_model(self,path = "model_full.pth"):
        self.model = torch.load(path)
        self.model = self.model.to(device)  # Ensure the model is on the GPU
        self.model.eval()
    
    def initialise_grid(self,input):
        width,height = input.shape
        self.grid = torch.zeros(height,width,self.model.n_channels,device=device)
        self.grid[:,:,0]=torch.tensor(input)[:,:]
        
