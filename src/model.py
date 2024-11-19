import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NCA(nn.Module):
    def __init__(self,width, height,n_channels,n_filters,n_dense,Tmin=50,Tmax=75):
        super().__init__()
        # defining attributes:
        self.width =width
        self.height = height
        self.n_channels = n_channels
        self.n_filters = n_filters
        self.n_dense = n_dense
        self.Tmin = Tmin
        self.Tmax = Tmax
        #defining layers
        self.conv = nn.Conv2d(in_channels=n_channels, out_channels=n_filters, kernel_size=3, stride=1, padding=1)
        self.dense1 = nn.Linear(n_filters, n_dense)
        self.dense2 = nn.Linear(n_dense, n_channels)
        
        #hidden variable
    def update_time(self): #returns the numbers of periods before giving the output
        return  np.random.randint(self.Tmin, self.Tmax)
            
    def forward(self, input):
        """_summary_

        Args:
            input (tensor): size (batch_size,height,width,channels)

        Returns:
            tensor:  size (batch_size,height,width,10)
        """
        #batch_size = input.shape[0]
        grid = torch.zeros((self.height,self.width,self.n_channels), dtype=torch.float32)
        grid[:,:,:,0]=input 
        for period in range(self.update_time()):
            # Apply convolution (input: batch_size, channels, height, width)
            convolved_grid = self.conv(grid.permute(0, 3, 1, 2))  # Permute to (batch_size, channels, height, width)
            convolved_grid = torch.nn.functional.relu(convolved_grid)  # Apply ReLU activation
            
            # Flatten the convolved grid for dense layer (batch_size, height * width, n_filters)
            reshaped_grid = convolved_grid.permute(0, 2, 3, 1).reshape(-1, self.n_filters)  # Flatten spatial dimensions
            
            # Apply dense1 to reduce dimensionality
            dense_output = self.dense1(reshaped_grid)  # Shape: (batch_size * height * width, n_dense)
            
            # Apply Relu
            dense_output = torch.nn.functional.relu(dense_output)
            
            # Apply dense2 to map back to n_channels
            dense_output = self.dense2(dense_output)  # Shape: (batch_size * height * width, n_channels)
            
            # Reshape back to grid shape (batch_size, height, width, n_channels)
            delta_grid = dense_output.view(-1, self.height, self.width, self.n_channels)
            delta_grid_c = delta_grid.clone()
            delta_grid_c[:,:,:,0]=0
            
            # Creating alive cells mask
            # Create the mask for positive `grid[..., 0]`
            mask = grid[:, :, :, 0] > 0.1  # Shape: (batch_size, height, width)

            # Expand the mask to match delta_grid's last dimension
            mask = mask.unsqueeze(-1).expand_as(delta_grid)  # Shape: (batch_size, height, width, n_channels)

            # Apply the mask
            delta_grid_c = delta_grid_c * mask

            #add the delta grid to the grid
            grid = grid + delta_grid_c
        
        proba_grid = grid[:,:,:,1:11].view(-1,10)
        proba_grid = torch.nn.functional.softmax(proba_grid , dim=1)
        proba_grid = proba_grid.view(-1,self.height,self.width,10)
        
        return proba_grid
