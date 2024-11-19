import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Define device (use GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
        batch_size = input.shape[0]
        grid = torch.zeros((batch_size, self.height, self.width, self.n_channels), dtype=torch.float32, device=input.device)
        grid[:,:,:,0]=(input>0.1).float()
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

def transform_labels_to_probagrid(inputs,labels):
        """_summary_

        Args:
            input (labels): size (batch_size)
            input (inputs): size (batch_size,height,width)

        Returns:
            tensor:  size (batch_size,height,width,10)
        """
        batch_size, height, width = inputs.shape
        transformed_labels = torch.zeros((batch_size, height, width, 10), device=inputs.device)
        for i in range(batch_size):
            transformed_labels [i,:,:,labels[i]]=(inputs[i, :, :] > 0.1).float()
        return transformed_labels

if __name__ == "__main__":
    # Define the model
    model = NCA(width=28, height=28, n_channels=20, n_filters=64, n_dense=128*4, Tmin=50, Tmax=75).to(device)

    # Define optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()

    # Import datasets
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.squeeze(0))
        ])
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    # Create data loaders
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    print(next(iter(train_loader))[0].shape)

    # Training Loop
    epochs = 10
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)  # Shape: (batch_size, 28, 28, 10)

            #transform labels
            transformed_labels = transform_labels_to_probagrid(inputs,labels)

            # Compute loss
            loss = criterion(outputs, transformed_labels)  # Flatten predictions and labels
            loss.backward()

            # Update weights
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")
        torch.save(model, "model_full.pth")



    # Evaluation Loop
    # model.eval()
    # correct = 0
    # total = 0
    # with torch.no_grad():
    #     for inputs, labels in test_loader:
    #         inputs = inputs  # Shape: (batch_size, 1, 28, 28)
    #         proba_grid = model(inputs)
    #         predictions = torch.argmax(proba_grid, dim=-1)  # Get class predictions
    #         correct += (predictions.view(-1) == labels.view(-1)).sum().item()
    #         total += labels.numel()

    # accuracy = 100 * correct / total
    # print(f"Test Accuracy: {accuracy:.2f}%")