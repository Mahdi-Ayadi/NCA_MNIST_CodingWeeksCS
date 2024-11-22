import torch
import numpy as np
from torchvision import datasets, transforms
from model import *
import matplotlib.pyplot as plt
from torch.utils.data import Subset

# Import the MNIST dataset
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.squeeze(0))
    ])
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# Separate the dataset into ten datasets
digit_datasets = {digit: [] for digit in range(10)}  # Dictionary to hold datasets for each digit

for idx, (image, label) in enumerate(test_dataset):
    digit_datasets[label].append(idx)

# Convert each list of indices into a Subset
digit_subsets = {digit: Subset(test_dataset, indices) for digit, indices in digit_datasets.items()}

# Create DataLoaders for each digit
digit_loaders = {digit: DataLoader(subset, batch_size=1, shuffle=True) for digit, subset in digit_subsets.items()}

# Create a DataLoader for the whole test dataset (in case we want to calculate the accuracy for the whole dataset without separating numbers)
test_loader = DataLoader(test_dataset,batch_size=1, shuffle=True, pin_memory=True)

def accuracy_per_image(input,label,n_steps):
    # Grid initialization
    height, width = input.shape
    grid = torch.zeros(height, width, model.n_channels, device=device)
    grid[:, :, 0] = input.clone().detach().to(device)
    percentages = np.zeros((n_steps), dtype=np.float64)
    
    for step in range(n_steps):
        grid = model.update_grid(grid)
        proba_grid = grid[:, :, 1:11]  # Extract probability grid
        number_grid = proba_grid.argmax(dim=-1) + 1  # Compute number grid (1 to 10)
        mask = grid[:, :, 0] > 0.1  # Mask for active cells
        number_grid = number_grid * mask - 1  # Apply mask and shift to range (-1 to 9)
        n_alive = torch.sum(mask).item()
        percentages[step] = torch.sum(number_grid == label).item()/n_alive
    
    return percentages
            
def accuracy(dataset, n_steps, model):
    N = len(dataset)
    accuracy_table = np.zeros((n_steps), dtype=np.float64)
    percentages_table = np.zeros((N, n_steps), dtype=np.float64)
    for i, (input, label) in tqdm(enumerate(dataset), total=len(dataset), desc="Processing"):
        percentages_table[i] = accuracy_per_image(input.squeeze(0), label.squeeze(0) ,n_steps)
    accuracy_table = np.mean(percentages_table,axis=0)
    return accuracy_table
    
if __name__ == "__main__":
    # To choose to calculate the accuracy for each number independently or for the entire dataset as a whole
    accuracy_per_number = False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load('model_full_delta_scaled.pth', map_location=device)
    model = model.to(device)  # Ensure the model is on the correct device
    model.eval()
    
    # Compute the accuracy array
    n_steps = 300
    if accuracy_per_number: 
        for i in range(10):
            dataset = digit_loaders[i]
            accuracy_array = accuracy(dataset, n_steps, model)
            plt.figure(figsize=(8, 6))
            plt.plot(range(n_steps), accuracy_array, marker='o', linestyle='-', color='b', label='Accuracy')
            plt.title(f'Accuracy vs Steps for number {i}', fontsize=16)
            plt.xlabel('Steps', fontsize=14)
            plt.ylabel('Accuracy', fontsize=14)
            plt.grid(True)
            plt.legend(fontsize=12)
            plt.tight_layout()
            
            # Save the plot
            output_path = f"accuracy_vs_steps_number_{i}.png"
            plt.savefig(output_path)
            plt.show()
            print(f"Plot saved as {output_path}")
    
    else:
        dataset = test_loader
        accuracy_array = accuracy(dataset, n_steps, model)

        plt.figure(figsize=(8, 6))
        plt.plot(range(n_steps), accuracy_array, marker='o', linestyle='-', color='b', label='Accuracy')
        plt.title('Accuracy vs Steps', fontsize=16)
        plt.xlabel('Steps', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)
        plt.grid(True)
        plt.legend(fontsize=12)
        plt.tight_layout()

        # Save the plot
        output_path = "accuracy_vs_steps.png"
        plt.savefig(output_path)
        plt.show()

        print(f"Plot saved as {output_path}")
    