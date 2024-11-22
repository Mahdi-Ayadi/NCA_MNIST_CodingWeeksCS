"""
Script to calculate and visualize the accuracy of an NCA model on the MNIST dataset.
"""

import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import NCA 


# Global Constants
ACCURACY_PER_NUMBER = True
N_STEPS = 300
TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.squeeze(0))
])
TEST_DATASET = datasets.MNIST(root="./data", train=False, download=True, transform=TRANSFORM)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Separate the dataset into ten datasets
DIGIT_DATASETS = {digit: [] for digit in range(10)}

for idx, (_, label) in enumerate(TEST_DATASET):
    DIGIT_DATASETS[label].append(idx)

# Convert each list of indices into a Subset
DIGIT_SUBSETS = {
    digit: Subset(TEST_DATASET, indices) for digit, indices in DIGIT_DATASETS.items()
}

# Create DataLoaders for each digit
DIGIT_LOADERS = {
    digit: DataLoader(subset, batch_size=1, shuffle=True)
    for digit, subset in DIGIT_SUBSETS.items()
}

# Create a DataLoader for the whole test dataset
TEST_LOADER = DataLoader(TEST_DATASET, batch_size=1, shuffle=True, pin_memory=True)


def accuracy_per_image(input_grid, label, n_steps, model_instance):
    """
    Compute accuracy for a single image across multiple steps.

    Args:
        input_grid (torch.Tensor): Input tensor of shape (height, width).
        label (torch.Tensor): Ground truth label for the image.
        n_steps (int): Number of steps to compute accuracy.
        model_instance (NCA): The NCA model instance.

    Returns:
        np.ndarray: Array of accuracies for each step.
    """
    height, width = input_grid.shape
    grid = torch.zeros(height, width, model_instance.n_channels, device=DEVICE)
    grid[:, :, 0] = input_grid.clone().detach().to(DEVICE)
    percentages = np.zeros(n_steps, dtype=np.float64)

    for step in range(n_steps):
        grid = model_instance.update_grid(grid)
        proba_grid = grid[:, :, 1:11]
        number_grid = proba_grid.argmax(dim=-1) + 1
        mask = grid[:, :, 0] > 0.1
        number_grid = number_grid * mask - 1
        n_alive = torch.sum(mask).item()
        percentages[step] = torch.sum(number_grid == label).item() / n_alive if n_alive > 0 else 0.0

    return percentages


def accuracy(data_loader, n_steps, model_instance):
    """
    Compute accuracy for an entire dataset across multiple steps.

    Args:
        data_loader (DataLoader): DataLoader for the dataset.
        n_steps (int): Number of steps to compute accuracy.
        model_instance (NCA): The NCA model instance.

    Returns:
        np.ndarray: Array of accuracies averaged over all images for each step.
    """
    num_samples = len(data_loader)
    percentages_table = np.zeros((num_samples, n_steps), dtype=np.float64)

    for i, (input_grid, label) in tqdm(
        enumerate(data_loader), total=num_samples, desc="Processing"
    ):
        percentages_table[i] = accuracy_per_image(
            input_grid.squeeze(0), label.squeeze(0), n_steps, model_instance
        )

    return np.mean(percentages_table, axis=0)


def plot_accuracy(accuracy_array, digit=None):
    """
    Plot the accuracy vs steps for a given digit or entire dataset.

    Args:
        accuracy_array (np.ndarray): Array of accuracy values.
        digit (int, optional): Digit for which the accuracy is plotted. If None, plot for all digits.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(range(N_STEPS), accuracy_array, marker='o', linestyle='-', color='b', label='Accuracy')
    title = f'Accuracy vs Steps for Digit {digit}' if digit is not None else 'Accuracy vs Steps'
    plt.title(title, fontsize=16)
    plt.xlabel('Steps', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()

    output_path = f"accuracy_vs_steps_digit_{digit}.png" if digit is not None else "accuracy_vs_steps.png"
    plt.savefig(output_path)
    plt.show()
    print(f"Plot saved as {output_path}")


if __name__ == "__main__":
    # Load the model
    model_instance = torch.load('model_full_1.pth', map_location=DEVICE)
    model_instance = model_instance.to(DEVICE)
    model_instance.eval()

    # Compute and visualize accuracy
    if ACCURACY_PER_NUMBER:
        for digit in range(10):
            dataset_loader = DIGIT_LOADERS[digit]
            accuracy_array = accuracy(dataset_loader, N_STEPS, model_instance)
            plot_accuracy(accuracy_array, digit=digit)
    else:
        accuracy_array = accuracy(TEST_LOADER, N_STEPS, model_instance)
        plot_accuracy(accuracy_array)
