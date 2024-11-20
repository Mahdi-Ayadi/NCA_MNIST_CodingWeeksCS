import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
from model import *
import matplotlib.pyplot as plt


# import validation data
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.squeeze(0))
    ])
batch_size = 8
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

# Load the entire model and move it to GPU
model = torch.load("model_full.pth")
model = model.to(device)  # Ensure the model is on the GPU
model.eval()

# Get a batch from the test loader

example = next(iter(test_loader))
input_images = example[0].to(device)  # Move the input image to GPU
output_grids = model(example[0].to(device))  # Model inference on GPU
labels = example[1].to(device)  # Move the label to GPU

# Process output
output_images = torch.zeros_like(input_images, device=device)  # Ensure the output tensor is on GPU
input_mask = input_images > 0.1
_,height, width = input_images.shape
print(input_images.shape)
for b in range(batch_size):
    for x in range(width):
        for y in range(height):
            if not input_mask[b,y, x]:
                continue
            p_label = torch.argmax(output_grids[b,y, x, :])
            if p_label == labels[b]:
                output_images[b,y, x] = 1

# Visualize input and output images
import matplotlib.pyplot as plt

# Ensure input and output tensors are on CPU for visualization
input_images_np = input_images.cpu().numpy()  # Convert batch of input images to numpy
output_images_np = output_images.cpu().numpy()  # Convert batch of output images to numpy

# Number of images per half
half_batch_size = batch_size // 2

# Combine the input and output into one figure with 4 columns (Input 1, Output 1, Input 2, Output 2)
fig, axes = plt.subplots(half_batch_size, 4, figsize=(16, 4 * half_batch_size))

for i in range(half_batch_size):
    # Input image (first half)
    axes[i, 0].imshow(input_images_np[i], cmap='gray')
    axes[i, 0].set_title(f"Input {i+1}")
    axes[i, 0].axis('off')

    # Output image (first half)
    axes[i, 1].imshow(output_images_np[i], cmap='gray')
    axes[i, 1].set_title(f"Output {i+1}")
    axes[i, 1].axis('off')

    # Input image (second half)
    axes[i, 2].imshow(input_images_np[i + half_batch_size], cmap='gray')
    axes[i, 2].set_title(f"Input {i + 1 + half_batch_size}")
    axes[i, 2].axis('off')

    # Output image (second half)
    axes[i, 3].imshow(output_images_np[i + half_batch_size], cmap='gray')
    axes[i, 3].set_title(f"Output {i + 1 + half_batch_size}")
    axes[i, 3].axis('off')

plt.tight_layout()
plt.show()