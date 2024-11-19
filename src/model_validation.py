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
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, pin_memory=True)

# Load the entire model and move it to GPU
model = torch.load("model_full.pth")
model = model.to("cuda")  # Ensure the model is on the GPU
model.eval()

# Get a batch from the test loader
example = next(iter(test_loader))
input_image = example[0][0, :, :].to("cuda")  # Move the input image to GPU
output_grid = model(example[0].to("cuda"))[0, :, :, :]  # Model inference on GPU
label = example[1][0].to("cuda")  # Move the label to GPU

# Process output
output_image = torch.zeros_like(input_image, device="cuda")  # Ensure the output tensor is on GPU
input_mask = input_image > 0.1
height, width = input_image.shape
for x in range(width):
    for y in range(height):
        if not input_mask[y, x]:
            continue
        p_label = torch.argmax(output_grid[y, x, :])
        if p_label == label:
            output_image[y, x] = 1

# Visualize input and output images
import matplotlib.pyplot as plt

input_image_np = input_image.cpu().numpy()  # Convert to CPU for visualization
output_image_np = output_image.cpu().numpy()

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(input_image_np, cmap='gray')
axes[0].set_title("Input Image")
axes[0].axis('off')
axes[1].imshow(output_image_np, cmap='gray')
axes[1].set_title("Output Image")
axes[1].axis('off')
plt.tight_layout()
plt.show()
