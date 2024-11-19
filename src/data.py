import os
import numpy as np
import torchvision
from torchvision import datasets, transforms
import random
import matplotlib.pyplot as plt

# Define file paths
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, '..', 'data')
path = os.path.join(data_dir, "MNIST")

if not (os.path.exists(path)):
    # Download the dataset
    dataloader = torchvision.datasets.MNIST(root="", download=True, transform=torchvision.transforms.ToTensor())

else:
    transform = transforms.ToTensor()
    mnist_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    i = random.randint(0, len(mnist_dataset))
    image, label = mnist_dataset[i]
    plt.imshow(image.squeeze(), cmap='gray')
    plt.show()
    print(f"Label: {label}")
    