import os
import numpy as np
import torchvision
import random
import matplotlib.pyplot as plt

# Define file paths
X_train_path = "X_train.npy"
Y_train_path = "Y_train.npy"
X_test_path = "X_test.npy"
Y_test_path = "Y_test.npy"

if not (os.path.exists(X_train_path) and os.path.exists(Y_train_path) and os.path.exists(X_test_path) and os.path.exists(Y_test_path)):
    # Download the dataset
    dataloader = torchvision.datasets.MNIST(root="", download=True, transform=torchvision.transforms.ToTensor())
    train_size = int(len(dataloader) * 0.8)

    # Create empty numpy arrays to store the data: X_dataset are the images and Y_dataset are the labels
    X_dataset = np.empty((len(dataloader), 28, 28))
    Y_dataset = np.empty((len(dataloader), 1))

    # Fill the numpy arrays with the data
    for i, (image, label) in enumerate(dataloader):
        X_dataset[i] = image.numpy().squeeze()
        Y_dataset[i] = label
    
    # Make all the images black and white
    X_dataset = (X_dataset > 0.2).astype(np.float32)
    
    # Split the data into training and testing sets
    X_train = X_dataset[:train_size]
    Y_train = Y_dataset[:train_size]
    X_test = X_dataset[train_size:]
    Y_test = Y_dataset[train_size:]

    # Save the datasets to disk
    np.save(X_train_path, X_train)
    np.save(Y_train_path, Y_train)
    np.save(X_test_path, X_test)
    np.save(Y_test_path, Y_test)
else:
    # Load the datasets from disk
    X_train = np.load(X_train_path)
    Y_train = np.load(Y_train_path)
    X_test = np.load(X_test_path)
    Y_test = np.load(Y_test_path)

    # Show example
    i = random.randint(0, len(X_train))
    plt.imshow(X_train[i], cmap="gray")
    plt.show()
    print("Label:", Y_train[i])