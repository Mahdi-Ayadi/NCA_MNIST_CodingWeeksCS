import os
import numpy as np
import torchvision
import random
import matplotlib.pyplot as plt

# Define file paths
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, '..', 'data')
X_train_path = os.path.join(data_dir, "X_train.npy")
Y_train_path = os.path.join(data_dir, "Y_train.npy")
X_test_path = os.path.join(data_dir, "X_test.npy")
Y_test_path = os.path.join(data_dir, "Y_test.npy")

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
    
    # Show example
    i = random.randint(0, len(X_train))
    plt.imshow(X_train[i], cmap="gray")
    plt.show()
    print("Label:", Y_train[i])