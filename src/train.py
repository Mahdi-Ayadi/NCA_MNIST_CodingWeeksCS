"""This module contains the training function for the NCA model."""
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
def train(model, width=28, height=28, n_channels=20, n_filters=64, n_dense=128*4, tmin=50, tmax=75):
    """
    This function trains the NCA model on the MNIST dataset.
    It is imported when the user wants to train or retrain the model.
    """
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # Define optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    # Import datasets
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.squeeze(0))
    ])
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

    # Create data loaders
    BATCH_SIZE = 16
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    # Training Loop
    EPOCHS = 10
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(inputs)

            labels = labels.repeat_interleave(inputs.shape[1] * inputs.shape[2])
            loss = criterion(logits, labels)
            loss.backward()

            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss / len(train_loader):.4f}")
        torch.save(model.state_dict(), "model_cross_entropy_sigmoid.pth")
