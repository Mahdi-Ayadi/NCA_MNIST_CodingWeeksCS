from torch.utils.data import Dataset
import torch

class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.transform = transform  # Optional: Data augmentation or preprocessing

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Extract features and labels
        features = torch.tensor(self.data.iloc[idx, :-1].values, dtype=torch.float32)  # All columns except the last
        label = torch.tensor(self.data.iloc[idx, -1], dtype=torch.long)  # The last column

        # Apply optional transformation
        if self.transform:
            features = self.transform(features)

        return features, label
