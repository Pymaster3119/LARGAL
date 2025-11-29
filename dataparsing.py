import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

# Global variables that are safe (won't trigger side-effects on spawn)
imgs = []

# Define a transform object for augmentation
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=360),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
])

# Load data
with open("traincubicfit_areas.pkl", 'rb') as f:
    data = pickle.load(f)
    train = data

with open("testcubicfit_areas.pkl", 'rb') as f:
    data = pickle.load(f)
    test = data

with open("valcubicfit_areas.pkl", 'rb') as f:
    data = pickle.load(f)
    val = data

class AugmentedDataset(Dataset):
    def __init__(self, data, transforms=False, returnArea = False):
        self.data = data
        self.transforms = transforms
        self.returnArea = returnArea

    def __len__(self):
        return len(self.data[0]) * 10 if self.transforms else len(self.data[0])
        
    def __getitem__(self, idx):
        idx = idx % len(self.data[0])
        X_cropped = self.data[0][idx].astype(np.float32)
        X_cropped = np.transpose(X_cropped, (2, 0, 1))
        # convert labels from 1-based to 0-based (cross_entropy expects 0..C-1)
        selected_channel = int(self.data[1][idx]) - 1
        if self.transforms:
            # Apply transforms (on CPU) and add noise
            X_cropped = transform(torch.tensor(X_cropped)).numpy() / 255
            noise = np.random.normal(0, 0.1, X_cropped.shape).astype(np.float32)
            X_cropped = X_cropped + noise
            X_cropped = np.clip(X_cropped, 0, 1)
        # return label as numpy int64 so DataLoader collate will convert to torch.LongTensor
        if self.returnArea:
            area = self.data[2][idx]
            return X_cropped, np.int64(selected_channel), area
        return X_cropped, np.int64(selected_channel)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")