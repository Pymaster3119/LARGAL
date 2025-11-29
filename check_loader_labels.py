import torch
from torch.utils.data import DataLoader
import dataparsing

ds = dataparsing.AugmentedDataset(dataparsing.train, transforms=False)
loader = DataLoader(ds, batch_size=16, shuffle=False)
batch = next(iter(loader))
images, labels = batch
print('labels type:', type(labels), 'dtype:', labels.dtype)
print('labels min/max:', labels.min().item(), labels.max().item())
print('labels sample:', labels[:20])
