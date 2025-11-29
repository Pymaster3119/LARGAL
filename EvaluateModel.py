import pickle
from collections import namedtuple
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import random
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle



imgs = []
annotationclass = namedtuple('annotationclass', ['bbox','category','keypoints','segmentation'])
imageclass = namedtuple('imageclass', ['image', 'normalized', 'annotations', 'mask'])

with open("train.obj", 'rb') as f:
    train = pickle.load(f)

with open("test.obj", 'rb') as f:
    test = pickle.load(f)

with open("val.obj", 'rb') as f:
    val = pickle.load(f)

class Dataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.transforms = False

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        X = sample.normalized.astype(np.float32)
        Y = sample.mask.astype(np.float32)
        
        if self.transforms:
            if random.random() < 0.5:
                X = np.fliplr(X).copy()
                Y = np.fliplr(Y).copy()
            if random.random() < 0.5:
                X = np.flipud(X).copy()
                Y = np.flipud(Y).copy()

        X = X[:448, :448, :]
        Y = Y[:448, :448, :]
        # C,W,H
        X = np.transpose(X, (2, 0, 1))
        Y = np.transpose(Y, (2, 0, 1))
        return X, Y

train_dataset = Dataset(train)
test_dataset = Dataset(test)
val_dataset = Dataset(val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

# # Get one batch of data from train_loader
# data_iter = iter(train_loader)
# images, masks = next(data_iter)

# # Print the shapes of the images and masks
# print(images.shape, masks.shape)
# import matplotlib.pyplot as plt

# # Display the first 32 images and their masks
# while True:
#     images, masks = next(data_iter)
#     fig, axes = plt.subplots(4, 8)
#     for x in range(4):
#         for y in range(8):
#             i = x * 8 + y
#             print(masks[i].numpy().shape)
#             array = np.repeat(masks[i].numpy(), 3, axis=-1)
#             axes[x, y].imshow((images[i].numpy() + array)/2, cmap='gray')
#             axes[x, y].axis('off')

#     plt.tight_layout()
#     plt.show()

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )   

def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    
    return loss.mean()

class UNet(nn.Module):

    def __init__(self, n_class):
        super().__init__()
                
        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        
        self.conv_last = nn.Conv2d(64, n_class, 1)
        
        
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)
        
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return out
   
device = torch.device("cpu")

model = UNet(n_class=4)
model.load_state_dict(torch.load('unet_epoch.pth'))

import matplotlib.pyplot as plt

data_iter = iter(val_loader)

while True:
    images, masks = next(data_iter)

    images = images.to(device)

    model.eval()
    with torch.no_grad():
        outputs = F.sigmoid(model(images))

    outputs = outputs.cpu().numpy()
    # Threshold to 0.5
    outputs[outputs > 0.1] = 1
    outputs[outputs <= 0.1] = 0
    masks = masks.cpu().numpy()

    fig, axes = plt.subplots(1, 9, figsize=(25, 5))  # 1 row, 9 columns

    # Display input image
    axes[0].imshow(np.transpose(images[0].cpu().numpy(), (1, 2, 0)))
    axes[0].set_title('Input Image')
    axes[0].axis('off')

    # Display ground truth masks
    for j in range(4):
        axes[j + 1].imshow(masks[0, j], cmap='gray')
        axes[j + 1].set_title(f'Ground Truth Mask - Channel {j+1}')
        axes[j + 1].axis('off')

    # Display predicted masks
    for j in range(4):
        axes[j + 5].imshow(outputs[0, j], cmap='gray')
        axes[j + 5].set_title(f'Predicted Mask - Channel {j+1}')
        axes[j + 5].axis('off')

    plt.tight_layout()
    plt.show()