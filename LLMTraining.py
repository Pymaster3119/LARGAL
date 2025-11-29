'''
Todo:
- Histogram the sizes
- Rotate by arbitrary angles
'''

import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm
import timm
import schedulefree.adamw_schedulefree

import wandb
import loss
import dataparsing

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create model
model = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=4, img_size=64)
model = model.to(device)


if __name__ == '__main__':
    # Initialize wandb inside main so that it is not re-run in child processes.
    wandb.init(project="radiowaves-classifier-LLM")
    wandb.config = {
        "learning_rate": 1e-4,
        "epochs": 50,
        "batch_size": 64
    }
    
    # Instantiate datasets and DataLoaders in the main process
    train_dataset = dataparsing.AugmentedDataset(dataparsing.train, transforms=True)
    test_dataset = dataparsing.AugmentedDataset(dataparsing.test)
    val_dataset = dataparsing.AugmentedDataset(dataparsing.val)
    
    train_loader = DataLoader(train_dataset, batch_size=wandb.config["batch_size"], shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=wandb.config["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=wandb.config["batch_size"], shuffle=True, num_workers=4)

    criterion = nn.CrossEntropyLoss()
    optimizer = schedulefree.adamw_schedulefree.AdamWScheduleFree(
        model.parameters(),
        lr=wandb.config["learning_rate"],
        weight_decay=1e-4
    )

    for epoch in range(wandb.config["epochs"]):
        model.train()
        running_loss = 0.0
        running_accuracy = 0
        total = 0
        running_unweighted_loss = 0.0

        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            # (Note: optimizer.train() is not a standard call but is kept per original code)
            optimizer.train()
            outputs = model(images)
            loss = loss.CB_loss(labels, outputs, "normal", loss.precomputed_weights)
            running_unweighted_loss += criterion(outputs, labels).item()
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            running_loss += loss.item()
            running_accuracy += (predicted == labels).sum().item()
        
        wandb.log({
            "weighted loss": running_loss / len(train_loader), 
            "accuracy": running_accuracy / total, 
            "loss": running_unweighted_loss / len(train_loader)
        }, step=epoch)
        
        # Validation loop
        model.eval()
        optimizer.eval()
        val_loss = 0.0
        val_unweighted_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_unweighted_loss += loss.item()
                val_loss += loss.CB_loss(labels, outputs, "normal", loss.precomputed_weights)
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        wandb.log({
            "val_weighted_loss": val_loss / len(val_loader), 
            "val_accuracy": correct / total, 
            "val_loss": val_unweighted_loss / len(val_loader)
        }, step=epoch)