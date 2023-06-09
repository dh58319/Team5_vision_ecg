import os

import numpy as np
from torcheval.metrics.aggregation.auc import AUC
from torch.utils.data import DataLoader, random_split
import torch
from tqdm import tqdm
from dataset import ECG_dataset
import torchvision.transforms as transforms
from torch import optim
from torch import nn
import torchvision
import sys
import timm
import wandb

img_path = '/home/dk58319/private/workbench/results/pngfiles'
csv_path = '/home/dk58319/private/workbench/results/output_files/MI_annotation.csv'

args = {
        "MODEL" : 'vit_tiny_patch16_384.augreg_in21k_ft_in1k', #384
        "MODEL_CNN" : 'resnet152.tv2_in1k',  
        "MODEL_PATH" : "../model",
        "NUM_FOLDS" : 1,
        "DEVICE" : torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')}

transform = transforms.Compose(
    [
        transforms.RandomRotation(15),
        transforms.ToTensor(),
    ]
)
transform_cnn = transforms.Compose(
    [
        transforms.RandomCrop(384),
        transforms.RandomHorizontalFlip(p = 0.3),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
    ]
)

dataset = ECG_dataset(csv_path, img_path, transform=transform_cnn)
dataset_len = len(dataset)
print(dataset_len)
train_size = int(17440)
val_size = int(dataset_len - 17440)

train_dataset, validation_dataset = random_split(
    dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8)
valid_loader = DataLoader(validation_dataset, batch_size=64, shuffle=True, num_workers=8)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = timm.create_model(args["MODEL"], pretrained=False, num_classes=2).to(device)
model.load_state_dict(torch.load('your/model/path.pt'))

criterion = nn.CrossEntropyLoss()
print(model)


def validation(model, valid_loader, criterion):
    accuracy = 0
    valid_loss = 0
    model.eval()
    metric = AUC()
    for i, (X, y) in enumerate(valid_loader):
        if torch.cuda.is_available():
            X = X.to(args["DEVICE"])
            y = y.type(torch.LongTensor)
            y = y.to(args["DEVICE"])

        outputs = model(X)
        loss = criterion(outputs, y)
        
        valid_loss += loss.item()
        outputs_ = torch.argmax(outputs, dim=1)
        metric.update(outputs_, y)
        
        accuracy += (outputs_ == y).float().sum()
        
    auc_ = metric.compute()
    metric.reset()
    

    return valid_loss, accuracy, auc_

loss, accuracy, auc = validation(model, valid_loader, criterion)

print("loss", loss)
print("auc", auc)
print("acc", accuracy)