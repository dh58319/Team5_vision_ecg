import os
from torcheval.metrics.functional import binary_auprc
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
        "MODEL" : 'vit_tiny_patch16_384.augreg_in21k_ft_in1k', #384 # 파라미터와 동일한 모델을 사용하셔야합니다.
        "MODEL_CNN" : 'resnet152.tv2_in1k',  
        "MODEL_PATH" : "../model",
        "NUM_FOLDS" : 1,
        "DEVICE" : torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')}

transform = transforms.Compose(
    [
        transforms.Resize(384),
        transforms.ToTensor(),
    ]
)

dataset = ECG_dataset(csv_path, img_path, transform=transform)
dataset_len = len(dataset)
print(dataset_len)
train_size = int(17440)
val_size = int(dataset_len - 17440)

train_dataset, validation_dataset = random_split(
    dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8)
valid_loader = DataLoader(validation_dataset, batch_size=64, shuffle=True, num_workers=8)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = timm.create_model(args["MODEL_CNN"], pretrained=False, num_classes=2).to(device) # 모델 명은 args에 정의되어있습니다. []안을 수정하시면 다른 모델을 가지고 올 수 있습니다.
#model.load_state_dict(torch.load('../model/vit_tiny_patch16_384.augreg_in21k_ft_in1k_crop_flip.pt')) ## 여기에 모델 파라미터 path를 넣으시면 됩니다.

criterion = nn.CrossEntropyLoss()


def validation(model, valid_loader, criterion):
    accuracy = 0
    valid_loss = 0
    model.eval()
    auprc= []
    for i, (X, y) in enumerate(valid_loader):
        if torch.cuda.is_available():
            X = X.to(args["DEVICE"])
            y = y.type(torch.LongTensor)
            y = y.to(args["DEVICE"])

        outputs = model(X)

        loss = criterion(outputs, y)
        valid_loss += loss.item()
        outputs_ = torch.argmax(outputs, dim=1)
        a= binary_auprc(outputs_, y)
        #print(a)
        auprc.append(a)
        accuracy += (outputs_ == y).float().sum() 
        
    auc_ = sum(auprc)/len(auprc)
    
    return valid_loss, accuracy, auc_

loss, accuracy, auc = validation(model, valid_loader, criterion)

print(                "auc: {:.5f}.. ".format(auc) +
                      "Valid Loss: {:.5f}.. ".format(loss / len(valid_loader)) +
                      "Valid Accuracy: {:.5f}.. ".format(accuracy / len(valid_loader.dataset)) )