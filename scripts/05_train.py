import os

import numpy as np
from torchmetrics.classification import MultilabelAUROC
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
import torch.nn.functional as F

wandb.init(project='medical_ecg')
wandb.config = {
  "learning_rate": 0.001,
  "epochs": 150,
  "batch_size": 64
}
img_path = '/home/dk58319/private/workbench/results/pngfiles'
csv_path = '/home/dk58319/private/workbench/results/output_files/MI_annotation.csv'

args = {
        "LEARNING_RATE" : 0.001,
        "WEIGHT_DECAY" : 0.003,
        "BATCH_SIZE" : 64,
        "NUM_EPOCHS" : 150,
        "MEAN" : (0.485, 0.456, 0.406),
        "STD" : (0.229, 0.224, 0.225),
        "BETA" : 0,
        "MODEL" : 'vit_tiny_patch16_384.augreg_in21k_ft_in1k',
        "MODEL_PATH" : "../model",
        "NUM_FOLDS" : 1,
        "DEVICE" : torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')}

transform = transforms.Compose(
    [
        transforms.RandomRotation(15),
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
valid_loader = DataLoader(
    validation_dataset, batch_size=64, shuffle=True, num_workers=8)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = timm.create_model(args["MODEL"], pretrained=True, num_classes=2).to(device)


optimizer = optim.AdamW(model.parameters(), lr=0.00001)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,total_steps=args["NUM_EPOCHS"],steps_per_epoch=len(train_loader), epochs=args["NUM_EPOCHS"])

train_losses, validation_losses = [], []



def validation(model, valid_loader, criterion):
    auroc = MultilabelAUROC(num_labels=2, average="macro", thresholds=None)
    accuracy = 0
    valid_loss = 0

    for i, (X, y) in enumerate(valid_loader):
        if torch.cuda.is_available():
            X = X.to(args["DEVICE"])
            y = y.type(torch.LongTensor)
            y = y.to(args["DEVICE"])

        outputs = model(X)
        loss = criterion(outputs, y)
        output = F.softmax(outputs, dim=1)
        valid_loss += loss.item()
        outputs_ = torch.argmax(outputs, dim=1)
        print(output)
        print("-----------------------")
        print(y)
        
        out_auroc= auroc(output,y)
        accuracy += (outputs_ == y).float().sum()
        print(accuracy)
    

    return valid_loss, accuracy, out_auroc


def train_model(model, train_loader, valid_loader, criterion, optimizer, args, fold_num=1):
    steps = 0
    total_step = len(train_loader)
    best_val = np.inf
    
    
    if torch.cuda.is_available():
        model = model.to(args["DEVICE"])

    for epoch in range(args['NUM_EPOCHS']):
        running_loss = 0
        for i, (X, y) in tqdm(enumerate(train_loader)):
            
            if torch.cuda.is_available():
                X = X.to(args["DEVICE"])
                y = y.type(torch.LongTensor)
                y = y.to(args["DEVICE"])
            
            outputs = model(X)
            loss = criterion(outputs, y)                  
                
            steps += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if steps % total_step == 0:
                model.eval()
                with torch.no_grad():
                    valid_loss, accuracy, auroc = validation(model, valid_loader, criterion)

                print("Epoch: {}/{}.. ".format(epoch + 1, args['NUM_EPOCHS']) +
                      "Training Loss: {:.5f}.. ".format(running_loss / total_step) +
                      "Valid Loss: {:.5f}.. ".format(valid_loss / len(valid_loader)) +
                      "Valid Accuracy: {:.5f}.. ".format(accuracy / len(valid_loader.dataset)) )
                wandb.log({"loss": (running_loss / total_step),"Valid loss":valid_loss / len(valid_loader) ,"Valid Accuracy": (accuracy / len(valid_loader.dataset)), "AUROC":auroc})
                # Save Model
                if (valid_loss / len(valid_loader)) < best_val:
                    best_val = (valid_loss / len(valid_loader))
                    torch.save(model.state_dict(), f"{args['MODEL_PATH']}/"+f"{args['MODEL']}.pt")
                    print("------ model saved ------- : {:.5f}".format((accuracy/len(valid_loader.dataset))*100))
                
                train_losses.append(running_loss / len(train_loader))
                validation_losses.append(valid_loss / len(valid_loader))
                steps = 0
                running_loss = 0
                model.train()
                
        scheduler.step()

    return 


with torch.no_grad():
                    valid_loss, accuracy, auroc = validation(model, valid_loader, criterion)
#train_model(model, train_loader, valid_loader, criterion, optimizer, args, fold_num=1)

