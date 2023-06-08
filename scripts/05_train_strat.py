import os

import numpy as np
from torcheval.metrics.aggregation.auc import AUC
from torch.utils.data import DataLoader, random_split
import torch
from tqdm import tqdm
from dataset_strat import ECG_dataset
import torchvision.transforms as transforms
from torch import optim
from torch import nn
import torchvision
import sys
import timm
import wandb
from torch.utils.data.sampler import SubsetRandomSampler

from sklearn.model_selection import train_test_split,StratifiedShuffleSplit


wandb.init(project='medical_ecg')
wandb.config = {
  "learning_rate": 0.001,
  "epochs": 100,
  "batch_size": 64
}
img_path = '/home/dk58319/private/workbench/results/pngfiles'
csv_path = '/home/dk58319/private/workbench/results/output_files/MI_annotation.csv'

args = {
        "LEARNING_RATE" : 0.001,
        "WEIGHT_DECAY" : 0.003,
        "BATCH_SIZE" : 64,
        "NUM_EPOCHS" : 100,
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
test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

dataset = ECG_dataset(csv_path, img_path, transform=transform)
dataset_len = len(dataset)
print(dataset_len)

train_dataset = ECG_dataset(csv_path, img_path, transform=transform)
test_dataset = ECG_dataset(csv_path, img_path, transform= test_transform)

train_indices = list(train_dataset.train_df.index)
test_indices = list(test_dataset.test_df.index)

train_sampler = SubsetRandomSampler(train_indices)
train_loader = DataLoader(train_dataset, batch_size=args["BATCH_SIZE"], sampler=train_sampler, num_workers= 8)

test_sampler = SubsetRandomSampler(test_indices)
test_loader = DataLoader(test_dataset, batch_size=args["BATCH_SIZE"], sampler=test_sampler, num_workers= 8)



device = "cuda" if torch.cuda.is_available() else "cpu"
model = timm.create_model(args["MODEL"], pretrained=True, num_classes=2).to(device)


optimizer = optim.AdamW(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.008,total_steps=args["NUM_EPOCHS"],steps_per_epoch=len(train_loader), epochs=args["NUM_EPOCHS"])

train_losses, validation_losses = [], []


# def validation(model, test_loader, criterion):

#     accuracy = 0
#     valid_loss = 0
#     auroc = 0
    
    
#     for i, (X, y) in enumerate(test_loader):
#         if torch.cuda.is_available():
#             X = X.to(args["DEVICE"])
#             y = y.type(torch.LongTensor)
#             y = y.to(args["DEVICE"])
        
#         outputs = model(X)
#         loss = criterion(outputs, y)
#         auroc= auroc(outputs,y)
#         #auroc = metric.compute()
#         valid_loss += loss.item()
#         outputs_ = torch.argmax(outputs, dim=1)
        
#         accuracy += (outputs_ == y).float().sum()
    

#     return valid_loss, accuracy, auroc

def validation(model, test_loader, criterion):
    accuracy = 0
    valid_loss = 0
    auc = 0
    metric = AUC()

    for i, (X, y) in enumerate(test_loader):
        if torch.cuda.is_available():
            X = X.to(args["DEVICE"])
            y = y.type(torch.LongTensor)
            y = y.to(args["DEVICE"])

        outputs = model(X)
        loss = criterion(outputs, y)
        
        valid_loss += loss.item()
        outputs_ = torch.argmax(outputs, dim=1)
        print(outputs_)
        metric.update(outputs_, y)
        auc = metric.compute()
        accuracy += (outputs_ == y).float().sum()
    
    return valid_loss, accuracy, auc


def train_model(model, train_loader, test_loader, criterion, optimizer, args, fold_num=1):
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
                    valid_loss, accuracy, auc_out = validation(model, test_loader, criterion)

                print("Epoch: {}/{}.. ".format(epoch + 1, args['NUM_EPOCHS']) +
                      "Training Loss: {:.5f}.. ".format(running_loss / total_step) +
                      "Valid Loss: {:.5f}.. ".format(valid_loss / len(test_loader)) +
                      "Valid Accuracy: {:.5f}.. ".format(accuracy / (len(test_loader.dataset)*0.2)) )
                wandb.log({"loss": (running_loss / total_step),"Valid loss":(valid_loss / len(test_loader)) ,"Valid Accuracy": (accuracy / (len(test_loader.dataset)*0.2)), "AUC":auc_out})
                # Save Model
                if (valid_loss / len(test_loader)) < best_val:
                    best_val = (valid_loss / len(test_loader))
                    torch.save(model.state_dict(), f"{args['MODEL_PATH']}/"+f"{args['MODEL']}.pt")
                    print("------ model saved ------- : {:.5f}".format((accuracy/(len(test_loader.dataset)*0.2))*100))
            
            # if steps % total_step == 0:
            #     model.eval()
            #     with torch.no_grad():
            #         valid_loss, accuracy= validation(model, test_loader, criterion)

            #     print("Epoch: {}/{}.. ".format(epoch + 1, args['NUM_EPOCHS']) +
            #           "Training Loss: {:.5f}.. ".format(running_loss / total_step) +
            #           "Valid Loss: {:.5f}.. ".format(valid_loss / len(test_loader)) +
            #           "Valid Accuracy: {:.5f}.. ".format(accuracy / len(test_loader.dataset)) )
            #     wandb.log({"loss": (running_loss / total_step),"Valid loss":valid_loss / len(test_loader) ,"Valid Accuracy": (accuracy / len(test_loader.dataset))})
            #     # Save Model
            #     if (valid_loss / len(test_loader)) < best_val:
            #         best_val = (valid_loss / len(test_loader))
            #         torch.save(model.state_dict(), f"{args['MODEL_PATH']}/"+f"{args['MODEL']}.pt")
            #         print("------ model saved ------- : {:.5f}".format((accuracy/len(test_loader.dataset))*100))
                
                train_losses.append(running_loss / len(train_loader))
                validation_losses.append(valid_loss / len(test_loader))
                steps = 0
                running_loss = 0
                model.train()
                
        scheduler.step()

    return 


train_model(model, train_loader, test_loader, criterion, optimizer, args, fold_num=1)

