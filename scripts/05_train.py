from torch.utils.data import DataLoader, random_split
import torch
from tqdm import tqdm
from dataset import ECG_dataset
import torchvision.transforms as transforms
from torch import optim
from torch import nn
import torchvision
from vit_pytorch import ViT
import sys
import timm


img_path = '/home/donghyun/workbench/results/pngfiles'
csv_path = '/home/donghyun/workbench/results/output_files/MI_annotation.csv'


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

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
validation_dataloader = DataLoader(
    validation_dataset, batch_size=16, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = timm.create_model(
    'vit_tiny_patch16_384.augreg_in21k_ft_in1k', pretrained=True, num_classes=2).to(device)
# criterion = nn.BCEWithLogitsLoss()
# optimizer = optim.RMSprop(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(device)
for epoch in range(2):
    cost = 0.0

    for x, y in tqdm(train_dataloader):
        y = y.type(torch.LongTensor)
        x, y = x.to(device), y.to(device)

        output = model(x)
        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cost += loss

    cost = cost / len(train_dataloader)

with torch.no_grad():
    model.eval()
    for x, y in tqdm(validation_dataloader):
        x = x.to(device)
        y = y.to(device)

        outputs = model(x)
        print(f"X : {x}")
        print(f"Y : {y}")
        print(f"Outputs : {outputs}")
        print("--------------------")


# model = timm.create_model('vit_base_patch16_224_miil.in21k', pretrained=True, num_classes=10)
