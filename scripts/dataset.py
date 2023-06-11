import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchvision
import torch.nn.functional as F


path = '/home/dk58319/private/workbench/results/pngfiles'
csv_path = '/home/dk58319/private/workbench/results/output_files/MI_annotation.csv'


class ECG_dataset(Dataset):
    def __init__(self, csv_path, path, transform=None):
        self.csv_path = csv_path
        self.path = path
        self.transform = transform
        self.df = pd.read_csv(csv_path)
        self.df.fillna(0)
        self.image_names = self.df['ecg_id']
        # self.labels = self.df.iloc[:, 1:10].values
        self.labels = self.df['non']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.path, str(
            self.image_names[index]).zfill(5)+'.png'))
        image = image.convert('RGB')
        image = image.resize((1000, 1000))
        if self.transform is not None:
            image = self.transform(image)
        label = self.df.iloc[index]['non']
        # label = self.labels
        # label = F.one_hot(label.to(torch.int64), num_classes=9)

        return image, label
