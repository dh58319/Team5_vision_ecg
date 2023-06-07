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
from sklearn.model_selection import StratifiedShuffleSplit

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
        self.labels = self.df['non']

        # Stratified Shuffle Split 수행하여 train 및 test 인덱스 얻기
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(sss.split(self.df, self.labels))

        # train 및 test 데이터셋 생성
        self.train_df = self.df.iloc[train_idx]
        self.test_df = self.df.iloc[test_idx]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.path, str(self.image_names[index]).zfill(5)+'.png'))
        image = image.convert('RGB')
        image = image.resize((384, 384))
        if self.transform is not None:
            image = self.transform(image)
        
        label = self.labels[index]
        
        return image, label