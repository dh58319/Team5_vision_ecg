#!/usr/bin/python3

# %pip install einops
# %pip install torch==1.12.1
# %pip install torchvision==0.13.1

import sys
sys.path.append('/home/donghyun/workbench/git/VITpytorch')
import torch
from vit_pytorch import ViT
import numpy as np
from PIL import Image

# Initialize the ViT model
v = ViT(
    image_size=256,
    patch_size=32,
    num_classes=10,
    dim=1024,
    depth=6,
    heads=16,
    mlp_dim=2048,
    dropout=0.1,
    emb_dropout=0.1
)

image = Image.open('/home/donghyun/workbench/results/pngfiles/00001.png')
image = image.resize((256, 256))
image_rgb = image.convert('RGB')  # Convert image to RGB format
image_array = np.array(image_rgb)

image_3d = np.moveaxis(image_array, -1, 0)[:3]  # Extract RGB channels and discard alpha channel if present
image_4d = np.expand_dims(image_3d, axis=0)

image_4d_tensor = torch.from_numpy(image_4d).float()
preds = v(image_4d_tensor)  # Forward pass through the ViT model

preds