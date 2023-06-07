# Vision Transformer-Based Approach for MI Subclass Classification from ECG Image Data

Final project of Team 5.
- HyeonJin Hwang
- DongHyun Kim
- Han Lee

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install vit-torch.

```bash
git clone https://github.com/lucidrains/vit-pytorch
```

## Dataset
TODO


## Usage

```python
v = ViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

img = torch.randn(1, 3, 256, 256)

preds = v(img) # (1, 1000)
```

## License

[vit-torch](https://github.com/lucidrains/vit-pytorch)