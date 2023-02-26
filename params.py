import torch
from torchvision import transforms

PARAMS = {
    "dataset_path": "dataset",
    "epochs": 50,
    "lr": 0.0001,
    "criterion": nn.MSELoss(),
    "accelerator": "gpu",
    "transform": transforms.Resize([144, 144]),
}