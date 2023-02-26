import numpy as np
from PIL import Image
from torchvision import transforms, models
from skimage import color, io
import torch
import pytorch_lightning as pl
import os
from sklearn.model_selection import train_test_split

def load_files(dataset_path):
    
    img_paths = []
    for r, d, f in os.walk(dataset_path):
        for file in f:
            if file[-3:]=='jpg' or file[-4:]=='JPEG':
                img_paths.append(os.path.join(r, file))

    train_images, test_images = train_test_split(img_paths, test_size=0.1)

    return train_images, test_images

def preprocess_img(img_path, transform = transforms.Resize([144, 144])):
    
    img = Image.open(img_path)
    img = np.array(transform(img))/255
    if len(img.shape) < 3:
        img = np.stack([img, img, img], axis=2)
    
    img = color.rgb2lab(img)
    img = (img + np.array([0, 128, 128]))
    
    image_gray = torch.Tensor(img[:,:,:1]/100).permute([2,0,1])
    image_col = torch.Tensor(img[:,:,1:]/255).permute([2,0,1])
    
    return image_gray, image_col

def get_rgb(gray, ab_channel):
    
    gray = gray.permute([1,2,0])
    ab_channel = ab_channel.permute([1,2,0])
    img = torch.cat([gray, ab_channel], dim=0)
    
    rgb_image = color.lab2rgb(img.cpu().numpy())
    return rgb_image


def train(model, data_module, epochs = 50, accelerator = None):

    trainer = pl.Trainer(max_epochs = epochs, accelerator = accelerator, enable_progress_bar = True)
    trainer.fit(model, data_module)

    return model
