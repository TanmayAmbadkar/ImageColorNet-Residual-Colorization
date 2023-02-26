from utils.util import train, load_files
from dataset.color_data_module import ColorDataModule
from models.image_color_net import ImageColorNet
from params import PARAMS
import os

train_images, test_images = load_files(PARAMS['dataset_path'])
data_module = ColorDataModule(train_images_dir=train_images, test_images_dir=test_images)

model = ImageColorNet(criterion = PARAMS['criterion'], lr = PARAMS['lr'])

model = train(model=model, data_module=data_module, epochs = PARAMS['epochs'], accelerator=PARAMS['accelerator'])
