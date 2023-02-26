import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from residual_block import ResidualBlock


class ImageColorNet(pl.LightningModule):
    def __init__(self, criterion = nn.MSELoss(), lr = 0.001):
        """Initialise network"""
        super(ImageColorNet, self).__init__()
        
        self.criterion = criterion
        self.lr = lr

        self.conv1 = nn.Conv2d(
            in_channels = 1, 
            out_channels = 128, 
            kernel_size = 19, 
            stride=1, 
            padding="same", 
        )
        self.prelu1 = nn.PReLU()
        residual_blocks = []
        for _ in range(16):
            residual_blocks.append(ResidualBlock(128, 128))
        
        self.residual_block = nn.Sequential(*residual_blocks)
        
        self.conv2 = nn.Conv2d(
            in_channels = 128, 
            out_channels = 128, 
            kernel_size = 19, 
            stride=1, 
            padding="same", 
        )
        self.conv3 = nn.Conv2d(
            in_channels = 128, 
            out_channels = 64, 
            kernel_size = 19, 
            stride=1, 
            padding="same", 
        )
        self.conv4 = nn.Conv2d(
            in_channels = 64, 
            out_channels = 64, 
            kernel_size = 19, 
            stride=1, 
            padding="same", 
        )
        
        self.conv5 = nn.Conv2d(
            in_channels = 64, 
            out_channels = 32, 
            kernel_size = 19, 
            stride=1, 
            padding="same", 
        )
        self.conv6 = nn.Conv2d(
            in_channels = 32, 
            out_channels = 16, 
            kernel_size = 19, 
            stride=1, 
            padding="same", 
        )
        
        self.conv7 = nn.Conv2d(
            in_channels = 16, 
            out_channels = 2, 
            kernel_size = 3, 
            stride=1, 
            padding="same", 
        )
        
    def forward(self, X):
        
        X1 = self.prelu1(self.conv1(X))
        X = self.residual_block(X1)
        X = self.conv2(X)
        X = X + X1
        X = self.conv3(X)
        X = self.conv4(X)
        X = self.conv5(X)
        X = self.conv6(X)
        X = torch.sigmoid(self.conv7(X))
       
        
        return X
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer]
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        return loss
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("test_loss", loss)
        
    def predict_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        return pred
        