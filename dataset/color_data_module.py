import pytorch_lightning as pl
from color_dataset import ColorDataset
from torch.utils.data import DataLoader

class ColorDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_images_dir,
        test_images_dir,
        batch_size = 16,
        dataloader_num_workers = 8,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.dataloader_num_workers = dataloader_num_workers
        self.train_images_dir = train_images_dir
        self.test_images_dir = test_images_dir


    def train_dataloader(self):
        
        return DataLoader(
            ColorDataset(self.train_images_dir),
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.dataloader_num_workers
        )
    
    def test_dataloader(self):
        return DataLoader(
            ColorDataset(self.test_images_dir),
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.dataloader_num_workers
        )

