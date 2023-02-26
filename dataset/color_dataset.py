from torch.utils.data import Dataset
from utils.util import preprocess_img

class ColorDataset(Dataset):

    def __init__(self, root_dir):
        self.images_dir = root_dir
        
    def __len__(self):
        return len(self.images_dir)

    def __getitem__(self, idx):
        
        return preprocess_img(self.images_dir[idx])