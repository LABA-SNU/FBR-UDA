import os
import pickle
from typing import Optional

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class CustomDataset(Dataset):
    def __init__(
        self, 
        img_folder, 
        json_pickle, 
        flag : Optional[str] = None, 
        img_names : Optional[list] = None,
        transform = None, 
        augmentation = None,
    ):
        self.img_folder = img_folder
        
        if img_names is not None:
            self.img_names = img_names    
        else:
            self.img_names = sorted(os.listdir(self.img_folder))
            
        self.transform = transform

        self.augmentation = augmentation
        
        self.flag = flag
        if self.flag == 'labeled':
            self.json_pickle = json_pickle

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_folder, img_name)
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        if self.augmentation:
            img = self.augmentation(img)
            
        if self.flag == 'labeled':
            json_name = img_name
            json_path = os.path.join(self.json_pickle, json_name)
            with open(self.json_pickle, 'rb') as fr:
                data = pickle.load(fr)
    
            # data 처리
            data = data[json_name]

            return img, data
        else:
            return img, 0