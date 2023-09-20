import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms, utils
from PIL import Image

class OxfordIIITPet(Dataset):
    """OxfordIIITPet dataset."""

    def __init__(self, img_dir, mask_dir, transform=None):

        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.imgs = sorted([img for img in os.listdir(img_dir) if img.endswith(".jpg")])
        self.masks = sorted([mask for mask in os.listdir(mask_dir) if mask.endswith(".png")])
        self.size = (128,128)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        
        img = Image.open(os.path.join(self.img_dir, self.imgs[idx])).convert('RGB').resize(self.size)
        img = np.reshape(img, (self.size[0],self.size[1],3))
        mask = Image.open(os.path.join(self.mask_dir, self.masks[idx])).resize(self.size)
        mask = np.reshape(mask, (1, self.size[0], self.size[1]))
        mask = mask - 1
        if self.transform:
            img = self.transform(img)
            # mask = self.transform(mask)
        return img, torch.Tensor(mask)
