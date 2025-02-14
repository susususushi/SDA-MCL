import os
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from PIL import Image
import json

#imagenet dataset
class ImageNet(ImageFolder):
    def __init__(self, root, train, transform):
         super().__init__(os.path.join(root, 'train' if train else 'val'), transform)
         self.transform_in = transforms.Compose([
             transforms.ConvertImageDtype(torch.float32),
             transforms.Resize(size=(224, 224)),
         ])
        
    def __getitem__(self, idx):
        img, label = super().__getitem__(idx)
        img = self.transform_in(img)
        label = torch.tensor(label)
        return img, label