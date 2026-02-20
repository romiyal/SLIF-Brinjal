import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

class TransformingDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        img, label = self.subset[index]
        if self.transform is not None:
            # Check if img is already a PIL Image or needs conversion
            if not isinstance(img, Image.Image):
                if isinstance(img, torch.Tensor):
                    img = transforms.ToPILImage()(img)
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.subset)

def get_transforms(input_size):
    transform_train = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transform_val_test = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform_train, transform_val_test

def create_dataloaders(train_subsets, val_subsets, test_subset, batch_size, input_size):
    transform_train, transform_val_test = get_transforms(input_size)
    
    train_loaders = []
    val_loaders = []
    
    for train_subset in train_subsets:
        train_dataset = TransformingDataset(train_subset, transform=transform_train)
        train_loaders.append(DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True))
        
    for val_subset in val_subsets:
        val_dataset = TransformingDataset(val_subset, transform=transform_val_test)
        val_loaders.append(DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True))
        
    test_dataset = TransformingDataset(test_subset, transform=transform_val_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loaders, val_loaders, test_loader
