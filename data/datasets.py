# Handle the dataset preparation and feature extraction

import torch
import torch.nn as nn
from torchvision.datasets import STL10
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy
import numpy as np
from tqdm.notebook import tqdm

class ContrastiveTransformations(object):
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]

# Function to mix two images with a random alpha
def mixup_augmentation(img):
   alpha = 0.2  # Mixing factor
   lam = np.random.beta(alpha, alpha)
   mixed_img = lam * img + (1 - lam) * torch.flip(img, dims=[2])  # Simple example with flip
   return mixed_img

def get_transformations():
    # Augmentation policy 1
    contrast_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=9),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Augmentation policy 2
    mixup_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)
        ], p=0.8),
        transforms.ToTensor(),
        transforms.Lambda(lambda img: mixup_augmentation(img)),  # Custom function for mixup
        transforms.Normalize((0.5,), (0.5,))
    ])
    return contrast_transforms, mixup_transforms

def get_stl10_datasets_clr(data_path):
    contrast_transforms, mixup_transforms = get_transformations()

    unlabeled_data = STL10(root=data_path, split="unlabeled", download=True, transform=ContrastiveTransformations(contrast_transforms, n_views=2))
    train_data_contrast = STL10(root=data_path, split="train", download=True, transform=ContrastiveTransformations(contrast_transforms, n_views=2))
    return unlabeled_data, train_data_contrast

def get_stl10_datasets(data_path, train_transforms, test_transforms):
    train_img_data = STL10(root=data_path, split='train', download=True,
                        transform=train_transforms)
    test_img_data = STL10(root=data_path, split='test', download=True,
                        transform=test_transforms)
    
    return train_img_data, test_img_data

@torch.no_grad()
def prepare_data_features(model, dataset, num_workers):
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # Prepare model
    network = deepcopy(model.convnet)
    network.fc = nn.Identity()  # Removing projection head g(.)
    network.eval()
    network.to(device)

    # Encode all images
    data_loader = DataLoader(dataset, batch_size=64, num_workers=num_workers, shuffle=False, drop_last=False)
    feats, labels = [], []
    for batch_imgs, batch_labels in tqdm(data_loader):
        batch_imgs = batch_imgs.to(device)
        batch_feats = network(batch_imgs)
        feats.append(batch_feats.detach().cpu())
        labels.append(batch_labels)

    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)

    # Sort images by labels
    labels, idxs = labels.sort()
    feats = feats[idxs]

    return TensorDataset(feats, labels)
