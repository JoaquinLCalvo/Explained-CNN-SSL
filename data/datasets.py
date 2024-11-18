# Handle the dataset preparation and feature extraction

import torch
from torchvision.datasets import STL10
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy

def get_stl10_datasets(data_path):
    # Data augmentations for SimCLR
    contrast_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(size=96),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=9),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    unlabeled_data = STL10(root=data_path, split="unlabeled", download=True, transform=contrast_transforms)
    train_data_contrast = STL10(root=data_path, split="train", download=True, transform=contrast_transforms)
    return unlabeled_data, train_data_contrast

@torch.no_grad()
def prepare_data_features(model, dataset, device="cuda"):
    # Prepare model
    network = deepcopy(model.convnet)
    network.fc = torch.nn.Identity()  # Removing projection head
    network.eval()
    network.to(device)

    # Encode all images
    data_loader = DataLoader(dataset, batch_size=64, num_workers=4, shuffle=False)
    feats, labels = [], []
    for batch_imgs, batch_labels in data_loader:
        batch_imgs = batch_imgs.to(device)
        batch_feats = network(batch_imgs)
        feats.append(batch_feats.detach().cpu())
        labels.append(batch_labels)

    return TensorDataset(torch.cat(feats), torch.cat(labels))
