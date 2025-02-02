from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
import wandb
import torch
from models.resnet import ResNet

# Train ResNet-18 Model

def train_resnet18(train_dataset, val_dataset, batch_size, lr, weight_decay, num_workers, num_classes=10, max_epochs=500, **kwargs):

    wandb.init(
        project="Explained-CNN-SimCLR",
        config={
            "name": "baseline_resnet18",
            "max_epochs": max_epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "weight_decay": weight_decay,
        }
    )
    
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # Initialize the trainer
    trainer = pl.Trainer(
        accelerator="gpu" if str(device).startswith("cuda") else "cpu",
        devices=1,
        max_epochs=max_epochs,
        callbacks=[
            LearningRateMonitor(logging_interval='epoch')
        ]
    )

    trainer.logger._default_hp_metric = None

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=num_workers
    )

    # Seed for reproducibility
    pl.seed_everything(47)

    # Initialize the ResNet-18 model
    model = ResNet(
        lr=lr,
        weight_decay=weight_decay,
        num_classes=num_classes,
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    wandb.finish()

    return model
