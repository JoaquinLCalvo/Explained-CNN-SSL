import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from models.simclr import SimCLR
import wandb

## Train SimCLR Model

def train_simclr(batch_size, unlabeled_data, train_data_contrast, num_workers, lr, weight_decay, temperature, hidden_dim, max_epochs=500):

    wandb.init(
        project="Explained-CNN-SimCLR",
        config={
            "name": "simclr",
            "backbone": "resnet18",
            "max_epochs": max_epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "temperature": temperature,
            "hidden_dim": hidden_dim
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
        unlabeled_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        train_data_contrast,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=num_workers
    )

    # Seed for reproducibility
    pl.seed_everything(47)

    # Initialize the SimCLR model
    model = SimCLR(max_epochs=max_epochs, lr=lr, hidden_dim=hidden_dim, weight_decay=weight_decay, temperature=temperature)

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    wandb.finish()

    return model
