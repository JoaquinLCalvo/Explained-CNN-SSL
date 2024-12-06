import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from data import get_stl10_datasets
from models import SimCLR
from configs.config import Config

def train_simclr():
    # Prepare datasets
    unlabeled_data, _ = get_stl10_datasets(Config.data_path)

    # Initialize SimCLR model
    model = SimCLR()

    # Prepare callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=Config.saved_models_path,
        filename="simclr-{epoch:02d}-{train_loss:.2f}",
        save_top_k=1,
        monitor="train_loss",
        mode="min"
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Initialize Trainer
    trainer = Trainer(
        max_epochs=Config.simclr_max_epochs,
        devices=1,
        callbacks=[checkpoint_callback, lr_monitor]
    )

    # Data loader for unlabeled data
    train_loader = torch.utils.data.DataLoader(
        unlabeled_data,
        batch_size=256,  # Adapt as needed for T4 GPU
        shuffle=True,
        num_workers=Config.num_workers
    )

    # Train the model
    trainer.fit(model, train_loader)

    # Save the trained model
    trainer.save_checkpoint(f"{Config.saved_models_path}/simclr_final.ckpt")
    print("SimCLR model training completed and saved.")

if __name__ == "__main__":
    train_simclr()
