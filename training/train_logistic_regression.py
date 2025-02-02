from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from models.logistic_regression_with_logging import LogisticRegressionWithLogging
import wandb
import torch

# Training function

def train_logreg(batch_size, train_feats_data, test_feats_data, lr, weight_decay, feature_dim, num_classes, max_epochs=500):
    wandb.init(
        project="Explained-CNN-SimCLR",
        config={
            "name": "logistic_regression",
            "max_epochs": max_epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "feature_dim": feature_dim
        }
    )

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # Define the Trainer
    trainer = pl.Trainer(
        accelerator="gpu" if str(device).startswith("cuda") else "cpu",
        devices=1,
        max_epochs=max_epochs,
        callbacks=[LearningRateMonitor("epoch")],
        enable_progress_bar=True,
        check_val_every_n_epoch=1
    )
    trainer.logger._default_hp_metric = None  # Disable default metric tracking

    # Data loaders
    train_loader = DataLoader(
        train_feats_data, batch_size=batch_size, shuffle=True,
        drop_last=False, pin_memory=True, num_workers=0
    )
    test_loader = DataLoader(
        test_feats_data, batch_size=batch_size, shuffle=False,
        drop_last=False, pin_memory=True, num_workers=0
    )

    # Instantiate the model
    model = LogisticRegressionWithLogging(lr=lr, weight_decay=weight_decay, feature_dim=feature_dim, num_classes=num_classes)

    # Train the model and log epoch-wise metrics
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)

    # Test the model for final accuracy results
    train_result = trainer.test(model, train_loader, verbose=False)
    test_result = trainer.test(model, test_loader, verbose=False)

    # Log final train and test accuracy to wandb
    wandb.log({
        "final_train_accuracy": train_result[0]["test_acc"],
        "final_test_accuracy": test_result[0]["test_acc"]
    })

    # Finish wandb run
    wandb.finish()

    return model