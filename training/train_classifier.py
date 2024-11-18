from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from data import get_stl10_datasets, prepare_data_features
from models import MLPClassifier, SimCLR
from configs.config import Config
import torch

def train_classifier():
    # Load pre-trained SimCLR model
    simclr_model = SimCLR.load_from_checkpoint(f"{Config.saved_models_path}/simclr_final.ckpt")
    simclr_model.eval()

    # Prepare datasets
    _, train_data = get_stl10_datasets(Config.data_path)
    test_data = get_stl10_datasets(Config.data_path)[1]  # For test data

    # Extract features
    train_feats = prepare_data_features(simclr_model, train_data)
    test_feats = prepare_data_features(simclr_model, test_data)

    # Initialize MLP classifier
    classifier = MLPClassifier(
        feature_dim=train_feats.tensors[0].shape[1],
        num_classes=10  # Number of STL10 classes
    )

    # Prepare callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=Config.saved_models_path,
        filename="classifier-{epoch:02d}-{val_acc:.2f}",
        save_top_k=1,
        monitor="val_acc",
        mode="max"
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Initialize Trainer
    trainer = Trainer(
        max_epochs=Config.classifier_max_epochs,
        accelerator="gpu",
        devices=1,
        callbacks=[checkpoint_callback, lr_monitor]
    )

    # Data loaders for features
    train_loader = torch.utils.data.DataLoader(
        train_feats,
        batch_size=Config.classifier_batch_size,
        shuffle=True,
        num_workers=Config.num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_feats,
        batch_size=Config.classifier_batch_size,
        shuffle=False,
        num_workers=Config.num_workers
    )

    # Train the model
    trainer.fit(classifier, train_loader, test_loader)

    # Save the trained classifier
    trainer.save_checkpoint(f"{Config.saved_models_path}/classifier_final.ckpt")
    print("Classifier training completed and saved.")

if __name__ == "__main__":
    train_classifier()
