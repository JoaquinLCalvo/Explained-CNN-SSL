from configs.config import Config
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.optim as optim

class MLPClassifier(pl.LightningModule):
    def __init__(self, feature_dim, num_classes):
        super().__init__()

        # Use hyperparameters from Config
        self.hidden_dim = Config.mlp_hidden_dim
        self.lr = Config.mlp_lr
        self.weight_decay = Config.mlp_weight_decay
        self.max_epochs = Config.mlp_max_epochs

        # Define the MLP architecture
        self.model = nn.Sequential(
            nn.Linear(feature_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim, num_classes)
        )

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs, eta_min=self.lr / 50
        )
        return [optimizer], [scheduler]

    def _calculate_loss(self, batch, mode="train"):
        feats, labels = batch
        preds = self.model(feats)  # Forward pass
        loss = F.cross_entropy(preds, labels)  # Compute loss
        acc = (preds.argmax(dim=-1) == labels).float().mean()  # Compute accuracy

        # Log metrics
        self.log(f"{mode}_loss", loss, prog_bar=True)
        self.log(f"{mode}_acc", acc, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")