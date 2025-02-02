from models.logistic_regression import LogisticRegression
import wandb
import torch.nn.functional as F

# Define LogisticRegression model
class LogisticRegressionWithLogging(LogisticRegression):
    def training_step(self, batch, batch_idx):
        feats, labels = batch
        preds = self.model(feats)
        loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        # Log training accuracy and loss
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        self.log("train_accuracy", acc, prog_bar=True, on_epoch=True)
        wandb.log(
            {"train_loss": loss},
            {"train_accuracy": acc}
        )
        return loss

    def validation_step(self, batch, batch_idx):
        feats, labels = batch
        preds = self.model(feats)
        loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        # Log validation accuracy
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_accuracy", acc, prog_bar=True, on_epoch=True)
        wandb.log(
            {"val_loss": loss},
            {"val_accuracy": acc}
        )
        return loss