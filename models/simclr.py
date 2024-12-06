from configs.config import Config
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.optim as optim

class SimCLR(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.hidden_dim = Config.simclr_hidden_dim
        self.lr = Config.simclr_lr
        self.temperature = Config.simclr_temperature
        self.weight_decay = Config.simclr_weight_decay
        self.max_epochs = Config.simclr_max_epochs

        # INTERNAL NOTES (TO BE DELETED IN FINAL VERSION)
        # 1. In the future, try with other backbones (could be Resnet50, could be some EfficientNet)
        # 2. Since SimCLR learns representations directly from the data, I'm not using the pre-trained weights by now
        # to avoid the bias learned from ImageNet-like datasets. However, this should also be tested.
        # Tip: for larger datasets, pretrained=False should work better (because of what I've just exposed)
        # while for small datasets, the pretrained weights might provide a performance boost

        # Define ResNet18 backbone
        self.convnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
        self.convnet.fc = nn.Sequential(
            nn.Linear(self.convnet.fc.in_features, 4 * self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(4 * self.hidden_dim, self.hidden_dim),
        )

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs, eta_min=self.lr / 50
        )
        return [optimizer], [scheduler]

    def info_nce_loss(self, batch):
        imgs, _ = batch
        # imgs = torch.cat(imgs, dim=0)

        # Encode all images
        feats = self.convnet(imgs)

        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)

        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)

        # Find positive example (batch_size//2 away from original example)
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)

        # Compute InfoNCE loss
        cos_sim = cos_sim / self.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        # Logging metrics
        self.log("train_loss", nll, prog_bar=True)

        # Ranking metrics
        comb_sim = torch.cat(
            [cos_sim[pos_mask][:, None], cos_sim.masked_fill(pos_mask, -9e15)],
            dim=-1
        )
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)

        self.log("train_acc_top1", (sim_argsort == 0).float().mean(), prog_bar=True)
        self.log("train_acc_top5", (sim_argsort < 5).float().mean(), prog_bar=True)

        return nll

    def training_step(self, batch, batch_idx):
        return self.info_nce_loss(batch)