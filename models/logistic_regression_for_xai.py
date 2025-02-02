from models.logistic_regression import LogisticRegression
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy

# Redefine the classifier but for the explainablity part

class LogisticRegressionForXai(LogisticRegression):
  def __init__(self, feature_dim, num_classes, simclr_model, lr, weight_decay, max_epochs=500):
        super().__init__(feature_dim=feature_dim, num_classes=num_classes, lr=lr, weight_decay=weight_decay)
        network = deepcopy(simclr_model.convnet)
        network.fc = nn.Identity()
        network.eval()
        self.simclr_model = network
        # Mapping from representation h to classes
        self.model = nn.Linear(feature_dim, num_classes)
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

  def configure_optimizers(self):
      optimizer = optim.AdamW(self.parameters(),
                              lr=self.hparams.lr,
                              weight_decay=self.hparams.weight_decay)
      lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                    milestones=[int(self.hparams.max_epochs*0.6),
                                                                int(self.hparams.max_epochs*0.8)],
                                                    gamma=0.1)
      return [optimizer], [lr_scheduler]

  # function for the explainability call
  def forward(self, batch):
      images_features = []
      for img in batch:
          img = img.to(self.device)
          feats = self.simclr_model(img.unsqueeze(0))
          images_features.append(feats)
      images_features = torch.cat(images_features, dim=0)
      preds = self.model(images_features)
      return preds