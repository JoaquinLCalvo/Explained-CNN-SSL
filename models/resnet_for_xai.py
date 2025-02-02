from models.resnet import ResNet

# ResNet-18 class for explainability with forward function

class ResNetForXai(ResNet):
    def __init__(self, num_classes, lr, weight_decay, max_epochs=500):
        super().__init__(num_classes=num_classes, lr=lr, weight_decay=weight_decay)
    
    def forward(self, batch):
        return self.model(batch)