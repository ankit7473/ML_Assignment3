import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, widths=(512,)):
        super().__init__(); layers=[]; prev=3072
        for w in widths:
            layers += [nn.Linear(prev,w), nn.ReLU()]; prev=w
        layers.append(nn.Linear(prev,10)); self.net=nn.Sequential(nn.Flatten(),*layers)
    def forward(self,x): return self.net(x)

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(nn.Conv2d(3,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(1))
        self.classifier=nn.Linear(128,10)
    def forward(self,x): return self.classifier(self.features(x).flatten(1))

class ConvAutoencoder(nn.Module):
    """Convolutional autoencoder with selectable bottleneck width."""
    def __init__(self, bottleneck=64):
        super().__init__()
        self.enc=nn.Sequential(nn.Conv2d(3,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(64,bottleneck,3,padding=1),nn.ReLU())
        self.dec=nn.Sequential(nn.Conv2d(bottleneck,64,3,padding=1),nn.ReLU(),nn.Upsample(scale_factor=2,mode="nearest"),
            nn.Conv2d(64,32,3,padding=1),nn.ReLU(),nn.Upsample(scale_factor=2,mode="nearest"),
            nn.Conv2d(32,3,3,padding=1),nn.Sigmoid())
    def forward(self,x): return self.dec(self.enc(x))
    def encode(self,x): return self.enc(x)
