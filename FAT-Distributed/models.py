import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ModelCifar(nn.Module):
    def __init__(self):
        super(ModelCifar, self).__init__()
        self.feature_extract = models.resnet152(pretrained=True)
        num_ftrs = self.feature_extract.fc.in_features
        self.feature_extract.fc = nn.Linear(num_ftrs, 10)

    def forward(self, x):
        x = self.feature_extract(x)
        return x


class ModelMnist(nn.Module):
    """A model of Mnist dataset."""

    def __init__(self, n_classes=10):
        super(ModelMnist, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=4, stride=1),
            nn.Tanh(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=n_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        output = self.classifier(x)
        return output
