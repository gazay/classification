import torch.nn as nn
from torchvision import models

from catalyst.dl import registry


@registry.Model
def resnet18(classes: int = 10, pretrained: bool = True):
    assert classes > 0, f"Classes must be > 0, got {classes}"
    model = models.resnet18(pretrained=pretrained)
    if classes != 1000:
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, classes)

    return model


@registry.Model
def resnet34(classes: int = 10, pretrained: bool = True):
    assert classes > 0, f"Classes must be > 0, got {classes}"
    model = models.resnet34(pretrained=pretrained)
    if classes != 1000:
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, classes)

    return model


@registry.Model
def resnet50(classes: int = 10, pretrained: bool = True):
    assert classes > 0, f"Classes must be > 0, got {classes}"
    model = models.resnet50(pretrained=pretrained)
    if classes != 1000:
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, classes)

    return model
