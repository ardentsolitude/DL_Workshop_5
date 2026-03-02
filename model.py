import torch
import torch.nn as nn
from torchvision import models


def freeze_layers(model):
    for param in model.parameters():
        param.requires_grad = False

def get_model(model_name="resnet18", num_classes=2, feature_extract=True):
    if model_name=="resnet18":
        model = models.resnet18(weights="IMAGENET1K_V1")
    elif model_name=="resnet34":
        model = models.resnet34(weights="IMAGENET1K_V1")
    else:
        raise ValueError("Supported models: resnet18, resnet34")
    if feature_extract:
        freeze_layers(model)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model