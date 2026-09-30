import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


def create_resnet50(num_classes=6):
    weights = ResNet50_Weights.DEFAULT

    model = resnet50(weights=weights)

    for parameter in model.parameters():
        parameter.requires_grad = False

    input_features = model.fc.in_features

    model.fc = nn.Linear(
        input_features,
        num_classes,
    )

    return model