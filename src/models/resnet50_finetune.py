import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


def create_resnet50_finetune(num_classes=6):
    weights = ResNet50_Weights.DEFAULT

    model = resnet50(weights=weights)

    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.layer4.parameters():
        parameter.requires_grad = True

    input_features = model.fc.in_features

    model.fc = nn.Linear(
        input_features,
        num_classes,
    )

    return model