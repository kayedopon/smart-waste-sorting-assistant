
import torch
from torch import nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

from src.dataset import CLASS_NAMES


class BaselineCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Reduce each feature map to one value
            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, len(CLASS_NAMES)),
        )


    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x

def create_mobilenet():

    model = mobilenet_v3_small(
        weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1
    )

    # Replace ImageNet's 1000-class output with 7 waste classes
    model.classifier[-1] = nn.Linear(
        model.classifier[-1].in_features,
        len(CLASS_NAMES),
    )

    return model

def configure_mobilenet(model, stage):

    if stage not in ("frozen", "fine_tune"):
        raise ValueError(f"Unknown training stage: {stage}")

    # Freeze the complete feature extractor
    for parameter in model.features.parameters():
        parameter.requires_grad = False

    # Train the complete classifier
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True

    # Fine-tuning: unfreeze the final three feature blocks
    if stage == "fine_tune":
        for layer in model.features[-3:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True

    return model

if __name__ == "__main__":

    model = create_mobilenet()

    for stage in ("frozen", "fine_tune"):

        configure_mobilenet(model, stage)

        trainable = sum(
            p.numel() for p in model.parameters()
            if p.requires_grad
        )

        print(f"{stage}: {trainable:,} trainable parameters")

    model.eval()

    with torch.no_grad():
        output = model(torch.randn(1, 3, 224, 224))

    print(f"Output shape: {output.shape}")