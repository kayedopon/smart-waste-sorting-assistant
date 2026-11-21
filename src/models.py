
import torch
from torch import nn

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

if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = BaselineCNN().to(device)

    # Simulate a batch of 4 RGB images
    images = torch.randn(4, 3, 224, 224).to(device)

    model.eval()

    with torch.no_grad():
        outputs = model(images)

    parameters = sum(
        p.numel() for p in model.parameters()
    )

    print(f"Device: {device}")
    print(f"Parameters: {parameters:,}")
    print(f"Input shape: {images.shape}")
    print(f"Output shape: {outputs.shape}")
