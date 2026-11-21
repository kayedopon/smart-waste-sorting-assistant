
import random
import time
from pathlib import Path

import mlflow
import torch
from sklearn.metrics import f1_score
from torch import nn, optim

from src.dataset import CLASS_NAMES, create_loader
from src.models import create_mobilenet, configure_mobilenet


SEEDS = [42, 123, 2026]

BATCH_SIZE = 16

FROZEN_EPOCHS = 10
FINETUNE_EPOCHS = 15

FROZEN_LR = 0.001
FINETUNE_LR = 0.0001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Smart Waste Sorting - Transfer Learning")


def set_seed(seed):

    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_epoch(model, loader, criterion, optimizer, stage):

    model.train()

    # Frozen feature blocks must remain in evaluation mode.
    # This also prevents updates to their BatchNorm statistics.
    model.features.eval()

    if stage == "fine_tune":
        for layer in model.features[-3:]:
            layer.train()

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total

def validate(model, loader, criterion):

    model.eval()

    total_loss = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)

            predictions = outputs.argmax(dim=1)

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    accuracy = sum(
        pred == label
        for pred, label in zip(all_predictions, all_labels)
    ) / len(all_labels)

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        labels=list(range(len(CLASS_NAMES))),
        average="macro",
        zero_division=0,
    )

    return total_loss / len(all_labels), accuracy, macro_f1

def run_experiment(seed, stage, initial_checkpoint=None):

    set_seed(seed)

    train_loader = create_loader("train", BATCH_SIZE)
    val_loader = create_loader("validation", BATCH_SIZE)

    model = create_mobilenet()

    # Fine-tuning starts from the best frozen checkpoint
    if initial_checkpoint is not None:

        checkpoint = torch.load(
            initial_checkpoint,
            map_location="cpu",
            weights_only=True,
        )

        model.load_state_dict(checkpoint["model_state_dict"])

    model = configure_mobilenet(model, stage).to(DEVICE)

    epochs = FROZEN_EPOCHS if stage == "frozen" else FINETUNE_EPOCHS
    learning_rate = FROZEN_LR if stage == "frozen" else FINETUNE_LR

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
    )

    checkpoint_path = MODEL_DIR / f"mobilenet_{stage}_seed_{seed}.pth"

    best_f1 = -1
    best_epoch = 0
    best_accuracy = 0

    with mlflow.start_run(run_name=f"mobilenet_{stage}_seed_{seed}"):

        mlflow.log_params({
            "model": "MobileNetV3-Small",
            "stage": stage,
            "seed": seed,
            "epochs": epochs,
            "batch_size": BATCH_SIZE,
            "learning_rate": learning_rate,
            "optimizer": "Adam",
            "precision": "FP32",
            "device": str(DEVICE),
            "pretrained": True,
            "unfrozen_feature_blocks": 0 if stage == "frozen" else 3,
            "num_classes": len(CLASS_NAMES),
            "train_images": len(train_loader.dataset),
            "validation_images": len(val_loader.dataset),
            "trainable_parameters": sum(
                p.numel() for p in model.parameters()
                if p.requires_grad
            ),
            "initial_checkpoint": str(initial_checkpoint or "ImageNet"),
        })

        mlflow.set_tag("dataset_type", "external_development")

        start_time = time.perf_counter()

        for epoch in range(1, epochs + 1):

            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, stage
            )

            val_loss, val_acc, val_f1 = validate(
                model, val_loader, criterion
            )

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_macro_f1": val_f1,
            }, step=epoch)

            print(
                f"{stage} | Seed {seed} | Epoch {epoch:02d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f} | "
                f"Val F1: {val_f1:.4f}"
            )

            # Keep the checkpoint with the highest validation F1
            if val_f1 > best_f1:

                best_f1 = val_f1
                best_epoch = epoch
                best_accuracy = val_acc

                torch.save({
                    "model_state_dict": model.state_dict(),
                    "model": "MobileNetV3-Small",
                    "stage": stage,
                    "seed": seed,
                    "epoch": epoch,
                    "class_names": CLASS_NAMES,
                    "val_macro_f1": val_f1,
                    "val_accuracy": val_acc,
                }, checkpoint_path)

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        training_time = time.perf_counter() - start_time

        mlflow.log_metrics({
            "best_val_macro_f1": best_f1,
            "best_val_accuracy": best_accuracy,
            "best_epoch": best_epoch,
            "training_seconds": training_time,
        })

        mlflow.log_artifact(
            str(checkpoint_path),
            artifact_path="checkpoints",
        )

        print(f"\n{stage} | Seed {seed} completed")
        print(f"Best epoch: {best_epoch}")
        print(f"Best validation accuracy: {best_accuracy:.4f}")
        print(f"Best validation macro-F1: {best_f1:.4f}")
        print(f"Training time: {training_time:.1f} seconds\n")

    return checkpoint_path

if __name__ == "__main__":

    print(f"Training device: {DEVICE}")
    print("Precision: FP32 (AMP disabled)")

    for seed in SEEDS:

        # Experiment A: Frozen feature extractor
        frozen_checkpoint = run_experiment(
            seed=seed,
            stage="frozen",
        )

        # Experiment B: Partially fine-tuned feature extractor
        run_experiment(
            seed=seed,
            stage="fine_tune",
            initial_checkpoint=frozen_checkpoint,
        )

    print("All transfer learning experiments completed.")
