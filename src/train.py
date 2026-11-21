
import random
import time
from pathlib import Path

import mlflow
import torch
from sklearn.metrics import f1_score
from torch import nn, optim

from src.dataset import CLASS_NAMES, create_loader
from src.models import BaselineCNN


EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.001

SEEDS = [42]

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Smart Waste Sorting - Baseline CNN")


def set_seed(seed):

    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_epoch(model, loader, criterion, optimizer):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy

def validate(model, loader, criterion):

    model.eval()

    total_loss = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)

            predictions = outputs.argmax(dim=1)

            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    average_loss = total_loss / len(all_labels)

    correct = sum(
        pred == label
        for pred, label in zip(all_predictions, all_labels)
    )

    accuracy = correct / len(all_labels)

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        labels=list(range(len(CLASS_NAMES))),
        average="macro",
        zero_division=0,
    )

    return average_loss, accuracy, macro_f1

def run_experiment(seed):

    set_seed(seed)

    train_loader = create_loader("train", BATCH_SIZE)
    val_loader = create_loader("validation", BATCH_SIZE)

    model = BaselineCNN().to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_f1 = -1
    best_epoch = 0

    checkpoint_path = MODEL_DIR / f"baseline_seed_{seed}.pth"

    with mlflow.start_run(run_name=f"baseline_seed_{seed}"):

        mlflow.log_params({
            "model": "BaselineCNN",
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "optimizer": "Adam",
            "seed": seed,
            "device": str(DEVICE),
            "num_classes": len(CLASS_NAMES),
            "train_images": len(train_loader.dataset),
            "validation_images": len(val_loader.dataset),
            "parameters": sum(p.numel() for p in model.parameters()),
        })

        start_time = time.perf_counter()

        for epoch in range(1, EPOCHS + 1):

            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer
            )

            val_loss, val_acc, val_f1 = validate(
                model, val_loader, criterion
            )

            # Log epoch metrics
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_macro_f1": val_f1,
            }, step=epoch)

            print(
                f"Seed {seed} | Epoch {epoch:02d}/{EPOCHS} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f} | "
                f"Val F1: {val_f1:.4f}"
            )

            # Save best checkpoint
            if val_f1 > best_f1:

                best_f1 = val_f1
                best_epoch = epoch

                torch.save({
                    "model_state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "epoch": epoch,
                    "seed": seed,
                    "val_macro_f1": val_f1,
                }, checkpoint_path)

        training_time = time.perf_counter() - start_time

        mlflow.log_metrics({
            "best_val_macro_f1": best_f1,
            "best_epoch": best_epoch,
            "training_seconds": training_time,
        })

        mlflow.log_artifact(
            str(checkpoint_path),
            artifact_path="checkpoints",
        )

        print(f"\nSeed {seed} completed")
        print(f"Best epoch: {best_epoch}")
        print(f"Best validation macro-F1: {best_f1:.4f}")
        print(f"Training time: {training_time:.1f} seconds\n")

if __name__ == "__main__":

    print(f"Training device: {DEVICE}")

    if DEVICE.type == "cpu":
        print("WARNING: CUDA unavailable. Training will use CPU.")

    for seed in SEEDS:
        run_experiment(seed)

    print("All baseline CNN experiments completed.")
