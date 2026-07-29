from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

from model import DigitClassifier


MODEL_PATH = Path("digit_model.pth")
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001


def get_device() -> torch.device:
    """Use an available accelerator, otherwise use the CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> float:
    """Return classification accuracy as a value between 0 and 1."""
    model.eval()

    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            predictions = logits.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return correct / total


def build_confusion_matrix(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> torch.Tensor:
    """
    Build a 10x10 confusion matrix.

    Rows represent actual digits.
    Columns represent predicted digits.
    """
    model.eval()

    matrix = torch.zeros(
        10,
        10,
        dtype=torch.int64,
    )

    with torch.inference_mode():
        for images, labels in data_loader:
            images = images.to(device)

            logits = model(images)
            predictions = logits.argmax(dim=1).cpu()

            # Move labels to the CPU so they can index the CPU matrix.
            labels = labels.cpu()

            for actual, predicted in zip(labels, predictions):
                matrix[actual.item(), predicted.item()] += 1

    return matrix


def main() -> None:
    device = get_device()
    print(f"Using device: {device}")

    training_transform = transforms.Compose([
        transforms.RandomAffine(
            degrees=10,
            translate=(0.10, 0.10),
            scale=(0.90, 1.10),
            shear=5,
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.1307,),
            std=(0.3081,),
        ),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.1307,),
            std=(0.3081,),
        ),
    ])

    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=training_transform,
    )

    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=test_transform,
    )

    training_loader = DataLoader(
        training_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = DigitClassifier().to(device)

    loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    for epoch in range(EPOCHS):
        model.train()

        running_loss = 0.0

        for images, labels in training_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Clear gradients from the previous batch.
            optimizer.zero_grad()

            # Run the images through the network.
            logits = model(images)

            # Measure prediction error.
            loss = loss_function(logits, labels)

            # Calculate gradients.
            loss.backward()

            # Update weights and biases.
            optimizer.step()

            running_loss += loss.item()

        average_loss = running_loss / len(training_loader)
        accuracy = evaluate(model, test_loader, device)

        # Reduce the learning rate if accuracy stops improving.
        scheduler.step(accuracy)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Loss: {average_loss:.4f} | "
            f"Test accuracy: {accuracy:.2%} | "
            f"Learning rate: {current_lr:.6f}"
        )

    matrix = build_confusion_matrix(
        model,
        test_loader,
        device,
    )

    print("\nConfusion matrix")
    print("Rows = actual digit, columns = predicted digit")
    print(matrix)

    print("\nImportant confusions")
    print(f"Actual 2 predicted as 4: {matrix[2, 4].item()}")
    print(f"Actual 2 predicted as 7: {matrix[2, 7].item()}")
    print(f"Actual 4 predicted as 9: {matrix[4, 9].item()}")
    print(f"Actual 7 predicted as 2: {matrix[7, 2].item()}")
    print(f"Actual 7 predicted as 9: {matrix[7, 9].item()}")
    print(f"Actual 9 predicted as 4: {matrix[9, 4].item()}")

    # Move parameters to the CPU before saving.
    model = model.to("cpu")

    torch.save(
        model.state_dict(),
        MODEL_PATH,
    )

    print(f"\nSaved model to {MODEL_PATH.resolve()}")


if __name__ == "__main__":
    main()