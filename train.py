from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import InterpolationMode

from model import DigitClassifier


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

MODEL_PATH = Path("digit_model.pth")
PERSONAL_DATA_PATH = Path("personal_digits")
PERSONAL_VALIDATION_PATH = Path("personal_validation")


# ---------------------------------------------------------------------
# Training settings
# ---------------------------------------------------------------------

BATCH_SIZE = 64
EPOCHS = 15

# Use a smaller learning rate when loading an existing trained model.
NEW_MODEL_LEARNING_RATE = 0.001
FINE_TUNE_LEARNING_RATE = 0.0003

WEIGHT_DECAY = 1e-4

# Repeating a Dataset reference does not copy its images on disk.
# It only makes personal samples appear more frequently each epoch.
PERSONAL_DATA_REPEATS = 15

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

EXPECTED_CLASS_MAPPING = {
    str(digit): digit
    for digit in range(10)
}


def get_device() -> torch.device:
    """Use CUDA or Apple MPS when available, otherwise use the CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def directory_contains_images(directory: Path) -> bool:
    """Return True when a directory contains at least one image."""
    if not directory.exists():
        return False

    supported_extensions = {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff",
    }

    return any(
        path.is_file() and path.suffix.lower() in supported_extensions
        for path in directory.rglob("*")
    )


def verify_class_mapping(dataset: ImageFolder, dataset_name: str) -> None:
    """
    Verify that folder names 0 through 9 map to integer labels 0 through 9.
    """
    if dataset.class_to_idx != EXPECTED_CLASS_MAPPING:
        raise ValueError(
            f"{dataset_name} has an unexpected class mapping.\n"
            f"Expected: {EXPECTED_CLASS_MAPPING}\n"
            f"Found:    {dataset.class_to_idx}\n\n"
            "Make sure the dataset contains folders named exactly "
            "0, 1, 2, 3, 4, 5, 6, 7, 8, and 9."
        )


def create_transforms() -> tuple[
    transforms.Compose,
    transforms.Compose,
    transforms.Compose,
]:
    """
    Return transforms for:

    1. MNIST training data
    2. Personal training data
    3. Validation and testing data
    """

    mnist_training_transform = transforms.Compose([
        transforms.RandomAffine(
            degrees=8,
            translate=(0.08, 0.08),
            scale=(0.92, 1.08),
            shear=4,
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(MNIST_MEAN,),
            std=(MNIST_STD,),
        ),
    ])

    # Personal images already resemble the drawing application,
    # so use less aggressive augmentation.
    personal_training_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomAffine(
            degrees=4,
            translate=(0.04, 0.04),
            scale=(0.96, 1.04),
            shear=2,
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(MNIST_MEAN,),
            std=(MNIST_STD,),
        ),
    ])

    evaluation_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(MNIST_MEAN,),
            std=(MNIST_STD,),
        ),
    ])

    return (
        mnist_training_transform,
        personal_training_transform,
        evaluation_transform,
    )


def create_datasets() -> tuple[
    Dataset,
    Dataset,
    Optional[Dataset],
]:
    """
    Create the combined training dataset, MNIST test dataset,
    and optional personal validation dataset.
    """
    (
        mnist_training_transform,
        personal_training_transform,
        evaluation_transform,
    ) = create_transforms()

    mnist_training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=mnist_training_transform,
    )

    mnist_test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=evaluation_transform,
    )

    training_datasets: list[Dataset] = [
        mnist_training_data
    ]

    print(f"MNIST training images: {len(mnist_training_data)}")
    print(f"MNIST test images:     {len(mnist_test_data)}")

    if directory_contains_images(PERSONAL_DATA_PATH):
        personal_training_data = ImageFolder(
            root=PERSONAL_DATA_PATH,
            transform=personal_training_transform,
        )

        verify_class_mapping(
            personal_training_data,
            "Personal training dataset",
        )

        print(
            f"Personal training images: "
            f"{len(personal_training_data)}"
        )
        print(
            f"Personal repeat factor:   "
            f"{PERSONAL_DATA_REPEATS}"
        )

        # Each repeat increases how often personal images appear relative
        # to the much larger MNIST dataset.
        training_datasets.extend(
            [personal_training_data] * PERSONAL_DATA_REPEATS
        )
    else:
        print(
            "\nNo personal training images were found."
            f"\nExpected them under: {PERSONAL_DATA_PATH.resolve()}"
            "\nTraining with MNIST only."
        )

    combined_training_data = ConcatDataset(
        training_datasets
    )

    personal_validation_data: Optional[Dataset] = None

    if directory_contains_images(PERSONAL_VALIDATION_PATH):
        validation_dataset = ImageFolder(
            root=PERSONAL_VALIDATION_PATH,
            transform=evaluation_transform,
        )

        verify_class_mapping(
            validation_dataset,
            "Personal validation dataset",
        )

        personal_validation_data = validation_dataset

        print(
            f"Personal validation images: "
            f"{len(validation_dataset)}"
        )
    else:
        print(
            "\nNo separate personal validation data found."
            f"\nOptional location: "
            f"{PERSONAL_VALIDATION_PATH.resolve()}"
        )

    print(
        f"\nEffective training samples per epoch: "
        f"{len(combined_training_data)}"
    )

    return (
        combined_training_data,
        mnist_test_data,
        personal_validation_data,
    )


def create_data_loader(
    dataset: Dataset,
    device: torch.device,
    *,
    shuffle: bool,
) -> DataLoader:
    """Create a DataLoader with sensible device-dependent settings."""
    use_cuda = device.type == "cuda"

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=use_cuda,
    )


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
            images = images.to(
                device,
                non_blocking=device.type == "cuda",
            )
            labels = labels.to(
                device,
                non_blocking=device.type == "cuda",
            )

            logits = model(images)
            predictions = logits.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    if total == 0:
        return 0.0

    return correct / total


def build_confusion_matrix(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> torch.Tensor:
    """
    Create a 10x10 confusion matrix.

    Rows are actual labels.
    Columns are predicted labels.
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
            labels = labels.to(device)

            predictions = model(images).argmax(dim=1)

            # Convert each actual/predicted pair into one index:
            #
            # actual 4, predicted 9:
            # 4 * 10 + 9 = 49
            pair_indices = labels * 10 + predictions

            batch_counts = torch.bincount(
                pair_indices,
                minlength=100,
            )

            batch_matrix = batch_counts.reshape(
                10,
                10,
            )

            matrix += batch_matrix.cpu()

    return matrix


def print_confusion_matrix(
    matrix: torch.Tensor,
    title: str,
) -> None:
    """Print a labeled confusion matrix."""
    print(f"\n{title}")
    print("Rows = actual digit")
    print("Columns = predicted digit\n")

    header = "     " + " ".join(
        f"{digit:5d}"
        for digit in range(10)
    )
    print(header)
    print("-" * len(header))

    for actual_digit, row in enumerate(matrix):
        values = " ".join(
            f"{value.item():5d}"
            for value in row
        )

        print(f"{actual_digit:2d} | {values}")


def print_problem_digit_summary(
    matrix: torch.Tensor,
) -> None:
    """Print common confusions involving 2, 4, 7, and 9."""
    problem_digits = (2, 4, 7, 9)

    print("\nConfusions among 2, 4, 7, and 9")

    found_confusion = False

    for actual in problem_digits:
        for predicted in problem_digits:
            if actual == predicted:
                continue

            count = matrix[actual, predicted].item()

            if count > 0:
                found_confusion = True
                print(
                    f"Actual {actual} predicted as "
                    f"{predicted}: {count}"
                )

    if not found_confusion:
        print("No mistakes among these four classes.")


def load_or_create_model(
    device: torch.device,
) -> tuple[DigitClassifier, bool]:
    """
    Load existing compatible weights when possible.

    Return:
        model
        True when fine-tuning existing weights
    """
    model = DigitClassifier().to(device)

    if not MODEL_PATH.exists():
        print("\nNo existing model found. Starting a new model.")
        return model, False

    try:
        state_dict = torch.load(
            MODEL_PATH,
            map_location=device,
            weights_only=True,
        )

        model.load_state_dict(state_dict)

        print(
            f"\nLoaded existing model from "
            f"{MODEL_PATH.resolve()}"
        )
        print("Fine-tuning existing weights.")

        return model, True

    except RuntimeError as error:
        print(
            "\nThe existing model file is incompatible with "
            "the current architecture."
        )
        print(
            "A new model will be trained instead."
        )
        print(f"Details: {error}")

        return model, False


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch and return the average batch loss."""
    model.train()

    running_loss = 0.0

    for images, labels in data_loader:
        images = images.to(
            device,
            non_blocking=device.type == "cuda",
        )
        labels = labels.to(
            device,
            non_blocking=device.type == "cuda",
        )

        optimizer.zero_grad()

        logits = model(images)
        loss = loss_function(logits, labels)

        loss.backward()

        # Prevent unusually large gradient updates.
        nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )

        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(data_loader)


def save_model(
    model: nn.Module,
    output_path: Path,
) -> None:
    """Save model weights in a CPU-compatible form."""
    cpu_state_dict = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
    }

    torch.save(
        cpu_state_dict,
        output_path,
    )


def main() -> None:
    # Makes repeated runs somewhat easier to compare.
    torch.manual_seed(42)

    device = get_device()
    print(f"Using device: {device}")

    (
        training_data,
        mnist_test_data,
        personal_validation_data,
    ) = create_datasets()

    training_loader = create_data_loader(
        training_data,
        device,
        shuffle=True,
    )

    mnist_test_loader = create_data_loader(
        mnist_test_data,
        device,
        shuffle=False,
    )

    personal_validation_loader: Optional[DataLoader] = None

    if personal_validation_data is not None:
        personal_validation_loader = create_data_loader(
            personal_validation_data,
            device,
            shuffle=False,
        )

    model, is_fine_tuning = load_or_create_model(
        device
    )

    learning_rate = (
        FINE_TUNE_LEARNING_RATE
        if is_fine_tuning
        else NEW_MODEL_LEARNING_RATE
    )

    print(f"Learning rate: {learning_rate}")

    loss_function = nn.CrossEntropyLoss(
        label_smoothing=0.05,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    best_score = -1.0
    best_epoch = 0

    print("\nStarting training\n")

    for epoch in range(EPOCHS):
        average_loss = train_one_epoch(
            model,
            training_loader,
            loss_function,
            optimizer,
            device,
        )

        mnist_accuracy = evaluate(
            model,
            mnist_test_loader,
            device,
        )

        personal_accuracy: Optional[float] = None

        if personal_validation_loader is not None:
            personal_accuracy = evaluate(
                model,
                personal_validation_loader,
                device,
            )

            # Personal performance is the main objective for the app.
            score_for_saving = personal_accuracy
        else:
            score_for_saving = mnist_accuracy

        scheduler.step(score_for_saving)

        current_learning_rate = (
            optimizer.param_groups[0]["lr"]
        )

        message = (
            f"Epoch {epoch + 1:2d}/{EPOCHS} | "
            f"Loss: {average_loss:.4f} | "
            f"MNIST: {mnist_accuracy:.2%}"
        )

        if personal_accuracy is not None:
            message += (
                f" | Personal: {personal_accuracy:.2%}"
            )

        message += (
            f" | LR: {current_learning_rate:.6f}"
        )

        print(message)

        if score_for_saving > best_score:
            best_score = score_for_saving
            best_epoch = epoch + 1

            save_model(
                model,
                MODEL_PATH,
            )

            print(
                f"  Saved new best model "
                f"(score: {best_score:.2%})"
            )

    print(
        f"\nBest model came from epoch {best_epoch} "
        f"with score {best_score:.2%}."
    )
    print(f"Saved to: {MODEL_PATH.resolve()}")

    # Reload the best model before producing final reports.
    best_state_dict = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(best_state_dict)

    mnist_matrix = build_confusion_matrix(
        model,
        mnist_test_loader,
        device,
    )

    print_confusion_matrix(
        mnist_matrix,
        "MNIST confusion matrix",
    )

    print_problem_digit_summary(
        mnist_matrix
    )

    if personal_validation_loader is not None:
        personal_matrix = build_confusion_matrix(
            model,
            personal_validation_loader,
            device,
        )

        print_confusion_matrix(
            personal_matrix,
            "Personal validation confusion matrix",
        )

        print_problem_digit_summary(
            personal_matrix
        )


if __name__ == "__main__":
    main()