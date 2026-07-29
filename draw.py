from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
from PIL import Image, ImageChops, ImageDraw
import torch
from torch import nn

from model import DigitClassifier


MODEL_PATH = Path("digit_model.pth")

CANVAS_SIZE = 280
BRUSH_SIZE = 22
MNIST_SIZE = 28
DIGIT_SIZE = 20

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


class DigitDrawingApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Handwritten Digit Classifier")
        self.root.resizable(False, False)

        self.model = self.load_model()

        self.last_x: int | None = None
        self.last_y: int | None = None

        self.canvas = tk.Canvas(
            root,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            background="black",
            cursor="crosshair",
            highlightthickness=1,
        )
        self.canvas.grid(
            row=0,
            column=0,
            columnspan=3,
            padx=12,
            pady=12,
        )

        self.result_label = ttk.Label(
            root,
            text="Draw a digit, then press Predict",
            font=("Arial", 16),
        )
        self.result_label.grid(
            row=1,
            column=0,
            columnspan=3,
            padx=12,
            pady=(0, 10),
        )

        self.predict_button = ttk.Button(
            root,
            text="Predict",
            command=self.predict,
        )
        self.predict_button.grid(
            row=2,
            column=0,
            padx=8,
            pady=(0, 12),
        )

        self.clear_button = ttk.Button(
            root,
            text="Clear",
            command=self.clear_canvas,
        )
        self.clear_button.grid(
            row=2,
            column=1,
            padx=8,
            pady=(0, 12),
        )

        self.quit_button = ttk.Button(
            root,
            text="Quit",
            command=root.destroy,
        )
        self.quit_button.grid(
            row=2,
            column=2,
            padx=8,
            pady=(0, 12),
        )

        # Hidden grayscale image containing the same drawing as the canvas.
        self.image = Image.new(
            mode="L",
            size=(CANVAS_SIZE, CANVAS_SIZE),
            color=0,
        )
        self.image_draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)

    @staticmethod
    def load_model() -> nn.Module:
        """Load the trained model from disk."""
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"{MODEL_PATH} was not found. Run train.py first."
            )

        model = DigitClassifier()

        state_dict = torch.load(
            MODEL_PATH,
            map_location="cpu",
            weights_only=True,
        )

        model.load_state_dict(state_dict)
        model.eval()

        return model

    def start_drawing(self, event: tk.Event) -> None:
        """Start a new brush stroke."""
        self.last_x = event.x
        self.last_y = event.y

        radius = BRUSH_SIZE // 2

        bounds = (
            event.x - radius,
            event.y - radius,
            event.x + radius,
            event.y + radius,
        )

        self.canvas.create_oval(
            *bounds,
            fill="white",
            outline="white",
        )

        self.image_draw.ellipse(
            bounds,
            fill=255,
        )

    def draw(self, event: tk.Event) -> None:
        """Continue drawing while the left mouse button is held."""
        if self.last_x is None or self.last_y is None:
            return

        points = (
            self.last_x,
            self.last_y,
            event.x,
            event.y,
        )

        self.canvas.create_line(
            *points,
            fill="white",
            width=BRUSH_SIZE,
            capstyle=tk.ROUND,
            smooth=True,
        )

        self.image_draw.line(
            points,
            fill=255,
            width=BRUSH_SIZE,
        )

        # Add a circle at the endpoint so Pillow strokes stay rounded.
        radius = BRUSH_SIZE // 2

        self.image_draw.ellipse(
            (
                event.x - radius,
                event.y - radius,
                event.x + radius,
                event.y + radius,
            ),
            fill=255,
        )

        self.last_x = event.x
        self.last_y = event.y

    def stop_drawing(self, _event: tk.Event) -> None:
        """Finish the current brush stroke."""
        self.last_x = None
        self.last_y = None

    def clear_canvas(self) -> None:
        """Remove the visible and hidden drawings."""
        self.canvas.delete("all")

        self.image = Image.new(
            mode="L",
            size=(CANVAS_SIZE, CANVAS_SIZE),
            color=0,
        )
        self.image_draw = ImageDraw.Draw(self.image)

        self.result_label.config(
            text="Draw a digit, then press Predict"
        )

    @staticmethod
    def center_by_mass(image: Image.Image) -> Image.Image:
        """
        Shift a 28x28 grayscale image so its center of mass
        is approximately at the center.
        """
        array = np.asarray(
            image,
            dtype=np.float32,
        )

        total = array.sum()

        if total == 0:
            return image

        rows, columns = np.indices(array.shape)

        center_y = float(
            (rows * array).sum() / total
        )
        center_x = float(
            (columns * array).sum() / total
        )

        target = (MNIST_SIZE - 1) / 2

        shift_x = round(target - center_x)
        shift_y = round(target - center_y)

        return ImageChops.offset(
            image,
            shift_x,
            shift_y,
        )

    def preprocess_image(self) -> torch.Tensor:
        """
        Convert the drawing into a normalized tensor with shape
        (1, 1, 28, 28).
        """
        bounding_box = self.image.getbbox()

        if bounding_box is None:
            raise ValueError("The canvas is empty.")

        cropped = self.image.crop(bounding_box)

        width, height = cropped.size

        # Resize while preserving the digit's aspect ratio.
        scale = DIGIT_SIZE / max(width, height)

        resized_width = max(
            1,
            round(width * scale),
        )
        resized_height = max(
            1,
            round(height * scale),
        )

        cropped = cropped.resize(
            (resized_width, resized_height),
            Image.Resampling.LANCZOS,
        )

        centered = Image.new(
            mode="L",
            size=(MNIST_SIZE, MNIST_SIZE),
            color=0,
        )

        left = (MNIST_SIZE - resized_width) // 2
        top = (MNIST_SIZE - resized_height) // 2

        centered.paste(
            cropped,
            (left, top),
        )

        centered = self.center_by_mass(centered)

        pixel_array = np.asarray(
            centered,
            dtype=np.float32,
        ) / 255.0

        # Apply the same normalization used during training.
        pixel_array = (
            pixel_array - MNIST_MEAN
        ) / MNIST_STD

        tensor = torch.from_numpy(pixel_array)

        # (28, 28) -> (1, 28, 28) -> (1, 1, 28, 28)
        tensor = tensor.unsqueeze(0).unsqueeze(0)

        return tensor

    def predict(self) -> None:
        """Run the drawing through the model and show the top 3 results."""
        try:
            image_tensor = self.preprocess_image()
        except ValueError as error:
            messagebox.showwarning(
                title="Nothing to predict",
                message=str(error),
            )
            return

        with torch.inference_mode():
            logits = self.model(image_tensor)
            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            top_probabilities, top_classes = torch.topk(
                probabilities,
                k=3,
                dim=1,
            )

        results = []

        for probability, digit in zip(
            top_probabilities[0],
            top_classes[0],
        ):
            results.append(
                f"{digit.item()}: "
                f"{probability.item() * 100:.1f}%"
            )

        self.result_label.config(
            text=" | ".join(results)
        )


def main() -> None:
    root = tk.Tk()

    try:
        DigitDrawingApp(root)
    except (FileNotFoundError, RuntimeError) as error:
        messagebox.showerror(
            title="Unable to load model",
            message=str(error),
        )
        root.destroy()
        return

    root.mainloop()


if __name__ == "__main__":
    main()