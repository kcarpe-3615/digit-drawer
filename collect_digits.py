from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk
from uuid import uuid4

import numpy as np
from PIL import Image, ImageDraw


DATA_DIRECTORY = Path("personal_digits")

CANVAS_SIZE = 280
BRUSH_SIZE = 22
MNIST_SIZE = 28
DIGIT_SIZE = 20


class DigitCollectorApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Digit Training Data Collector")
        self.root.resizable(False, False)

        self.last_x: int | None = None
        self.last_y: int | None = None

        # The currently selected correct label.
        self.selected_label = tk.IntVar(value=0)

        self.create_directories()
        self.create_widgets()
        self.reset_hidden_image()
        self.update_count_label()

    @staticmethod
    def create_directories() -> None:
        """Create personal_digits/0 through personal_digits/9."""
        for digit in range(10):
            (DATA_DIRECTORY / str(digit)).mkdir(
                parents=True,
                exist_ok=True,
            )

    def create_widgets(self) -> None:
        self.canvas = tk.Canvas(
            self.root,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            background="black",
            cursor="crosshair",
            highlightthickness=1,
        )
        self.canvas.grid(
            row=0,
            column=0,
            columnspan=10,
            padx=12,
            pady=12,
        )

        instruction_label = ttk.Label(
            self.root,
            text="Choose the correct label, draw it, then save",
            font=("Arial", 14),
        )
        instruction_label.grid(
            row=1,
            column=0,
            columnspan=10,
            pady=(0, 8),
        )

        # One radio button for each possible digit label.
        for digit in range(10):
            button = ttk.Radiobutton(
                self.root,
                text=str(digit),
                value=digit,
                variable=self.selected_label,
            )
            button.grid(
                row=2,
                column=digit,
                padx=5,
                pady=5,
            )

        self.save_button = ttk.Button(
            self.root,
            text="Save Drawing",
            command=self.save_drawing,
        )
        self.save_button.grid(
            row=3,
            column=0,
            columnspan=4,
            padx=8,
            pady=10,
            sticky="ew",
        )

        self.clear_button = ttk.Button(
            self.root,
            text="Clear",
            command=self.clear_canvas,
        )
        self.clear_button.grid(
            row=3,
            column=4,
            columnspan=3,
            padx=8,
            pady=10,
            sticky="ew",
        )

        self.quit_button = ttk.Button(
            self.root,
            text="Quit",
            command=self.root.destroy,
        )
        self.quit_button.grid(
            row=3,
            column=7,
            columnspan=3,
            padx=8,
            pady=10,
            sticky="ew",
        )

        self.status_label = ttk.Label(
            self.root,
            text="",
            font=("Arial", 11),
        )
        self.status_label.grid(
            row=4,
            column=0,
            columnspan=10,
            pady=(0, 12),
        )

        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)

        # Pressing a number key changes the selected label.
        for digit in range(10):
            self.root.bind(
                str(digit),
                lambda _event, value=digit:
                    self.selected_label.set(value),
            )

        # Useful keyboard shortcuts.
        self.root.bind("<Return>", lambda _event: self.save_drawing())
        self.root.bind("<space>", lambda _event: self.clear_canvas())

    def reset_hidden_image(self) -> None:
        """Create the Pillow image that mirrors the visible canvas."""
        self.image = Image.new(
            mode="L",
            size=(CANVAS_SIZE, CANVAS_SIZE),
            color=0,
        )
        self.image_draw = ImageDraw.Draw(self.image)

    def start_drawing(self, event: tk.Event) -> None:
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
        self.last_x = None
        self.last_y = None

    @staticmethod
    def center_by_mass(image: Image.Image) -> Image.Image:
        """
        Shift an image into a new blank canvas so its pixel mass
        is centered, without wrapping pixels around the edges.
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

        shifted = Image.new(
            mode="L",
            size=image.size,
            color=0,
        )

        shifted.paste(
            image,
            (shift_x, shift_y),
        )

        return shifted

    def process_drawing(self) -> Image.Image:
        """
        Convert the 280x280 drawing into an MNIST-like 28x28 image.

        The returned image is not normalized. Normalization should
        happen later in the PyTorch training transform.
        """
        bounding_box = self.image.getbbox()

        if bounding_box is None:
            raise ValueError("Draw a digit before saving.")

        cropped = self.image.crop(bounding_box)

        width, height = cropped.size
        scale = DIGIT_SIZE / max(width, height)

        resized_width = max(
            1,
            round(width * scale),
        )
        resized_height = max(
            1,
            round(height * scale),
        )

        resized = cropped.resize(
            (resized_width, resized_height),
            Image.Resampling.LANCZOS,
        )

        processed = Image.new(
            mode="L",
            size=(MNIST_SIZE, MNIST_SIZE),
            color=0,
        )

        left = (MNIST_SIZE - resized_width) // 2
        top = (MNIST_SIZE - resized_height) // 2

        processed.paste(
            resized,
            (left, top),
        )

        return self.center_by_mass(processed)

    def save_drawing(self) -> None:
        try:
            processed = self.process_drawing()
        except ValueError as error:
            messagebox.showwarning(
                title="Nothing to save",
                message=str(error),
            )
            return

        label = self.selected_label.get()
        label_directory = DATA_DIRECTORY / str(label)

        filename = f"{uuid4().hex}.png"
        output_path = label_directory / filename

        processed.save(output_path)

        self.clear_canvas()
        self.update_count_label()

        self.status_label.config(
            text=f"Saved digit {label}"
        )

    def clear_canvas(self) -> None:
        self.canvas.delete("all")
        self.reset_hidden_image()

    def update_count_label(self) -> None:
        counts = []

        for digit in range(10):
            count = len(
                list((DATA_DIRECTORY / str(digit)).glob("*.png"))
            )
            counts.append(f"{digit}: {count}")

        self.status_label.config(
            text=" | ".join(counts)
        )


def main() -> None:
    root = tk.Tk()
    DigitCollectorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()