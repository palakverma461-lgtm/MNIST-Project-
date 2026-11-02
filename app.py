
import tkinter as tk
import torch
import torch.nn.functional as F

from PIL import Image, ImageDraw
from model import DigitCNN
from torchvision import transforms

# Device selection
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Load trained CNN model
model = DigitCNN().to(device)
model.load_state_dict(
    torch.load("digit_cnn.pth", map_location=device)
)
model.eval()

# MNIST normalization
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Create GUI
root = tk.Tk()
root.title("Handwritten Digit Recognition")
root.geometry("500x600")

tk.Label(
    root,
    text="Handwritten Digit Recognition",
    font=("Arial", 20, "bold")
).pack(pady=15)

tk.Label(
    root,
    text="Draw a Digit (0-9)",
    font=("Arial", 15)
).pack(pady=10)

# Drawing canvas
canvas = tk.Canvas(
    root,
    width=280,
    height=280,
    bg="black",
    cursor="pencil"
)
canvas.pack(pady=10)

# PIL image buffer
image = Image.new("L", (280, 280), 0)
draw = ImageDraw.Draw(image)

previous_x = None
previous_y = None

# Smooth drawing function
def paint(event):

    global previous_x, previous_y

    x, y = event.x, event.y

    if previous_x is not None:

        canvas.create_line(
            previous_x, previous_y, x, y,
            fill="white",
            width=16,
            capstyle=tk.ROUND,
            smooth=True
        )

        draw.line(
            (previous_x, previous_y, x, y),
            fill=255,
            width=16
        )

    else:

        canvas.create_oval(
            x-8, y-8, x+8, y+8,
            fill="white",
            outline="white"
        )

        draw.ellipse(
            (x-8, y-8, x+8, y+8),
            fill=255
        )

    previous_x = x
    previous_y = y


def reset_position(event):

    global previous_x, previous_y

    previous_x = None
    previous_y = None


canvas.bind("<B1-Motion>", paint)
canvas.bind("<ButtonRelease-1>", reset_position)


# Image preprocessing function
def preprocess_image(img):

    # Find digit boundaries
    bbox = img.getbbox()

    if bbox is None:
        return None

    # Crop unnecessary blank space
    cropped = img.crop(bbox)

    width, height = cropped.size

    # Reject an isolated horizontal line
    if width > height * 4 and height < 80:
        return None

    # Make image square with padding
    size = max(width, height) + 20

    square = Image.new("L", (size, size), 0)

    x = (size - width) // 2
    y = (size - height) // 2

    square.paste(cropped, (x, y))

    # Resize digit to 20x20
    resized = square.resize(
        (20, 20),
        Image.Resampling.LANCZOS
    )

    # Create final 28x28 image
    final_image = Image.new("L", (28, 28), 0)

    final_image.paste(resized, (4, 4))

    return final_image


# Prediction function
def predict():

    processed = preprocess_image(image)

    if processed is None:
        result_label.config(
            text="Please draw a valid digit!"
        )
        return

    # Convert image into tensor
    img_tensor = transform(processed)

    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # CNN prediction
    with torch.no_grad():

        output = model(img_tensor)

        probabilities = F.softmax(output, dim=1)

        confidence, prediction = torch.max(
            probabilities, dim=1
        )

        digit = prediction.item()
        score = confidence.item() * 100

    # Display prediction
    if score < 55:

        result_label.config(
            text=f"Uncertain Prediction ({score:.1f}%)"
        )

    else:

        result_label.config(
            text=f"Predicted Digit: {digit}\n"
                 f"Confidence: {score:.2f}%"
        )


# Clear canvas
def clear():

    global previous_x, previous_y

    canvas.delete("all")

    draw.rectangle(
        (0, 0, 280, 280),
        fill=0
    )

    previous_x = None
    previous_y = None

    result_label.config(
        text="Draw a Digit (0-9)"
    )


# Buttons
tk.Button(
    root,
    text="Predict",
    command=predict,
    font=("Arial", 13),
    width=12
).pack(pady=10)

tk.Button(
    root,
    text="Clear",
    command=clear,
    font=("Arial", 13),
    width=12
).pack(pady=5)

# Result
result_label = tk.Label(
    root,
    text="Draw a Digit (0-9)",
    font=("Arial", 16, "bold")
)

result_label.pack(pady=15)

root.mainloop()
