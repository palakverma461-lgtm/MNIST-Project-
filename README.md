<<<<<<< HEAD

# Handwritten Digit Recognition Using CNN

## Project Overview

This project implements a Handwritten Digit Recognition System
using Convolutional Neural Networks (CNN) and PyTorch.

The system recognizes handwritten digits from 0 to 9.
Users can draw digits on an interactive Tkinter canvas,
and the trained CNN model predicts the digit.

## Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Pillow (PIL)
- Tkinter
- MNIST Dataset

## Dataset

The project uses the MNIST dataset containing:

- 60,000 training images
- 10,000 testing images
- 28x28 grayscale images
- 10 digit classes (0-9)

## Project Phases

### Phase 1: Data Acquisition and Preprocessing

- Downloading the MNIST dataset.
- Converting images into PyTorch tensors.
- Normalizing images.
- Preparing training and testing DataLoaders.

### Phase 2: Building and Training the CNN Model

- Implementing convolutional layers.
- Applying ReLU activation.
- Using Max Pooling for downsampling.
- Flattening extracted features.
- Implementing fully connected layers.
- Training using Adam optimizer.
- Calculating loss using Cross Entropy Loss.
- Saving trained weights in digit_cnn.pth.

### Phase 3: Interactive GUI Development

- Creating a drawing canvas using Tkinter.
- Synchronizing the canvas with a PIL image buffer.
- Resizing drawings to 28x28 pixels.
- Applying normalization.
- Predicting digits using the trained CNN model.

## Installation

Clone the repository:

git clone YOUR_GITHUB_REPOSITORY_URL

Navigate to the project folder:

cd MNIST_Digit_Recognition

Install dependencies:

pip install -r requirements.txt

## Run the Project

First, train the model:

python train.py

After training, launch the GUI:

python app.py

## Model Architecture

- Input: 28x28 grayscale image
- Convolutional Layer 1: 16 filters
- Convolutional Layer 2: 32 filters
- Max Pooling Layers
- Fully Connected Layer: 128 neurons
- Output Layer: 10 neurons

## Output

The application displays the predicted handwritten
digit from 0 to 9.

## Conclusion

This project demonstrates the complete machine learning
pipeline, including data preprocessing, CNN model development,
model training, evaluation, and real-time handwritten
digit prediction through an interactive graphical interface.
=======
# MNIST-Project-
>>>>>>> fd4a431e12c7ea318be48d6bea32d195e41730e7
