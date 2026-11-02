
import torch
import torch.nn as nn
import torch.optim as optim

from model import DigitCNN
from preprocessing import train_loader, test_loader

# Select device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Initialize model
model = DigitCNN().to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
epochs = 5

for epoch in range(epochs):

    model.train()
    running_loss = 0.0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        # Forward propagation
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backward propagation
        optimizer.zero_grad()
        loss.backward()

        # Update weights
        optimizer.step()

        running_loss += loss.item()

    print(
        f"Epoch [{epoch+1}/{epochs}], "
        f"Loss: {running_loss/len(train_loader):.4f}"
    )

# Testing
model.eval()
correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total

print(f"Test Accuracy: {accuracy:.2f}%")

# Save trained model
torch.save(model.state_dict(), "digit_cnn.pth")

print("Model saved successfully!")
