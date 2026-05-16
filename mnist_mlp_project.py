# =========================================
# Neural Networks Project
# Handwritten Digit Recognition using MLP
# MNIST Dataset
# =========================================

# =============== IMPORT LIBRARIES ===============

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

import matplotlib.pyplot as plt

# =============== DEVICE ===============

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

# =============== DATA PREPROCESSING ===============

transform = transforms.Compose([
    transforms.ToTensor(),

    # Normalize Data
    transforms.Normalize((0.1307,), (0.3081,))
])

# =============== LOAD DATASET ===============

train_data_full = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_data = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# =============== SPLIT DATA ===============
# 80% Train
# 20% Validation

train_size = int(0.8 * len(train_data_full))
val_size = len(train_data_full) - train_size

train_data, val_data = random_split(
    train_data_full,
    [train_size, val_size]
)

# =============== DATALOADERS ===============

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# =========================================
# EXPERIMENT 1
# Optimizer = SGD
# =========================================

print("\n================================")
print("EXPERIMENT 1 : SGD")
print("================================")

# =============== MODEL ===============

class MLP_SGD(nn.Module):

    def __init__(self):
        super(MLP_SGD, self).__init__()

        self.fc1 = nn.Linear(28*28, 256)

        self.fc2 = nn.Linear(256, 128)

        self.fc3 = nn.Linear(128, 10)

        self.relu = nn.ReLU()

    def forward(self, x):

        x = x.view(-1, 28*28)

        x = self.relu(self.fc1(x))

        x = self.relu(self.fc2(x))

        x = self.fc3(x)

        return x

# Create Model
model_sgd = MLP_SGD().to(device)

# Loss Function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer_sgd = optim.SGD(
    model_sgd.parameters(),
    lr=0.01
)

# Lists for Curves
sgd_train_loss = []
sgd_val_loss = []
sgd_test_loss = []

sgd_train_acc = []
sgd_val_acc = []
sgd_test_acc = []

epochs = 10

# =============== TRAINING LOOP ===============

for epoch in range(epochs):

    # ================= TRAIN =================

    model_sgd.train()

    correct = 0
    total = 0
    running_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model_sgd(images)

        loss = criterion(outputs, labels)

        optimizer_sgd.zero_grad()

        loss.backward()

        optimizer_sgd.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)

    train_accuracy = 100 * correct / total

    sgd_train_loss.append(train_loss)

    sgd_train_acc.append(train_accuracy)

    # ================= VALIDATION =================

    model_sgd.eval()

    correct = 0
    total = 0
    running_loss = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model_sgd(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    val_loss = running_loss / len(val_loader)

    val_accuracy = 100 * correct / total

    sgd_val_loss.append(val_loss)

    sgd_val_acc.append(val_accuracy)

    # ================= TEST =================

    correct = 0
    total = 0
    running_loss = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model_sgd(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    test_loss = running_loss / len(test_loader)

    test_accuracy = 100 * correct / total

    sgd_test_loss.append(test_loss)

    sgd_test_acc.append(test_accuracy)

    # ================= PRINT =================

    print(f"\nEpoch {epoch+1}/{epochs}")

    print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.2f}%")

    print(f"Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_accuracy:.2f}%")

    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.2f}%")

# =========================================
# EXPERIMENT 2
# Optimizer = Adam
# =========================================

print("\n================================")
print("EXPERIMENT 2 : Adam")
print("================================")

# =============== MODEL ===============

class MLP_Adam(nn.Module):

    def __init__(self):
        super(MLP_Adam, self).__init__()

        self.fc1 = nn.Linear(28*28, 256)

        self.fc2 = nn.Linear(256, 128)

        self.fc3 = nn.Linear(128, 10)

        self.relu = nn.ReLU()

    def forward(self, x):

        x = x.view(-1, 28*28)

        x = self.relu(self.fc1(x))

        x = self.relu(self.fc2(x))

        x = self.fc3(x)

        return x

# Create Model
model_adam = MLP_Adam().to(device)

# Optimizer
optimizer_adam = optim.Adam(
    model_adam.parameters(),
    lr=0.001
)

# Lists for Curves
adam_train_loss = []
adam_val_loss = []
adam_test_loss = []

adam_train_acc = []
adam_val_acc = []
adam_test_acc = []

# =============== TRAINING LOOP ===============

for epoch in range(epochs):

    # ================= TRAIN =================

    model_adam.train()

    correct = 0
    total = 0
    running_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model_adam(images)

        loss = criterion(outputs, labels)

        optimizer_adam.zero_grad()

        loss.backward()

        optimizer_adam.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)

    train_accuracy = 100 * correct / total

    adam_train_loss.append(train_loss)

    adam_train_acc.append(train_accuracy)

    # ================= VALIDATION =================

    model_adam.eval()

    correct = 0
    total = 0
    running_loss = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model_adam(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    val_loss = running_loss / len(val_loader)

    val_accuracy = 100 * correct / total

    adam_val_loss.append(val_loss)

    adam_val_acc.append(val_accuracy)

    # ================= TEST =================

    correct = 0
    total = 0
    running_loss = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model_adam(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    test_loss = running_loss / len(test_loader)

    test_accuracy = 100 * correct / total

    adam_test_loss.append(test_loss)

    adam_test_acc.append(test_accuracy)

    # ================= PRINT =================

    print(f"\nEpoch {epoch+1}/{epochs}")

    print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.2f}%")

    print(f"Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_accuracy:.2f}%")

    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.2f}%")

# =========================================
# VISUALIZATION - SGD
# =========================================

epochs_range = range(1, epochs + 1)

plt.figure(figsize=(12,5))

# Loss Curve
plt.subplot(1,2,1)

plt.plot(epochs_range, sgd_train_loss, label="Train")
plt.plot(epochs_range, sgd_val_loss, label="Validation")
plt.plot(epochs_range, sgd_test_loss, label="Test")

plt.title("SGD Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# Accuracy Curve
plt.subplot(1,2,2)

plt.plot(epochs_range, sgd_train_acc, label="Train")
plt.plot(epochs_range, sgd_val_acc, label="Validation")
plt.plot(epochs_range, sgd_test_acc, label="Test")

plt.title("SGD Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.show()

# =========================================
# VISUALIZATION - Adam
# =========================================

plt.figure(figsize=(12,5))

# Loss Curve
plt.subplot(1,2,1)

plt.plot(epochs_range, adam_train_loss, label="Train")
plt.plot(epochs_range, adam_val_loss, label="Validation")
plt.plot(epochs_range, adam_test_loss, label="Test")

plt.title("Adam Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# Accuracy Curve
plt.subplot(1,2,2)

plt.plot(epochs_range, adam_train_acc, label="Train")
plt.plot(epochs_range, adam_val_acc, label="Validation")
plt.plot(epochs_range, adam_test_acc, label="Test")

plt.title("Adam Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.show()

# =========================================
# FINAL RESULTS
# =========================================

print("\n================================")
print("FINAL RESULTS")
print("================================")

print("SGD Final Test Accuracy:",
      round(sgd_test_acc[-1], 2), "%")

print("Adam Final Test Accuracy:",
      round(adam_test_acc[-1], 2), "%")


# =========================================
# COMPARISON BETWEEN SGD AND ADAM
# =========================================

plt.figure(figsize=(12,5))

# ========= LOSS COMPARISON =========

plt.subplot(1,2,1)

plt.plot(epochs_range, sgd_test_loss, label="SGD Test Loss")

plt.plot(epochs_range, adam_test_loss, label="Adam Test Loss")

plt.title("SGD vs Adam Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

# ========= ACCURACY COMPARISON =========

plt.subplot(1,2,2)

plt.plot(epochs_range, sgd_test_acc, label="SGD Test Accuracy")

plt.plot(epochs_range, adam_test_acc, label="Adam Test Accuracy")

plt.title("SGD vs Adam Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.show()


# =========================================
# RESULTS TABLE
# =========================================

import pandas as pd

results = pd.DataFrame({

    "Model": ["MLP + SGD", "MLP + Adam"],

    "Final Test Accuracy": [
        round(sgd_test_acc[-1], 2),
        round(adam_test_acc[-1], 2)
    ],

    "Final Test Loss": [
        round(sgd_test_loss[-1], 4),
        round(adam_test_loss[-1], 4)
    ]
})

print("\n================================")
print("RESULTS TABLE")
print("================================")

print(results)