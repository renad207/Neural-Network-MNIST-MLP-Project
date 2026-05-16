# Handwritten Digit Recognition using MLP

## Problem Description

This project implements a Multilayer Perceptron (MLP) model for handwritten digit recognition using the MNIST dataset.

Two experiments were conducted using different optimizers:
- SGD
- Adam

The dataset was divided into:
- Training Set
- Validation Set
- Testing Set

The model performance was evaluated using:
- Accuracy
- Loss

Training, validation, and testing curves were visualized for comparison.

---

## Dataset Link

MNIST Dataset:
https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html

---

## Experiments

### Experiment 1
Optimizer: SGD

### Experiment 2
Optimizer: Adam

---

## Results Comparison

| Optimizer | Final Test Accuracy | Final Test Loss  |
|-----------|---------------------|------------------|
| SGD       | 96.23               | 0.1268           |
| Adam      | 97.72               | 0.1022           |


---

## Technologies Used

- Python
- PyTorch
- Matplotlib

---

## Instructions for Running the Project

1. Install Python
2. Install required libraries:

```bash
pip install torch torchvision matplotlib
```

3. Run the Python file:

```bash
python mnist_mlp_project.py
```

---

## Author
Renad Amr Ahmed
