"""
Utilities for training a Convolutional Neural Network (CNN) on the MNIST dataset with PyTorch.

Contents:
    get_args()   - command-line argument parser (argparse)
    ConvNet      - the CNN model
    MNISTData    - downloads MNIST and builds the train/test DataLoaders
    Trainer      - trains the model and tracks loss/accuracy
    plot()       - plots and saves the loss & accuracy curves

References:
    tensors:                    https://pytorch.org/docs/stable/tensors.html
    torch.nn:                   https://pytorch.org/docs/stable/nn.html
    torch.utils.data:           https://pytorch.org/docs/stable/data.html
    torch.optim:                https://pytorch.org/docs/stable/optim.html
    torchvision.datasets:       https://pytorch.org/vision/main/datasets.html
    torchvision.transforms:     https://pytorch.org/vision/stable/transforms.html
    argparse:                   https://docs.python.org/3/library/argparse.html
"""

import argparse

# Import the PyTorch library
# PyTorch is an open-source machine learning library used for applications such as computer vision & natural language processing
import torch

# Import the neural network module from PyTorch
# torch.nn provides classes & functions to build neural networks
import torch.nn as nn

# Import the optimization module from PyTorch
# torch.optim provides various optimization algorithms such as SGD, Adam, etc.
import torch.optim as optim

# Import the torchvision library
# torchvision provides tools to handle image & video data
import torchvision

# Import the transforms module from torchvision
# torchvision.transforms provides common image transformations for data augmentation & preprocessing
import torchvision.transforms as transforms

# Use a non-interactive backend so the figure can be saved from a terminal
# (e.g., a codespace or any machine without a display)
import matplotlib
matplotlib.use('Agg')

# Import the matplotlib library for plotting
# matplotlib.pyplot provides functions to create various types of plots & visualizations
import matplotlib.pyplot as plt

# Import the tqdm library for progress bars
# tqdm provides a fast, extensible progress bar for loops & tasks
from tqdm import tqdm


def get_args():
    """Parse the three training hyperparameters from the command line."""
    parser = argparse.ArgumentParser(
        description='Train a Convolutional Neural Network on the MNIST dataset with PyTorch.')

    parser.add_argument('-e', '--num_epochs', type=int, default=2,
                        help='Number of times the entire dataset is passed through the model (default: 2)')
    parser.add_argument('-b', '--batch_size', type=int, default=1000,
                        help='Number of samples per batch to be passed through the model (default: 1000)')
    parser.add_argument('-l', '--learning_rate', type=float, default=0.001,
                        help='Step size for parameter updates (default: 0.001)')

    return parser.parse_args()


# leveraged verbatim from Module 7, Lecture B, Section 3
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # First convolutional layer: 1 input channel (grayscale), 32 output channels, 5x5 kernel
        self.layer1 = nn.Sequential(
            #https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            #https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
            nn.BatchNorm2d(32),  # Batch normalization for faster convergence
            # https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
            nn.ReLU(),  # Activation function
            # https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
            nn.MaxPool2d(kernel_size=2, stride=2))  # Max pooling layer

        # Second convolutional layer: 32 input channels, 64 output channels, 5x5 kernel
        self.layer2 = nn.Sequential(
            #https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            #https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
            nn.BatchNorm2d(64),  # Batch normalization
            # https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
            nn.ReLU(),  # Activation function
            # https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
            nn.MaxPool2d(kernel_size=2, stride=2))  # Max pooling layer

        # Fully connected layer: input size 7*7*64, output size 1000
        self.fc1 = nn.Linear(7*7*64, 1000)
        # Fully connected layer: input size 1000, output size 10 (number of classes)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        # Forward pass through the first convolutional layer
        out = self.layer1(x)
        # Forward pass through the second convolutional layer
        out = self.layer2(out)
        # Flatten the output for the fully connected layer
        out = out.reshape(out.size(0), -1) # also look at https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html
        # Forward pass through the first fully connected layer
        out = self.fc1(out)
        # Forward pass through the second fully connected layer
        out = self.fc2(out)
        return out


class MNISTData:
    """Downloads MNIST and wraps the train/test sets in DataLoaders."""

    def __init__(self, batch_size, root='./data'):
        # MNIST dataset
        # Download & load the training dataset, applying a transformation to convert images to PyTorch tensors
        self.train_dataset = torchvision.datasets.MNIST(root=root,
                                                        train=True,
                                                        transform=transforms.ToTensor(),
                                                        download=True)

        # Download & load the test dataset, applying a transformation to convert images to PyTorch tensors
        # download=True here as well so the test split is fetched even if the folder only holds the train split
        self.test_dataset = torchvision.datasets.MNIST(root=root,
                                                       train=False,
                                                       transform=transforms.ToTensor(),
                                                       download=True)

        # Data loader
        # DataLoader provides an iterable over the dataset with support for batching, shuffling, & parallel data loading
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                        batch_size=batch_size,
                                                        shuffle=True)

        # DataLoader for the test dataset, used for evaluating the model
        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                       batch_size=batch_size,
                                                       shuffle=False)


class Trainer:
    """Trains a model on the given data and records loss/accuracy history."""

    def __init__(self, model, data, learning_rate, device):
        self.device = device
        self.data = data
        # Move the model to the configured device (GPU/CPU)
        self.model = model.to(device)

        # Loss & optimizer
        # CrossEntropyLoss combines nn.LogSoftmax & nn.NLLLoss in one single class
        self.criterion = nn.CrossEntropyLoss()
        # Adam optimizer with the specified learning rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.loss_list = []  # List to store loss values
        self.acc_train = []  # List to store training accuracy values
        self.acc_test = []  # List to store testing accuracy values

    # leveraged from Module 7, Lecture B, Section 3 ("Track accuracy")
    def track_accuracy(self, loader):
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient computation
            correct = 0
            total = 0
            for images, labels in loader:
                images = images.to(self.device)  # Move images to the configured device
                labels = labels.to(self.device)  # Move labels to the configured device
                outputs = self.model(images)  # Forward pass
                _, predicted = torch.max(outputs.data, 1)  # Get the predicted class
                total += labels.size(0)  # Increment total by the number of labels
                correct += (predicted == labels).sum().item()  # Increment correct by the number of correct predictions
            accuracy = 100 * correct / total  # Calculate accuracy
        self.model.train()  # Set the model back to training mode
        return accuracy

    # leveraged from Module 7, Lecture B, Section 3 ("Training")
    def train(self, num_epochs):
        # Loop over the number of epochs
        for epoch in range(num_epochs):
            # Loop over each batch in the training DataLoader

            pbar = tqdm(self.data.train_loader,
                        desc = f"Training: {epoch + 1:03d}/{num_epochs}",
                        ncols = 125,
                        leave = True)

            # Creating running loss empty list
            running_loss = []

            for i, (images, labels) in enumerate(pbar, 1):
                # Important note: In PyTorch, the images in a batch is typically represented as (batch_size, channels, height, width)
                # For example, (100, 1, 28, 28) for the MNIST data
                images = images.to(self.device)  # Move images to the configured device
                labels = labels.to(self.device)  # Move labels to the configured device

                # Forward pass: compute predicted y by passing x to the model
                outputs = self.model(images)
                # Compute the loss
                loss = self.criterion(outputs, labels)
                running_loss.append(loss.item())

                # Backward pass & optimize
                self.optimizer.zero_grad()  # Zero the gradients
                loss.backward()  # Backward pass
                self.optimizer.step()  # Update weights

                # Add the running loss
                pbar.set_postfix(loss = f"{sum(running_loss)/i : 10.6f}")

            # Get the average of all losses in the running_loss as the loss of the current epoch
            self.loss_list.append(sum(running_loss)/i)

            # Track accuracy on the train set
            self.acc_train.append(self.track_accuracy(self.data.train_loader))

            # Track accuracy on the test set
            self.acc_test.append(self.track_accuracy(self.data.test_loader))

            # Plain per-epoch summary so a record remains in the terminal log after the progress bar
            print(f"Epoch {epoch + 1:03d}/{num_epochs} | Loss: {self.loss_list[-1]:.6f} | "
                  f"Train Acc: {self.acc_train[-1]:.2f}% | Test Acc: {self.acc_test[-1]:.2f}%")

        return self.model, self.loss_list, self.acc_train, self.acc_test


#@title Plot
# leveraged from the assignment prompt and Module 7, Lecture B, Section 3 ("Plot")
def plot(loss_list, acc_train, acc_test, save_path='./figure.png'):
    # Plot the loss & accuracy curves
    plt.figure(figsize=(10, 4))

    # Plot the training loss over iterations
    plt.subplot(1, 2, 1)
    plt.plot(loss_list)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(f"Training Loss")

    # Plot the train & test accuracies over epochs
    plt.subplot(1, 2, 2)
    plt.plot(acc_train)
    plt.plot(acc_test)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f"Model Accuracy")
    plt.legend(['Training', 'Testing'])

    # Save the figure to disk, since plt.show() displays nothing when run from a terminal
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
