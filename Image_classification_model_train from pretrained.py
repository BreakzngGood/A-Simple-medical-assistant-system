import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os
import torchvision.models as models

class ClassificationNetwork(nn.Module):
    def __init__(self, num_classes=22):
        super(ClassificationNetwork, self).__init__()
        self.model = models.resnet18(pretrained=True)  # Loading the ResNet18 pre-trained model
        num_ftrs = self.model.fc.in_features  # Get the number of features in the last layer of ResNet18.
        self.model.fc = nn.Linear(num_ftrs, num_classes)  # Replace the last layer to accommodate 22 categories

    def forward(self, x):
        return self.model(x)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameter
num_epochs = 10
num_classes = 22
batch_size = 64
learning_rate = 0.0001

train_path = r"Datasets\SkinDisease\train"
test_path = r"Datasets\SkinDisease\test"

# Training Data Augmentation
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=(-30, 30), translate=(0.2, 0.2), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    transforms.RandomResizedCrop((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Test Data Augmentation (Without Random Changes)
test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Dataset
train_dataset = ImageFolder(train_path, transform=train_transform)
test_dataset = ImageFolder(test_path, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize Model
model = ClassificationNetwork(num_classes=num_classes).to(device)

# Cross-Entropy Loss Function & Adam Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

best_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    test_loss /= len(test_loader)
    accuracy = correct / total

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.3f}, Test Loss: {test_loss:.3f}, Accuracy: {accuracy:.2%}")

    # Save the Best Model
    if test_loss < best_loss:
        best_loss = test_loss
        torch.save(model.state_dict(), 'best_resnet18_model.pt')    