import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import medical_assistant_package.config as cfg


device = 'cpu'
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameter
momentum = 0.9
num_epochs = 10
num_classes = 22  
batch_size = 100
learn_rate = 0.001
train_path = cfg.SKIN_DISEASE_TRAIN
test_path = cfg.SKIN_DISEASE_TEST

# Training Data Augmentation
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=(-30, 30), translate=(0.25, 0.25), scale=(0.85, 1.15)),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    transforms.RandomResizedCrop(size=(64, 64), antialias=True),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Test Data Augmentation (Without Random Changes)
test_transform = transforms.Compose([
    transforms.Resize(64, antialias=True),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Dataset
train_dataset = ImageFolder(train_path, transform=train_transform)
test_dataset = ImageFolder(test_path, transform=test_transform)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# CNN Model
class ClassificationNetwork(nn.Module):
    def __init__(self, num_classes=22):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.conv3 = nn.Conv2d(128, 128, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(2048, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize Model
model = ClassificationNetwork(num_classes=num_classes).to(device)

# Cross-Entropy Loss
criterion = nn.CrossEntropyLoss()

# Stochastic Gradient Descent
optimizer = optim.SGD(model.parameters(), lr=learn_rate, momentum=momentum)

# Train
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
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

    test_loss /= len(test_loader)

    
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.3f}, Test Loss: {test_loss:.3f}")

    # Save the Best Model
    if test_loss < best_loss:
        best_loss = test_loss
        torch.save(model.state_dict(), cfg.SKIN_DISEASE_MODEL)

    
    # The result of train loss and test loss is not good, so I have to choose the pretrained model.