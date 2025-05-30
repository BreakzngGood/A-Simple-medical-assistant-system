import torch.nn as nn
import torchvision.models as models

# Save CNN model structure for futher use in `Medical_System_streamlit.py`
class ClassificationNetwork(nn.Module):
    def __init__(self, num_classes=22):
        super(ClassificationNetwork, self).__init__()
        self.model = models.resnet18(pretrained=True)  # Loading the ResNet18 pre-trained model
        num_ftrs = self.model.fc.in_features  
        self.model.fc = nn.Linear(num_ftrs, num_classes)  # Replace the last layer to accommodate 22 categories

    def forward(self, x):
        return self.model(x)