import torch

class SpecialistNN(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(SpecialistNN, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 128)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  
        return x