import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import medical_assistant_package.config as cfg
import pickle

# -------------------------------------------------
# For that the project require the netural network model, so I trained another model for the doctor prediction
# -------------------------------------------------

# Load data and preprocess
def load_data(path):
    data = pd.read_excel(path)
    X = data.drop(['Disease', 'Unnamed: 0'], axis=1).values
    y = data['Disease'].values

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return X, y_encoded, le

# Define neural network model
class SpecialistNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SpecialistNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Training function
def train_model(model, criterion, optimizer, x_train, y_train, epochs=50):
    model.train()  # Set model to training mode
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Evaluation function
def evaluate_model(model, x_test, y_test):
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        outputs = model(x_test)
        preds = torch.argmax(outputs, dim=1)
        acc = accuracy_score(y_test.numpy(), preds.numpy())
    return acc

def main():
    X, y, label_encoder = load_data(cfg.SPECIALIST_EXCEL_PATH)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

    # Convert to torch tensors
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    input_size = x_train.shape[1]
    output_size = len(np.unique(y))

    model = SpecialistNN(input_size, output_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    train_model(model, criterion, optimizer, x_train_tensor, y_train_tensor, epochs=50)

    accuracy = evaluate_model(model, x_test_tensor, y_test_tensor)
    print(f"Model Accuracy: {accuracy:.4f}")

    # Save the model and label encoder
    torch.save(model.state_dict(), "specialist_nn.pth")
    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

if __name__ == "__main__":
    main()