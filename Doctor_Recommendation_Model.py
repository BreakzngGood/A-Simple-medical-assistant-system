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



# The features in the original dataset have already been encoded (One-Hot Encoding), so no further feature engineering is needed.
# Based on these features, I trained and compared two models: Random Forest and Logistic Regression.
# The final result shows that the model can predict the most suitable doctor based on disease symptoms.

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

data = pd.read_excel("Datasets\Specialist.xlsx")
# data.head()
x = data.drop(['Disease', 'Unnamed: 0'], axis = 1)
y = data.Disease

x_train, x_test, y_train , y_test = train_test_split(x ,y, random_state = 50)

# model = RandomForestClassifier(n_estimators=200)
# model.fit(x_train, y_train)
# predictions = model.predict(x_test)
# accuracy_score(y_test, predictions)

model = LogisticRegression(max_iter = 1000)
model.fit(x_train, y_train)
predictions = model.predict(x_test)
accuracy_score(y_test, predictions)

pickle.dump(model, open('Specalist.pkl', 'wb'))

def load_symptom_feature():
    df = pd.read_excel("Datasets\Specialist.xlsx")
    x = df.drop(['Disease', 'Unnamed: 0'], axis = 1)
    feature_names = x.columns.tolist()
    return feature_names

feature_names = load_symptom_feature()

print(feature_names)

# -------------------------------------------------
# For that the project require the netural network model, so I trained another model for the doctor prediction
# -------------------------------------------------

data = pd.read_excel("Datasets/Specialist.xlsx")

x = data.drop(['Disease', 'Unnamed: 0'], axis=1).values
y = data['Disease'].values

# Label encoding (converting symptoms categories to numbers)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=50)

# Convert to PyTorch Tensor
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Defining Neural Networks
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

input_size = x_train.shape[1]
output_size = len(np.unique(y))  # Number of categories

model = SpecialistNN(input_size, output_size)

# Defining loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training the model
epochs = 50
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(x_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

with torch.no_grad():
    test_outputs = model(x_test_tensor)
    predictions = torch.argmax(test_outputs, dim=1)
    accuracy = accuracy_score(y_test, predictions.numpy())

print(f"Model Accuracy: {accuracy:.4f}")

# save the model
torch.save(model.state_dict(), "specialist_nn.pth")

# Save the LabelEncoder 
import pickle
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)