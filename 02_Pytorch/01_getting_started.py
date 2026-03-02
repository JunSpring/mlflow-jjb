import mlflow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Set up MLflow experiment
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("pytorch-getting-started")

# Enable autologging
mlflow.pytorch.autolog()

# Create synthetic data
X = torch.randn(1000, 784)
y = torch.randint(0, 10, (1000,))
train_loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)

# Your existing PyTorch code works unchanged
model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop - metrics, parameters, and models logged automatically
for epoch in range(10):
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()