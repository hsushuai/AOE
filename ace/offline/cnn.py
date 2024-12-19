import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


def same_seeds(seed):
    """Fixed random seed for reproducibility."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_loader():
    # Load the payoff data
    df = pd.read_csv("ace/data/payoff/payoff_data.csv")
    df["strategy"] = df["strategy"].apply(lambda x: np.array(eval(x)))
    df["opponent"] = df["opponent"].apply(lambda x: np.array(eval(x)))

    strategy = np.stack(df["strategy"].values)
    opponent = np.stack(df["opponent"].values)
    X = np.concatenate((strategy, opponent), axis=1)
    y = df["win_loss"].values
    y[np.where(y < 1)] = 0

    # Convert data to PyTorch tensors and adjust dimensions to (samples, 1, features) to fit 1D CNN
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (samples, 1, features)
    y_tensor = torch.tensor(y, dtype=torch.long)

    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.3)

    # Dataset
    train_set = TensorDataset(X_train, y_train)
    test_set = TensorDataset(X_test, y_test)

    # Data loader
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    return train_loader, test_loader


class CNN1D(nn.Module):
    def __init__(self):
        super(CNN1D, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=128, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(128 * 4, 64)
        self.fc2 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(p=0.5)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.act(self.bn1(self.conv1(x))))
        x = self.pool(self.act(self.bn2(self.conv2(x))))
        x = x.view(-1, 128 * 4)  # Flatten
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def train_model():
    same_seeds(520)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_loader()
    # Instance the model, define the loss function and optimizer
    model = CNN1D()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience=10)

    # train the model
    max_epochs = 1000
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    best_acc = 0.0
    patience = 200
    patience_counter = 0

    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            # inputs: (batch_size, features, 1), labels: (batch_size,)
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # calculate average loss and accuracy for the epoch
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        scheduler.step(train_acc)

        # evaluate on test set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_acc = correct / total
        test_accuracies.append(test_acc)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'ace/data/payoff/cnn1d.pth')
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping, best test accuracy: {best_acc:.4f}")
            break

        print(f"Epoch [{epoch+1}/{max_epochs}], Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")

    # plotting
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(train_accuracies)+1), train_accuracies, label='train set accuracy')
    plt.plot(range(1, len(test_accuracies)+1), test_accuracies, label='test set accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()
    plt.tight_layout()

    plt.savefig('results/training.png', dpi=300)


if __name__ == "__main__":
    train_model()
