import os
import pandas as pd
import numpy as np
import json
import random
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
from ace.strategy import Strategy
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import scikitplot as skplt


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


def prepare_data(runs_dir = "runs/offline_runs", augment=True):
    df = pd.DataFrame()
    runs = os.listdir(runs_dir)
    runs = sorted(runs, key=lambda x: int(x.split("_")[0]) * 100 + int(x.split("_")[1]))
    strategy_space = []
    for run_name in runs:
        idx_strategy = run_name.split("_")[0]
        idx_opponent = run_name.split("_")[1]
        strategy_dir = "ace/data/train"
        strategy = Strategy.load_from_json(f"{strategy_dir}/strategy_{idx_strategy}.json")
        opponent = Strategy.load_from_json(f"{strategy_dir}/strategy_{idx_opponent}.json")
        with open(f"{runs_dir}/{run_name}/metric.json") as f:
            metric = json.load(f)
        win_loss = metric["win_loss"]
        raw = pd.DataFrame({
            "id": [run_name],
            "strategy": [strategy.feats.tolist()],
            "opponent": [opponent.feats.tolist()],
            "win_loss": [win_loss[0]]
        })
        df = pd.concat([df, raw])

        if augment:
            aug_df = pd.DataFrame({
                "id": [f"{idx_opponent}_{idx_strategy}"],
                "strategy": [opponent.feats.tolist()],
                "opponent": [strategy.feats.tolist()],
                "win_loss": [win_loss[1]]
            })
            if idx_strategy not in strategy_space:
                aug_df = pd.concat([
                    aug_df,
                    pd.DataFrame({
                        "id": [f"{idx_strategy}_{idx_strategy}"],
                        "strategy": [strategy.feats.tolist()],
                        "opponent": [strategy.feats.tolist()],
                        "win_loss": [0]
                    })
                ])
                strategy_space.append(idx_strategy)
            if idx_opponent not in strategy_space:
                aug_df = pd.concat([
                    aug_df,
                    pd.DataFrame({
                        "id": [f"{idx_opponent}_{idx_opponent}"],
                        "strategy": [opponent.feats.tolist()],
                        "opponent": [opponent.feats.tolist()],
                        "win_loss": [0]
                    })
                ])
                strategy_space.append(idx_opponent)
            df = pd.concat([df, aug_df])
    df.to_csv("ace/data/payoff/payoff_data.csv", index=False)


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
    y_tensor = torch.tensor(y, dtype=torch.float32)

    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2)

    # Dataset
    train_set = TensorDataset(X_train, y_train)
    test_set = TensorDataset(X_test, y_test)

    # Data loader
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    return train_loader, test_loader


class PayoffNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(24, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 64)
        self.bn_fc2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = F.leaky_relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return torch.sigmoid(x.squeeze(1))
    
    @staticmethod
    def load(filename: str) -> "PayoffNet":
        model = PayoffNet().to("cuda")
        model.load_state_dict(torch.load(filename, weights_only=True))
        model.eval()
        return model
    
    def search_best_response(self, feat_space, opponent_feats) -> tuple[Strategy, float]:
        # Shuffle the feature space
        np.random.seed(520)
        shuffled_space = feat_space[np.random.permutation(feat_space.shape[0])]
        opponent_feats = np.tile(opponent_feats, (shuffled_space.shape[0], 1))
        X = torch.tensor(np.hstack([shuffled_space, opponent_feats]), dtype=torch.float32, device="cuda")
        X = X.unsqueeze(1)
        with torch.no_grad():
            win_rate = self(X)
        max_idx = win_rate.argmax().item()
        response = Strategy.decode(shuffled_space[max_idx]).strategy
        return response, win_rate[max_idx].item()


def train_model():
    same_seeds(520)
    train_loader, test_loader = get_loader()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PayoffNet().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    max_epochs = 1000
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    best_acc = 0.0
    patience = 100
    patience_counter = 0

    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            predictions = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                predictions = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predictions == labels).sum().item()

        test_acc = correct / total
        test_accuracies.append(test_acc)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "ace/data/payoff/payoff_net.pth")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping, best test accuracy: {best_acc:.4f}")
            break

        print(f"Epoch [{epoch+1}/{max_epochs}], Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")

    os.environ["QT_QPA_PLATFORM"] = "offscreen"  # without GUI
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(train_accuracies)+1), train_accuracies, label="Train Accuracy")
    plt.plot(range(1, len(test_accuracies)+1), test_accuracies, label="Test Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over Epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/training_curve.png", dpi=300)

    y_probas = []
    y_true = []
    model.load_state_dict(torch.load("ace/data/payoff/payoff_net.pth", weights_only=True))
    model.eval()
    model.to(device)
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            probas = model(inputs)
            y_probas.extend(probas.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    y_true = np.array(y_true)
    y_probas = np.array(y_probas)
    y_preds = (y_probas > 0.5).astype(int)
    os.environ["QT_QPA_PLATFORM"] = "offscreen"  # without GUI
    skplt.metrics.plot_confusion_matrix(y_true, y_preds, normalize=True)
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png", dpi=300)


if __name__ == "__main__":
    # prepare_data()
    train_model()  # 0.7944