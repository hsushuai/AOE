import os
import pandas as pd
import numpy as np
import json
import random
from ace.strategy import Strategy

from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import VotingClassifier
from lightgbm import LGBMClassifier
import scikitplot as skplt
from sklearn.metrics import accuracy_score, log_loss
import matplotlib.pyplot as plt
import joblib

import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


def prepare_data(runs_dir = "runs/pre_match_runs", augment=True):
    df = pd.DataFrame()
    runs = os.listdir(runs_dir)
    runs = sorted(runs, key=lambda x: int(x.split("_")[0]) * 100 + int(x.split("_")[1]))
    strategy_space = []
    for run_name in runs:
        idx_strategy = run_name.split("_")[0]
        idx_opponent = run_name.split("_")[1]
        strategy_dir = "ace/data/train"
        strategy = Strategy.load_from_json(f"{strategy_dir}/strategies/strategy_{idx_strategy}.json")
        opponent = Strategy.load_from_json(f"{strategy_dir}/opponents/strategy_{idx_opponent}.json")
        with open(f"{runs_dir}/{run_name}/metric.json") as f:
            metric = json.load(f)
        win_loss = metric["win_loss"]
        raw = pd.DataFrame({
            "id": [run_name],
            "strategy": [strategy.one_hot_feats.tolist()],
            "opponent": [opponent.one_hot_feats.tolist()],
            "win_loss": [win_loss[0]]
        })
        df = pd.concat([df, raw])

        if augment:
            aug_df = pd.DataFrame({
                "id": [f"{idx_opponent}_{idx_strategy}"],
                "strategy": [opponent.one_hot_feats.tolist()],
                "opponent": [strategy.one_hot_feats.tolist()],
                "win_loss": [win_loss[1]]
            })
            if idx_strategy not in strategy_space:
                aug_df = pd.concat([
                    aug_df,
                    pd.DataFrame({
                        "id": [f"{idx_strategy}_{idx_strategy}"],
                        "strategy": [strategy.one_hot_feats.tolist()],
                        "opponent": [strategy.one_hot_feats.tolist()],
                        "win_loss": [0]
                    })
                ])
                strategy_space.append(idx_strategy)
            if idx_opponent not in strategy_space:
                aug_df = pd.concat([
                    aug_df,
                    pd.DataFrame({
                        "id": [f"{idx_opponent}_{idx_opponent}"],
                        "strategy": [opponent.one_hot_feats.tolist()],
                        "opponent": [opponent.one_hot_feats.tolist()],
                        "win_loss": [0]
                    })
                ])
                strategy_space.append(idx_opponent)
            df = pd.concat([df, aug_df])
    df.to_csv("ace/data/payoff/payoff_data.csv", index=False)


def load_data(file_path="ace/data/payoff/payoff_data.csv"):
    df = pd.read_csv(file_path)
    df["strategy"] = df["strategy"].apply(lambda x: np.array(eval(x)))
    df["opponent"] = df["opponent"].apply(lambda x: np.array(eval(x)))
    return df


def fit_payoff_model():
    same_seeds(520)
    OUTPUT_DIM = 2
    data = load_data()
    strategy = np.stack(data["strategy"].values)
    opponent = np.stack(data["opponent"].values)
    X = np.concatenate((strategy, opponent), axis=1)
    y = data["win_loss"].values + 1  # -1, 0, 1 -> 0, 1, 2 for CrossEntropyLoss which requires label >= 0
    if OUTPUT_DIM == 2:
        y[np.where(y < 2)] = 0
        y[np.where(y == 2)] = 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    sgd_model1 = SGDClassifier(
        max_iter=8000,
        tol=1e-4,
        loss="modified_huber",
        n_jobs=-1,
        random_state=42,
        verbose=10,
        early_stopping=True,
        validation_fraction=0.1,
        class_weight="balanced",
    )
    sgd_model2 = SGDClassifier(
        max_iter=15000,
        tol=1e-4,
        loss="modified_huber",
        n_jobs=-1,
        random_state=71,
        verbose=10,
        early_stopping=True,
        validation_fraction=0.1,
        class_weight="balanced",
    )
    sgd_model3 = SGDClassifier(
        max_iter=20000,
        tol=1e-4,
        loss="modified_huber",
        n_jobs=-1,
        random_state=22,
        verbose=10,
        early_stopping=True,
        validation_fraction=0.1,
        class_weight="balanced",
    )
    params = {
        "n_iter": 300,
        "verbose": -1,
        "learning_rate": 0.005689066836106983,
        "colsample_bytree": 0.8915976762048253,
        "colsample_bynode": 0.5942203285139224,
        "lambda_l1": 7.6277555139102864,
        "lambda_l2": 6.6591278779517808,
        "min_data_in_leaf": 156,
        "max_depth": 11,
        "max_bin": 813,
    }
    lgb = LGBMClassifier(**params)

    ensemble = VotingClassifier(
        estimators=[
            ("sgd1", sgd_model1),
            ("sgd2", sgd_model2),
            ("sgd3", sgd_model3),
            ("lgb", lgb),
        ],
        weights=[0.15, 0.15, 0.15, 0.55],
        voting="soft",
        n_jobs=-1,
    )

    ensemble.fit(X_train, y_train)
    joblib.dump(ensemble, "ace/data/payoff/ensemble_model.pkl")
    loaded_ensemble = joblib.load("ace/data/payoff/ensemble_model.pkl")
    
    # test
    y_pred = loaded_ensemble.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))

    # plot confusion matrix
    os.environ["QT_QPA_PLATFORM"] = "offscreen"  # run without GUI
    skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True)
    plt.savefig("results/ml_confusion_matrix.png")

    # plot roc curve
    y_proba = loaded_ensemble.predict_proba(X_test)
    print("Log Loss:", log_loss(y_test, y_proba))
    skplt.metrics.plot_roc(y_test, y_proba)
    plt.savefig("results/ml_roc_curve.png")

    # skplt.estimators.plot_learning_curve(loaded_ensemble, X, y)
    # plt.savefig("results/learning_curve.png")


class PayoffNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PayoffNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)


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


def fit_payoff_nn():
    same_seeds(520)
    OUTPUT_DIM = 2
    data = load_data()
    strategy = np.stack(data["strategy"].values)
    opponent = np.stack(data["opponent"].values)
    X = np.concatenate((strategy, opponent), axis=1)
    y = data["win_loss"].values + 1  # -1, 0, 1 -> 0, 1, 2 for CrossEntropyLoss which requires label >= 0
    if OUTPUT_DIM == 2:
        y[np.where(y < 2)] = 0
        y[np.where(y == 2)] = 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PayoffNet(input_dim=X.shape[1], hidden_dim=64, output_dim=OUTPUT_DIM)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y),
        y=y
    )
    weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model = model.to(device)
    X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train, dtype=torch.long, device=device)
    X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test = torch.tensor(y_test, dtype=torch.long, device=device)

    max_acc = 0
    best_model = None
    for epoch in range(1000):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                y_pred = model(X_test)
                loss = criterion(y_pred, y_test)
                y_pred_labels = torch.argmax(y_pred, dim=1).cpu().numpy()
                y_test_labels = y_test.cpu().numpy()
                accuracy = accuracy_score(y_test_labels, y_pred_labels)
                if accuracy > max_acc:
                    max_acc = accuracy
                    best_model = model
                print(f"Epoch {epoch + 1}, Test Loss: {loss.item()}, Accuracy: {accuracy:.2f}")
    print(f"Best Accuracy: {max_acc:.2f}")
    torch.save(best_model, "ace/data/payoff/payoff_net.pth")

    model = torch.load("ace/data/payoff/payoff_net.pth")
    model.to(device)
    model.eval()
    y_test_labels = y_test.cpu().numpy()
    with torch.no_grad():
        y_pred = model(X_test)
    y_pred_labels = torch.argmax(y_pred, dim=1).cpu().numpy()
    y_pred_probs = torch.softmax(y_pred, dim=1).cpu().numpy()
    # plot confusion matrix
    os.environ["QT_QPA_PLATFORM"] = "offscreen"  # run without GUI
    skplt.metrics.plot_confusion_matrix(y_test_labels, y_pred_labels, normalize=True)
    plt.savefig("results/dnn_confusion_matrix.png")

    # plot roc curve
    print("Log Loss:", log_loss(y_test_labels, y_pred_probs))
    skplt.metrics.plot_roc(y_test_labels, y_pred_probs)
    plt.savefig("results/dnn_roc_curve.png")

if __name__ == "__main__":
    prepare_data()
    fit_payoff_nn()