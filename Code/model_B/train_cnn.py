# Code/model_B/train_cnn.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_cnn(model, train_loader, val_loader, epochs=10, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        "train_loss": [],
        "val_loss": []
    }

    for epoch in range(epochs):
        # ===== training =====
        model.train()
        running_train_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.float().to(device).squeeze()

            optimizer.zero_grad()
            outputs = model(x).squeeze()
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        train_loss = running_train_loss / len(train_loader)
        history["train_loss"].append(train_loss)

        # ===== validation =====
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.float().to(device).squeeze()

                outputs = model(x).squeeze()
                loss = criterion(outputs, y)
                running_val_loss += loss.item()

        val_loss = running_val_loss / len(val_loader)
        history["val_loss"].append(val_loss)

        print(
            f"Epoch {epoch+1}/{epochs}, "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}"
        )

    return history



def evaluate_cnn(model, data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.float().to(device).squeeze()

            logits = model(x).squeeze()
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()

            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds),
        "recall": recall_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds),
    }