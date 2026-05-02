"""
experiments/train_cnn.py

Training loop for the CNN baseline classifier.
Loss: BCEWithLogitsLoss (numerically more stable than BCE + sigmoid).
Evaluation is done after every epoch on the test split.
"""

import torch
import torch.nn as nn
from tqdm import tqdm

import config
from models.cnn_baseline import CNNClassifier
from utils.metrics import compute_metrics, format_metrics


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs   = imgs.to(device)
        labels = labels.float().to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = (torch.sigmoid(logits) >= 0.5).long().cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.long().cpu().tolist())

    metrics = compute_metrics(all_preds, all_labels)
    return total_loss / len(loader), metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs   = imgs.to(device)
        labels = labels.float().to(device)

        logits = model(imgs)
        loss   = criterion(logits, labels)

        total_loss += loss.item()
        preds = (torch.sigmoid(logits) >= 0.5).long().cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.long().cpu().tolist())

    metrics = compute_metrics(all_preds, all_labels)
    return total_loss / len(loader), metrics


def run_cnn_training(train_loader, test_loader,
                     epochs=config.EPOCHS,
                     lr=config.LR,
                     dropout=config.DROPOUT,
                     weight_decay=config.WEIGHT_DECAY,
                     device=None,
                     verbose=True):
    """
    Trains the CNN baseline and returns:
        model, train_acc_history, test_acc_history, best_test_metrics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model     = CNNClassifier(dropout=dropout).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=epochs)

    train_acc_hist, test_acc_hist = [], []
    best_acc = 0.0
    best_metrics = {}

    for epoch in range(1, epochs + 1):
        tr_loss, tr_m = train_one_epoch(model, train_loader,
                                        optimizer, criterion, device)
        te_loss, te_m = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        train_acc_hist.append(tr_m["accuracy"])
        test_acc_hist.append(te_m["accuracy"])

        if te_m["accuracy"] > best_acc:
            best_acc     = te_m["accuracy"]
            best_metrics = te_m.copy()

        if verbose and epoch % 100 == 0:
            print(f"[CNN] Epoch {epoch:4d}/{epochs}  "
                  f"train_loss={tr_loss:.4f}  "
                  + format_metrics(te_m, prefix="test"))

    return model, train_acc_hist, test_acc_hist, best_metrics
