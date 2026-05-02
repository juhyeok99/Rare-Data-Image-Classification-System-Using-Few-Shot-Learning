"""
experiments/train_proposed.py

Episodic training loop for the proposed few-shot system.

Each epoch consists of N_EPISODES randomly sampled few-shot tasks (episodes).
For each episode:
  - Sample N_WAY classes, N_SUPPORT images per class → support set
  - Sample N_QUERY images per class → query set
  - Forward pass: encode all images, compute prototypes, score queries
  - Backprop cross-entropy loss over query predictions

After episodic training, final evaluation is done on the full test set by
treating the entire training split as the support set (transductive setting),
which is a reasonable approximation for the binary real/fake task.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import config
from models.proposed_system import ProposedFewShotSystem, build_proposed_system
from data.dataset import FewShotEpisodeSampler, RareImageDataset
from utils.metrics import compute_metrics, format_metrics


def run_episode(model, sampler, device):
    """
    Executes one few-shot episode and returns the loss + query predictions.
    """
    s_imgs, s_labels, q_imgs, q_labels = sampler.sample_episode()

    s_imgs   = s_imgs.to(device)
    s_labels = s_labels.to(device)
    q_imgs   = q_imgs.to(device)
    q_labels = q_labels.to(device)

    logits = model(s_imgs, s_labels, q_imgs)       # [Q, N_WAY]
    loss   = ProposedFewShotSystem.episode_loss(logits, q_labels)

    preds = logits.argmax(dim=-1).cpu()
    return loss, preds, q_labels.cpu()


@torch.no_grad()
def evaluate_on_testset(model, train_dataset, test_loader, device):
    """
    Full test-set evaluation.

    We build prototypes from the entire training split (mean of all real feats
    and all fake feats), then classify every test image by cosine similarity.
    This mirrors how the system would actually be deployed on rare data.
    """
    model.eval()

    # ── collect training features per class ───────────────────────────────
    all_feats  = {0: [], 1: []}
    train_tmp  = torch.utils.data.DataLoader(train_dataset, batch_size=32,
                                             shuffle=False)
    for imgs, labels in train_tmp:
        imgs = imgs.to(device)
        with torch.no_grad():
            feats = model.backbone(imgs).cpu()
        for f, l in zip(feats, labels):
            all_feats[int(l)].append(f)

    support_feats  = []
    support_labels = []
    for cls, feats in all_feats.items():
        if feats:
            support_feats.append(torch.stack(feats).mean(0))
            support_labels.append(cls)

    support_feats  = torch.stack(support_feats).to(device)
    support_labels = torch.tensor(support_labels, dtype=torch.long).to(device)
    model.build_support_cache(support_feats, support_labels)

    # ── classify test images ───────────────────────────────────────────────
    all_preds, all_labels = [], []
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        preds = model.predict(imgs).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

    return compute_metrics(all_preds, all_labels)


def run_proposed_training(train_dataset, test_loader,
                          epochs=config.EPOCHS,
                          n_episodes=config.N_EPISODES,
                          lr=config.LR,
                          dropout=config.DROPOUT,
                          weight_decay=config.WEIGHT_DECAY,
                          feature_dim=256,
                          device=None,
                          verbose=True):
    """
    Episodic training for the proposed system.

    Returns:
        model, episode_loss_history, test_acc_history, best_test_metrics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model    = build_proposed_system(device=device, feature_dim=feature_dim,
                                     dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=epochs)
    sampler   = FewShotEpisodeSampler(train_dataset)

    ep_loss_hist, test_acc_hist = [], []
    best_acc     = 0.0
    best_metrics = {}

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        all_ep_preds, all_ep_labels = [], []

        for _ in range(n_episodes):
            optimizer.zero_grad()
            loss, preds, labels = run_episode(model, sampler, device)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            all_ep_preds.extend(preds.tolist())
            all_ep_labels.extend(labels.tolist())

        scheduler.step()
        avg_loss = epoch_loss / n_episodes
        ep_loss_hist.append(avg_loss)

        # ── test evaluation every 50 epochs (expensive) ───────────────────
        if epoch % 50 == 0 or epoch == epochs:
            te_m = evaluate_on_testset(model, train_dataset,
                                       test_loader, device)
            test_acc_hist.append(te_m["accuracy"])

            if te_m["accuracy"] > best_acc:
                best_acc     = te_m["accuracy"]
                best_metrics = te_m.copy()

            if verbose:
                print(f"[Proposed] Epoch {epoch:4d}/{epochs}  "
                      f"ep_loss={avg_loss:.4f}  "
                      + format_metrics(te_m, prefix="test"))
        else:
            test_acc_hist.append(test_acc_hist[-1] if test_acc_hist else 0.0)

    return model, ep_loss_hist, test_acc_hist, best_metrics
