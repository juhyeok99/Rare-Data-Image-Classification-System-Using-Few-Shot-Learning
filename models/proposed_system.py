"""
models/proposed_system.py

Proposed rare-data image classification system.

Two-stage design as described in the paper:
  1. CNN backbone: feature extractor (Conv2D -> ReLU -> MaxPool, repeated)
  2. Few-Shot Learning head: prototype computation + cosine similarity scoring

The key difference from a plain CNN is that at inference time we don't need
to retrain for a new data distribution — we just update the support set and
recompute prototypes.  This is what makes the system effective on rare data.

Equation (1) from the paper:
    cosine_similarity(A, B) = (A · B) / (||A|| * ||B||)

The model is trained in an episodic fashion: each episode provides a small
support set (a few labelled images per class) and a query set whose labels
we predict via similarity to class prototypes.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import config


# ── Backbone CNN ───────────────────────────────────────────────────────────

class _ConvBlock(nn.Module):
    """
    Conv2D -> ReLU -> MaxPool unit as described in Section 3.2.
    Optional BatchNorm is added between conv and activation to stabilise
    training on small datasets — not explicitly in the paper but without it
    the model diverges on 100-sample runs.
    """

    def __init__(self, in_ch, out_ch, use_bn=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=not use_bn)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers += [nn.ReLU(inplace=True), nn.MaxPool2d(2, 2)]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class CNNBackbone(nn.Module):
    """
    Feature extractor shared across the whole system.

    Channel progression: 3 -> 32 -> 64 -> 128
    For 64x64 input: spatial dims go 64 -> 32 -> 16 -> 8
    Final feature dimension: 128 * 8 * 8 -> projected to `feature_dim`

    The paper mentions 32 filters of 3x3 for the first layer; we expand
    deeper layers for richer representations while keeping the same pattern.
    """

    def __init__(self, in_channels=3, feature_dim=256,
                 img_size=config.IMG_SIZE, dropout=config.DROPOUT):
        super().__init__()

        self.conv_blocks = nn.Sequential(
            _ConvBlock(in_channels, 32),    # paper's first conv: 32 filters 3x3
            _ConvBlock(32, 64),
            _ConvBlock(64, 128),
        )

        # determine flat size without hardcoding
        with torch.no_grad():
            dummy   = torch.zeros(1, in_channels, img_size, img_size)
            flat_sz = self.conv_blocks(dummy).flatten(1).size(1)

        self.flatten  = nn.Flatten()
        self.dropout  = nn.Dropout(dropout)
        self.fc       = nn.Linear(flat_sz, feature_dim)
        self.feat_dim = feature_dim

    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x     # [B, feature_dim]


# ── Few-Shot Learning Head ─────────────────────────────────────────────────

class PrototypeHead(nn.Module):
    """
    Prototype-based few-shot classification head.

    Given a support set (a few labelled examples per class), builds one
    prototype vector per class as the mean of the support feature vectors.
    Query images are then classified by their cosine similarity to each
    prototype — whichever prototype is closest wins.

    Cosine similarity is used as specified in Equation (1) of the paper.
    A learned temperature parameter τ scales the logits before softmax so
    the model can control confidence without changing the similarity space.
    """

    def __init__(self, feature_dim=256, n_classes=config.N_WAY):
        super().__init__()
        self.n_classes   = n_classes
        self.feature_dim = feature_dim

        # learnable temperature for scaling similarity scores
        # initialised to log(10) ≈ 2.3 so softmax isn't too peaked at start
        self.log_tau = nn.Parameter(torch.tensor(math.log(10.0)))

        # small projection applied to prototypes only — helps the prototype
        # space stay well-separated even when feature_dim is large
        self.proto_proj = nn.Linear(feature_dim, feature_dim, bias=False)

    def _cosine_similarity(self, query_feats, prototypes):
        """
        Compute cosine similarity between every query and every prototype.

        query_feats : [Q, D]
        prototypes  : [C, D]
        returns     : [Q, C] similarity matrix
        """
        q_norm = F.normalize(query_feats, dim=-1)           # [Q, D]
        p_norm = F.normalize(prototypes, dim=-1)             # [C, D]
        return torch.mm(q_norm, p_norm.t())                  # [Q, C]

    def build_prototypes(self, support_feats, support_labels):
        """
        Average support features per class to form prototype vectors.

        support_feats  : [S, D]
        support_labels : [S]  (integer class ids)
        returns        : [C, D] one prototype per class
        """
        protos = []
        for c in range(self.n_classes):
            mask = (support_labels == c)
            if mask.sum() == 0:
                # edge case: if a class has no support samples, use zeros
                protos.append(torch.zeros(self.feature_dim,
                                          device=support_feats.device))
            else:
                protos.append(support_feats[mask].mean(0))
        return torch.stack(protos)       # [C, D]

    def forward(self, support_feats, support_labels, query_feats):
        """
        Returns logits [Q, C] for each query over all classes.
        """
        prototypes = self.build_prototypes(support_feats, support_labels)
        prototypes = self.proto_proj(prototypes)          # projected prototypes

        sim = self._cosine_similarity(query_feats, prototypes)  # [Q, C]

        # temperature scaling: τ = exp(log_tau), clamped to avoid explosion
        tau = self.log_tau.exp().clamp(min=1.0, max=100.0)
        return tau * sim                                  # [Q, C] logits


# ── Full Proposed System ───────────────────────────────────────────────────

class ProposedFewShotSystem(nn.Module):
    """
    Complete proposed classification system: CNN backbone + FSL prototype head.

    For training (episodic):
        Call forward(support_imgs, support_labels, query_imgs) in each episode.
        Loss = cross_entropy(logits, query_labels)

    For inference on a new image without retraining:
        1. Collect a small support set (even 1–5 images per class)
        2. build_support_cache(support_imgs, support_labels)
        3. predict(query_imgs) → predicted class labels

    The CNN backbone can optionally be used standalone (baseline mode) via
    the classify_binary() method which adds a linear head.
    """

    def __init__(self, feature_dim=256, n_classes=config.N_WAY,
                 dropout=config.DROPOUT):
        super().__init__()
        self.backbone  = CNNBackbone(feature_dim=feature_dim, dropout=dropout)
        self.fsl_head  = PrototypeHead(feature_dim=feature_dim, n_classes=n_classes)

        # binary classification head for the standalone CNN comparison
        # (kept inside this module so the backbone weights are shared when needed)
        self.binary_head = nn.Linear(feature_dim, 1)

        # cached prototypes for inference without support set re-computation
        self._cached_protos = None

    # -- episodic forward (used during training) ----------------------------

    def forward(self, support_imgs, support_labels, query_imgs):
        """
        Episode forward pass.

        support_imgs   : [N_way * N_support, C, H, W]
        support_labels : [N_way * N_support]   (0 or 1 for binary)
        query_imgs     : [N_way * N_query,   C, H, W]
        returns        : logits [N_way * N_query, N_way]
        """
        s_feats = self.backbone(support_imgs)   # [S, D]
        q_feats = self.backbone(query_imgs)     # [Q, D]
        logits  = self.fsl_head(s_feats, support_labels, q_feats)
        return logits

    # -- inference helpers --------------------------------------------------

    @torch.no_grad()
    def build_support_cache(self, support_imgs, support_labels):
        """
        Pre-compute and cache prototypes from a support set.
        Call this once before running predict() on multiple batches.
        """
        self.eval()
        feats = self.backbone(support_imgs)
        protos = self.fsl_head.build_prototypes(feats, support_labels)
        protos = self.fsl_head.proto_proj(protos)
        self._cached_protos = F.normalize(protos, dim=-1)

    @torch.no_grad()
    def predict(self, query_imgs):
        """
        Classify query images using cached prototypes.
        Returns predicted class indices [Q].
        """
        if self._cached_protos is None:
            raise RuntimeError("Call build_support_cache() before predict().")

        self.eval()
        q_feats = self.backbone(query_imgs)
        q_norm  = F.normalize(q_feats, dim=-1)
        sim     = torch.mm(q_norm, self._cached_protos.t())
        return sim.argmax(dim=-1)

    # -- loss ---------------------------------------------------------------

    @staticmethod
    def episode_loss(logits, query_labels):
        """Cross-entropy over query predictions. Standard for prototypical nets."""
        return F.cross_entropy(logits, query_labels)


# ── Weight initialisation ──────────────────────────────────────────────────

def _init_weights(module):
    """
    Kaiming init for conv layers, Xavier for linear.
    Default PyTorch init is usually fine but on small datasets explicit init
    can shave a few epochs off convergence.
    """
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out",
                                nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def build_proposed_system(device=None, feature_dim=256,
                           dropout=config.DROPOUT):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProposedFewShotSystem(feature_dim=feature_dim, dropout=dropout)
    model.apply(_init_weights)
    return model.to(device)
