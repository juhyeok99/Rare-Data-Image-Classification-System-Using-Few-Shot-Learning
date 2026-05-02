"""
data/dataset.py

Handles loading painting / plant images and generating few-shot episodes.
Labels: 0 = real/authentic, 1 = fake/generated
"""

import os
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image

import config


# standard augmentation for training set to avoid overfitting on small data
_train_tf = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

_eval_tf = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


class RareImageDataset(Dataset):
    """
    Flat binary dataset: real images (label=0) mixed with generated/fake
    images (label=1).  Both real_dir and fake_dir are expected to contain
    only image files.
    """

    def __init__(self, real_dir, fake_dir, n_samples=None, split="train",
                 train_ratio=0.8, seed=config.SEED):
        self.samples = []
        self.transform = _train_tf if split == "train" else _eval_tf

        real_paths = sorted(Path(real_dir).glob("*.jpg")) + \
                     sorted(Path(real_dir).glob("*.png")) + \
                     sorted(Path(real_dir).glob("*.jpeg"))
        fake_paths = sorted(Path(fake_dir).glob("*.jpg")) + \
                     sorted(Path(fake_dir).glob("*.png")) + \
                     sorted(Path(fake_dir).glob("*.jpeg"))

        rng = random.Random(seed)

        if n_samples is not None:
            half = n_samples // 2
            real_paths = rng.sample(real_paths, min(half, len(real_paths)))
            fake_paths = rng.sample(fake_paths, min(half, len(fake_paths)))

        all_samples = [(p, 0) for p in real_paths] + [(p, 1) for p in fake_paths]
        rng.shuffle(all_samples)

        cut = int(len(all_samples) * train_ratio)
        self.samples = all_samples[:cut] if split == "train" else all_samples[cut:]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)


class FewShotEpisodeSampler:
    """
    Generates N-way K-shot episodes from a RareImageDataset.
    Each episode yields:
        support_imgs  [N*K, C, H, W]
        support_labels[N*K]
        query_imgs    [N*Q, C, H, W]
        query_labels  [N*Q]
    """

    def __init__(self, dataset, n_way=config.N_WAY,
                 n_support=config.N_SUPPORT, n_query=config.N_QUERY):
        self.dataset   = dataset
        self.n_way     = n_way
        self.n_support = n_support
        self.n_query   = n_query

        # group indices by class
        self.class_indices = {}
        for i, (_, label) in enumerate(dataset.samples):
            self.class_indices.setdefault(int(label), []).append(i)

        assert len(self.class_indices) >= n_way, \
            f"Need at least {n_way} classes, got {len(self.class_indices)}"

    def sample_episode(self):
        chosen_classes = random.sample(list(self.class_indices.keys()), self.n_way)

        support_imgs, support_labels = [], []
        query_imgs,   query_labels   = [], []

        for new_label, cls in enumerate(chosen_classes):
            pool = random.sample(self.class_indices[cls],
                                 self.n_support + self.n_query)
            s_pool = pool[:self.n_support]
            q_pool = pool[self.n_support:]

            for idx in s_pool:
                img, _ = self.dataset[idx]
                support_imgs.append(img)
                support_labels.append(new_label)

            for idx in q_pool:
                img, _ = self.dataset[idx]
                query_imgs.append(img)
                query_labels.append(new_label)

        return (
            torch.stack(support_imgs),
            torch.tensor(support_labels, dtype=torch.long),
            torch.stack(query_imgs),
            torch.tensor(query_labels, dtype=torch.long),
        )


def build_dataloaders(real_dir, fake_dir, n_samples=None,
                      batch_size=config.BATCH_SIZE):
    """Returns (train_loader, test_loader) for baseline CNN training."""
    train_ds = RareImageDataset(real_dir, fake_dir, n_samples=n_samples, split="train")
    test_ds  = RareImageDataset(real_dir, fake_dir, n_samples=n_samples, split="test")

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader
