"""
inference.py

Run inference with a trained proposed system or CNN baseline.

Loads model weights and a small support set, then classifies images
in a query directory.

Usage (proposed system):
    python inference.py \
        --model      proposed \
        --ckpt       checkpoints/proposed_painting_n100.pt \
        --support_real  data/raw/paintings \
        --support_fake  data/augmented/paintings \
        --query_dir     path/to/new_images \
        --n_support  5

Usage (CNN):
    python inference.py \
        --model      cnn \
        --ckpt       checkpoints/cnn_painting_n100.pt \
        --query_dir  path/to/new_images
"""

import os
import argparse
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image

import config
from models.cnn_baseline import CNNClassifier
from models.proposed_system import ProposedFewShotSystem


_eval_tf = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

LABELS = {0: "real", 1: "fake"}


def load_images_from_dir(img_dir, max_n=None):
    paths = (sorted(Path(img_dir).glob("*.jpg")) +
             sorted(Path(img_dir).glob("*.png")) +
             sorted(Path(img_dir).glob("*.jpeg")))
    if max_n:
        paths = paths[:max_n]
    imgs = [_eval_tf(Image.open(p).convert("RGB")) for p in paths]
    return torch.stack(imgs), paths


def run_cnn_inference(ckpt, query_dir, device):
    model = CNNClassifier()
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device).eval()

    query_imgs, query_paths = load_images_from_dir(query_dir)

    with torch.no_grad():
        logits = model(query_imgs.to(device))
        preds  = (torch.sigmoid(logits) >= 0.5).long().cpu()

    for path, pred in zip(query_paths, preds):
        print(f"  {path.name:<40}  → {LABELS[int(pred)]}")


def run_proposed_inference(ckpt, support_real, support_fake,
                            query_dir, n_support, device):
    model = ProposedFewShotSystem()
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device).eval()

    # build support set
    real_imgs, _ = load_images_from_dir(support_real, max_n=n_support)
    fake_imgs, _ = load_images_from_dir(support_fake, max_n=n_support)

    support_imgs   = torch.cat([real_imgs, fake_imgs]).to(device)
    support_labels = torch.tensor([0]*len(real_imgs) + [1]*len(fake_imgs),
                                   dtype=torch.long).to(device)

    model.build_support_cache(support_imgs, support_labels)

    query_imgs, query_paths = load_images_from_dir(query_dir)

    with torch.no_grad():
        preds = model.predict(query_imgs.to(device)).cpu()

    for path, pred in zip(query_paths, preds):
        print(f"  {path.name:<40}  → {LABELS[int(pred)]}")


def parse_args():
    p = argparse.ArgumentParser(description="Classify images with trained model")
    p.add_argument("--model",        choices=["cnn", "proposed"], required=True)
    p.add_argument("--ckpt",         required=True)
    p.add_argument("--query_dir",    required=True)
    p.add_argument("--support_real", default=None)
    p.add_argument("--support_fake", default=None)
    p.add_argument("--n_support",    type=int, default=5)
    p.add_argument("--device",       default=None)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device) if args.device \
             else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nClassifying images in: {args.query_dir}")
    print("-" * 55)

    if args.model == "cnn":
        run_cnn_inference(args.ckpt, args.query_dir, device)
    else:
        if not args.support_real or not args.support_fake:
            raise ValueError(
                "--support_real and --support_fake are required for proposed system"
            )
        run_proposed_inference(
            ckpt         = args.ckpt,
            support_real = args.support_real,
            support_fake = args.support_fake,
            query_dir    = args.query_dir,
            n_support    = args.n_support,
            device       = device,
        )


if __name__ == "__main__":
    main()
