"""
prepare_data.py

Downloads / organises the Wikiart dataset and runs the data augmentation
model to generate fake images for the experiments.

Steps:
  1. Point this script at a directory of raw artwork images.
  2. It trains the StructureAwareAugmentor on those images.
  3. Generated images that pass PSNR/SSIM quality gates are saved to
     data/augmented/<category>.
  4. The same raw images + generated images form the binary dataset used
     in run_experiments.py.

If you have the ArtGAN-generated images from Wikiart, place them in
  data/artgan/<category>/
and pass that as --fake_dir when running experiments.

Usage:
    python prepare_data.py \
        --raw_dir   data/raw/paintings \
        --out_dir   data/augmented/paintings \
        --aug_ckpt  checkpoints/augmentor_painting.pt
"""

import os
import argparse
import torch

import config
from data.augmentation import train_augmentor, generate_augmented_dataset, \
                              StructureAwareAugmentor


def parse_args():
    p = argparse.ArgumentParser(description="Prepare augmented dataset")
    p.add_argument("--raw_dir",  required=True,
                   help="Directory with real source images")
    p.add_argument("--out_dir",  required=True,
                   help="Output directory for generated images")
    p.add_argument("--aug_ckpt", default=None,
                   help="Path to save/load augmentor weights")
    p.add_argument("--epochs",   type=int, default=config.AUG_EPOCHS)
    p.add_argument("--load_ckpt", action="store_true",
                   help="Load existing augmentor instead of retraining")
    p.add_argument("--no_filter", action="store_true",
                   help="Skip PSNR/SSIM quality filtering")
    p.add_argument("--device",   default=None)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device) if args.device \
             else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.out_dir, exist_ok=True)
    if args.aug_ckpt:
        os.makedirs(os.path.dirname(args.aug_ckpt) or ".", exist_ok=True)

    # ── train or load augmentor ───────────────────────────────────────────
    if args.load_ckpt and args.aug_ckpt and os.path.exists(args.aug_ckpt):
        print(f"Loading augmentor from {args.aug_ckpt}")
        model = StructureAwareAugmentor().to(device)
        model.load_state_dict(torch.load(args.aug_ckpt, map_location=device))
    else:
        print("Training augmentor ...")
        model = train_augmentor(
            img_dir   = args.raw_dir,
            save_path = args.aug_ckpt,
            epochs    = args.epochs,
            device    = device,
        )

    # ── generate augmented dataset ────────────────────────────────────────
    print("Generating augmented images ...")
    n_accepted = generate_augmented_dataset(
        model         = model,
        img_dir       = args.raw_dir,
        out_dir       = args.out_dir,
        device        = device,
        filter_quality= not args.no_filter,
    )
    print(f"Done. {n_accepted} images saved to {args.out_dir}")


if __name__ == "__main__":
    main()
