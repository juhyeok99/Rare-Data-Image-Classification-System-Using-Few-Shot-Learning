"""
run_experiments.py

Runs the full set of experiments from the paper:
  - CNN baseline vs Proposed system
  - Across data sizes: 1000 / 500 / 100 samples
  - On painting and plant datasets (augmented + ArtGAN)

Usage:
    python run_experiments.py \
        --real_dir  data/raw/paintings \
        --fake_dir  data/augmented/paintings \
        --dataset_name  painting_proposed \
        --data_sizes 1000 500 100

    python run_experiments.py \
        --real_dir  data/raw/paintings \
        --fake_dir  data/artgan/paintings \
        --dataset_name  painting_artgan \
        --data_sizes 1000 500 100

    python run_experiments.py \
        --real_dir  data/raw/plants \
        --fake_dir  data/augmented/plants \
        --dataset_name  plant_proposed \
        --data_sizes 1000 500 100

Set --bayes_opt flag to run Bayesian hyperparameter search first.
"""

import os
import json
import argparse
import random

import torch
import numpy as np

import config
from data.dataset import RareImageDataset, build_dataloaders
from experiments.train_cnn import run_cnn_training
from experiments.train_proposed import run_proposed_training
from experiments.bayesian_opt import optimise_hyperparams
from utils.visualization import plot_comparison_bars, plot_accuracy_curves
from utils.metrics import format_metrics


def set_seed(seed=config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser(description="Few-shot rare data experiments")
    p.add_argument("--real_dir",      required=True,
                   help="Directory containing real/authentic images")
    p.add_argument("--fake_dir",      required=True,
                   help="Directory containing generated/fake images")
    p.add_argument("--dataset_name",  default="experiment",
                   help="Label for output files")
    p.add_argument("--data_sizes",    nargs="+", type=int,
                   default=config.DATA_SIZES)
    p.add_argument("--epochs",        type=int, default=config.EPOCHS)
    p.add_argument("--batch_size",    type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr",            type=float, default=config.LR)
    p.add_argument("--dropout",       type=float, default=config.DROPOUT)
    p.add_argument("--bayes_opt",     action="store_true",
                   help="Run Bayesian hyperparameter optimisation first")
    p.add_argument("--device",        default=None)
    return p.parse_args()


def run_single_experiment(real_dir, fake_dir, n_samples,
                           epochs, lr, dropout, batch_size,
                           device, dataset_name, result_dir):
    """
    Train CNN and proposed system on n_samples images, evaluate, save results.
    Returns (cnn_metrics, proposed_metrics).
    """
    print(f"\n{'='*60}")
    print(f" n_samples={n_samples}  lr={lr:.4f}  dropout={dropout:.2f}")
    print(f"{'='*60}")

    train_loader, test_loader = build_dataloaders(
        real_dir, fake_dir,
        n_samples=n_samples,
        batch_size=batch_size,
    )
    train_dataset = RareImageDataset(real_dir, fake_dir,
                                     n_samples=n_samples, split="train")

    # ── CNN baseline ───────────────────────────────────────────────────────
    print("\n[1/2] Training CNN baseline ...")
    _, cnn_tr_hist, cnn_te_hist, cnn_m = run_cnn_training(
        train_loader, test_loader,
        epochs=epochs, lr=lr, dropout=dropout,
        device=device, verbose=True,
    )
    print(format_metrics(cnn_m, prefix="[CNN final]"))

    # ── Proposed system ────────────────────────────────────────────────────
    print("\n[2/2] Training proposed system ...")
    _, prop_loss_hist, prop_te_hist, prop_m = run_proposed_training(
        train_dataset, test_loader,
        epochs=epochs, lr=lr, dropout=dropout,
        device=device, verbose=True,
    )
    print(format_metrics(prop_m, prefix="[Proposed final]"))

    # ── save curves ────────────────────────────────────────────────────────
    tag = f"{dataset_name}_n{n_samples}"
    plot_accuracy_curves(
        cnn_te_hist, prop_te_hist,
        title=f"Test accuracy — {dataset_name} ({n_samples} samples)",
        save_path=os.path.join(result_dir, f"{tag}_accuracy_curve.png"),
    )

    return cnn_m, prop_m


def main():
    args   = parse_args()
    set_seed()

    device = torch.device(args.device) if args.device \
             else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    result_dir = os.path.join(config.RESULT_DIR, args.dataset_name)
    os.makedirs(result_dir, exist_ok=True)

    lr      = args.lr
    dropout = args.dropout

    # optional Bayesian optimisation (done once on 500-sample split)
    if args.bayes_opt:
        print("\nRunning Bayesian optimisation on CNN ...")
        best_cnn = optimise_hyperparams(
            "cnn", args.real_dir, args.fake_dir, n_samples=500, device=device
        )
        print("\nRunning Bayesian optimisation on proposed system ...")
        best_prop = optimise_hyperparams(
            "proposed", args.real_dir, args.fake_dir, n_samples=500, device=device
        )
        # use the proposed system's best params for the full experiment
        lr      = best_prop["lr"]
        dropout = best_prop["dropout"]
        print(f"\nSelected lr={lr:.6f}  dropout={dropout:.3f}")

    # ── run across all data sizes ──────────────────────────────────────────
    cnn_accs, prop_accs  = [], []
    all_results          = {}

    for n in args.data_sizes:
        cnn_m, prop_m = run_single_experiment(
            real_dir     = args.real_dir,
            fake_dir     = args.fake_dir,
            n_samples    = n,
            epochs       = args.epochs,
            lr           = lr,
            dropout      = dropout,
            batch_size   = args.batch_size,
            device       = device,
            dataset_name = args.dataset_name,
            result_dir   = result_dir,
        )
        cnn_accs.append(cnn_m["accuracy"])
        prop_accs.append(prop_m["accuracy"])
        all_results[n] = {"cnn": cnn_m, "proposed": prop_m}

    # ── summary bar chart ──────────────────────────────────────────────────
    plot_comparison_bars(
        data_sizes    = args.data_sizes,
        cnn_accs      = cnn_accs,
        proposed_accs = prop_accs,
        title         = f"Classification accuracy — {args.dataset_name}",
        save_path     = os.path.join(result_dir, f"{args.dataset_name}_summary.png"),
    )

    # ── save JSON summary ──────────────────────────────────────────────────
    summary_path = os.path.join(result_dir, "results.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {summary_path}")

    # ── print table ────────────────────────────────────────────────────────
    print(f"\n{'Data':>8}  {'CNN Acc':>10}  {'Prop Acc':>10}")
    print("-" * 34)
    for n, ca, pa in zip(args.data_sizes, cnn_accs, prop_accs):
        print(f"{n:>8}  {ca:>10.4f}  {pa:>10.4f}")


if __name__ == "__main__":
    main()
