"""
utils/visualization.py

Plotting helpers for accuracy curves and comparison bar charts.
"""

import os
import matplotlib
matplotlib.use("Agg")   # headless
import matplotlib.pyplot as plt
import numpy as np

import config


def plot_accuracy_curves(cnn_hist, proposed_hist, title="Training Accuracy",
                         save_path=None):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(cnn_hist, label="CNN (baseline)", color="#2c7bb6", lw=1.8)
    ax.plot(proposed_hist, label="Proposed system", color="#d7191c",
            lw=1.8, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_comparison_bars(data_sizes, cnn_accs, proposed_accs,
                         title="Classification accuracy by data size",
                         save_path=None):
    """
    Replicates the bar chart style from Figures 6-9 in the paper.
    data_sizes : [1000, 500, 100]
    cnn_accs   : matching accuracy list
    proposed_accs: matching accuracy list
    """
    x = np.arange(len(data_sizes))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars_cnn  = ax.bar(x - w/2, cnn_accs,      w, label="CNN",
                       color="#555555", alpha=0.9)
    bars_prop = ax.bar(x + w/2, proposed_accs, w, label="Proposed system",
                       color="#888888", alpha=0.9)

    # value labels on bars
    for bar in bars_cnn:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.4f}", ha="center", va="bottom",
                fontsize=8, fontweight="bold")
    for bar in bars_prop:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.4f}", ha="center", va="bottom",
                fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in data_sizes])
    ax.set_xlabel("Number of data")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_loss_curve(losses, label="Loss", save_path=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses, lw=1.5, color="#1a9641")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(label)
    ax.set_title(f"{label} curve")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()
