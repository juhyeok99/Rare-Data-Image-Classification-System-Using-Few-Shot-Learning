"""
config.py

Central place for hyperparameters and paths so we're not hunting through
files to change a learning rate or data directory.
"""

import os

# ── paths ──────────────────────────────────────────────────────────────────
ROOT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(ROOT_DIR, "data", "raw")
AUG_DIR    = os.path.join(ROOT_DIR, "data", "augmented")
CKPT_DIR   = os.path.join(ROOT_DIR, "checkpoints")
RESULT_DIR = os.path.join(ROOT_DIR, "results")

for _d in [DATA_DIR, AUG_DIR, CKPT_DIR, RESULT_DIR]:
    os.makedirs(_d, exist_ok=True)

# ── image settings ─────────────────────────────────────────────────────────
IMG_SIZE   = 64       # resize target; 64x64 keeps things manageable
CHANNELS   = 3

# ── augmentation model ─────────────────────────────────────────────────────
AUG_LATENT_DIM  = 256
AUG_EPOCHS      = 50
AUG_LR          = 1e-3
AUG_BATCH_SIZE  = 16

# quality thresholds for generated image filtering
PSNR_THRESHOLD  = 20.0   # dB
SSIM_THRESHOLD  = 0.5

# ── training (CNN baseline & proposed system) ──────────────────────────────
EPOCHS      = 1000
BATCH_SIZE  = 32
LR          = 1e-3
DROPOUT     = 0.5
WEIGHT_DECAY = 1e-4    # light L2, wasn't specified in paper but helps

# ── few-shot learning settings ─────────────────────────────────────────────
N_WAY       = 2    # binary: real vs. fake
N_SUPPORT   = 5    # support samples per class during meta-training episodes
N_QUERY     = 15   # query samples per class per episode
N_EPISODES  = 100  # episodes per epoch

# ── experiment sizes (data counts) ────────────────────────────────────────
DATA_SIZES  = [1000, 500, 100]

# ── reproducibility ────────────────────────────────────────────────────────
SEED        = 42

# ── Bayesian optimisation search space ────────────────────────────────────
BAYES_PBOUNDS = {
    "lr":      (1e-4, 1e-2),
    "dropout": (0.3, 0.7),
}
BAYES_INIT_POINTS = 5
BAYES_N_ITER      = 15
