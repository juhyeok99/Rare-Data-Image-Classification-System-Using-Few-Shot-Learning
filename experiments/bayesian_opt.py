"""
experiments/bayesian_opt.py

Bayesian optimisation over learning rate and dropout for both models.
Uses the bayesian-optimization library as a black-box optimiser.

The paper explicitly states hyperparameters were chosen via Bayesian
optimisation (Section 4.1 / [21]), so we replicate that selection process
rather than treating the paper's values as fixed magic numbers.
"""

import torch
from bayes_opt import BayesianOptimization

import config
from data.dataset import RareImageDataset, build_dataloaders
from experiments.train_cnn import run_cnn_training
from experiments.train_proposed import run_proposed_training


def _cnn_objective(real_dir, fake_dir, n_samples, device):
    """Returns a callable for the CNN objective (accuracy to maximise)."""

    def _obj(lr, dropout):
        train_loader, test_loader = build_dataloaders(
            real_dir, fake_dir, n_samples=n_samples,
            batch_size=config.BATCH_SIZE,
        )
        # use fewer epochs during search to keep wall time reasonable
        _, _, _, metrics = run_cnn_training(
            train_loader, test_loader,
            epochs=200,
            lr=lr,
            dropout=dropout,
            device=device,
            verbose=False,
        )
        return metrics.get("accuracy", 0.0)

    return _obj


def _proposed_objective(real_dir, fake_dir, n_samples, device):
    """Returns a callable for the proposed system objective."""

    def _obj(lr, dropout):
        train_ds = RareImageDataset(real_dir, fake_dir,
                                    n_samples=n_samples, split="train")
        _, test_loader = build_dataloaders(
            real_dir, fake_dir, n_samples=n_samples,
            batch_size=config.BATCH_SIZE,
        )
        # keep epochs low during search
        _, _, _, metrics = run_proposed_training(
            train_ds, test_loader,
            epochs=200,
            lr=lr,
            dropout=dropout,
            device=device,
            verbose=False,
        )
        return metrics.get("accuracy", 0.0)

    return _obj


def optimise_hyperparams(model_type, real_dir, fake_dir, n_samples,
                          device=None,
                          init_points=config.BAYES_INIT_POINTS,
                          n_iter=config.BAYES_N_ITER):
    """
    Run Bayesian optimisation for the specified model type.

    model_type : "cnn" or "proposed"
    Returns dict with best lr and dropout.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_type == "cnn":
        objective = _cnn_objective(real_dir, fake_dir, n_samples, device)
    elif model_type == "proposed":
        objective = _proposed_objective(real_dir, fake_dir, n_samples, device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    optimizer = BayesianOptimization(
        f=objective,
        pbounds=config.BAYES_PBOUNDS,
        random_state=config.SEED,
        verbose=2,
    )
    optimizer.maximize(init_points=init_points, n_iter=n_iter)

    best_params = optimizer.max["params"]
    best_score  = optimizer.max["target"]
    print(f"\n[BayesOpt/{model_type}]  best accuracy={best_score:.4f}  "
          f"lr={best_params['lr']:.6f}  dropout={best_params['dropout']:.3f}")

    return best_params
