"""
Shared helpers: seeding, parameter counting, training/eval loops.
"""

import hashlib
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.amp import autocast
from tqdm import tqdm

__all__ = ["set_seed", "get_hash", "count_parameters", "run_epoch"]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_hash(hparams: dict, length=8) -> str:
    pairs = "&".join(f"{k}={v}" for k, v in sorted(hparams.items()))
    return hashlib.sha1(pairs.encode()).hexdigest()[:length]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_epoch(model, dataloader, criterion, optimizer=None, scaler=None, device="cpu"):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss = 0.0

    with torch.set_grad_enabled(is_train):
        for X, y in tqdm(dataloader):
            X, y = X.to(device), y.to(device)
            if is_train:
                optimizer.zero_grad()

            with autocast(device_type=device):
                logits = model(X)
                loss = criterion(logits, y)

            if is_train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item()

    return total_loss / len(dataloader)


def load_checkpoint(model, ckpt_path, log_path, summary_path, device):
    """
    Loads model checkpoint and training history.

    Returns:
        - model: The model with loaded state_dict.
        - results (list): List of dictionaries of historical results.
        - start_epoch (int): The epoch to resume training from.
        - old_total_training_time (float): Cumulative training time from old runs.
    """
    start_epoch = 1
    results = []
    old_total_training_time = 0.0

    if os.path.exists(ckpt_path) and os.path.exists(log_path):
        print(f"Found existing checkpoint. Loading from {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

        # Load old epoch-by-epoch results
        results_df = pd.read_csv(log_path)
        results = results_df.to_dict("records")  # Load old results
        start_epoch = results_df["epoch"].max() + 1
        print(f"Resuming from epoch {start_epoch}")

        # Load old summary file to get cumulative time
        if os.path.exists(summary_path):
            try:
                old_summary_df = pd.read_csv(summary_path)
                old_total_training_time = old_summary_df["total_training_time_s"].iloc[
                    0
                ]
            except Exception as e:
                print(
                    f"Warning: Could not read old summary file {summary_path}. Resetting time. {e}"
                )
                old_total_training_time = 0.0
    else:
        print("Starting new training run.")

    return model, results, start_epoch, old_total_training_time


def save_training_summary(
    summary_path,
    h_params,
    file_hash,
    old_total_training_time,
    this_run_training_time,
    total_validation_time,
    total_ngram_eval_time,
    parameter_count,
    ngram_results,
):
    """Saves the final training summary CSV."""
    summary_data = h_params.copy()  # Start with all hyperparameters
    summary_data["hash"] = file_hash

    # Add time from this run to the old total time
    cumulative_training_time = old_total_training_time + this_run_training_time
    summary_data["total_training_time_s"] = cumulative_training_time

    summary_data["total_validation_time_s"] = total_validation_time
    summary_data["total_ngram_eval_time_s"] = total_ngram_eval_time
    summary_data["parameter_count"] = parameter_count

    summary_data.update(ngram_results)
    summary_df = pd.DataFrame([summary_data])
    summary_df.to_csv(summary_path, index=False)
    print(f"Training summary saved to {summary_path}")
