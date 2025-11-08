"""
Train classic reservoir network on Shakespeare.
"""

import argparse
import os
import time

import pandas as pd
import torch
from torch import nn, optim
from torch.cuda.amp import GradScaler

from evaluation import evaluate_ngram_overlap
from models.reservoir import ReservoirNet
from utils.data_utils import setup_dataloaders
from utils.training_utils import (
    count_parameters,
    get_hash,
    load_checkpoint,
    run_epoch,
    save_training_summary,
    set_seed,
)

DATA_PATH = "data/shakespeare.txt"
RESULTS_CSV = "results_reservoir.csv"
CKPT_DIR = "checkpoints"
LOG_DIR = "training_logs"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

h = {
    "TRAIN_SPLIT": 0.9,
    "BATCH": 1024,
    "EPOCHS": 2,
    "LR": 1e-4,
    "SEQ": 32,
    "EMB": 16,
    "RES": 750,
    "NUM_RES": 2,
    "DEEP_TYPE": "grouped_esn",
}

parser = argparse.ArgumentParser(description="Train a Stacked Reservoir Network.")
parser.add_argument("--EPOCHS", type=int, help="Number of training epochs.")
parser.add_argument("--RES", type=int, help="Reservoir size (number of units).")
parser.add_argument("--NUM_RES", type=int, help="Number of stacked reservoirs.")
parser.add_argument(
    "--DEEP_TYPE",
    type=str,
    help="Type of deep architecture (deep, deep_input, deep_esn, deep_ia, deep_esn_d, grouped_esn).",
)

args = parser.parse_args()

for key, value in vars(args).items():
    if value is not None:
        h[key] = value

print(h)


def main():
    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    h_for_hash = h.copy()
    h_for_hash.pop("EPOCHS", None)
    file_name = f"reservoir_{get_hash(h_for_hash)}"

    ckpt_path = os.path.join(CKPT_DIR, file_name + ".pth")
    log_path = os.path.join(LOG_DIR, file_name + ".csv")
    summary_path = os.path.join(LOG_DIR, f"{file_name}_summary.csv")

    train_dl, test_dl, vocab, c2i, i2c, reference_text = setup_dataloaders(
        DATA_PATH, h["TRAIN_SPLIT"], h["SEQ"], h["BATCH"]
    )

    model = ReservoirNet(
        len(vocab), h["EMB"], h["RES"], h["NUM_RES"], h["DEEP_TYPE"]
    ).to(device)
    parameter_count = count_parameters(model)
    print("Params:", parameter_count)
    opt = optim.Adam(model.classifier.parameters(), lr=h["LR"])
    crit = nn.CrossEntropyLoss()
    scaler = GradScaler()

    model, results, start_epoch, old_total_training_time = load_checkpoint(
        model, ckpt_path, log_path, summary_path, device
    )

    total_start_time = time.time()
    # Timer for *this run*

    total_desired_epochs = h["EPOCHS"]

    if start_epoch > total_desired_epochs:
        print(
            f"Model already trained for {start_epoch - 1} epochs. Skipping training loop."
        )
    else:
        print(f"Training from epoch {start_epoch} to {total_desired_epochs}...")
        for epoch in range(start_epoch, total_desired_epochs + 1):

            train_start_time = time.time()
            tr = run_epoch(model, train_dl, crit, opt, scaler, device)
            train_time = time.time() - train_start_time

            val_start_time = time.time()
            te = run_epoch(model, test_dl, crit, None, None, device)
            val_time = time.time() - val_start_time

            print(
                f"Epoch {epoch}: train {tr:.4f} ({train_time:.2f}s) | test {te:.4f} ({val_time:.2f}s)"
            )
            results.append(
                {
                    "epoch": epoch,
                    "train_loss": tr,
                    "test_loss": te,
                    "train_time_s": train_time,
                    "val_time_s": val_time,
                }
            )

    this_run_training_time = time.time() - total_start_time

    results_df = pd.DataFrame(results)
    results_df.to_csv(log_path, index=False)
    total_validation_time = results_df["val_time_s"].sum()
    torch.save(model.state_dict(), ckpt_path)

    seed_text = reference_text[: h["SEQ"]]

    ngram_results, total_ngram_eval_time = evaluate_ngram_overlap(
        model=model,
        c2i=c2i,
        i2c=i2c,
        reference_text=reference_text,
        seed_text=seed_text,
        seq_len=h["SEQ"],
        device=device,
        gen_length=20000,
        n_values=[3, 5, 7, 8],
        temperature=0.8,
    )

    save_training_summary(
        summary_path=summary_path,
        h_params=h,
        file_hash=file_name,
        old_total_training_time=old_total_training_time,
        this_run_training_time=this_run_training_time,
        total_validation_time=total_validation_time,
        total_ngram_eval_time=total_ngram_eval_time,
        parameter_count=parameter_count,
        ngram_results=ngram_results,
    )


if __name__ == "__main__":
    main()
