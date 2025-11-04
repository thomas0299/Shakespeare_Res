"""
Train classic reservoir network on Shakespeare.
"""

import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.cuda.amp import GradScaler
import time

from utils.data_utils import load_corpus, build_vocab, TextDataset
from utils.training_utils import set_seed, get_hash, count_parameters, run_epoch
from models.reservoir import ReservoirNet

DATA_PATH = "data/shakespeare.txt"
RESULTS_CSV = "results_reservoir.csv"
CKPT_DIR = "checkpoints"
LOG_DIR = "training_logs"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

h = {
    "TRAIN_SPLIT": 0.9, "BATCH": 1024, "EPOCHS": 3,
    "LR": 1e-4, "SEQ": 32, "EMB": 16, "RES": 50, "NUM_RES": 1,
    "DEEP_TYPE": "deep",
}


def get_char_ngrams(text, n):
    """Helper function to extract a set of unique character n-grams from text."""
    ngrams = set()
    for i in range(len(text) - n + 1):
        ngrams.add(text[i:i+n])  # Use string slices as n-grams
    return ngrams


def calculate_overlap(generated_text, reference_text, n):
    """Calculates Overlap-n metric: |Gn ∩ Rn| / |Gn|"""
    Gn = get_char_ngrams(generated_text, n)
    if not Gn:  # Avoid division by zero
        return 0.0

    Rn = get_char_ngrams(reference_text, n)

    intersection = Gn.intersection(Rn)

    return len(intersection) / len(Gn)


@torch.no_grad()
def generate_text(model, c2i, i2c, seed_text, length, seq_len, device, temperature=0.8):
    """Auto-regressively generate text from the model using temperature sampling."""
    model.eval() 
    
    tokens = [c2i[char] for char in seed_text]
    generated_chars = []

    for _ in range(length):
        input_seq = tokens[-seq_len:]
        input_tensor = torch.tensor([input_seq], device=device)
        
        logits = model(input_tensor) # Shape [1, vocab_size]
        
        # --- MODIFICATION START ---
        # Apply temperature
        # A higher temp (e.g., 1.0) is more random.
        # A lower temp (e.g., 0.2) is more deterministic (like argmax).
        logits_with_temp = logits / temperature
        
        # Get probabilities using softmax
        probabilities = torch.softmax(logits_with_temp, dim=1)
        
        # Sample from the probability distribution
        next_token = torch.multinomial(probabilities, num_samples=1).item()
        # --- MODIFICATION END ---
        
        tokens.append(next_token)
        generated_chars.append(i2c[next_token])

    return "".join(generated_chars)


def main():
    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    text = load_corpus(DATA_PATH)
    vocab, c2i, i2c = build_vocab(text)  # Capture i2c (index-to-char)
    split = int(len(text) * h["TRAIN_SPLIT"])
    train_ds = TextDataset(text[:split], h["SEQ"], c2i)
    test_ds = TextDataset(text[split:], h["SEQ"], c2i)

    train_dl = DataLoader(train_ds, h["BATCH"], shuffle=True,  num_workers=2)
    test_dl = DataLoader(test_ds,  h["BATCH"], shuffle=False, num_workers=2)

    model = ReservoirNet(len(vocab), h["EMB"], h["RES"], h["NUM_RES"], h["DEEP_TYPE"]).to(device)
    parameter_count = count_parameters(model)
    print("Params:", parameter_count)
    opt = optim.Adam(model.classifier.parameters(), lr=h["LR"])
    crit = nn.CrossEntropyLoss()
    scaler = GradScaler()

    results = []

    total_start_time = time.time()

    for epoch in range(1, h["EPOCHS"] + 1):
        
        # --- MODIFICATION START ---
        # Time training and validation (inference) separately
        train_start_time = time.time()
        tr = run_epoch(model, train_dl, crit, opt, scaler, device)
        train_time = time.time() - train_start_time

        val_start_time = time.time()
        te = run_epoch(model, test_dl,  crit, None, None,  device)
        val_time = time.time() - val_start_time

        print(f"Epoch {epoch}: train {tr:.4f} ({train_time:.2f}s) | test {te:.4f} ({val_time:.2f}s)")
        results.append({
            "epoch": epoch, 
            "train_loss": tr, "test_loss": te, 
            "train_time_s": train_time, "val_time_s": val_time
        })

    total_training_time = time.time() - total_start_time
    file_name = f"reservoir_{get_hash(h)}"

    results_df = pd.DataFrame(results)
    results_df.to_csv(file_name+".csv", index=False)
    total_validation_time = results_df['val_time_s'].sum()
    torch.save(model.state_dict(), os.path.join(CKPT_DIR, file_name+".pth"))

    print("\n--- N-gram Overlap Evaluation ---")

    ngram_eval_start_time = time.time()

    # 1. Define reference text (the test set)
    reference_text = text[split:]

    # 2. Define a seed from the start of the test set
    seed_text = reference_text[:h["SEQ"]]

    # --- MODIFY THIS BLOCK ---
    GEN_LENGTH = 20000 # Define a much smaller, fixed length
    print(f"Generating {GEN_LENGTH} characters...")
    generated_text = generate_text(
        model, c2i, i2c, 
        seed_text=seed_text, 
        length=GEN_LENGTH, # Use the fixed length
        seq_len=h["SEQ"], 
        device=device,
        temperature=0.8
    )
    # --- END MODIFICATION ---

    # 4. Calculate and print overlap
    n_values = [3, 5, 7, 8]

    # --- ADD THIS DICT ---
    ngram_results = {}
    # --- END ADDITION ---

    for n in n_values:
        overlap = calculate_overlap(generated_text, reference_text, n)
        print(f"Overlap-{n}: {overlap:.4f}")

        # --- ADD THIS LINE ---
        ngram_results[f'overlap_{n}'] = overlap
        # --- END ADDITION ---
    total_ngram_eval_time = time.time() - ngram_eval_start_time
    print(f"Total n-gram evaluation time: {total_ngram_eval_time:.2f}s")

    # --- END EVALUATION BLOCK ---

    # --- ADD THIS NEW BLOCK ---
    # Create new summary CSV with parameters and total time
    SUMMARY_CSV = os.path.join(LOG_DIR, f"{file_name}_summary.csv") # <-- Use unique name
    summary_data = h.copy()  # Start with all hyperparameters
    summary_data['hash'] = file_name
    summary_data['total_training_time_s'] = total_training_time
    summary_data['total_validation_time_s'] = total_validation_time # Sum of all val epochs
    summary_data['total_ngram_eval_time_s'] = total_ngram_eval_time # Time for gen + n-gram
    summary_data['parameter_count'] = parameter_count

    summary_data.update(ngram_results)

    summary_df = pd.DataFrame([summary_data])

    summary_df.to_csv(SUMMARY_CSV, index=False)


if __name__ == "__main__":
    main()
