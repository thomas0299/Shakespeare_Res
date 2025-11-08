import time

import torch


def get_char_ngrams(text, n):
    """Helper function to extract a set of unique character n-grams from text."""
    ngrams = set()
    for i in range(len(text) - n + 1):
        ngrams.add(text[i : i + n])  # Use string slices as n-grams
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

        logits = model(input_tensor)  # Shape [1, vocab_size]

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


def evaluate_ngram_overlap(
    model,
    c2i,
    i2c,
    reference_text,
    seed_text,
    seq_len,
    device,
    gen_length=20000,
    n_values=[3, 5, 7, 8],
    temperature=0.8,
):
    """
    Evaluates n-gram overlap of a model's generated text against a reference.

    Returns:
        - ngram_results (dict): A dictionary with overlap_{n} keys.
        - total_ngram_eval_time (float): Time taken for the evaluation.
    """
    print(f"\n--- N-gram Overlap Evaluation (Gen length: {gen_length}) ---")
    ngram_eval_start_time = time.time()

    # Ensure model is in eval mode for generation
    model.eval()

    print(f"Generating {gen_length} characters...")
    generated_text = generate_text(
        model,
        c2i,
        i2c,
        seed_text=seed_text,
        length=gen_length,
        seq_len=seq_len,
        device=device,
        temperature=temperature,
    )

    ngram_results = {}
    for n in n_values:
        overlap = calculate_overlap(generated_text, reference_text, n)
        print(f"Overlap-{n}: {overlap:.4f}")
        ngram_results[f"overlap_{n}"] = overlap

    total_ngram_eval_time = time.time() - ngram_eval_start_time
    print(f"Total n-gram evaluation time: {total_ngram_eval_time:.2f}s")

    return ngram_results, total_ngram_eval_time
