"""
Neural-net-generated attention weights on top of a reservoir.
"""

import torch.nn as nn

from .reservoir import _ReservoirExtractor

__all__ = ["AttReservoirNet"]


class _DynAttentionHead(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size):
        super().__init__()
        self.generate = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size * input_size),
        )
        self.classifier = nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size
        self.input_size = input_size  # Store the generic input_size

    def forward(self, r_state):
        B = r_state.size(0)
        W = self.generate(r_state).view(B, self.hidden_size, self.input_size)
        out = (W @ r_state.unsqueeze(-1)).squeeze(-1)
        return self.classifier(out)


class AttReservoirNet(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_size,
        reservoir_size,
        num_reservoirs,
        deep_type,
        hidden_size,
    ):
        super().__init__()
        self.extractor = _ReservoirExtractor(
            vocab_size, embed_size, reservoir_size, num_reservoirs, deep_type
        )

        # --- Determine the extractor's output size ---
        # This logic must match the extractor's readout strategy
        all_layer_readout_types = ["deep_esn", "deep_ia", "grouped_esn", "deep_esn_d"]
        if deep_type in all_layer_readout_types:
            extractor_output_size = reservoir_size * num_reservoirs
        else:  # 'deep', 'deep_input', or (implicitly) num_reservoirs=1
            extractor_output_size = reservoir_size

        self.head = _DynAttentionHead(extractor_output_size, hidden_size, vocab_size)

    def forward(self, x):
        r = self.extractor(x)
        return self.head(r)
