"""
Classic (static) reservoir feature extractor + linear classifier.
NOW STACKABLE with multiple deep types.
"""

import math

import torch
import torch.nn as nn

__all__ = ["ReservoirNet"]


class _ReservoirExtractor(nn.Module):
    def __init__(
        self, vocab_size, embed_size, reservoir_size, num_reservoirs, deep_type="deep"
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.embed.weight.requires_grad = False

        self.num_reservoirs = num_reservoirs
        self.deep_type = deep_type  # Store the deep_type

        # --- Define which models use "all layer" readout (like the paper) ---
        self.all_layer_readout = ["deep_esn", "deep_ia", "grouped_esn", "deep_esn_d"]

        self.Win_list = nn.ParameterList()
        self.R_list = nn.ParameterList()

        # --- Create the stack ---
        current_input_size = embed_size
        for i in range(num_reservoirs):

            # --- MODIFICATION: Adjust input size based on deep_type ---
            if self.deep_type == "grouped_esn":
                # All layers get the same 'embed_size' input
                current_input_size = embed_size
            elif self.deep_type == "deep_esn_d" and i > 0:
                # Input is [embed_input, state_0, ..., state_i-1]
                current_input_size = embed_size + i * reservoir_size
            elif (
                self.deep_type == "deep_input" or self.deep_type == "deep_ia"
            ) and i > 0:
                # Input is concatenated: [prev_state, embed_input]
                current_input_size = reservoir_size + embed_size
            elif (self.deep_type == "deep" or self.deep_type == "deep_esn") and i > 0:
                # Input is just [prev_state]
                current_input_size = reservoir_size

            else:  # i == 0 (for all deep types except grouped_esn)
                # Input is [embed_input]
                current_input_size = embed_size
            # --- END MODIFICATION ---

            # Input matrix
            self.Win_list.append(
                nn.Parameter(
                    torch.randn(current_input_size, reservoir_size)
                    / math.sqrt(current_input_size)
                    / 0.9,
                    requires_grad=False,
                )
            )
            # Reservoir matrix
            self.R_list.append(
                nn.Parameter(
                    torch.randn(reservoir_size, reservoir_size)
                    / math.sqrt(reservoir_size)
                    / 0.9,
                    requires_grad=False,
                )
            )

        # --- MODIFICATION: Norm layer size depends on readout strategy ---
        if self.deep_type in self.all_layer_readout:
            # Readout concatenates all layer states
            self.norm = nn.LayerNorm(reservoir_size * num_reservoirs)
        else:
            # 'deep' & 'deep_input' only read out from the last layer
            self.norm = nn.LayerNorm(reservoir_size)
        # --- END MODIFICATION ---

    def forward(self, x):
        # x: [B, T]
        x_embed = self.embed(x)  # [B, T, embed_size]
        B, T, _ = x_embed.size()

        states = [
            torch.zeros(B, self.R_list[i].size(0), device=x.device)
            for i in range(self.num_reservoirs)
        ]

        for t in range(T):
            embedded_input_t = x_embed[:, t, :]  # [B, embed_size]

            # This is the input for the *first* layer (or all layers in grouped_esn)
            current_layer_input = embedded_input_t

            for i in range(self.num_reservoirs):
                state_part = states[i] @ self.R_list[i]
                if self.deep_type == "deep_esn_d" and i > 0:
                    # 'deep_esn_d' gets [embed_input, state_0, ..., state_i-1]
                    current_layer_input = torch.cat(
                        [embedded_input_t] + states[0:i], dim=-1
                    )

                # --- MODIFICATION: Handle all 5 stacking types ---
                elif (
                    self.deep_type == "deep_input" or self.deep_type == "deep_ia"
                ) and i > 0:
                    # 'deep_input' / 'deep_ia' get [prev_state, embed_input]
                    current_layer_input = torch.cat(
                        (states[i - 1], embedded_input_t), dim=-1
                    )
                elif (
                    self.deep_type == "deep" or self.deep_type == "deep_esn"
                ) and i > 0:
                    # 'deep' / 'deep_esn' get [prev_state]
                    current_layer_input = states[i - 1]
                elif self.deep_type == "grouped_esn":
                    # 'grouped_esn' gets [embed_input] for all layers
                    current_layer_input = embedded_input_t
                # else for i==0, current_layer_input is already embedded_input_t
                # --- END MODIFICATION ---

                input_part = current_layer_input @ self.Win_list[i]

                states[i] = torch.tanh(state_part + input_part)

        # --- MODIFICATION: Return state based on readout strategy ---
        if self.deep_type in self.all_layer_readout:
            # Concatenate all final states (as in the paper)
            final_state = torch.cat(states, dim=-1)
            return self.norm(final_state)
        else:
            # 'deep' and 'deep_input' only return the last state (original behavior)
            return self.norm(states[-1])
        # --- END MODIFICATION ---


class ReservoirNet(nn.Module):
    def __init__(
        self, vocab_size, embed_size, reservoir_size, num_reservoirs=1, deep_type="deep"
    ):
        super().__init__()

        # --- MODIFICATION: Pass all arguments to the extractor ---
        self.extractor = _ReservoirExtractor(
            vocab_size, embed_size, reservoir_size, num_reservoirs, deep_type
        )

        # --- MODIFICATION: Classifier input size depends on readout strategy ---
        if deep_type in ["deep_esn", "deep_ia", "grouped_esn", "deep_esn_d"]:
            classifier_input_size = reservoir_size * num_reservoirs
        else:  # 'deep' or 'deep_input'
            classifier_input_size = reservoir_size

        self.classifier = nn.Linear(classifier_input_size, vocab_size)
        # --- END MODIFICATION ---

    def forward(self, x):
        r = self.extractor(x)
        return self.classifier(r)
