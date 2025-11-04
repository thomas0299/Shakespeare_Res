"""
Classic (static) reservoir feature extractor + linear classifier.
NOW STACKABLE with multiple deep types.
"""

import torch, torch.nn as nn
import math

__all__ = ["ReservoirNet"]

class _ReservoirExtractor(nn.Module):
    # --- MODIFICATION ---
    def __init__(self, vocab_size, embed_size, reservoir_size, num_reservoirs, deep_type="deep"):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.embed.weight.requires_grad = False
        
        self.num_reservoirs = num_reservoirs
        self.deep_type = deep_type # Store the deep_type
        
        self.Win_list = nn.ParameterList()
        self.R_list = nn.ParameterList()

        # --- Create the stack ---
        current_input_size = embed_size
        for i in range(num_reservoirs):
            
            # --- MODIFICATION: Adjust input size based on deep_type ---
            if self.deep_type == 'deep_input' and i > 0:
                # Input is concatenated: [prev_state, embed_input]
                current_input_size = reservoir_size + embed_size
            elif i > 0: # 'deep' type
                # Input is just [prev_state]
                current_input_size = reservoir_size
            else: # i == 0
                # Input is [embed_input]
                current_input_size = embed_size
            # --- END MODIFICATION ---

            # Input matrix
            self.Win_list.append(nn.Parameter(
                torch.randn(current_input_size, reservoir_size) / math.sqrt(current_input_size) / 0.9,
                requires_grad=False
            ))
            # Reservoir matrix
            self.R_list.append(nn.Parameter(
                torch.randn(reservoir_size, reservoir_size) / math.sqrt(reservoir_size) / 0.9,
                requires_grad=False
            ))
            
            # --- This logic is now handled above ---
            # current_input_size = reservoir_size 
        
        self.norm = nn.LayerNorm(reservoir_size)
        
        # --- REMOVED (now in lists) ---
        # self.R = ...
        # self.Win = ...

    def forward(self, x):
        # x: [B, T]
        x_embed = self.embed(x) # [B, T, embed_size]
        B, T, _ = x_embed.size()
        
        states = [torch.zeros(B, self.R_list[i].size(0), device=x.device) for i in range(self.num_reservoirs)]

        for t in range(T):
            # --- MODIFICATION: Handle 'deep' vs 'deep_input' logic ---
            embedded_input_t = x_embed[:, t, :] # [B, embed_size]
            
            # This is the input for the *first* layer
            current_layer_input = embedded_input_t 
            
            for i in range(self.num_reservoirs):
                state_part = states[i] @ self.R_list[i]
                
                # For 'deep_input' type, subsequent layers also get the original input
                if self.deep_type == 'deep_input' and i > 0:
                    current_layer_input = torch.cat((states[i-1], embedded_input_t), dim=-1)
                
                input_part = current_layer_input @ self.Win_list[i]
                
                states[i] = torch.tanh(state_part + input_part)
                
                # The output state of this layer is the input for the next (for 'deep' type)
                current_layer_input = states[i] 
            # --- END MODIFICATION ---
        
        return self.norm(states[-1])

class ReservoirNet(nn.Module):
    # --- MODIFICATION ---
    def __init__(self, vocab_size, embed_size, reservoir_size, num_reservoirs=1, deep_type="deep"):
        super().__init__()
        # --- MODIFICATION ---
        # Pass all arguments to the extractor
        self.extractor = _ReservoirExtractor(vocab_size, embed_size, reservoir_size, num_reservoirs, deep_type)
        self.classifier = nn.Linear(reservoir_size, vocab_size)

    def forward(self, x):
        r = self.extractor(x)
        return self.classifier(r)