"""
JumpReLU Sparse Autoencoder

Implementation of the JumpReLU SAE architecture used by GemmaScope.
"""

import torch
import torch.nn as nn


class JumpReLUSAE(nn.Module):
    """
    JumpReLU Sparse Autoencoder.

    Adapted from Google DeepMind's Gemma Scope tutorial.
    """

    def __init__(self, d_in: int, d_sae: int):
        super().__init__()
        self.w_enc = nn.Parameter(torch.zeros(d_in, d_sae))
        self.w_dec = nn.Parameter(torch.zeros(d_sae, d_in))
        self.threshold = nn.Parameter(torch.zeros(d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

    def encode(self, input_acts: torch.Tensor) -> torch.Tensor:
        """Encode input activations to sparse latent space."""
        pre_acts = input_acts @ self.w_enc + self.b_enc
        mask = (pre_acts > self.threshold)
        acts = mask * torch.nn.functional.relu(pre_acts)
        return acts

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        """Decode sparse latents back to activation space."""
        return acts @ self.w_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass: encode then decode."""
        acts = self.encode(x)
        recon = self.decode(acts)
        return recon
