import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, List, Optional, Sequence, Tuple, Union

import config as gem_config


class Sampler(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size

    @torch.no_grad()
    def forward(self,
                embedding: torch.Tensor,
                hidden_states: torch.Tensor,
                output_position: torch.Tensor,
                temperatures: torch.Tensor,
                top_ps: torch.Tensor,
                tops_ks: torch.Tensor,
                embedding_bias) -> torch.Tensor:
        # on choisi  le dernier element de chaque sequence
        hidden_states = hidden_states.index_select(1, output_position).squeeze(dim=1)
        logits = torch.matmul(hidden_states, embedding.t())
        if embedding_bias is not None:
            logits += embedding_bias

        if temperatures is None:
            return torch.argmax(logits, dim=-1).squeeze(dim=-1)

        # applique la temperature
        logits.div_(temperatures.unsqueeze(dim=1))

        # calcule la temperature avec  SOFTMAX
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)

        # on applique top-p et top-k
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        top_ps_mask = (probs_sum - probs_sort) > top_ps.unsqueeze(dim=1)
        probs_sort = torch.where(top_ps_mask, 0, probs_sort)

        tops_ks_mask = torch.arange(probs_idx.shape[-1], device=probs_idx.device)
        tops_ks_mask = tops_ks_mask.expand(probs_idx.shape[0], -1)
        top_ps_mask = top_ps_mask >= tops_ks.unsqueeze(dim=1)
        probs_sort = torch.where(tops_ks_mask, 0, probs_sort)

        # renormalisation

        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        probs = torch.gather(probs_sort, dim=-1, index=torch.argsort(probs_idx, dim=-1))

        next_token_ids = torch.multinomial(probs, num_samples=1, replacement=True).squeeze(dim=-1)

        return next_token_ids


def precompute_freqs_ids(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    pass

def apply_rotary_emb():
    pass