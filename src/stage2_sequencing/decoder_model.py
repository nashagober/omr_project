"""
Stage 2: Decoder-Only Transformer Language Model
Used when training on PrIMuS — no encoder needed since PrIMuS provides
ground truth token sequences directly (no symbol detection input).

Architecture: GPT-style decoder-only transformer
    - Token embedding + sinusoidal positional encoding
    - N layers of masked self-attention + feedforward
    - Causal (autoregressive) — each token only attends to previous tokens
    - Output projection → vocab logits

This model learns the statistical structure of music notation:
what tokens follow which other tokens, what valid sequences look like.
After training on PrIMuS it can be used in two ways:
    1. Standalone: generate sequences token by token
    2. Scoring: compute log-probability of a sequence from stage 2
       (this is how it feeds into stage 3 error correction)
"""

import math
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.stage2_sequencing.vocabulary import (
    VOCAB_SIZE, TOKEN_TO_IDX, IDX_TO_TOKEN,
    PAD_IDX, SOS_IDX, EOS_IDX,
)
from src.stage2_sequencing.primus_dataset import MAX_SEQ_LEN


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = MAX_SEQ_LEN,
                 dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Decoder-Only Transformer (GPT-style)
# ---------------------------------------------------------------------------

class MusicDecoderTransformer(nn.Module):
    """
    Decoder-only transformer language model for music token sequences.

    Input : [B, S] token IDs
    Output: [B, S, vocab_size] logits

    Args:
        vocab_size   : vocabulary size (default from vocabulary.py)
        d_model      : embedding dimension
        nhead        : number of attention heads
        num_layers   : number of transformer decoder layers
        ffn_dim      : feedforward network hidden dimension
        dropout      : dropout rate
        max_seq_len  : maximum sequence length
    """

    def __init__(
        self,
        vocab_size:  int   = VOCAB_SIZE,
        d_model:     int   = 256,
        nhead:       int   = 8,
        num_layers:  int   = 6,
        ffn_dim:     int   = 1024,
        dropout:     float = 0.1,
        max_seq_len: int   = MAX_SEQ_LEN,
    ):
        super().__init__()
        self.d_model     = d_model
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, d_model,
                                             padding_idx=PAD_IDX)
        self.pos_encoding    = PositionalEncoding(d_model, max_seq_len, dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout, batch_first=True,
        )
        self.decoder     = nn.TransformerDecoder(decoder_layer,
                                                  num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for teacher-forced training.

        Args:
            input_ids    : [B, S] token IDs (SOS + sequence)
            padding_mask : [B, S] BoolTensor, True = padding position

        Returns:
            logits : [B, S, vocab_size]
        """
        S   = input_ids.size(1)
        tgt = self.pos_encoding(
            self.token_embedding(input_ids) * math.sqrt(self.d_model)
        )

        # Causal mask — prevent attending to future tokens
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            S, device=input_ids.device)

        # Decoder-only: memory = tgt (self-attention over input)
        # We use TransformerDecoder with tgt == memory to simulate
        # a decoder-only model — the cross-attention becomes a no-op
        # since both inputs are identical
        out = self.decoder(
            tgt=tgt,
            memory=tgt,
            tgt_mask=causal_mask,
            memory_mask=causal_mask,
            tgt_key_padding_mask=padding_mask,
            memory_key_padding_mask=padding_mask,
        )
        return self.output_proj(out)   # [B, S, vocab_size]

    @torch.no_grad()
    def sequence_log_prob(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute the log-probability of a token sequence.
        Used by stage 3 corrector to score candidate sequences.

        Args:
            token_ids : [S] or [B, S] token IDs (including SOS)
        Returns:
            log_prob  : scalar (or [B]) — sum of log-probs over sequence
        """
        self.eval()
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)  # [1, S]

        logits     = self.forward(token_ids)          # [B, S, V]
        log_probs  = F.log_softmax(logits, dim=-1)    # [B, S, V]

        # Shift: input is [SOS, t1, t2...], target is [t1, t2, ..., EOS]
        input_ids  = token_ids[:, :-1]
        target_ids = token_ids[:, 1:]

        # Gather log-prob of each target token
        gathered = log_probs[:, :-1, :].gather(
            2, target_ids.unsqueeze(-1)).squeeze(-1)  # [B, S-1]

        # Sum over non-PAD positions
        non_pad = target_ids != PAD_IDX
        log_prob = (gathered * non_pad).sum(dim=-1)   # [B]

        return log_prob.squeeze(0) if log_prob.size(0) == 1 else log_prob

    @torch.no_grad()
    def generate(self, prompt_ids: Optional[torch.Tensor] = None,
                 max_new_tokens: int = 256,
                 temperature: float = 1.0,
                 top_k: int = 50,
                 device: Optional[torch.device] = None) -> List[int]:
        """
        Autoregressive generation — sample a music token sequence.

        Args:
            prompt_ids     : optional [S] starting token IDs (defaults to SOS)
            max_new_tokens : maximum tokens to generate
            temperature    : sampling temperature (lower = more conservative)
            top_k          : top-k sampling (0 = greedy)
            device         : torch device

        Returns:
            generated token IDs (excluding SOS, including EOS if generated)
        """
        self.eval()
        if device is None:
            device = next(self.parameters()).device

        if prompt_ids is None:
            ids = [SOS_IDX]
        else:
            ids = prompt_ids.tolist() if isinstance(prompt_ids, torch.Tensor) \
                  else list(prompt_ids)

        for _ in range(max_new_tokens):
            input_t = torch.tensor([ids], dtype=torch.long, device=device)
            logits  = self.forward(input_t)[0, -1, :]  # [V]

            logits = logits / max(temperature, 1e-8)

            if top_k > 0:
                # Zero out everything outside top-k
                topk_vals, _ = logits.topk(top_k)
                logits[logits < topk_vals[-1]] = float("-inf")

            probs   = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            ids.append(next_id)

            if next_id == EOS_IDX:
                break

        return ids[1:]  # strip SOS


# ---------------------------------------------------------------------------
# Inference wrapper (used by main.py and stage 3)
# ---------------------------------------------------------------------------

class MusicSequenceLM:
    """
    Wrapper around MusicDecoderTransformer for inference.
    Used by main.py and the Stage 3 corrector.
    """

    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = self._load_model(config.get("model_path"))
        self.model.to(self.device)
        self.model.eval()

    def _load_model(self, model_path: Optional[str]) -> MusicDecoderTransformer:
        model = MusicDecoderTransformer(
            d_model    = self.config.get("d_model",    256),
            nhead      = self.config.get("nhead",      8),
            num_layers = self.config.get("num_layers", 6),
            ffn_dim    = self.config.get("ffn_dim",    1024),
        )
        if model_path and Path(model_path).exists():
            ckpt = torch.load(model_path, map_location=self.device)
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"[MusicSequenceLM] Loaded from {model_path}")
        else:
            print("[MusicSequenceLM] No checkpoint — using random weights.")
        return model

    def score(self, tokens: List[str]) -> float:
        """Score a token sequence — higher = more musically plausible."""
        ids = [TOKEN_TO_IDX.get(t, TOKEN_TO_IDX["<UNK>"]) for t in tokens]
        ids = [SOS_IDX] + ids
        t   = torch.tensor(ids, dtype=torch.long, device=self.device)
        return self.model.sequence_log_prob(t).item()

    def generate(self, prompt_tokens: Optional[List[str]] = None,
                 max_new_tokens: int = 256,
                 temperature: float = 0.9) -> List[str]:
        """Generate a music token sequence."""
        prompt_ids = None
        if prompt_tokens:
            prompt_ids = [TOKEN_TO_IDX.get(t, TOKEN_TO_IDX["<UNK>"])
                          for t in prompt_tokens]
        ids = self.model.generate(
            prompt_ids=torch.tensor(prompt_ids) if prompt_ids else None,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device=self.device,
        )
        return [IDX_TO_TOKEN.get(i, "<UNK>") for i in ids
                if i not in (PAD_IDX, EOS_IDX)]
