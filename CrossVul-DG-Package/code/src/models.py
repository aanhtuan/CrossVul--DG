"""Encoder + 2-layer MLP classifier.

The same architecture wraps any RoBERTa-family code encoder so the proposed
method can be shown to transfer across backbones:

  * microsoft/codebert-base       (Tier 1, main results)
  * microsoft/unixcoder-base      (Tier 2, generality)
  * microsoft/graphcodebert-base  (Tier 2, generality)

All three expose a 768-d pooled representation taken from the first token
(<s>/[CLS]) of ``last_hidden_state``.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class EncoderClassifier(nn.Module):
    """A pretrained code encoder followed by a 2-layer MLP head.

    The IRMv1 "scalar dummy weight" is applied inside ``losses.irm_penalty``
    (the logits are scaled by a differentiable 1.0 there), so this module
    needs no special wiring for IRM.
    """

    def __init__(self, model_name: str = "microsoft/codebert-base",
                 hidden_size: int = 768, dropout: float = 0.1):
        super().__init__()
        self.model_name = model_name
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(model_name)
        enc_dim = getattr(self.encoder.config, "hidden_size", hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(enc_dim, enc_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(enc_dim, 1),
        )

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0]   # first token (<s>/[CLS])

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor
                ) -> torch.Tensor:
        cls = self.encode(input_ids, attention_mask)
        return self.classifier(cls).squeeze(-1)


# Backwards-compatible alias (older scripts import CodeBERTClassifier).
CodeBERTClassifier = EncoderClassifier


class GradientReversal(torch.autograd.Function):
    """Used by DANN: identity forward, negated-scaled gradient backward."""

    @staticmethod
    def forward(ctx, x, alpha: float):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def grad_reverse(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    return GradientReversal.apply(x, alpha)


class DomainHead(nn.Module):
    """Simple 2-layer domain classifier used by DANN."""

    def __init__(self, in_dim: int, n_domains: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim // 2, n_domains),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
