# --------------------------------------------------------
# BEATs: Audio Pre-Training with Acoustic Tokenizers (https://arxiv.org/abs/2212.09058)
# Github source: https://github.com/microsoft/unilm/tree/master/beats
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq code bases
# https://github.com/pytorch/fairseq
# --------------------------------------------------------


import torch
import torch.nn as nn
from torch.nn import LayerNorm
import torchaudio.compliance.kaldi as ta_kaldi

from .backbone import (
    TransformerEncoder,
)

import logging
from typing import Optional, Literal

logger = logging.getLogger(__name__)


class BEATsConfig:
    def __init__(self, cfg=None):
        self.input_patch_size: int = -1  # path size of patch embedding
        self.embed_dim: int = 512  # patch embedding dimension
        self.conv_bias: bool = False  # include bias in conv encoder

        self.encoder_layers: int = 12  # num encoder layers in the transformer
        self.encoder_embed_dim: int = 768  # encoder embedding dimension
        self.encoder_ffn_embed_dim: int = 3072  # encoder embedding dimension for FFN
        self.encoder_attention_heads: int = 12  # num encoder attention heads
        self.activation_fn: str = "gelu"  # activation function to use

        self.layer_wise_gradient_decay_ratio: float = 1.0  # ratio for layer-wise gradient decay
        self.layer_norm_first: bool = False  # apply layernorm first in the transformer
        self.deep_norm: bool = False  # apply deep_norm first in the transformer

        # dropouts
        self.dropout: float = 0.1  # dropout probability for the transformer
        self.attention_dropout: float = 0.1  # dropout probability for attention weights
        self.activation_dropout: float = 0.0  # dropout probability after activation in FFN
        self.encoder_layerdrop: float = 0.0  # probability of dropping a tarnsformer layer
        self.dropout_input: float = 0.0  # dropout to apply to the input (after feat extr)

        # positional embeddings
        self.conv_pos: int = 128  # number of filters for convolutional positional embeddings
        self.conv_pos_groups: int = 16  # number of groups for convolutional positional embedding

        # relative position embedding
        self.relative_position_embedding: bool = False  # apply relative position embedding
        self.num_buckets: int = 320  # number of buckets for relative position embedding
        self.max_distance: int = 1280  # maximum distance for relative position embedding
        self.gru_rel_pos: bool = False  # apply gated relative position embedding

        # label predictor
        self.finetuned_model: bool = False  # whether the model is a fine-tuned model.
        self.predictor_dropout: float = 0.1  # dropout probability for the predictor
        self.predictor_class: int = 527  # target class number for the predictor

        if cfg is not None:
            self.update(cfg)

    def update(self, cfg: dict):
        self.__dict__.update(cfg)


class BEATs(nn.Module):
    def __init__(
            self,
            cfg: BEATsConfig,
    ) -> None:
        super().__init__()
        logger.info(f"BEATs Config: {cfg.__dict__}")

        self.cfg = cfg

        self.embed = cfg.embed_dim
        self.post_extract_proj = (
            nn.Linear(self.embed, cfg.encoder_embed_dim)
            if self.embed != cfg.encoder_embed_dim
            else None
        )

        self.input_patch_size = cfg.input_patch_size
        self.patch_embedding = nn.Conv2d(1, self.embed, kernel_size=self.input_patch_size, stride=self.input_patch_size,
                                         bias=cfg.conv_bias)

        self.dropout_input = nn.Dropout(cfg.dropout_input)

        assert not cfg.deep_norm or not cfg.layer_norm_first
        self.encoder = TransformerEncoder(cfg)
        self.layer_norm = LayerNorm(self.embed)

        if cfg.finetuned_model:
            self.predictor_dropout = nn.Dropout(cfg.predictor_dropout)
            self.predictor = nn.Linear(cfg.encoder_embed_dim, cfg.predictor_class)
        else:
            self.predictor = None

    def forward_padding_mask(
            self,
            features: torch.Tensor,
            padding_mask: torch.Tensor,
            threshold: Literal['any', 'all'] | float = 0.5
    ) -> torch.Tensor:
        if padding_mask.shape[:2] == features.shape[:2] :
            return padding_mask
        
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            # drop up to extra (endpoints dont match)
            padding_mask = padding_mask[:, :-extra]
        # Reshape to match feature shape
        # -1: left over act as downsample/stride
        padding_mask = padding_mask.view(
            padding_mask.size(0), features.size(1), -1
        )
        # if all samples in dim is true, then downsample should be true
        if threshold == 'all':
            # consider mask, only if all subtokens were masked
            padding_mask = padding_mask.all(dim=-1)
        elif threshold == 'any':
            # consider mask, if any of the subtokens were masked
            padding_mask = padding_mask.any(dim=-1)
        else:
            # consider mask, if a percentage of subtokens were masked
            padding_mask = padding_mask.mean(dim=-1, dtype=torch.float) > threshold

        return padding_mask

    def preprocess(
        self,
        source: torch.Tensor,
        *,
        fbank_mean: float = 0,
        fbank_std: float = 0.5,
        num_mel_bins = 128,
        sample_frequency = 16_000,
        frame_length = 25, #ms
        frame_shift = 10, #ms
        **kwargs
    ) -> torch.Tensor:
        fbanks = []
        for waveform in source:
            waveform = waveform.unsqueeze(0) * 2 ** 15
            fbank = ta_kaldi.fbank(waveform, num_mel_bins=num_mel_bins, sample_frequency=sample_frequency, frame_length=frame_length, frame_shift=frame_shift)
            fbanks.append(fbank)
        fbank = torch.stack(fbanks, dim=0)
        fbank = (fbank - fbank_mean) / (2 * fbank_std)
        return fbank

    def forward(self,
                fbank: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None,
                attn_mask: torch.Tensor | None = None,
                return_encoder_layer: Optional[int] = None,
                ):
        # Divide melspec into patches
        features = self.patch_embedding(fbank)
        # B - Batch
        # C - Channel (default 512)
        # T - Time patches
        # M - Mel frequency patches (default 8)
        B, C, T, M = features.size()

        # Combine mel and time patches
        features = features.reshape(B, C, -1)
        # Swap channel and (mel&time) dim
        features = features.transpose(1, 2)
        # (B, T*M, C)


        if padding_mask is not None:
            # Expand out mel and time dim
            features = features.reshape(B, T, M, C)
            # (B, T, M, C)
            # Downsample padding mask from frames to patches
            padding_mask = self.forward_padding_mask(features, padding_mask)
            # Expand (copy) time mask across associated freq patches
            padding_mask = padding_mask.unsqueeze(-1).expand(-1, -1, M)
            # Collapse mel and time dim
            padding_mask = padding_mask.reshape(B, -1)
            # (B, T*M)
            features = features.reshape(B, -1, C)
            # (B, T*M, C)

        if attn_mask is not None:
            padding_mask |= attn_mask

        # Apply layer wise normalisation
        features = self.layer_norm(features)

        # BUG: padding is computes over flattened spectrum
        # i.e. padding in time starts to mask some frequency tokens...
        # The above changes fixes this bug
        # if padding_mask is not None:
        #     padding_mask = self.forward_padding_mask(features, padding_mask)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        x = self.dropout_input(features)

        x, layer_results = self.encoder(
            x,
            padding_mask=padding_mask,
            layer=return_encoder_layer,
        )

        if self.predictor is not None:
            x = self.predictor_dropout(x)
            logits = self.predictor(x)

            if padding_mask is not None and padding_mask.any():
                logits[padding_mask] = 0
                logits = logits.sum(dim=1)
                logits = logits / (~padding_mask).sum(dim=1).unsqueeze(-1).expand_as(logits)
            else:
                logits = logits.mean(dim=1)

            lprobs = torch.sigmoid(logits)

            return lprobs, padding_mask
        else:
            if return_encoder_layer is not None:
                # layer_results -> [(x_i, attn)]
                # x_i : output of current layer fed to next layer
                # attn : attention weights, averaged over heads
                # index 0: is inputs to encoder layer
                return x, padding_mask, layer_results[1:]
            else:
                return x, padding_mask

    def extract_features(
            self,
            source: torch.Tensor,
            fbank_mean: float = 0,
            fbank_std: float = 0.5,
            padding_mask: Optional[torch.Tensor] = None,
            **kwargs,
    ):
        fbank = self.preprocess(source, fbank_mean=fbank_mean, fbank_std=fbank_std)

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(fbank, padding_mask)
        fbank = fbank.unsqueeze(1)
        return self.forward(fbank, padding_mask=padding_mask, **kwargs)

