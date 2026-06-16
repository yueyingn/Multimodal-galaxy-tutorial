"""SFHCodec — 1D ConvNext VQ-autoencoder for star formation histories.

Input: SimSFH.value shape (B, 2, 24)
  - channel 0: lookback time (fixed grid, same for all galaxies)
  - channel 1: log SFR values (the signal)

The codec encodes ONLY the SFR channel (index 1) as a (B, 1, 24) 1D
sequence. The fixed lookback time grid is stored as a buffer inside the
codec and restored during decoding.

Architecture:
  ConvNextEncoder1d(in_chans=1) → stem stride-4 → (B, 64, 6)
  LayerNorm + quant_conv → (B, lfq_dim, 6)
  LucidrainsLFQ(codebook_size=1024) → 6 discrete codes

  Decoder (simple ConvTranspose):
  post_quant_conv → (B, 64, 6) → ConvTranspose1d stride-4 → (B, 1, 24)

Token count: 6  (= 24 / stem stride 4)
Codebook size: 1024  (= 2^lfq_dim)

Training loss:
  F.mse_loss(sfr_pred, sfr_true) + lfq_weight * lfq_aux_loss
"""

from typing import Optional, Type

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from aion.codecs.base import Codec
from aion.codecs.modules.convnext import ConvNextEncoder1d
from aion.codecs.quantizers import Quantizer
from aion.codecs.quantizers import LucidrainsLFQ
from sim.codecs.mixin import SimCodecMixin
from sim.modalities import SimSFH


class SFHCodec(Codec, SimCodecMixin):
    """1D ConvNext VQ-autoencoder for star formation histories.

    The encoder uses ConvNextEncoder1d with a stride-4 stem, reducing 24
    timesteps to 6 tokens.  The decoder uses two transposed-convolution
    layers to upsample back to 24 timesteps (×2 then ×2, totalling ×4).

    Args:
        encoder_dims: Channel widths for ConvNext encoder stages.
        encoder_depths: Number of ConvNext blocks per encoder stage.
        latent_channels: Channels output by the encoder. Must equal encoder_dims[-1].
        lfq_dim: LFQ binary dimension. codebook_size = 2^lfq_dim.
        codebook_size: LFQ codebook size. Must be a power of 2 and equal 2^lfq_dim.
    """

    def __init__(
        self,
        encoder_dims: tuple[int, ...] = (64,),
        encoder_depths: tuple[int, ...] = (2,),
        latent_channels: int = 64,
        lfq_dim: int = 10,
        codebook_size: int = 1024,
    ):
        super().__init__()
        assert 2 ** lfq_dim == codebook_size, (
            f"codebook_size={codebook_size} must equal 2^lfq_dim=2^{lfq_dim}={2**lfq_dim}"
        )
        assert encoder_dims[-1] == latent_channels, (
            f"encoder_dims[-1]={encoder_dims[-1]} must equal latent_channels={latent_channels}"
        )

        self.latent_channels = latent_channels
        self.lfq_dim = lfq_dim

        # Encoder: (B, 1, 24) → (B, latent_channels, 6) via stride-4 stem
        self.encoder = ConvNextEncoder1d(
            in_chans=1,
            depths=encoder_depths,
            dims=encoder_dims,
        )

        # Pre-quantization projection and normalisation
        self.pre_quant_norm = nn.LayerNorm(latent_channels)
        self.quant_conv = nn.Conv1d(latent_channels, lfq_dim, kernel_size=1)

        # LFQ quantizer: (B, lfq_dim, 6) ↔ 6 integer codes in [0, codebook_size)
        self._quantizer = LucidrainsLFQ(dim=lfq_dim, codebook_size=codebook_size)

        # Post-quantization projection: (B, lfq_dim, 6) → (B, latent_channels, 6)
        self.post_quant_conv = nn.Conv1d(lfq_dim, latent_channels, kernel_size=1)

        # Decoder: (B, latent_channels, 6) → (B, 1, 24)
        # Two ×2 transposed-conv stages → total ×4 upsampling (6 → 12 → 24)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_channels, latent_channels // 2,
                               kernel_size=2, stride=2),  # 6 → 12
            nn.GELU(),
            nn.ConvTranspose1d(latent_channels // 2, 1,
                               kernel_size=2, stride=2),  # 12 → 24
        )

        # Fixed lookback time grid — set via set_time_grid() before use
        self.register_buffer("time_grid", torch.zeros(1, 1, 24))

    def set_time_grid(self, time_grid: Tensor) -> None:
        """Register the fixed lookback time axis.

        Args:
            time_grid: Tensor of shape (24,), (1, 24), or (1, 1, 24).
        """
        tg = time_grid.float()
        if tg.dim() == 1:
            tg = tg.reshape(1, 1, 24)
        elif tg.dim() == 2:
            tg = tg.unsqueeze(0)
        self.time_grid = tg.to(next(self.parameters()).device)

    @property
    def modality(self) -> Type[SimSFH]:
        return SimSFH

    @property
    def quantizer(self) -> Quantizer:
        return self._quantizer

    def _encode(self, x: SimSFH) -> Float[Tensor, "b c t"]:
        """Encode SFH to pre-quantization latent.

        Args:
            x: SimSFH with value shape (B, 2, 24).

        Returns:
            Latent tensor shape (B, lfq_dim, 6).
        """
        sfr = x.value[:, 1:2, :]              # (B, 1, 24) — SFR channel only
        h = self.encoder(sfr)                  # (B, latent_channels, 6)
        # LayerNorm expects (B, T, C) → transpose, norm, transpose back
        h = self.pre_quant_norm(h.transpose(1, 2)).transpose(1, 2)
        return self.quant_conv(h)              # (B, lfq_dim, 6)

    def _decode(
        self,
        z: Float[Tensor, "b c t"],
        **metadata,
    ) -> SimSFH:
        """Decode post-dequantization latent back to SFH.

        Args:
            z: Latent tensor shape (B, lfq_dim, 6).

        Returns:
            SimSFH with value shape (B, 2, 24).
        """
        h = self.post_quant_conv(z)            # (B, latent_channels, 6)
        sfr_pred = self.decoder(h)             # (B, 1, 24)
        B = sfr_pred.shape[0]
        tg = self.time_grid.expand(B, 1, 24).to(sfr_pred.device)
        return SimSFH(value=torch.cat([tg, sfr_pred], dim=1))
