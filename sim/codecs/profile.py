"""GasProfileCodec / DMProfileCodec — 1D ConvNext VQ-autoencoders for radial density profiles.

Both share an identical architecture (a _BaseProfileCodec), differing only in
the modality they serve.

Input: shape (B, 2, 20)
  - channel 0: normalized radius r/r200 (fixed grid, stored as a buffer)
  - channel 1: log10(density), clipped at -10 (the signal)

The codec encodes ONLY the density channel (index 1) as a (B, 1, 20) sequence.
The fixed radius grid is stored as a buffer and restored during decoding.

The log-density channel is **standardized** (zero mean, unit variance) before
the encoder and de-standardized after the decoder, using statistics calibrated
on the training split (`calibrate()`). log10(density) sits around -6.3 with a
std of ~0.75; feeding those raw offsets straight into the conv stack wastes
codebook capacity (the original raw-input codec collapsed to ~150 distinct
token rows out of 2048 for gas, and to a *single* row for DM). Standardizing
the input lets the LFQ use its codebook → ~5× lower reconstruction MSE and
~13× more distinct token rows.

Architecture:
  standardize density: (x - mean) / std
  ConvNextEncoder1d(in_chans=1) → stride-4 stem → (B, 64, 5)
  LayerNorm + quant_conv → (B, lfq_dim, 5)
  LucidrainsLFQ(codebook_size=1024, entropy_loss_weight=0.05) → 5 discrete codes

  Decoder (a ConvNext block between the two upsamples gives it enough capacity
  to reproduce the curved profile shape):
  post_quant_conv → (B, 64, 5)
  ConvTranspose1d stride-2 → (B, 64, 10)
  ConvNextBlock1d + GELU
  ConvTranspose1d stride-2 → (B, 32, 20)
  GELU + Conv1d(3) → (B, 1, 20)
  de-standardize: x * std + mean

Token count: 5  (= 20 / stem stride 4)
Codebook size: 1024 (= 2^10)
"""

from typing import Type

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from aion.codecs.base import Codec
from aion.codecs.modules.convnext import ConvNextEncoder1d, ConvNextBlock1d
from aion.codecs.quantizers import Quantizer
from aion.codecs.quantizers import LucidrainsLFQ
from sim.codecs.mixin import SimCodecMixin
from sim.modalities import SimGasProfile, SimDMProfile


class _BaseProfileCodec(Codec, SimCodecMixin):
    """Shared 1D VQ-autoencoder architecture for radial density profiles."""

    def __init__(
        self,
        encoder_dims: tuple[int, ...] = (64,),
        encoder_depths: tuple[int, ...] = (2,),
        latent_channels: int = 64,
        lfq_dim: int = 10,
        codebook_size: int = 1024,
        entropy_loss_weight: float = 0.05,
    ):
        super().__init__()
        assert 2 ** lfq_dim == codebook_size
        assert encoder_dims[-1] == latent_channels

        self.latent_channels = latent_channels
        self.lfq_dim = lfq_dim

        self.encoder = ConvNextEncoder1d(
            in_chans=1,
            depths=encoder_depths,
            dims=encoder_dims,
        )

        self.pre_quant_norm = nn.LayerNorm(latent_channels)
        self.quant_conv = nn.Conv1d(latent_channels, lfq_dim, kernel_size=1)

        self._quantizer = LucidrainsLFQ(
            dim=lfq_dim, codebook_size=codebook_size,
            entropy_loss_weight=entropy_loss_weight,
        )

        self.post_quant_conv = nn.Conv1d(lfq_dim, latent_channels, kernel_size=1)

        # 5 → 10 (ConvNext block) → 20.  The extra ConvNext block at the 10-step
        # stage gives the decoder enough capacity to reproduce the curved
        # profile rather than a piecewise-flat approximation.
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_channels, latent_channels,
                               kernel_size=2, stride=2),       # 5 → 10
            ConvNextBlock1d(latent_channels),
            nn.GELU(),
            nn.ConvTranspose1d(latent_channels, latent_channels // 2,
                               kernel_size=2, stride=2),       # 10 → 20
            nn.GELU(),
            nn.Conv1d(latent_channels // 2, 1, kernel_size=3, padding=1),
        )

        self.register_buffer("radius_grid", torch.zeros(1, 1, 20))
        # Standardization statistics for the log-density channel, set by
        # calibrate() on the training split. Defaults are an identity transform
        # so an un-calibrated codec still round-trips (just less efficiently).
        self.register_buffer("dens_mean", torch.zeros(()))
        self.register_buffer("dens_std", torch.ones(()))

    def set_radius_grid(self, radius_grid: Tensor) -> None:
        """Register the fixed r/r200 axis. Call once before training."""
        rg = radius_grid.float()
        if rg.dim() == 1:
            rg = rg.reshape(1, 1, 20)
        elif rg.dim() == 2:
            rg = rg.unsqueeze(0)
        self.radius_grid = rg.to(next(self.parameters()).device)

    def calibrate(self, values: Tensor) -> None:
        """Set the log-density standardization statistics from training data.

        Args:
            values: Either the full (B, 2, 20) modality tensor or the (B, 1, 20)
                / (B, 20) density channel. Mean and std are taken over the
                density channel of the whole batch. Call once before training.
        """
        v = values.float()
        if v.dim() == 3 and v.shape[1] == 2:
            v = v[:, 1, :]                                     # density channel
        elif v.dim() == 3 and v.shape[1] == 1:
            v = v[:, 0, :]
        dev = self.dens_mean.device
        self.dens_mean = v.mean().detach().to(dev).reshape(())
        self.dens_std = v.std().clamp_min(1e-6).detach().to(dev).reshape(())

    @property
    def quantizer(self) -> Quantizer:
        return self._quantizer

    def _encode(self, x) -> Float[Tensor, "b c t"]:
        density = x.value[:, 1:2, :]                          # (B, 1, 20)
        density = (density - self.dens_mean) / self.dens_std  # standardize
        h = self.encoder(density)                              # (B, latent, 5)
        h = self.pre_quant_norm(h.transpose(1, 2)).transpose(1, 2)
        return self.quant_conv(h)                              # (B, lfq_dim, 5)

    def _decode(self, z: Float[Tensor, "b c t"], **metadata):
        h = self.post_quant_conv(z)                            # (B, latent, 5)
        density_pred = self.decoder(h)                         # (B, 1, 20)
        density_pred = density_pred * self.dens_std + self.dens_mean
        B = density_pred.shape[0]
        rg = self.radius_grid.expand(B, 1, 20).to(density_pred.device)
        value = torch.cat([rg, density_pred], dim=1)           # (B, 2, 20)
        return self._make_modality(value)

    def _make_modality(self, value: Tensor):
        raise NotImplementedError


class GasProfileCodec(_BaseProfileCodec):
    """Profile codec for gas radial density profiles."""

    @property
    def modality(self) -> Type[SimGasProfile]:
        return SimGasProfile

    def _make_modality(self, value: Tensor) -> SimGasProfile:
        return SimGasProfile(value=value)


class DMProfileCodec(_BaseProfileCodec):
    """Profile codec for dark matter radial density profiles."""

    @property
    def modality(self) -> Type[SimDMProfile]:
        return SimDMProfile

    def _make_modality(self, value: Tensor) -> SimDMProfile:
        return SimDMProfile(value=value)
