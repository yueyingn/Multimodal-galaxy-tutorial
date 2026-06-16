"""SimGasFaceonCodec — MagVit VQ-autoencoder for 128×128×2 gas images.

Mirrors SimGalaxyImageCodec but with n_bands=2 (gas surface density +
mass-weighted temperature, both in log10 scale).

n_compressions=2 → 128/(2²) = 32 → 32×32 = 1024 tokens.
FiniteScalarQuantizer(levels=[5,5,5,5]) → vocab_size = 5^4 = 625.
"""

from typing import ClassVar, List, Optional, Type

import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from aion.codecs.base import Codec
from aion.codecs.modules.magvit import MagVitAE
from aion.codecs.quantizers import FiniteScalarQuantizer, Quantizer
from sim.codecs.mixin import SimCodecMixin
from sim.modalities import SimGasFaceon


class SimGasFaceonCodec(Codec, SimCodecMixin):
    """MagVit VQ-autoencoder for 128×128×2 gas face-on images."""

    def __init__(
        self,
        quantizer_levels: List[int] = None,
        hidden_dims: int = 256,
        n_compressions: int = 2,
        num_consecutive: int = 2,
        embedding_dim: int = 4,
        input_size: int = 128,
    ):
        super().__init__()
        if quantizer_levels is None:
            quantizer_levels = [5, 5, 5, 5]

        assert len(quantizer_levels) == embedding_dim

        model = MagVitAE(
            n_bands=2,
            hidden_dims=hidden_dims,
            n_compressions=n_compressions,
            num_consecutive=num_consecutive,
        )
        self._quantizer = FiniteScalarQuantizer(levels=quantizer_levels)
        self._encoder = model.encode
        self._decoder = model.decode
        self.embedding_dim = embedding_dim
        self.n_compressions = n_compressions
        self.input_size = input_size

        self.pre_quant_proj = nn.Conv2d(hidden_dims, embedding_dim, kernel_size=1)
        self.post_quant_proj = nn.Conv2d(embedding_dim, hidden_dims, kernel_size=1)

        self.model = model

    @property
    def modality(self) -> Type[SimGasFaceon]:
        return SimGasFaceon

    @property
    def quantizer(self) -> Quantizer:
        return self._quantizer

    def _spatial_size(self) -> int:
        return self.input_size // (2 ** self.n_compressions)

    def _encode(self, x: SimGasFaceon) -> Float[Tensor, "b c n_tokens"]:
        h = self._encoder(x.value)              # (B, hidden_dims, 32, 32)
        h = self.pre_quant_proj(h)              # (B, embedding_dim, 32, 32)
        B, C, H, W = h.shape
        return h.reshape(B, C, H * W)           # (B, embedding_dim, 1024)

    def _decode(self, z: Float[Tensor, "b c n_tokens"], **metadata) -> SimGasFaceon:
        B, C, N = z.shape
        s = self._spatial_size()
        assert N == s * s
        z = z.reshape(B, C, s, s)              # (B, embedding_dim, 32, 32)
        h = self.post_quant_proj(z)             # (B, hidden_dims, 32, 32)
        value = self._decoder(h)               # (B, 2, 128, 128)
        return SimGasFaceon(value=value)
