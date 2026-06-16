"""SimGalaxyImageCodec — MagVit VQ-autoencoder for 256×256×8 simulation images.

Design choices vs. the observational ImageCodec:
  - No CenterCrop: simulated galaxies fill the full 256×256 field.
  - No ImagePadder / SubsampledLinear: all 8 bands are always present.
  - No RescaleToLegacySurvey: simulation data has no survey zeropoints.
  - No _range_compress: images are already linearly normalized (mag_normalize).
  - n_compressions=3 → 32×32 = 1024 tokens covering the full field.
  - FiniteScalarQuantizer(levels=[5,5,5,5]) → vocab_size = 5^4 = 625.
"""

from typing import ClassVar, List, Optional, Type

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from aion.codecs.base import Codec
from aion.codecs.modules.magvit import MagVitAE
from aion.codecs.quantizers import FiniteScalarQuantizer, Quantizer
from sim.codecs.mixin import SimCodecMixin
from sim.modalities import SIM_BANDS, SimGalaxyImage


class SimGalaxyImageCodec(Codec, SimCodecMixin):
    """MagVit VQ-autoencoder for 256×256×8 simulation galaxy images.

    Args:
        quantizer_levels: FSQ level list. vocab_size = product(levels).
            Default [5,5,5,5] → 625.
        hidden_dims: Channel width inside MagVitAE. Reduce for less memory.
        n_compressions: Spatial downsampling stages. 3 → 256/8=32, 1024 tokens.
        num_consecutive: Residual blocks per compression stage.
        embedding_dim: Post-quantization latent channel depth (= len(levels)).
    """

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

        assert len(quantizer_levels) == embedding_dim, (
            f"len(quantizer_levels)={len(quantizer_levels)} must equal "
            f"embedding_dim={embedding_dim}"
        )

        model = MagVitAE(
            n_bands=8,
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

        # Project between MagVit hidden space and FSQ embedding space
        self.pre_quant_proj = nn.Conv2d(hidden_dims, embedding_dim, kernel_size=1)
        self.post_quant_proj = nn.Conv2d(embedding_dim, hidden_dims, kernel_size=1)

        # Store the full model so its parameters are included in state_dict
        self.model = model

    @property
    def modality(self) -> Type[SimGalaxyImage]:
        return SimGalaxyImage

    @property
    def quantizer(self) -> Quantizer:
        return self._quantizer

    def _spatial_size(self) -> int:
        """Compute the spatial size of the codec output."""
        return self.input_size // (2 ** self.n_compressions)

    def _encode(self, x: SimGalaxyImage) -> Float[Tensor, "b c n_tokens"]:
        """Encode images to pre-quantization latent.

        Args:
            x: SimGalaxyImage with flux shape (B, 8, 256, 256).

        Returns:
            Latent tensor shape (B, embedding_dim, 1024).
        """
        h = self._encoder(x.flux)          # (B, hidden_dims, 32, 32)
        h = self.pre_quant_proj(h)          # (B, embedding_dim, 32, 32)
        B, C, H, W = h.shape
        return h.reshape(B, C, H * W)       # (B, embedding_dim, 1024)

    def _decode(
        self,
        z: Float[Tensor, "b c n_tokens"],
        bands: Optional[List[str]] = None,
    ) -> SimGalaxyImage:
        """Decode post-dequantization latent back to image.

        Args:
            z: Latent tensor shape (B, embedding_dim, 1024).
            bands: Ignored; always returns SIM_BANDS. Kept for API compatibility.

        Returns:
            SimGalaxyImage with flux shape (B, 8, 256, 256).
        """
        B, C, N = z.shape
        s = self._spatial_size()
        assert N == s * s, f"Expected {s*s} tokens, got {N}"
        z = z.reshape(B, C, s, s)          # (B, embedding_dim, 32, 32)
        h = self.post_quant_proj(z)         # (B, hidden_dims, 32, 32)
        flux = self._decoder(h)             # (B, 8, 256, 256)
        return SimGalaxyImage(flux=flux, bands=SIM_BANDS)
