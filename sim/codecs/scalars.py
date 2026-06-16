"""Scalar codecs for simulation quantities.

All simulation scalars are assumed to already be in log scale (or a
suitably transformed space), so we use ScalarReservoirQuantizer which
builds an empirical CDF from streaming data — no range assumptions needed.

Usage:
    codec = SimScalarCodec(SimMhalo, codebook_size=1024, reservoir_size=50_000)

    # Calibrate by streaming data (no backprop):
    for batch in dataloader:
        codec.quantizer.forward(batch_values)   # updates reservoir in-place

    # Encode / decode:
    tokens = codec.encode(SimMhalo(value=values))   # → (B, 1) integer tensor
    recon  = codec.decode(tokens)                   # → SimMhalo(value=...)
"""

from typing import Type

import torch
from jaxtyping import Float
from torch import Tensor

from aion.codecs.base import Codec
from aion.codecs.quantizers import Quantizer
from aion.codecs.quantizers.scalar import ScalarReservoirQuantizer
from aion.modalities import Scalar
from sim.codecs.mixin import SimCodecMixin
from sim.modalities import (
    SimEgyRM,
    SimMbh,
    SimMhalo,
    SimMstar,
    SimR200,
    SimRedshift,
    SimRMpow,
    SimSFR,
)


class SimScalarCodec(Codec, SimCodecMixin):
    """Scalar codec for simulation quantities already in log / transformed space.

    Implements an identity encode/decode with reservoir-CDF quantization.
    No neural network; the only learned state is the codebook (histogram bins).

    Args:
        modality_class: The Scalar subclass this codec operates on
            (e.g. SimMhalo, SimSFR).
        codebook_size: Number of quantization bins.
        reservoir_size: Size of the reservoir for CDF estimation.
            Should be at least ~10× the dataset size for accurate CDFs.
        floor: Optional lower clip applied to values before quantization. Use
            this when a field has an explicit "non-detection" floor value many
            dex below the bulk distribution (e.g. log SFR=-10 for quenched
            galaxies). Re-basing the floor to just below the bulk prevents the
            CDF from spending most of its bins on the gap between floor and
            bulk. Recovered values are clipped to `floor` on decode by virtue
            of the codebook never going below it.
    """

    def __init__(
        self,
        modality_class: Type[Scalar],
        codebook_size: int = 1024,
        reservoir_size: int = 50_000,
        floor: float | None = None,
    ):
        super().__init__()
        self._modality_class = modality_class
        self._floor = floor
        self._quantizer = ScalarReservoirQuantizer(
            codebook_size=codebook_size,
            reservoir_size=reservoir_size,
        )

    @property
    def modality(self) -> Type[Scalar]:
        return self._modality_class

    @property
    def quantizer(self) -> Quantizer:
        return self._quantizer

    @property
    def floor(self) -> float | None:
        return self._floor

    def _apply_floor(self, val: Tensor) -> Tensor:
        if self._floor is not None:
            val = torch.clamp(val, min=self._floor)
        return val

    def calibrate(self, values: Tensor) -> None:
        """Stream a batch of raw scalar values into the reservoir, applying
        the configured floor first. Use this in place of
        `codec.quantizer.forward(values)` during calibration."""
        self._quantizer.forward(self._apply_floor(values))

    def _encode(self, x: Scalar) -> Float[Tensor, "b"]:
        """Identity: extract the scalar value tensor (with optional floor)."""
        val = x.value.squeeze(-1) if x.value.dim() > 1 else x.value
        return self._apply_floor(val)

    def _decode(self, z: Float[Tensor, "b"], **metadata) -> Scalar:
        """Identity: wrap the dequantized value back into the modality."""
        return self._modality_class(value=z.unsqueeze(-1))

    def encode(self, x: Scalar) -> Float[Tensor, "b 1"]:
        """Encode scalar to a (B, 1) integer token tensor."""
        if not isinstance(x, self.modality):
            raise ValueError(
                f"Input type {type(x).__name__} does not match "
                f"modality {self.modality.__name__}"
            )
        raw = self._encode(x)               # (B,)
        codes = self._quantizer.encode(raw) # (B,)
        return codes.unsqueeze(-1)          # (B, 1)

    def decode(self, z: Float[Tensor, "b 1"], **metadata) -> Scalar:
        """Decode (B, 1) integer tokens back to scalar modality."""
        codes = z.squeeze(-1)               # (B,)
        dequant = self._quantizer.decode(codes)  # (B,)
        return self._decode(dequant, **metadata)


# ---------------------------------------------------------------------------
# Convenience factory functions
#
# Floors are picked just below the non-floor minimum of each field, so the
# explicit "non-detection" population (e.g. log SFR=-10) collapses against the
# bulk instead of stealing CDF resolution. Other fields have no floor.
# ---------------------------------------------------------------------------

def make_sfr_codec(codebook_size: int = 1024, reservoir_size: int = 50_000) -> SimScalarCodec:
    # Raw floor is -10; non-floor min ~-3.58. Re-base to -4.
    return SimScalarCodec(SimSFR, codebook_size=codebook_size,
                          reservoir_size=reservoir_size, floor=-4.0)


def make_mstar_codec(codebook_size: int = 1024, reservoir_size: int = 50_000) -> SimScalarCodec:
    return SimScalarCodec(SimMstar, codebook_size=codebook_size, reservoir_size=reservoir_size)


def make_mhalo_codec(codebook_size: int = 1024, reservoir_size: int = 50_000) -> SimScalarCodec:
    return SimScalarCodec(SimMhalo, codebook_size=codebook_size, reservoir_size=reservoir_size)


def make_redshift_codec(codebook_size: int = 1024, reservoir_size: int = 50_000) -> SimScalarCodec:
    return SimScalarCodec(SimRedshift, codebook_size=codebook_size, reservoir_size=reservoir_size)


def make_r200_codec(codebook_size: int = 1024, reservoir_size: int = 50_000) -> SimScalarCodec:
    return SimScalarCodec(SimR200, codebook_size=codebook_size, reservoir_size=reservoir_size)


def make_mbh_codec(codebook_size: int = 1024, reservoir_size: int = 50_000) -> SimScalarCodec:
    return SimScalarCodec(SimMbh, codebook_size=codebook_size, reservoir_size=reservoir_size)


def make_egyRM_codec(codebook_size: int = 1024, reservoir_size: int = 50_000) -> SimScalarCodec:
    # Raw floor is -6; non-floor min ~-1.06. Re-base to -2.
    return SimScalarCodec(SimEgyRM, codebook_size=codebook_size,
                          reservoir_size=reservoir_size, floor=-2.0)


def make_RMpow_codec(codebook_size: int = 1024, reservoir_size: int = 50_000) -> SimScalarCodec:
    return SimScalarCodec(SimRMpow, codebook_size=codebook_size, reservoir_size=reservoir_size)
