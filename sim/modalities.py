"""Simulation-domain modality dataclasses for AION.

These modalities correspond to data from cosmological simulations
(e.g. IllustrisTNG, EAGLE) and are entirely separate from the
observational modalities in aion/modalities.py.
"""

from dataclasses import dataclass
from typing import ClassVar

from jaxtyping import Float
from torch import Tensor

from aion.modalities import Image, Scalar, Modality

__all__ = [
    "SIM_BANDS",
    "SimGalaxyImage",
    "SimGasFaceon",
    "SimGasProfile",
    "SimDMProfile",
    "SimSFH",
    "SimSFR",
    "SimMstar",
    "SimMhalo",
    "SimR200",
    "SimMbh",
    "SimEgyRM",
    "SimRMpow",
    "SimRedshift",
]

# Band labels for the 8 photometric bands in simulation images
SIM_BANDS = [f"band{i}" for i in range(8)]


class SimGalaxyImage(Image):
    """Face-on galaxy image from cosmological simulation.

    Raw data shape: (batch, 8, 256, 256) — 8 photometric bands,
    256×256 pixels, linearly normalized via mag_normalize.
    Tokenized to 32×32 = 1024 tokens (n_compressions=3 in MagVitAE).
    """

    name: ClassVar[str] = "sim_galaxy_image"
    token_key: ClassVar[str] = "tok_sim_galaxy_image"
    num_tokens: ClassVar[int] = 1024  # 32×32 after 3× spatial compression


@dataclass
class SimSFH(Modality):
    """Star formation history from cosmological simulation.

    Shape: (batch, 2, 24)
      - channel 0: lookback time grid (fixed, same for all galaxies)
      - channel 1: log SFR at each lookback time (clipped at -3)

    The codec encodes only channel 1; channel 0 is stored as a buffer
    and restored during decoding.

    Tokenized to 6 tokens (24 / stride-4 stem).
    """

    name: ClassVar[str] = "sim_sfh"
    token_key: ClassVar[str] = "tok_sim_sfh"
    num_tokens: ClassVar[int] = 6

    value: Float[Tensor, "batch 2 24"]

    def __repr__(self) -> str:
        return f"SimSFH(shape={list(self.value.shape)})"


@dataclass
class SimGasFaceon(Modality):
    """Face-on gas image from cosmological simulation.

    Shape: (batch, 2, 128, 128)
      - channel 0: log10(gas surface density + 1e-10)
      - channel 1: log10(mass-weighted temperature + 1e-10)

    Tokenized to 32×32 = 1024 tokens (n_compressions=2 in MagVitAE).
    """

    name: ClassVar[str] = "sim_gas_faceon"
    token_key: ClassVar[str] = "tok_sim_gas_faceon"
    num_tokens: ClassVar[int] = 1024

    value: Float[Tensor, "batch 2 128 128"]

    def __repr__(self) -> str:
        return f"SimGasFaceon(shape={list(self.value.shape)})"


@dataclass
class SimGasProfile(Modality):
    """Radial gas density profile from cosmological simulation.

    Shape: (batch, 2, 20)
      - channel 0: normalized radius r/r200 (fixed grid)
      - channel 1: log10(gas density), clipped at -10

    The codec encodes only channel 1; channel 0 is stored as a buffer.
    Tokenized to 5 tokens (20 / stride-4 stem).
    """

    name: ClassVar[str] = "sim_gas_profile"
    token_key: ClassVar[str] = "tok_sim_gas_profile"
    num_tokens: ClassVar[int] = 5

    value: Float[Tensor, "batch 2 20"]

    def __repr__(self) -> str:
        return f"SimGasProfile(shape={list(self.value.shape)})"


@dataclass
class SimDMProfile(Modality):
    """Radial dark matter density profile from cosmological simulation.

    Shape: (batch, 2, 20)
      - channel 0: normalized radius r/r200 (fixed grid)
      - channel 1: log10(DM density), clipped at -10

    The codec encodes only channel 1; channel 0 is stored as a buffer.
    Tokenized to 5 tokens (20 / stride-4 stem).
    """

    name: ClassVar[str] = "sim_dm_profile"
    token_key: ClassVar[str] = "tok_sim_dm_profile"
    num_tokens: ClassVar[int] = 5

    value: Float[Tensor, "batch 2 20"]

    def __repr__(self) -> str:
        return f"SimDMProfile(shape={list(self.value.shape)})"


class SimSFR(Scalar):
    """Instantaneous star formation rate (log scale) from simulation."""

    name: ClassVar[str] = "sim_sfr"
    token_key: ClassVar[str] = "tok_sim_sfr"
    num_tokens: ClassVar[int] = 1


class SimMstar(Scalar):
    """Stellar mass (log scale) from simulation."""

    name: ClassVar[str] = "sim_mstar"
    token_key: ClassVar[str] = "tok_sim_mstar"
    num_tokens: ClassVar[int] = 1


class SimMhalo(Scalar):
    """Halo mass (log scale) from simulation."""

    name: ClassVar[str] = "sim_mhalo"
    token_key: ClassVar[str] = "tok_sim_mhalo"
    num_tokens: ClassVar[int] = 1


class SimR200(Scalar):
    """Virial radius r_crit200 (log10, ckpc/h) from simulation."""

    name: ClassVar[str] = "sim_r200"
    token_key: ClassVar[str] = "tok_sim_r200"
    num_tokens: ClassVar[int] = 1


class SimMbh(Scalar):
    """Central black hole mass (log10, M_sun) from simulation."""

    name: ClassVar[str] = "sim_mbh"
    token_key: ClassVar[str] = "tok_sim_mbh"
    num_tokens: ClassVar[int] = 1


class SimEgyRM(Scalar):
    """Cumulative BH kinetic (radio-mode) energy injection (log10) from simulation."""

    name: ClassVar[str] = "sim_egyRM"
    token_key: ClassVar[str] = "tok_sim_egyRM"
    num_tokens: ClassVar[int] = 1


class SimRMpow(Scalar):
    """Instantaneous BH kinetic power (log10, erg/s) from simulation."""

    name: ClassVar[str] = "sim_RMpow"
    token_key: ClassVar[str] = "tok_sim_RMpow"
    num_tokens: ClassVar[int] = 1


class SimRedshift(Scalar):
    """Scale factor of the simulation snapshot."""

    name: ClassVar[str] = "sim_redshift"
    token_key: ClassVar[str] = "tok_sim_redshift"
    num_tokens: ClassVar[int] = 1
