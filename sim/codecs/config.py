"""Codec mapping for simulation modalities."""

from sim.codecs.gas_image import SimGasFaceonCodec
from sim.codecs.image import SimGalaxyImageCodec
from sim.codecs.profile import GasProfileCodec, DMProfileCodec
from sim.codecs.scalars import SimScalarCodec
from sim.codecs.sfh import SFHCodec
from sim.modalities import (
    SimGalaxyImage,
    SimGasFaceon,
    SimGasProfile,
    SimDMProfile,
    SimMhalo,
    SimMstar,
    SimR200,
    SimMbh,
    SimEgyRM,
    SimRMpow,
    SimRedshift,
    SimSFH,
    SimSFR,
)

SIM_MODALITY_CODEC_MAPPING = {
    SimGalaxyImage: SimGalaxyImageCodec,
    SimGasFaceon:   SimGasFaceonCodec,
    SimGasProfile:  GasProfileCodec,
    SimDMProfile:   DMProfileCodec,
    SimSFH:         SFHCodec,
    SimSFR:         SimScalarCodec,
    SimMstar:       SimScalarCodec,
    SimMhalo:       SimScalarCodec,
    SimR200:        SimScalarCodec,
    SimMbh:         SimScalarCodec,
    SimEgyRM:       SimScalarCodec,
    SimRMpow:       SimScalarCodec,
    SimRedshift:    SimScalarCodec,
}
