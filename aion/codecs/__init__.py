"""Codec subpackage (tutorial subset).

The upstream ``aion.codecs`` package eagerly imports the observational codecs
(ImageCodec, SpectrumCodec, CatalogCodec, ...) and the CodecManager. The
simulation tutorial does not use any of those, so we only re-export the abstract
``Codec`` base class here. The simulation codecs live under ``sim.codecs``.
"""

from .base import Codec

__all__ = ["Codec"]
