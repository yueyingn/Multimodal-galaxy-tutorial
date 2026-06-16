"""Transformer modality configuration for simulation domain.

Token counts:
  tok_sim_galaxy_image : 1024  (32×32 FSQ, vocab=625)
  tok_sim_gas_faceon   : 1024  (32×32 FSQ, vocab=625)
  tok_sim_gas_profile  :    5  (LFQ, vocab=1024)
  tok_sim_dm_profile   :    5  (LFQ, vocab=1024)
  tok_sim_sfh          :    6  (LFQ, vocab=1024)
  tok_sim_sfr          :    1  (scalar, vocab=4096)
  tok_sim_mstar        :    1  (scalar, vocab=4096)
  tok_sim_mhalo        :    1  (scalar, vocab=4096)
  tok_sim_r200         :    1  (scalar, vocab=4096)
  tok_sim_mbh          :    1  (scalar, vocab=4096)
  tok_sim_egyRM        :    1  (scalar, vocab=4096)
  tok_sim_RMpow        :    1  (scalar, vocab=4096)
  tok_sim_redshift     :    1  (scalar, vocab=4096)
"""

from functools import partial

from aion.fourm.decoder_embeddings import ImageTokenDecoderEmbedding
from aion.fourm.encoder_embeddings import ImageTokenEncoderEmbedding
from aion.fourm.modality_info import generate_uint15_hash


def _scalar_entry(token_key: str, vocab_size: int = 4096) -> dict:
    emb_kwargs = dict(vocab_size=vocab_size, patch_size=1, image_size=(1, 1))
    return {
        "input_size": 1,
        "patch_size": 1,
        "vocab_size": vocab_size,
        "encoder_embedding": partial(ImageTokenEncoderEmbedding, **emb_kwargs),
        "decoder_embedding": partial(ImageTokenDecoderEmbedding, **emb_kwargs),
        "min_tokens": 0,
        "max_tokens": 1,
        "type": "img",
        "id": generate_uint15_hash(token_key),
        "pretokenized": True,
        # Mark as ordinal: bin index k+1 is "close to" bin k. Used by the
        # FourM loss to enable Gaussian-smoothed cross-entropy. Scalar codecs
        # use a reservoir CDF, so adjacent bins have adjacent values; image
        # (FSQ) and 1D (LFQ) codebooks do not have this property.
        "is_ordinal": True,
    }


def _seq1d_entry(token_key: str, n_tokens: int, vocab_size: int = 1024) -> dict:
    """1D sequence treated as (n_tokens × 1) for positional encoding."""
    emb_kwargs = dict(vocab_size=vocab_size, patch_size=1, image_size=(n_tokens, 1))
    return {
        "input_size": n_tokens,
        "patch_size": 1,
        "vocab_size": vocab_size,
        "encoder_embedding": partial(ImageTokenEncoderEmbedding, **emb_kwargs),
        "decoder_embedding": partial(ImageTokenDecoderEmbedding, **emb_kwargs),
        "min_tokens": 0,
        "max_tokens": n_tokens,
        "type": "img",
        "id": generate_uint15_hash(token_key),
        "pretokenized": True,
    }


def _image2d_entry(token_key: str, input_size: int, patch_size: int,
                   vocab_size: int = 625) -> dict:
    """2D image tokenized to (input_size/patch_size)^2 tokens."""
    emb_kwargs = dict(vocab_size=vocab_size, patch_size=patch_size,
                      image_size=(input_size, input_size))
    return {
        "input_size": input_size,
        "patch_size": patch_size,
        "vocab_size": vocab_size,
        "encoder_embedding": partial(ImageTokenEncoderEmbedding, **emb_kwargs),
        "decoder_embedding": partial(ImageTokenDecoderEmbedding, **emb_kwargs),
        "min_tokens": 0,
        "max_tokens": None,
        "type": "img",
        "id": generate_uint15_hash(token_key),
        "pretokenized": True,
    }


SIM_MODALITY_INFO: dict = {
    # 128×128×8 star image → 32×32 = 1024 FSQ tokens
    "tok_sim_galaxy_image": _image2d_entry("tok_sim_galaxy_image",
                                           input_size=128, patch_size=4, vocab_size=625),

    # 128×128×2 gas image → 32×32 = 1024 FSQ tokens
    "tok_sim_gas_faceon":   _image2d_entry("tok_sim_gas_faceon",
                                           input_size=128, patch_size=4, vocab_size=625),

    # radial density profiles — 5 LFQ tokens each
    "tok_sim_gas_profile":  _seq1d_entry("tok_sim_gas_profile",  n_tokens=5),
    "tok_sim_dm_profile":   _seq1d_entry("tok_sim_dm_profile",   n_tokens=5),

    # SFH — 6 LFQ tokens
    "tok_sim_sfh":          _seq1d_entry("tok_sim_sfh",          n_tokens=6),

    # scalars — 1 token each
    "tok_sim_sfr":          _scalar_entry("tok_sim_sfr"),
    "tok_sim_mstar":        _scalar_entry("tok_sim_mstar"),
    "tok_sim_mhalo":        _scalar_entry("tok_sim_mhalo"),
    "tok_sim_r200":         _scalar_entry("tok_sim_r200"),
    "tok_sim_mbh":          _scalar_entry("tok_sim_mbh"),
    "tok_sim_egyRM":        _scalar_entry("tok_sim_egyRM"),
    "tok_sim_RMpow":        _scalar_entry("tok_sim_RMpow"),
    "tok_sim_redshift":     _scalar_entry("tok_sim_redshift"),
}


def build_sim_embeddings() -> tuple[dict, dict]:
    """Construct encoder and decoder embedding dicts for the sim FourM model."""
    encoder_embeddings = {
        key: info["encoder_embedding"]()
        for key, info in SIM_MODALITY_INFO.items()
    }
    decoder_embeddings = {
        key: info["decoder_embedding"]()
        for key, info in SIM_MODALITY_INFO.items()
    }
    return encoder_embeddings, decoder_embeddings
