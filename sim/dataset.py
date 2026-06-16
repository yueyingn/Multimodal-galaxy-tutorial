"""Dataset classes for simulation training.

GalaxyDataset       — loads raw .npz files; used for codec training.
SimTokenizedDataset — loads pre-tokenized .pt files; used for transformer training.
prepare_mod_dict    — converts a token dict into the mod_dict format FourM expects.
"""

import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

PROFILE_FLOOR = -10.0   # clip floor for log10 density profiles


def mag_normalize(x: np.ndarray, stats: Optional[dict] = None) -> np.ndarray:
    """IQR normalization for star images (N, 8, H, W).

    If `stats` (with keys 'median' and 'iqr') is given, those *fixed* constants
    are used. Otherwise the median/IQR are recomputed over `x`. Passing fixed
    stats is what keeps the normalization identical across the train set, the
    held-out val set, and downstream tokenization — recomputing per-load would
    silently rescale a different galaxy subset onto a different range and feed
    the codec out-of-distribution images (the bug behind the old z=0 subset).
    """
    if stats is not None:
        median = float(stats["median"])
        iqr = float(stats["iqr"])
    else:
        median = np.median(x)
        iqr = np.percentile(x, 75) - np.percentile(x, 25)
    return ((x - median) / iqr).astype(np.float32)


def gas_normalize(x: np.ndarray) -> np.ndarray:
    """IQR normalization for gas images (N, 2, H, W)."""
    global_median = np.median(x)
    q25 = np.percentile(x, 25)
    q75 = np.percentile(x, 75)
    return ((x - global_median) / (q75 - q25)).astype(np.float32)


class GalaxyDataset(Dataset):
    """Loads raw simulation data from .npz files.

    Expected .npz keys (TNG-100 training set):
        star_faceon   — (N, 8, 128, 128)
        gas_faceon    — (N, 2, 128, 128)   [optional; absent in the tutorial dataset]
        gas_profile   — (N, 2, 20)
        dm_profile    — (N, 2, 20)
        sfh           — (N, 2, 24)
        sfr           — (N,)
        mstar         — (N,)
        mhalo         — (N,)
        r200          — (N,)
        mbh           — (N,)
        egyRM         — (N,)
        RMpow         — (N,)
        scale_factor  — (N,)
    """

    def __init__(
        self,
        files: list[str | Path],
        normalize_images: bool = True,
        img_norm: Optional[dict] = None,
    ):
        # img_norm: optional {'median','iqr'} fixed constants for star-image
        # normalization. When given, the SAME scale is applied to every dataset
        # (train / val / tokenization) instead of recomputing per-load.
        self.img_norm = img_norm
        # gas_faceon (the 2-channel gas image) is optional: the tutorial dataset
        # ships without it. We only load it when the key is present.
        with np.load(files[0]) as _d0:
            self.has_gas_img = "gas_faceon" in _d0.files

        star_imgs, gas_imgs = [], []
        gas_profs, dm_profs = [], []
        sfhs = []
        sfr0, m4s, mhs, r200s, mbhs, egyRMs, RMpows, sas = (
            [], [], [], [], [], [], [], []
        )

        for f in files:
            with np.load(f) as data:
                star_imgs.append(data["star_faceon"])
                if self.has_gas_img:
                    gas_imgs.append(data["gas_faceon"])

                gp = data["gas_profile"].copy()
                gp[:, 1, :] = np.clip(gp[:, 1, :], PROFILE_FLOOR, None)
                gas_profs.append(gp)

                dp = data["dm_profile"].copy()
                dp[:, 1, :] = np.clip(dp[:, 1, :], PROFILE_FLOOR, None)
                dm_profs.append(dp)

                sfhs.append(data["sfh"])
                sfr0.append(data["sfr"])
                m4s.append(data["mstar"])
                mhs.append(data["mhalo"])
                r200s.append(data["r200"])
                mbhs.append(data["mbh"])
                egyRMs.append(data["egyRM"])
                RMpows.append(data["RMpow"])
                sas.append(data["scale_factor"])

        star_imgs = np.concatenate(star_imgs, axis=0)
        gas_profs = np.concatenate(gas_profs, axis=0)
        dm_profs  = np.concatenate(dm_profs,  axis=0)
        sfhs      = np.concatenate(sfhs,      axis=0)
        sfr0      = np.concatenate(sfr0,      axis=0)
        m4s       = np.concatenate(m4s,       axis=0)
        mhs       = np.concatenate(mhs,       axis=0)
        r200s     = np.concatenate(r200s,     axis=0)
        mbhs      = np.concatenate(mbhs,      axis=0)
        egyRMs    = np.concatenate(egyRMs,    axis=0)
        RMpows    = np.concatenate(RMpows,    axis=0)
        sas       = np.concatenate(sas,       axis=0)

        # clip SFH log SFR at -3
        sfhs[:, 1, :][sfhs[:, 1, :] < -3] = -3

        if normalize_images:
            star_imgs = mag_normalize(star_imgs, stats=img_norm)

        self.star_img   = torch.tensor(star_imgs, dtype=torch.float32)
        if self.has_gas_img:
            gas_imgs = np.concatenate(gas_imgs, axis=0)
            if normalize_images:
                gas_imgs = gas_normalize(gas_imgs)
            self.gas_img = torch.tensor(gas_imgs, dtype=torch.float32)
        else:
            self.gas_img = None
        self.gas_prof   = torch.tensor(gas_profs, dtype=torch.float32)
        self.dm_prof    = torch.tensor(dm_profs,  dtype=torch.float32)
        self.sfh        = torch.tensor(sfhs,      dtype=torch.float32)
        self.sfr0       = torch.tensor(sfr0,      dtype=torch.float32)
        self.m4s        = torch.tensor(m4s,       dtype=torch.float32)
        self.mhs        = torch.tensor(mhs,       dtype=torch.float32)
        self.r200       = torch.tensor(r200s,     dtype=torch.float32)
        self.mbh        = torch.tensor(mbhs,      dtype=torch.float32)
        self.egyRM      = torch.tensor(egyRMs,    dtype=torch.float32)
        self.RMpow      = torch.tensor(RMpows,    dtype=torch.float32)
        self.scale_factor = torch.tensor(sas,     dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.star_img)

    def __getitem__(self, idx: int) -> dict:
        sample = {
            "star_img":   self.star_img[idx],       # (8, 128, 128)
            "gas_prof":   self.gas_prof[idx],        # (2, 20)
            "dm_prof":    self.dm_prof[idx],         # (2, 20)
            "sfh":        self.sfh[idx],             # (2, 24)
            "meta": {
                "sfr0":         self.sfr0[idx].unsqueeze(-1),
                "m4s":          self.m4s[idx].unsqueeze(-1),
                "mhs":          self.mhs[idx].unsqueeze(-1),
                "r200":         self.r200[idx].unsqueeze(-1),
                "mbh":          self.mbh[idx].unsqueeze(-1),
                "egyRM":        self.egyRM[idx].unsqueeze(-1),
                "RMpow":        self.RMpow[idx].unsqueeze(-1),
                "scale_factor": self.scale_factor[idx].unsqueeze(-1),
            },
        }
        if self.has_gas_img:
            sample["gas_img"] = self.gas_img[idx]   # (2, 128, 128)
        return sample


class SimTokenizedDataset(Dataset):
    """Loads pre-tokenized integer tensors saved by tokenize_dataset()."""

    def __init__(self, token_file: str | Path):
        self.tokens: dict[str, Tensor] = torch.load(token_file, weights_only=True)
        lengths = [v.shape[0] for v in self.tokens.values()]
        assert len(set(lengths)) == 1, "Token tensors have mismatched lengths"
        self._len = lengths[0]

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        return {k: v[idx] for k, v in self.tokens.items()}


def prepare_mod_dict(
    tokens: dict[str, Tensor],
    mask_frac: float | tuple[float, float] = 0.5,
    min_encoder_modalities: int = 1,
    encoder_only_modalities: Optional[set[str]] = None,
) -> dict[str, dict[str, Tensor]]:
    """Convert a flat token dict into the mod_dict format FourM.forward() expects.

    Args:
        tokens: Dict of pre-tokenized integer tensors, one per modality.
        mask_frac: Fraction of the *remaining* (non-encoder-only) modalities
            assigned to the decoder per step. Either a scalar (used as-is
            every call, original behavior) or a `(lo, hi)` tuple — in which
            case the value is sampled uniformly from `[lo, hi]` per call so
            the model trains on a range of conditioning regimes instead of
            a single 50/50 split. The sampled value is then quantized to an
            integer modality count by the same rounding rule below.
        min_encoder_modalities: Lower bound on encoder modalities sampled from
            the remaining pool (encoder-only modalities are not counted toward
            this).
        encoder_only_modalities: Modalities that should *always* sit in the
            encoder and never be decoder targets. Useful for heavy modalities
            (e.g. tokenized galaxy images) that dominate the loss budget and
            are not needed as prediction targets in downstream tasks. When
            None or empty, behavior is identical to the previous version.
    """
    encoder_only = set(encoder_only_modalities or ())
    unknown = encoder_only - set(tokens)
    if unknown:
        raise ValueError(
            f"encoder_only_modalities {sorted(unknown)} not present in tokens"
        )

    sampleable_keys = [k for k in tokens.keys() if k not in encoder_only]
    random.shuffle(sampleable_keys)

    if isinstance(mask_frac, tuple):
        lo, hi = mask_frac
        mf = random.uniform(lo, hi)
    else:
        mf = mask_frac

    # Cap at len-1 so at least one sampleable modality is always a decoder
    # target (otherwise the decoder has nothing to predict and the loss is
    # ill-defined). Floor at min_encoder_modalities for the symmetric reason.
    n_enc_sampled = max(
        min_encoder_modalities,
        round(len(sampleable_keys) * (1.0 - mf)),
    )
    n_enc_sampled = min(n_enc_sampled, max(len(sampleable_keys) - 1, 0))
    encoder_keys = set(sampleable_keys[:n_enc_sampled]) | encoder_only

    mod_dict: dict[str, dict[str, Tensor]] = {}
    for key in tokens.keys():
        t = tokens[key].long()
        B, N = t.shape
        is_encoder = key in encoder_keys
        is_encoder_only = key in encoder_only

        input_mask  = torch.zeros(B, N, dtype=torch.bool, device=t.device) if is_encoder \
                      else torch.ones(B, N, dtype=torch.bool, device=t.device)
        # target_mask=1 excludes a token from the decoder-target pool (it is
        # sorted to the end by forward_mask_decoder's argsort and dropped by
        # the num_decoder_tokens budget). We only set this for explicitly
        # encoder-only modalities to preserve the previous training behavior
        # for everything else.
        target_mask = torch.ones(B, N, dtype=torch.bool, device=t.device) if is_encoder_only \
                      else torch.zeros(B, N, dtype=torch.bool, device=t.device)

        mod_dict[key] = {
            "tensor":                 t,
            "input_mask":             input_mask,
            "target_mask":            target_mask,
            "decoder_attention_mask": torch.zeros(B, N, dtype=torch.bool, device=t.device),
        }

    return mod_dict
