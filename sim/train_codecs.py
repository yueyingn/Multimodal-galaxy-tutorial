"""Codec training script — Phase 1 of simulation model training.

Phases (run in order or individually via --phase):

  star_image  — VQ-AE for 128×128×8 star face-on images    (~hours)
  gas_image   — VQ-AE for 128×128×2 gas face-on images     (~hours)
  sfh         — 1D VQ-AE for star formation histories       (~minutes)
  profiles    — 1D VQ-AE for gas and DM density profiles    (~minutes)
  scalars     — reservoir-CDF calibration for all scalars   (~minutes, no GPU)
  tokenize    — run all codecs over full dataset and save   (~minutes)
  all         — run all phases in sequence

star_image and gas_image are slow and independent — submit them as
separate SLURM jobs to run in parallel, then run the remaining phases
(sfh, profiles, scalars, tokenize) once both image codecs are done.
"""

import argparse
import os
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from sim.visualize_codecs import (
    plot_star_epoch,
    plot_gas_epoch,
    plot_sfh_epoch,
    plot_gas_profile_epoch,
    plot_dm_profile_epoch,
    plot_profile_epoch,
)

from sim.codecs.gas_image import SimGasFaceonCodec
from sim.codecs.image import SimGalaxyImageCodec
from sim.codecs.profile import GasProfileCodec, DMProfileCodec
from sim.codecs.scalars import (
    SimScalarCodec,
    make_egyRM_codec,
    make_mbh_codec,
    make_mhalo_codec,
    make_mstar_codec,
    make_r200_codec,
    make_redshift_codec,
    make_RMpow_codec,
    make_sfr_codec,
)
from sim.codecs.sfh import SFHCodec
from sim.dataset import GalaxyDataset
from sim.modalities import (
    SIM_BANDS,
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


# ---------------------------------------------------------------------------
# Fixed viz-batch helper
# ---------------------------------------------------------------------------

def _viz_batch(dataset, n: int = 6, seed: int = 42) -> dict:
    """Grab n fixed random samples as a dict of tensors for visualization."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=rng)[:n].tolist()
    samples = [dataset[i] for i in indices]
    out = {
        "star_img": torch.stack([s["star_img"] for s in samples]),
        "gas_prof": torch.stack([s["gas_prof"] for s in samples]),
        "dm_prof":  torch.stack([s["dm_prof"]  for s in samples]),
        "sfh":      torch.stack([s["sfh"]      for s in samples]),
    }
    # gas_faceon is out of scope for the tutorial dataset; include only if present.
    if "gas_img" in samples[0]:
        out["gas_img"] = torch.stack([s["gas_img"] for s in samples])
    return out


def load_img_norm(data_dir: str | Path) -> dict | None:
    """Load persisted star-image IQR normalization constants, if present.

    Returns {'median', 'iqr'} so train / val / tokenization all share one scale.
    """
    import json
    p = Path(data_dir) / "img_norm.json"
    if not p.exists():
        print(f"  [img_norm] none found at {p} — falling back to per-load IQR")
        return None
    d = json.loads(p.read_text())
    print(f"  [img_norm] using fixed median={d['median']:.4f} iqr={d['iqr']:.4f} from {p}")
    return {"median": d["median"], "iqr": d["iqr"]}


# ---------------------------------------------------------------------------
# Phase A1: Star image codec
# ---------------------------------------------------------------------------

@torch.no_grad()
def _val_recon(codec, val_loader, key, make_mod, field, device) -> float:
    """Mean reconstruction MSE of `codec` over a held-out val loader."""
    codec.eval()
    tot, nb = 0.0, 0
    for batch in val_loader:
        x_in = batch[key].to(device)
        mod  = make_mod(x_in)
        z          = codec._encode(mod)
        z_q, _, _  = codec.quantizer(z)
        x_hat      = codec._decode(z_q)
        pred = getattr(x_hat, field)
        if field == "flux":
            target = x_in
        else:
            # 1-D codecs only score the value (density / SFR) channel
            target = x_in[:, 1:2, :]
            pred = pred[:, 1:2, :]
        tot += F.mse_loss(pred, target).item(); nb += 1
    return tot / max(nb, 1)


def train_star_image_codec(
    files: list[str],
    save_dir: str,
    val_files: list[str] | None = None,
    img_norm: dict | None = None,
    epochs: int = 80,
    lr: float = 1e-4,
    batch_size: int = 8,
    device: str = "cuda",
    viz_n: int = 6,
) -> SimGalaxyImageCodec:
    """Train the star face-on galaxy image VQ-autoencoder.

    When `val_files` is given, the best checkpoint (lowest held-out val recon
    MSE) is saved — not just the latest — so reload picks a generalizing model.
    """
    print("=== Phase A1: Training star image codec ===")
    codec = SimGalaxyImageCodec().to(device)
    opt = AdamW(codec.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(opt, T_max=epochs)

    dataset  = GalaxyDataset(files, normalize_images=True, img_norm=img_norm)
    loader   = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                          num_workers=4, pin_memory=True)
    val_loader = None
    if val_files:
        val_ds = GalaxyDataset(val_files, normalize_images=True, img_norm=img_norm)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                num_workers=2, pin_memory=True)
        print(f"  train={len(dataset)}  val={len(val_ds)}")
    vbatch   = _viz_batch(dataset, viz_n)
    plot_dir = Path(save_dir) / "recon_plots"
    mk = lambda im: SimGalaxyImage(flux=im, bands=SIM_BANDS)
    best_val = float("inf")

    for epoch in range(epochs):
        codec.train()
        total_loss = 0.0
        for batch in loader:
            img = batch["star_img"].to(device)
            x   = mk(img)

            z         = codec._encode(x)
            z_q, _, _ = codec.quantizer(z)
            x_hat     = codec._decode(z_q)

            loss = F.mse_loss(x_hat.flux, img)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(codec.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()

        scheduler.step()
        avg = total_loss / len(loader)

        if val_loader is not None:
            vavg = _val_recon(codec, val_loader, "star_img", mk, "flux", device)
            improved = vavg < best_val
            tag = "  *best" if improved else ""
            print(f"  star_image  epoch {epoch+1:03d}/{epochs}  "
                  f"train={avg:.5f}  val={vavg:.5f}{tag}")
            if improved:
                best_val = vavg
                _save_codec(codec, save_dir, SimGalaxyImage)
                plot_star_epoch(codec, vbatch["star_img"], epoch + 1, plot_dir, device, n=viz_n)
        else:
            print(f"  star_image  epoch {epoch+1:03d}/{epochs}  recon={avg:.5f}")
            if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                _save_codec(codec, save_dir, SimGalaxyImage)
                plot_star_epoch(codec, vbatch["star_img"], epoch + 1, plot_dir, device, n=viz_n)

    if val_loader is not None:
        print(f"  star_image best val recon MSE: {best_val:.5f}")
    return codec


# ---------------------------------------------------------------------------
# Phase A2: Gas image codec
# ---------------------------------------------------------------------------

def train_gas_image_codec(
    files: list[str],
    save_dir: str,
    epochs: int = 80,
    lr: float = 1e-4,
    batch_size: int = 8,
    device: str = "cuda",
    viz_n: int = 6,
) -> SimGasFaceonCodec:
    """Train the gas face-on image VQ-autoencoder (2-channel log10 images)."""
    print("=== Phase A2: Training gas image codec ===")
    codec = SimGasFaceonCodec().to(device)
    opt = AdamW(codec.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(opt, T_max=epochs)

    dataset  = GalaxyDataset(files, normalize_images=True)
    loader   = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                          num_workers=4, pin_memory=True)
    vbatch   = _viz_batch(dataset, viz_n)
    plot_dir = Path(save_dir) / "recon_plots"

    for epoch in range(epochs):
        codec.train()
        total_loss = 0.0
        for batch in loader:
            img = batch["gas_img"].to(device)
            x   = SimGasFaceon(value=img)

            z         = codec._encode(x)
            z_q, _, _ = codec.quantizer(z)
            x_hat     = codec._decode(z_q)

            loss = F.mse_loss(x_hat.value, img)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(codec.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()

        scheduler.step()
        avg = total_loss / len(loader)
        print(f"  gas_image  epoch {epoch+1:03d}/{epochs}  recon={avg:.5f}")

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            _save_codec(codec, save_dir, SimGasFaceon)
            plot_gas_epoch(codec, vbatch["gas_img"], epoch + 1, plot_dir, device, n=viz_n)

    return codec


# ---------------------------------------------------------------------------
# Phase B: SFH codec
# ---------------------------------------------------------------------------

def train_sfh_codec(
    files: list[str],
    save_dir: str,
    val_files: list[str] | None = None,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 64,
    lfq_weight: float = 0.1,
    device: str = "cuda",
    viz_n: int = 6,
) -> SFHCodec:
    """Train the star formation history 1D VQ-autoencoder (best-by-val)."""
    print("=== Phase B: Training SFH codec ===")
    codec = SFHCodec().to(device)
    opt = AdamW(codec.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(opt, T_max=epochs)

    dataset  = GalaxyDataset(files, normalize_images=False)
    loader   = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                          num_workers=4, pin_memory=True)
    val_loader = None
    if val_files:
        val_ds = GalaxyDataset(val_files, normalize_images=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
        print(f"  train={len(dataset)}  val={len(val_ds)}")
    vbatch   = _viz_batch(dataset, viz_n)
    plot_dir = Path(save_dir) / "recon_plots"

    time_grid = dataset.sfh[0, 0, :]
    codec.set_time_grid(time_grid)
    mk = lambda v: SimSFH(value=v)
    best_val = float("inf")

    for epoch in range(epochs):
        codec.train()
        total_recon = 0.0; total_lfq = 0.0

        for batch in loader:
            sfh = batch["sfh"].to(device)
            x   = mk(sfh)

            z                = codec._encode(x)
            z_q, lfq_loss, _ = codec.quantizer(z)
            x_hat            = codec._decode(z_q)

            sfr_true = sfh[:, 1:2, :]
            sfr_pred = x_hat.value[:, 1:2, :]
            recon_loss = F.mse_loss(sfr_pred, sfr_true)
            loss = recon_loss + lfq_weight * lfq_loss

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(codec.parameters(), 1.0)
            opt.step()
            total_recon += recon_loss.item()
            total_lfq   += lfq_loss.item() if isinstance(lfq_loss, torch.Tensor) else 0.0

        scheduler.step()
        nb = len(loader)
        if val_loader is not None:
            vavg = _val_recon(codec, val_loader, "sfh", mk, "value", device)
            improved = vavg < best_val
            tag = "  *best" if improved else ""
            print(f"  SFH  epoch {epoch+1:03d}/{epochs}  train={total_recon/nb:.5f}  "
                  f"lfq={total_lfq/nb:.5f}  val={vavg:.5f}{tag}")
            if improved:
                best_val = vavg
                _save_codec(codec, save_dir, SimSFH)
                plot_sfh_epoch(codec, vbatch["sfh"], epoch + 1, plot_dir, device, n=viz_n)
        else:
            print(f"  SFH  epoch {epoch+1:03d}/{epochs}  "
                  f"recon={total_recon/nb:.5f}  lfq={total_lfq/nb:.5f}")
            if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                _save_codec(codec, save_dir, SimSFH)
                plot_sfh_epoch(codec, vbatch["sfh"], epoch + 1, plot_dir, device, n=viz_n)

    if val_loader is not None:
        print(f"  SFH best val recon MSE: {best_val:.5f}")
    return codec


# ---------------------------------------------------------------------------
# Phase C: Profile codecs
# ---------------------------------------------------------------------------

def train_profile_codecs(
    files: list[str],
    save_dir: str,
    val_files: list[str] | None = None,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 64,
    lfq_weight: float = 0.1,
    device: str = "cuda",
    viz_n: int = 6,
) -> tuple:
    """Train gas and DM radial density profile codecs sequentially.

    Each codec's log-density standardization stats are calibrated on the TRAIN
    split via codec.calibrate() before training (this is the upgrade that lets
    the LFQ actually use its codebook). Best checkpoint is chosen by held-out
    val recon MSE when val_files is given.
    """
    print("=== Phase C: Training profile codecs ===")
    dataset  = GalaxyDataset(files, normalize_images=False)
    loader   = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                          num_workers=4, pin_memory=True)
    val_loader = None
    if val_files:
        val_ds = GalaxyDataset(val_files, normalize_images=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
        print(f"  train={len(dataset)}  val={len(val_ds)}")
    vbatch   = _viz_batch(dataset, viz_n)
    plot_dir = Path(save_dir) / "recon_plots"

    radius_grid = dataset.gas_prof[0, 0, :]   # (20,) r/r200 bin centres

    trained = {}
    viz_fns = {
        "gas_prof": (plot_gas_profile_epoch, "gas_prof"),
        "dm_prof":  (plot_dm_profile_epoch,  "dm_prof"),
    }
    train_arr = {"gas_prof": dataset.gas_prof, "dm_prof": dataset.dm_prof}

    for CodecClass, modality, key in [
        (GasProfileCodec, SimGasProfile, "gas_prof"),
        (DMProfileCodec,  SimDMProfile,  "dm_prof"),
    ]:
        print(f"  Training {CodecClass.__name__} ...")
        codec = CodecClass().to(device)
        codec.set_radius_grid(radius_grid)
        # Standardize the log-density channel using TRAIN-split statistics.
        codec.calibrate(train_arr[key])
        print(f"    {CodecClass.__name__} calibrated: "
              f"dens_mean={float(codec.dens_mean):.3f} dens_std={float(codec.dens_std):.3f}")
        opt = AdamW(codec.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(opt, T_max=epochs)
        viz_fn, vkey = viz_fns[key]
        mk = lambda v, _m=modality: _m(value=v)
        best_val = float("inf")

        for epoch in range(epochs):
            codec.train()
            total_recon = 0.0; total_lfq = 0.0

            for batch in loader:
                prof = batch[key].to(device)
                x    = modality(value=prof)

                z                = codec._encode(x)
                z_q, lfq_loss, _ = codec.quantizer(z)
                x_hat            = codec._decode(z_q)

                density_true = prof[:, 1:2, :]
                density_pred = x_hat.value[:, 1:2, :]
                recon_loss = F.mse_loss(density_pred, density_true)
                loss = recon_loss + lfq_weight * lfq_loss

                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(codec.parameters(), 1.0)
                opt.step()
                total_recon += recon_loss.item()
                total_lfq   += lfq_loss.item() if isinstance(lfq_loss, torch.Tensor) else 0.0

            scheduler.step()
            nb = len(loader)
            if val_loader is not None:
                vavg = _val_recon(codec, val_loader, key, mk, "value", device)
                improved = vavg < best_val
                tag = "  *best" if improved else ""
                print(f"    {CodecClass.__name__}  epoch {epoch+1:03d}/{epochs}  "
                      f"train={total_recon/nb:.5f}  lfq={total_lfq/nb:.5f}  val={vavg:.5f}{tag}")
                if improved:
                    best_val = vavg
                    _save_codec(codec, save_dir, modality)
                    viz_fn(codec, vbatch[vkey], epoch + 1, plot_dir, device, n=viz_n)
            else:
                print(f"    {CodecClass.__name__}  epoch {epoch+1:03d}/{epochs}  "
                      f"recon={total_recon/nb:.5f}  lfq={total_lfq/nb:.5f}")
                if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                    _save_codec(codec, save_dir, modality)
                    viz_fn(codec, vbatch[vkey], epoch + 1, plot_dir, device, n=viz_n)

        if val_loader is not None:
            print(f"    {CodecClass.__name__} best val recon MSE: {best_val:.5f}")
        trained[key] = codec

    # Final joint panel with both codecs side-by-side
    plot_profile_epoch(
        trained["gas_prof"], trained["dm_prof"],
        vbatch["gas_prof"], vbatch["dm_prof"],
        epoch=epochs, out_dir=plot_dir, device=device, n=viz_n,
    )
    return trained["gas_prof"], trained["dm_prof"]


# ---------------------------------------------------------------------------
# Phase D: Calibrate scalar codecs
# ---------------------------------------------------------------------------

def calibrate_scalars(
    files: list[str],
    save_dir: str,
    codebook_size: int = 1024,
    reservoir_size: int = 50_000,
) -> dict:
    """Calibrate all scalar codecs via reservoir CDF estimation."""
    print("=== Phase D: Calibrating scalar codecs ===")
    codecs = {
        "tok_sim_sfr":      (make_sfr_codec(codebook_size, reservoir_size),      SimSFR,      "sfr0"),
        "tok_sim_mstar":    (make_mstar_codec(codebook_size, reservoir_size),    SimMstar,    "m4s"),
        "tok_sim_mhalo":    (make_mhalo_codec(codebook_size, reservoir_size),    SimMhalo,    "mhs"),
        "tok_sim_r200":     (make_r200_codec(codebook_size, reservoir_size),     SimR200,     "r200"),
        "tok_sim_mbh":      (make_mbh_codec(codebook_size, reservoir_size),      SimMbh,      "mbh"),
        "tok_sim_egyRM":    (make_egyRM_codec(codebook_size, reservoir_size),    SimEgyRM,    "egyRM"),
        "tok_sim_RMpow":    (make_RMpow_codec(codebook_size, reservoir_size),    SimRMpow,    "RMpow"),
        "tok_sim_redshift": (make_redshift_codec(codebook_size, reservoir_size), SimRedshift, "scale_factor"),
    }

    dataset = GalaxyDataset(files, normalize_images=False)
    loader  = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4)

    for batch in loader:
        for tok_key, (codec, _, meta_key) in codecs.items():
            values = batch["meta"][meta_key].squeeze(-1)
            codec.calibrate(values)

    for tok_key, (codec, modality, _) in codecs.items():
        floor_str = f" (floor={codec.floor})" if codec.floor is not None else ""
        _save_codec(codec, save_dir, modality)
        print(f"  {tok_key}: saved{floor_str}")

    return {k: v[0] for k, v in codecs.items()}


# ---------------------------------------------------------------------------
# Phase E: Pre-tokenize full dataset
# ---------------------------------------------------------------------------

def _load_weights(codec, codec_dir: str, modality_name: str, device: str):
    weights = Path(codec_dir) / "codecs" / modality_name / "pytorch_model.bin"
    codec.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    return codec.to(device).eval()


def _build_tokenizer_registry():
    """Spec for every modality the tokenizer can emit.

    Each entry is keyed by a short CLI name and supplies:
      token_key    — output dict key written to the .pt file
      codec_dir    — sub-directory under save_dir/codecs holding the weights
      make        — factory returning an un-initialised codec instance
      needs_gpu    — whether the codec must run on GPU (image / 1D codecs)
      encode       — fn(codec, batch, device) -> token tensor (still on device)
    """
    return {
        "star_image": dict(
            token_key="tok_sim_galaxy_image", codec_dir="sim_galaxy_image",
            make=SimGalaxyImageCodec, needs_gpu=True,
            encode=lambda c, b, d: c.encode(SimGalaxyImage(
                flux=b["star_img"].to(d), bands=SIM_BANDS)),
        ),
        "gas_image": dict(
            token_key="tok_sim_gas_faceon", codec_dir="sim_gas_faceon",
            make=SimGasFaceonCodec, needs_gpu=True,
            encode=lambda c, b, d: c.encode(SimGasFaceon(value=b["gas_img"].to(d))),
        ),
        "sfh": dict(
            token_key="tok_sim_sfh", codec_dir="sim_sfh",
            make=SFHCodec, needs_gpu=True,
            encode=lambda c, b, d: c.encode(SimSFH(value=b["sfh"].to(d))),
        ),
        "gas_profile": dict(
            token_key="tok_sim_gas_profile", codec_dir="sim_gas_profile",
            make=GasProfileCodec, needs_gpu=True,
            encode=lambda c, b, d: c.encode(SimGasProfile(value=b["gas_prof"].to(d))),
        ),
        "dm_profile": dict(
            token_key="tok_sim_dm_profile", codec_dir="sim_dm_profile",
            make=DMProfileCodec, needs_gpu=True,
            encode=lambda c, b, d: c.encode(SimDMProfile(value=b["dm_prof"].to(d))),
        ),
        "sfr": dict(
            token_key="tok_sim_sfr", codec_dir="sim_sfr",
            make=make_sfr_codec, needs_gpu=False,
            encode=lambda c, b, d: c.encode(SimSFR(value=b["meta"]["sfr0"])),
        ),
        "mstar": dict(
            token_key="tok_sim_mstar", codec_dir="sim_mstar",
            make=make_mstar_codec, needs_gpu=False,
            encode=lambda c, b, d: c.encode(SimMstar(value=b["meta"]["m4s"])),
        ),
        "mhalo": dict(
            token_key="tok_sim_mhalo", codec_dir="sim_mhalo",
            make=make_mhalo_codec, needs_gpu=False,
            encode=lambda c, b, d: c.encode(SimMhalo(value=b["meta"]["mhs"])),
        ),
        "r200": dict(
            token_key="tok_sim_r200", codec_dir="sim_r200",
            make=make_r200_codec, needs_gpu=False,
            encode=lambda c, b, d: c.encode(SimR200(value=b["meta"]["r200"])),
        ),
        "mbh": dict(
            token_key="tok_sim_mbh", codec_dir="sim_mbh",
            make=make_mbh_codec, needs_gpu=False,
            encode=lambda c, b, d: c.encode(SimMbh(value=b["meta"]["mbh"])),
        ),
        "egyRM": dict(
            token_key="tok_sim_egyRM", codec_dir="sim_egyRM",
            make=make_egyRM_codec, needs_gpu=False,
            encode=lambda c, b, d: c.encode(SimEgyRM(value=b["meta"]["egyRM"])),
        ),
        "RMpow": dict(
            token_key="tok_sim_RMpow", codec_dir="sim_RMpow",
            make=make_RMpow_codec, needs_gpu=False,
            encode=lambda c, b, d: c.encode(SimRMpow(value=b["meta"]["RMpow"])),
        ),
        "redshift": dict(
            token_key="tok_sim_redshift", codec_dir="sim_redshift",
            make=make_redshift_codec, needs_gpu=False,
            encode=lambda c, b, d: c.encode(SimRedshift(value=b["meta"]["scale_factor"])),
        ),
    }


ALL_TOKENIZE_MODALITIES = list(_build_tokenizer_registry().keys())


def tokenize_dataset(
    files: list[str],
    codec_dir: str,
    output_path: str,
    modalities: list[str] | None = None,
    batch_size: int = 32,
    device: str = "cuda",
    img_norm: dict | None = None,
) -> None:
    """Tokenize selected modalities of the dataset into a single .pt file.

    Args:
        modalities: subset of ALL_TOKENIZE_MODALITIES. None ⇒ all of them.
        img_norm: fixed star-image IQR norm constants (must match codec training).
    """
    print("=== Phase E: Pre-tokenizing dataset ===")
    registry = _build_tokenizer_registry()
    if modalities is None:
        modalities = ALL_TOKENIZE_MODALITIES
    unknown = [m for m in modalities if m not in registry]
    if unknown:
        raise ValueError(
            f"Unknown modalities: {unknown}. "
            f"Available: {ALL_TOKENIZE_MODALITIES}"
        )
    print(f"Tokenizing modalities: {modalities}")

    loaded = {}
    for name in modalities:
        spec = registry[name]
        dev = device if spec["needs_gpu"] else "cpu"
        loaded[name] = (
            _load_weights(spec["make"](), codec_dir, spec["codec_dir"], dev),
            spec, dev,
        )

    dataset = GalaxyDataset(files, normalize_images=True, img_norm=img_norm)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    all_tokens: dict[str, list[torch.Tensor]] = defaultdict(list)
    with torch.no_grad():
        for i, batch in enumerate(loader):
            for name, (codec, spec, dev) in loaded.items():
                tokens = spec["encode"](codec, batch, dev).cpu()
                all_tokens[spec["token_key"]].append(tokens)
            if (i + 1) % 50 == 0:
                print(f"  tokenized {(i+1)*batch_size} / {len(dataset)} samples")

    token_dict = {k: torch.cat(v, dim=0) for k, v in all_tokens.items()}
    for k, v in token_dict.items():
        print(f"  {k}: {tuple(v.shape)}  dtype={v.dtype}")

    torch.save(token_dict, output_path)
    print(f"  Tokens saved to {output_path}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_codec(codec, save_dir: str, modality) -> None:
    os.makedirs(save_dir, exist_ok=True)
    codec.save_pretrained(save_dir, modality=modality)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train simulation codecs")
    parser.add_argument("--files",    nargs="+", required=True)
    parser.add_argument("--val_files", nargs="+", default=None,
                        help="Held-out validation .npz file(s). When given, GPU "
                             "codecs save the best-by-val checkpoint.")
    parser.add_argument("--data_dir", default=None,
                        help="Dir holding img_norm.json (fixed star-image IQR "
                             "norm). Defaults to the dir of the first --files entry.")
    parser.add_argument("--save_dir", default="./checkpoints/codecs")
    parser.add_argument("--token_out", default="./data/tokens.pt")
    parser.add_argument("--phase",    default="all",
                        choices=["all", "star_image", "gas_image",
                                 "sfh", "profiles", "scalars", "tokenize"])
    parser.add_argument(
        "--modalities", nargs="+", default=None,
        choices=ALL_TOKENIZE_MODALITIES,
        help="Subset of modalities to tokenize in the 'tokenize' phase. "
             "Default: all modalities. Ignored for non-tokenize phases.",
    )
    parser.add_argument("--star_epochs",    type=int, default=80)
    parser.add_argument("--gas_epochs",     type=int, default=80)
    parser.add_argument("--sfh_epochs",     type=int, default=100)
    parser.add_argument("--profile_epochs", type=int, default=100)
    parser.add_argument("--image_bs",       type=int, default=8)
    parser.add_argument("--seq_bs",         type=int, default=64)
    parser.add_argument("--lr",             type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    from glob import glob
    files = []
    for pattern in args.files:
        files.extend(sorted(glob(pattern)))
    if not files:
        raise FileNotFoundError(f"No files matched: {args.files}")
    print(f"Found {len(files)} train data files.")

    val_files = None
    if args.val_files:
        val_files = []
        for pattern in args.val_files:
            val_files.extend(sorted(glob(pattern)))
        if not val_files:
            raise FileNotFoundError(f"No val files matched: {args.val_files}")
        print(f"Found {len(val_files)} val data files.")

    # Fixed star-image IQR normalization (shared by train / val / tokenize).
    data_dir = args.data_dir or str(Path(files[0]).parent)
    img_norm = load_img_norm(data_dir)

    if args.phase in ("all", "star_image"):
        train_star_image_codec(files, args.save_dir, val_files=val_files,
                               img_norm=img_norm,
                               epochs=args.star_epochs, lr=args.lr,
                               batch_size=args.image_bs, device=args.device)

    if args.phase in ("all", "gas_image"):
        train_gas_image_codec(files, args.save_dir,
                              epochs=args.gas_epochs, lr=args.lr,
                              batch_size=args.image_bs, device=args.device)

    if args.phase in ("all", "sfh"):
        train_sfh_codec(files, args.save_dir, val_files=val_files,
                        epochs=args.sfh_epochs, lr=args.lr * 10,
                        batch_size=args.seq_bs, device=args.device)

    if args.phase in ("all", "profiles"):
        train_profile_codecs(files, args.save_dir, val_files=val_files,
                             epochs=args.profile_epochs, lr=args.lr * 10,
                             batch_size=args.seq_bs, device=args.device)

    if args.phase in ("all", "scalars"):
        calibrate_scalars(files, args.save_dir)

    if args.phase in ("all", "tokenize"):
        tokenize_dataset(files, args.save_dir, args.token_out,
                         modalities=args.modalities, device=args.device,
                         img_norm=img_norm)


if __name__ == "__main__":
    main()
