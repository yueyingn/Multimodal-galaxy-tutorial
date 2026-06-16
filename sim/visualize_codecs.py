"""Codec reconstruction visualizations — standalone script and in-training hooks.

Standalone CLI (post-training):
    python -m sim.visualize_codecs \
        --data   /data/TNG-100/Snap99_000.npz \
        --ckpt   /checkpoints/codecs \
        --out    recon_check.png \
        --n      6

In-training hooks (called every 10 epochs from train_codecs.py):
    plot_star_epoch(codec, imgs, epoch, out_dir, device)
    plot_gas_epoch(codec, gas_imgs, epoch, out_dir, device)
    plot_sfh_epoch(codec, sfhs, epoch, out_dir, device)
    plot_profile_epoch(gas_codec, dm_codec, gas_profs, dm_profs, epoch, out_dir, device)
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from sim.codecs.gas_image import SimGasFaceonCodec
from sim.codecs.image import SimGalaxyImageCodec
from sim.codecs.profile import GasProfileCodec, DMProfileCodec
from sim.codecs.sfh import SFHCodec
from sim.dataset import mag_normalize, gas_normalize, PROFILE_FLOOR
from sim.modalities import (
    SIM_BANDS,
    SimGalaxyImage, SimGasFaceon,
    SimGasProfile, SimDMProfile,
    SimSFH,
)

__all__ = [
    "plot_star_epoch",
    "plot_gas_epoch",
    "plot_sfh_epoch",
    "plot_gas_profile_epoch",
    "plot_dm_profile_epoch",
    "plot_profile_epoch",
]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _to_display(t: torch.Tensor) -> np.ndarray:
    """Clip a single-channel 2D tensor to [0, 1] via percentile stretch."""
    arr = t.numpy()
    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
    return np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Star image  (8-channel, show 4 representative bands)
# ---------------------------------------------------------------------------

STAR_BANDS = [0, 2, 4, 6]   # U, V, g, z


@torch.no_grad()
def plot_star_epoch(
    codec: SimGalaxyImageCodec,
    imgs: torch.Tensor,
    epoch: int,
    out_dir: str | Path,
    device: str = "cpu",
    n: int = 6,
) -> None:
    """Save star image orig/recon grid for a given training epoch.

    Args:
        codec: Trained (or in-training) SimGalaxyImageCodec.
        imgs:  Fixed sample batch, shape (N, 8, 128, 128), IQR-normalised.
        epoch: Current epoch number (used in filename).
        out_dir: Directory to write PNG files.
        device: torch device string.
        n:     Number of galaxies to show.
    """
    codec.eval()
    imgs = imgs[:n].to(device)
    x = SimGalaxyImage(flux=imgs, bands=SIM_BANDS)
    z = codec._encode(x)
    z_q, _, _ = codec.quantizer(z)
    recon = codec._decode(z_q).flux.cpu()
    orig  = imgs.cpu()
    codec.train()

    nb = len(STAR_BANDS)
    fig, axes = plt.subplots(2 * n, nb, figsize=(nb * 1.8, n * 3.2), squeeze=False)

    for gi in range(n):
        for ci, b in enumerate(STAR_BANDS):
            axes[2*gi][ci].imshow(_to_display(orig[gi, b]),  cmap="inferno", origin="lower")
            axes[2*gi+1][ci].imshow(_to_display(recon[gi, b]), cmap="inferno", origin="lower")
            if gi == 0:
                axes[0][ci].set_title(f"band {b}", fontsize=8)
            for ax in (axes[2*gi][ci], axes[2*gi+1][ci]):
                ax.axis("off")
        axes[2*gi][0].set_ylabel(f"gal {gi}\norig",  fontsize=7, rotation=0, labelpad=35, va="center")
        axes[2*gi+1][0].set_ylabel("recon", fontsize=7, rotation=0, labelpad=35, va="center")

    mse = ((orig - recon) ** 2).mean().item()
    fig.suptitle(f"Star image — epoch {epoch:03d}  MSE={mse:.4f}", fontsize=10)
    fig.tight_layout()
    out = _ensure_dir(out_dir) / f"star_epoch_{epoch:03d}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [viz] star recon saved → {out}  (MSE={mse:.4f})")


# ---------------------------------------------------------------------------
# Gas face-on image  (2-channel: Σ_gas, T_mw)
# ---------------------------------------------------------------------------

GAS_CHAN_LABELS = ["Σ_gas", "T_mw"]


@torch.no_grad()
def plot_gas_epoch(
    codec: SimGasFaceonCodec,
    gas_imgs: torch.Tensor,
    epoch: int,
    out_dir: str | Path,
    device: str = "cpu",
    n: int = 6,
) -> None:
    """Save gas image orig/recon grid for a given training epoch."""
    codec.eval()
    gas_imgs = gas_imgs[:n].to(device)
    x = SimGasFaceon(value=gas_imgs)
    z = codec._encode(x)
    z_q, _, _ = codec.quantizer(z)
    recon = codec._decode(z_q).value.cpu()
    orig  = gas_imgs.cpu()
    codec.train()

    nc = 2  # channels
    fig, axes = plt.subplots(2 * n, nc, figsize=(nc * 2.2, n * 3.2), squeeze=False)

    for gi in range(n):
        for ci in range(nc):
            axes[2*gi][ci].imshow(_to_display(orig[gi, ci]),  cmap="viridis", origin="lower")
            axes[2*gi+1][ci].imshow(_to_display(recon[gi, ci]), cmap="viridis", origin="lower")
            if gi == 0:
                axes[0][ci].set_title(GAS_CHAN_LABELS[ci], fontsize=8)
            for ax in (axes[2*gi][ci], axes[2*gi+1][ci]):
                ax.axis("off")
        axes[2*gi][0].set_ylabel(f"gal {gi}\norig",  fontsize=7, rotation=0, labelpad=35, va="center")
        axes[2*gi+1][0].set_ylabel("recon", fontsize=7, rotation=0, labelpad=35, va="center")

    mse = ((orig - recon) ** 2).mean().item()
    fig.suptitle(f"Gas image — epoch {epoch:03d}  MSE={mse:.4f}", fontsize=10)
    fig.tight_layout()
    out = _ensure_dir(out_dir) / f"gas_epoch_{epoch:03d}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [viz] gas recon saved  → {out}  (MSE={mse:.4f})")


# ---------------------------------------------------------------------------
# SFH
# ---------------------------------------------------------------------------

@torch.no_grad()
def plot_sfh_epoch(
    codec: SFHCodec,
    sfhs: torch.Tensor,
    epoch: int,
    out_dir: str | Path,
    device: str = "cpu",
    n: int = 6,
) -> None:
    """Save SFH orig/recon line plots for a given training epoch."""
    codec.eval()
    sfhs = sfhs[:n].to(device)
    x = SimSFH(value=sfhs)
    z = codec._encode(x)
    z_q, _, _ = codec.quantizer(z)
    recon = codec._decode(z_q).value.cpu()
    orig  = sfhs.cpu()
    codec.train()

    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 2.2),
                             squeeze=False)
    axes_flat = axes.flatten()

    for gi in range(n):
        ax = axes_flat[gi]
        t  = orig[gi, 0].numpy()
        y  = orig[gi, 1].numpy()
        yr = recon[gi, 1].numpy()
        ax.plot(t, y,  color="steelblue", lw=1.5, label="orig")
        ax.plot(t, yr, color="tomato",    lw=1.5, ls="--", label="recon")
        ax.set_title(f"galaxy {gi}", fontsize=8)
        ax.set_xlabel("lookback [Gyr]", fontsize=7)
        ax.set_ylabel("log SFR", fontsize=7)
        ax.tick_params(labelsize=6)
        if gi == 0:
            ax.legend(fontsize=6)

    for ax in axes_flat[n:]:
        ax.axis("off")

    mse = ((orig[:, 1:2] - recon[:, 1:2]) ** 2).mean().item()
    fig.suptitle(f"SFH — epoch {epoch:03d}  MSE={mse:.4f}", fontsize=10)
    fig.tight_layout()
    out = _ensure_dir(out_dir) / f"sfh_epoch_{epoch:03d}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [viz] SFH recon saved  → {out}  (MSE={mse:.4f})")


# ---------------------------------------------------------------------------
# Radial density profiles  (gas + DM side-by-side)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _plot_single_profile(
    codec,
    profs: torch.Tensor,
    ModClass,
    label: str,
    fname: str,
    epoch: int,
    out_dir: str | Path,
    device: str = "cpu",
    n: int = 6,
) -> float:
    """Plot orig/recon for a single profile codec. Returns MSE."""
    codec.eval()
    profs = profs[:n].to(device)
    z = codec._encode(ModClass(value=profs))
    z_q, _, _ = codec.quantizer(z)
    recon = codec._decode(z_q).value.cpu()
    orig  = profs.cpu()
    codec.train()

    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.0, nrows * 2.2),
                             squeeze=False)
    axes_flat = axes.flatten()

    # y-axis: keep floor as lower bound but autoscale upper from data
    all_density = np.concatenate([orig[:n, 1].numpy().ravel(), recon[:n, 1].numpy().ravel()])
    ymax = float(np.percentile(all_density[all_density > PROFILE_FLOOR], 99)) + 0.5
    ymin = PROFILE_FLOOR - 0.5

    for gi in range(n):
        ax = axes_flat[gi]
        r  = orig[gi, 0].numpy()
        y  = orig[gi, 1].numpy()
        yr = recon[gi, 1].numpy()
        ax.plot(r, y,  color="steelblue", lw=1.5, label="orig")
        ax.plot(r, yr, color="tomato",    lw=1.5, ls="--", label="recon")
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("r / r200", fontsize=7)
        ax.set_ylabel("log ρ", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_title(f"gal {gi}", fontsize=8)
        if gi == 0:
            ax.legend(fontsize=6)

    for ax in axes_flat[n:]:
        ax.axis("off")

    mse = ((orig[:, 1:2] - recon[:, 1:2]) ** 2).mean().item()
    orig_range = float(orig[:n, 1].max() - orig[:n, 1].min())
    fig.suptitle(f"{label} — epoch {epoch:03d}  MSE={mse:.4f}  data_range={orig_range:.2f}", fontsize=10)
    fig.tight_layout()
    out = _ensure_dir(out_dir) / fname
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [viz] {label} recon saved → {out}  (MSE={mse:.4f})")
    return mse


def plot_gas_profile_epoch(
    codec: GasProfileCodec,
    gas_profs: torch.Tensor,
    epoch: int,
    out_dir: str | Path,
    device: str = "cpu",
    n: int = 6,
) -> None:
    """Save gas density profile orig/recon for a given training epoch."""
    _plot_single_profile(codec, gas_profs, SimGasProfile,
                         label="Gas profile",
                         fname=f"gas_profile_epoch_{epoch:03d}.png",
                         epoch=epoch, out_dir=out_dir, device=device, n=n)


def plot_dm_profile_epoch(
    codec: DMProfileCodec,
    dm_profs: torch.Tensor,
    epoch: int,
    out_dir: str | Path,
    device: str = "cpu",
    n: int = 6,
) -> None:
    """Save DM density profile orig/recon for a given training epoch."""
    _plot_single_profile(codec, dm_profs, SimDMProfile,
                         label="DM profile",
                         fname=f"dm_profile_epoch_{epoch:03d}.png",
                         epoch=epoch, out_dir=out_dir, device=device, n=n)


@torch.no_grad()
def plot_profile_epoch(
    gas_codec: GasProfileCodec,
    dm_codec: DMProfileCodec,
    gas_profs: torch.Tensor,
    dm_profs: torch.Tensor,
    epoch: int,
    out_dir: str | Path,
    device: str = "cpu",
    n: int = 6,
) -> None:
    """Save a combined gas+DM profile orig/recon panel (post-training overview)."""
    for codec in (gas_codec, dm_codec):
        codec.eval()

    gas_profs = gas_profs[:n].to(device)
    dm_profs  = dm_profs[:n].to(device)

    def _recon(codec, profs, ModClass):
        z = codec._encode(ModClass(value=profs))
        z_q, _, _ = codec.quantizer(z)
        return codec._decode(z_q).value.cpu()

    gas_recon = _recon(gas_codec, gas_profs, SimGasProfile)
    dm_recon  = _recon(dm_codec,  dm_profs,  SimDMProfile)
    gas_profs = gas_profs.cpu()
    dm_profs  = dm_profs.cpu()

    for codec in (gas_codec, dm_codec):
        codec.train()

    fig, axes = plt.subplots(n, 2, figsize=(6.0, n * 2.0), squeeze=False)

    for ci, (title, orig, recon) in enumerate([
        ("Gas density", gas_profs, gas_recon),
        ("DM density",  dm_profs,  dm_recon),
    ]):
        all_density = np.concatenate([orig[:n, 1].numpy().ravel(), recon[:n, 1].numpy().ravel()])
        above_floor = all_density[all_density > PROFILE_FLOOR]
        ymax = (float(np.percentile(above_floor, 99)) + 0.5) if len(above_floor) else 1.5
        ymin = PROFILE_FLOOR - 0.5

        for gi in range(n):
            ax = axes[gi][ci]
            r  = orig[gi, 0].numpy()
            ax.plot(r, orig[gi, 1].numpy(),  color="steelblue", lw=1.5, label="orig")
            ax.plot(r, recon[gi, 1].numpy(), color="tomato",    lw=1.5, ls="--", label="recon")
            ax.set_ylim(ymin, ymax)
            ax.set_xlabel("r / r200", fontsize=7)
            ax.set_ylabel("log ρ", fontsize=7)
            ax.tick_params(labelsize=6)
            if gi == 0:
                ax.set_title(title, fontsize=9)
                ax.legend(fontsize=6)

    gas_mse = ((gas_profs[:, 1:2] - gas_recon[:, 1:2]) ** 2).mean().item()
    dm_mse  = ((dm_profs[:, 1:2]  - dm_recon[:, 1:2])  ** 2).mean().item()
    fig.suptitle(f"Profiles — epoch {epoch:03d}  gas={gas_mse:.4f}  DM={dm_mse:.4f}",
                 fontsize=10)
    fig.tight_layout()
    out = _ensure_dir(out_dir) / f"profile_epoch_{epoch:03d}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [viz] profiles saved → {out}  (gas={gas_mse:.4f}, DM={dm_mse:.4f})")


# ---------------------------------------------------------------------------
# Checkpoint loaders  (used by CLI below)
# ---------------------------------------------------------------------------

def _load(codec, ckpt_dir: str, name: str, device: str):
    weights = Path(ckpt_dir) / "codecs" / name / "pytorch_model.bin"
    codec.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    return codec.to(device).eval()


def load_star_codec(ckpt_dir, device):
    return _load(SimGalaxyImageCodec(), ckpt_dir, "sim_galaxy_image", device)

def load_gas_codec(ckpt_dir, device):
    return _load(SimGasFaceonCodec(), ckpt_dir, "sim_gas_faceon", device)

def load_sfh_codec(ckpt_dir, time_grid, device):
    codec = _load(SFHCodec(), ckpt_dir, "sim_sfh", device)
    codec.set_time_grid(time_grid.to(device))
    return codec

def load_gas_profile_codec(ckpt_dir, radius_grid, device):
    codec = _load(GasProfileCodec(), ckpt_dir, "sim_gas_profile", device)
    codec.set_radius_grid(radius_grid.to(device))
    return codec

def load_dm_profile_codec(ckpt_dir, radius_grid, device):
    codec = _load(DMProfileCodec(), ckpt_dir, "sim_dm_profile", device)
    codec.set_radius_grid(radius_grid.to(device))
    return codec


# ---------------------------------------------------------------------------
# CLI — post-training reconstruction check for all codec types
# ---------------------------------------------------------------------------

def _load_samples(npz_path: str, n: int):
    """Load n samples from a single .npz shard."""
    from sim.dataset import PROFILE_FLOOR
    with np.load(npz_path) as d:
        star = d["star_faceon"][:n].copy()
        gas  = d["gas_faceon"][:n].copy()
        gp   = d["gas_profile"][:n].copy()
        dp   = d["dm_profile"][:n].copy()
        sfh  = d["sfh"][:n].copy()

    # Same preprocessing as GalaxyDataset
    sfh[:, 1, :][sfh[:, 1, :] < -3] = -3
    gp[:, 1, :] = np.clip(gp[:, 1, :], PROFILE_FLOOR, None)
    dp[:, 1, :] = np.clip(dp[:, 1, :], PROFILE_FLOOR, None)

    # Normalise images using the per-file stats (as done during training)
    star_norm = mag_normalize(star)
    gas_norm  = gas_normalize(gas)

    return {
        "star": torch.tensor(star_norm, dtype=torch.float32),
        "gas":  torch.tensor(gas_norm,  dtype=torch.float32),
        "gp":   torch.tensor(gp,        dtype=torch.float32),
        "dp":   torch.tensor(dp,        dtype=torch.float32),
        "sfh":  torch.tensor(sfh,       dtype=torch.float32),
    }


def main():
    parser = argparse.ArgumentParser(description="Post-training codec reconstruction check")
    parser.add_argument("--data",   required=True, help="Path to a single .npz shard")
    parser.add_argument("--ckpt",   required=True, help="Codec checkpoint directory")
    parser.add_argument("--out_dir", default="recon_check", help="Output directory for PNGs")
    parser.add_argument("--n",      type=int, default=6, help="Number of galaxies to visualise")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Loading {args.n} samples from {args.data}")
    samples = _load_samples(args.data, args.n)

    print(f"Loading codecs from {args.ckpt}")
    star_codec    = load_star_codec(args.ckpt, args.device)
    gas_codec     = load_gas_codec(args.ckpt, args.device)
    sfh_codec     = load_sfh_codec(args.ckpt, samples["sfh"][0, 0, :], args.device)
    gasprof_codec = load_gas_profile_codec(args.ckpt, samples["gp"][0, 0, :], args.device)
    dmprof_codec  = load_dm_profile_codec(args.ckpt,  samples["dp"][0, 0, :], args.device)

    epoch = 999   # sentinel epoch for standalone run
    out   = args.out_dir

    print("Plotting reconstructions ...")
    plot_star_epoch(star_codec,    samples["star"], epoch, out, args.device, n=args.n)
    plot_gas_epoch(gas_codec,     samples["gas"],  epoch, out, args.device, n=args.n)
    plot_sfh_epoch(sfh_codec,     samples["sfh"],  epoch, out, args.device, n=args.n)
    plot_profile_epoch(gasprof_codec, dmprof_codec,
                       samples["gp"], samples["dp"],
                       epoch, out, args.device, n=args.n)

    print(f"All figures saved to {out}/")


if __name__ == "__main__":
    main()
