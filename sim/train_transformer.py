"""Transformer training script — Phase 2 of simulation model training.

Trains a FourM masked multimodal transformer on pre-tokenized simulation data.
Run AFTER sim/train_codecs.py has completed all four phases.

Usage:
    python -m sim.train_transformer \
        --token_file ./data/tokens.pt \
        --save_dir   ./checkpoints/transformer \
        --model      tiny             # tiny | small | base

Training strategy:
  - Each batch has all 6 modalities tokenized.
  - prepare_mod_dict() randomly assigns ~half the modalities to the encoder
    (all tokens visible) and the rest to the decoder (all tokens predicted).
  - Loss = average cross-entropy per predicted modality (loss_type="mod").
  - A modality loss near log(vocab_size) means random guessing;
    decreasing loss means the model is learning cross-modal correlations.
"""

import argparse
import os
import random
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split

from aion.fourm.fm import (
    fm_tiny_6e_6d_swiglu_nobias,
    fm_small_8e_8d_swiglu_nobias,
    fm_base_12e_12d_swiglu_nobias,
)
from sim.dataset import SimTokenizedDataset, prepare_mod_dict
from sim.modality_info import SIM_MODALITY_INFO


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

MODEL_FACTORIES = {
    "tiny":  fm_tiny_6e_6d_swiglu_nobias,
    "small": fm_small_8e_8d_swiglu_nobias,
    "base":  fm_base_12e_12d_swiglu_nobias,
}


def build_model(
    model_size: str = "tiny",
    modality_info: dict | None = None,
    soft_ce_sigma: float = 0.0,
) -> nn.Module:
    """Build a FourM transformer for the simulation modalities.

    All embedding layers are initialized inside FourM.__init__ via
    emb.init(dim_tokens=dim), so no separate init call is needed.

    Args:
        model_size: One of "tiny" (384-d), "small" (512-d), "base" (768-d).
        modality_info: Subset of SIM_MODALITY_INFO. If None, all modalities are
            built. Pass a filtered dict so the model only allocates embeddings
            for modalities actually present in the token file.
        soft_ce_sigma: Width (in bin units) of the Gaussian-smoothed target
            distribution used for ordinal modalities (scalars). 0 disables
            smoothing — falls back to standard one-hot CE.

    Returns:
        Initialized FourM model.
    """
    if model_size not in MODEL_FACTORIES:
        raise ValueError(f"model_size must be one of {list(MODEL_FACTORIES)}")

    if modality_info is None:
        modality_info = SIM_MODALITY_INFO
    enc_emb = {key: info["encoder_embedding"]() for key, info in modality_info.items()}
    dec_emb = {key: info["decoder_embedding"]() for key, info in modality_info.items()}
    factory = MODEL_FACTORIES[model_size]
    model = factory(
        encoder_embeddings=enc_emb,
        decoder_embeddings=dec_emb,
        modality_info=modality_info,
        soft_ce_sigma=soft_ce_sigma,
    )
    return model


# ---------------------------------------------------------------------------
# Token count helpers
# ---------------------------------------------------------------------------

def count_tokens(mod_dict: dict) -> tuple[int, int]:
    """Count encoder and decoder tokens in a mod_dict.

    Returns:
        (n_encoder_tokens, n_decoder_tokens)
    """
    n_enc = 0
    n_dec = 0
    for d in mod_dict.values():
        n = d["tensor"].shape[1]
        if not d["input_mask"].all():
            n_enc += n
        else:
            n_dec += n
    # Ensure at least 1 decoder token to avoid division-by-zero
    return max(n_enc, 1), max(n_dec, 1)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_transformer(
    token_file: str,
    save_dir: str,
    val_file: str = "",
    model_size: str = "tiny",
    epochs: int = 200,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    batch_size: int = 16,
    mask_frac: float | tuple[float, float] = 0.5,
    loss_type: str = "mod",
    val_frac: float = 0.05,
    save_every: int = 20,
    device: str = "cuda",
    grad_clip: float = 1.0,
    resume_ckpt: str = "",
    soft_ce_sigma: float = 0.0,
    encoder_only_mods: list[str] | None = None,
    val_seed: int = 1234,
) -> nn.Module:
    """Train the FourM transformer on pre-tokenized simulation data.

    Args:
        token_file: Path to .pt token file produced by tokenize_dataset().
        save_dir: Directory to save transformer checkpoints.
        model_size: "tiny" | "small" | "base".
        epochs: Total training epochs.
        lr: Peak learning rate.
        weight_decay: AdamW weight decay.
        batch_size: Training batch size.
            Recommendation: 16–32 for tiny on 1 GPU.
        mask_frac: Fraction of modalities assigned to decoder per step. Either
            a scalar (fixed every step, original behavior) or a `(lo, hi)`
            tuple — sampled uniformly per call so the model trains across
            conditioning regimes instead of just 50/50 splits.
        loss_type: "mod" (equal weight per modality) or "token" (weight by count).
        val_frac: Fraction of data held out for validation.
        save_every: Save checkpoint every N epochs.
        device: Torch device string.
        grad_clip: Gradient norm clipping.
        encoder_only_mods: Modalities (e.g. ["tok_sim_galaxy_image"]) to put
            in the encoder every step and exclude from decoder targets. Helps
            balance the loss when one modality (typically galaxy_image with
            1024 tokens) would otherwise dominate, and we don't predict it
            downstream anyway. None / empty list ⇒ standard random encoder /
            decoder split, identical to the previous training behavior.

    Returns:
        Trained FourM model.
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f"=== Training FourM-{model_size} transformer ===")
    print(f"  token_file  : {token_file}")
    print(f"  save_dir    : {save_dir}")
    print(f"  epochs      : {epochs}   batch_size : {batch_size}")
    if isinstance(mask_frac, tuple):
        mf_str = f"uniform[{mask_frac[0]}, {mask_frac[1]}] per step"
    else:
        mf_str = f"{mask_frac} (fixed)"
    print(f"  lr          : {lr}       mask_frac  : {mf_str}")
    print(f"  weight_decay: {weight_decay}   val masking: deterministic (seed={val_seed})")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_full = SimTokenizedDataset(token_file)
    if val_file:
        train_ds = train_full
        val_ds   = SimTokenizedDataset(val_file)
        if set(val_ds.tokens.keys()) != set(train_full.tokens.keys()):
            raise ValueError(
                f"val_file modalities {sorted(val_ds.tokens)} do not match "
                f"train modalities {sorted(train_full.tokens)}"
            )
        print(f"  dataset: {len(train_ds)} train (from {token_file})")
        print(f"           {len(val_ds)} val   (from {val_file})")
    else:
        n_val = max(1, int(len(train_full) * val_frac))
        n_train = len(train_full) - n_val
        train_ds, val_ds = random_split(train_full, [n_train, n_val])
        print(f"  dataset: {n_train} train / {n_val} val (random split of token_file)")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)

    # ------------------------------------------------------------------
    # Model — only build embeddings for modalities present in the token file.
    # ------------------------------------------------------------------
    present = set(train_full.tokens.keys())
    mod_info = {k: v for k, v in SIM_MODALITY_INFO.items() if k in present}
    missing = present - set(mod_info)
    if missing:
        raise ValueError(f"Token file contains unknown modalities: {missing}")
    print(f"  modalities  : {sorted(mod_info)}")

    encoder_only_set = set(encoder_only_mods or ())
    unknown_eo = encoder_only_set - present
    if unknown_eo:
        raise ValueError(
            f"--encoder_only_mods refers to modalities not in the token file: "
            f"{sorted(unknown_eo)}"
        )
    if encoder_only_set:
        print(f"  enc-only    : {sorted(encoder_only_set)} "
              f"(always in encoder, never decoder targets)")
    else:
        print(f"  enc-only    : none (standard random enc/dec split)")
    if soft_ce_sigma > 0:
        ordinal_mods = sorted(
            m for m, info in mod_info.items() if info.get("is_ordinal", False)
        )
        print(f"  soft CE     : sigma={soft_ce_sigma} bins applied to "
              f"{len(ordinal_mods)} ordinal modalities: {ordinal_mods}")
    else:
        print(f"  soft CE     : disabled (one-hot CE on all modalities)")
    model = build_model(
        model_size, modality_info=mod_info, soft_ce_sigma=soft_ce_sigma
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  parameters  : {n_params:,}")

    # ------------------------------------------------------------------
    # Optimiser + scheduler
    # ------------------------------------------------------------------
    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))
    scheduler = CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)

    # ------------------------------------------------------------------
    # Resume from checkpoint (if requested)
    # ------------------------------------------------------------------
    start_epoch = 1
    best_val_loss = float("inf")

    if resume_ckpt:
        print(f"  Resuming from checkpoint: {resume_ckpt}")
        ckpt = torch.load(resume_ckpt, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        print(f"  Resuming at epoch {start_epoch} / {epochs}")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        t0 = time.time()
        train_loss = 0.0
        train_mod_loss: dict[str, float] = {}

        for tokens in train_loader:
            tokens = {k: v.to(device) for k, v in tokens.items()}
            mod_dict = prepare_mod_dict(
                tokens, mask_frac=mask_frac,
                encoder_only_modalities=encoder_only_set,
            )
            n_enc, n_dec = count_tokens(mod_dict)

            loss, mod_loss = model(mod_dict, n_enc, n_dec, loss_type=loss_type)

            opt.zero_grad()
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            train_loss += loss.item()
            for k, v in mod_loss.items():
                train_mod_loss[k] = train_mod_loss.get(k, 0.0) + v.item()

        scheduler.step()

        # ------------------------------------------------------------------
        # Validation  (deterministic)
        # ------------------------------------------------------------------
        # The encoder/decoder split inside prepare_mod_dict is drawn from the
        # global `random` stream. With a (lo, hi) mask_frac range that means a
        # *fresh* masking task every epoch, so the raw val loss is noisy and
        # `best.pt` can latch onto whichever epoch happened to draw an easy
        # (heavily conditioned) mask rather than the model that truly
        # generalizes. We pin a fixed seed for the duration of the val loop so
        # every epoch is scored on the *same* set of masking tasks, then
        # restore the RNG state so the training stream is unaffected.
        model.eval()
        val_loss = 0.0
        py_state = random.getstate()
        torch_state = torch.get_rng_state()
        random.seed(val_seed)
        torch.manual_seed(val_seed)
        with torch.no_grad():
            for tokens in val_loader:
                tokens = {k: v.to(device) for k, v in tokens.items()}
                mod_dict = prepare_mod_dict(
                    tokens, mask_frac=mask_frac,
                    encoder_only_modalities=encoder_only_set,
                )
                n_enc, n_dec = count_tokens(mod_dict)
                loss, _ = model(mod_dict, n_enc, n_dec, loss_type=loss_type)
                val_loss += loss.item()
        random.setstate(py_state)
        torch.set_rng_state(torch_state)

        train_loss /= len(train_loader)
        val_loss   /= max(len(val_loader), 1)
        elapsed     = time.time() - t0

        # Per-modality losses (for diagnostics)
        mod_loss_str = "  ".join(
            f"{k.replace('tok_sim_','')}: {v/len(train_loader):.3f}"
            for k, v in sorted(train_mod_loss.items())
        )

        print(
            f"  epoch {epoch:04d}/{epochs}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}  "
            f"t={elapsed:.1f}s"
        )
        print(f"    per-mod: {mod_loss_str}")

        # ------------------------------------------------------------------
        # Checkpointing
        # ------------------------------------------------------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {"epoch": epoch, "model": model.state_dict(),
                 "val_loss": val_loss, "modalities": sorted(mod_info),
                 "encoder_only_modalities": sorted(encoder_only_set),
                 "mask_frac": mask_frac},
                os.path.join(save_dir, "best.pt"),
            )

        if epoch % save_every == 0:
            torch.save(
                {"epoch": epoch, "model": model.state_dict(),
                 "opt": opt.state_dict(), "scheduler": scheduler.state_dict(),
                 "encoder_only_modalities": sorted(encoder_only_set),
                 "mask_frac": mask_frac},
                os.path.join(save_dir, f"checkpoint_ep{epoch:04d}.pt"),
            )

    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Checkpoints saved to {save_dir}/")
    return model


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, model_size: str = "tiny", device: str = "cpu") -> nn.Module:
    """Load a saved transformer checkpoint.

    Args:
        checkpoint_path: Path to a .pt file saved by train_transformer().
        model_size: Must match the size used during training.
        device: Target device.

    Returns:
        FourM model with loaded weights in eval mode.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    saved_mods = ckpt.get("modalities")
    if saved_mods is not None:
        mod_info = {k: SIM_MODALITY_INFO[k] for k in saved_mods}
    else:
        mod_info = None
    model = build_model(model_size, modality_info=mod_info).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded model from epoch {ckpt.get('epoch','?')} "
          f"(val_loss={ckpt.get('val_loss', '?'):.4f})")
    return model


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train simulation FourM transformer")
    parser.add_argument("--token_file", required=True,
                        help="Path to pre-tokenized train .pt file from sim.split_tokens")
    parser.add_argument("--val_file",   default="",
                        help="Optional held-out validation .pt file. If unset, "
                             "val_frac of token_file is used as validation.")
    parser.add_argument("--save_dir",   default="./checkpoints/transformer")
    parser.add_argument("--model",      default="tiny",
                        choices=["tiny", "small", "base"])
    parser.add_argument("--epochs",     type=int,   default=200)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="AdamW weight decay. Raise (e.g. 0.05) to combat "
                             "overfitting on small training sets.")
    parser.add_argument("--val_seed",   type=int,   default=1234,
                        help="Fixed RNG seed for the deterministic validation "
                             "masking, so val loss is comparable across epochs "
                             "and best.pt tracks real generalization.")
    parser.add_argument("--batch_size", type=int,   default=16)
    parser.add_argument("--mask_frac",  type=str,   default="0.5",
                        help="Fraction of modalities to predict per step. "
                             "Accepts a scalar (e.g. '0.5', fixed every step) "
                             "or a 'lo,hi' range (e.g. '0.15,0.85') sampled "
                             "uniformly per step so the model practices a "
                             "spread of conditioning regimes instead of just "
                             "50/50 splits.")
    parser.add_argument("--loss_type",  default="mod", choices=["mod", "token"])
    parser.add_argument("--save_every", type=int,   default=20)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume",     default="",
                        help="Path to a checkpoint_ep*.pt file to resume training from")
    parser.add_argument("--soft_ce_sigma", type=float, default=0.0,
                        help="Width (in bin units) of the Gaussian-smoothed CE "
                             "target for ordinal modalities (scalar codecs). "
                             "0 disables smoothing. Try 1.0 for vocab=1024 "
                             "scalar codecs to give the model partial credit "
                             "for off-by-one bin predictions.")
    parser.add_argument("--encoder_only_mods", default="",
                        help="Comma-separated modality names (e.g. "
                             "'tok_sim_galaxy_image') to keep in the encoder "
                             "every step and exclude from decoder targets. "
                             "Useful for heavy modalities you don't predict "
                             "downstream — they still inform the encoder but "
                             "stop dominating the per-modality loss budget. "
                             "Empty string ⇒ standard random enc/dec split.")
    args = parser.parse_args()
    encoder_only_mods = [
        m.strip() for m in args.encoder_only_mods.split(",") if m.strip()
    ]

    if "," in args.mask_frac:
        lo_s, hi_s = args.mask_frac.split(",", 1)
        lo, hi = float(lo_s), float(hi_s)
        if not (0.0 <= lo <= hi <= 1.0):
            raise ValueError(
                f"--mask_frac range '{args.mask_frac}' must satisfy "
                f"0 <= lo <= hi <= 1"
            )
        mask_frac: float | tuple[float, float] = (lo, hi)
    else:
        mask_frac = float(args.mask_frac)

    train_transformer(
        token_file=args.token_file,
        val_file=args.val_file,
        save_dir=args.save_dir,
        model_size=args.model,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_seed=args.val_seed,
        batch_size=args.batch_size,
        mask_frac=mask_frac,
        loss_type=args.loss_type,
        save_every=args.save_every,
        device=args.device,
        resume_ckpt=args.resume,
        soft_ce_sigma=args.soft_ce_sigma,
        encoder_only_mods=encoder_only_mods,
    )


if __name__ == "__main__":
    main()
