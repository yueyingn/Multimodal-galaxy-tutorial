"""CLI for Tutorial 3 — train the galaxy-image <-> SFH CLIP model(s) on the
z=0 dataset and save checkpoints + retrieval metrics.

Each run trains one config (`baseline` or `improved`, defined in
sim.contrastive.CONFIGS) and writes, under <save_dir>/<config>/:

    best.pt      — {model_state, config, sfh_mean, sfh_std, img_norm,
                    val_idx, train_idx, best_metrics, history}
    metrics.json — human-readable summary (final + best retrieval metrics)

The notebook (tutorial-3) loads best.pt for the load pass. Train both with:

    python -m sim.train_contrastive --config baseline --save_dir checkpoints/clip
    python -m sim.train_contrastive --config improved --save_dir checkpoints/clip
"""

import argparse
import dataclasses
import json
from pathlib import Path

import torch

from sim.contrastive import (
    CONFIGS, load_clip_data, make_model, train_clip,
    encode_all, retrieval_metrics,
)


def load_img_norm(data_dir: str) -> dict:
    p = Path(data_dir) / "img_norm.json"
    if p.exists():
        with open(p) as f:
            j = json.load(f)
        return {"median": j["median"], "iqr": j["iqr"]}
    # Fallback to the published tutorial constants.
    print(f"[warn] {p} not found; using built-in img_norm constants")
    return {"median": 7.5859375, "iqr": 4.05078125}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=list(CONFIGS), default="improved")
    ap.add_argument("--npz", default="data/Snap99.npz",
                    help="z=0 dataset (star_faceon + sfh)")
    ap.add_argument("--img_norm_dir", default="data",
                    help="directory holding img_norm.json (fixed IQR stats)")
    ap.add_argument("--save_dir", default="checkpoints/clip")
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None, help="override config epochs")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    cfg = CONFIGS[args.config]
    if args.epochs is not None:
        cfg = dataclasses.replace(cfg, epochs=args.epochs)

    img_norm = load_img_norm(args.img_norm_dir)
    data = load_clip_data(args.npz, img_norm,
                          standardize_sfh=cfg.standardize_sfh,
                          val_frac=args.val_frac, seed=args.seed)

    print("=" * 70)
    print(f"Config: {cfg.name}")
    print(f"  device={args.device}  epochs={cfg.epochs}  loss={cfg.loss_type}")
    print(f"  augment={cfg.augment}  standardize_sfh={cfg.standardize_sfh}  "
          f"select_by={cfg.select_by}")
    print(f"  conv={cfg.conv_kwargs}")
    print(f"  train={len(data.img_train)}  val={len(data.img_val)}")
    print("=" * 70)

    model = make_model(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model params: {n_params/1e6:.2f}M")

    history, best_state, best_metrics = train_clip(
        model, data, cfg, args.device, verbose=True, log_every=10)

    # Final-epoch metrics (in addition to the best-selected ones).
    model.load_state_dict(best_state)
    img_emb, sfh_emb = encode_all(model, data.img_val, data.sfh_val, args.device)
    final_rm = retrieval_metrics(img_emb, sfh_emb)

    out_dir = Path(args.save_dir) / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model_state": best_state,
        "config": dataclasses.asdict(cfg),
        "sfh_mean": data.sfh_mean, "sfh_std": data.sfh_std,
        "img_norm": data.img_norm,
        "train_idx": data.train_idx, "val_idx": data.val_idx,
        "best_metrics": best_metrics, "history": history,
        "n_params": n_params,
    }
    torch.save(ckpt, out_dir / "best.pt")

    summary = {
        "config": cfg.name,
        "n_params": n_params,
        "best_epoch": best_metrics["epoch"],
        "best_val_loss": best_metrics["val_loss"],
        "best_retrieval": best_metrics["retrieval"],
        "mean_recall@1": 0.5 * (final_rm["i2s"]["recall@1"] + final_rm["s2i"]["recall@1"]),
        "mean_recall@5": 0.5 * (final_rm["i2s"]["recall@5"] + final_rm["s2i"]["recall@5"]),
        "retrieval_at_best": final_rm,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print(f"[{cfg.name}] best epoch {best_metrics['epoch']}  "
          f"mean R@1={summary['mean_recall@1']:.3f}  "
          f"mean R@5={summary['mean_recall@5']:.3f}")
    print("  image->SFH:", {k: (round(v, 3) if isinstance(v, float) else v)
                            for k, v in final_rm["i2s"].items()})
    print("  SFH->image:", {k: (round(v, 3) if isinstance(v, float) else v)
                            for k, v in final_rm["s2i"].items()})
    print(f"  saved -> {out_dir}/best.pt  +  metrics.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
