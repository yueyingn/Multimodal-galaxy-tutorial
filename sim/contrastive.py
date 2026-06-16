"""Contrastive (CLIP-style) alignment of z=0 galaxy images and star-formation
histories — the model + training code behind Tutorial 3.

The idea (CLIP / SigLIP): train two encoders — one for the 8-band face-on
*galaxy image*, one for the 1-D *star-formation history* (SFH) — that map each
modality into a shared unit-sphere embedding. A contrastive loss pulls the
image and SFH of the *same* galaxy together and pushes mismatched pairs apart.
Because the same galaxy's morphology/colour and its assembly history are
physically correlated, the aligned space supports cross-modal retrieval
(image -> SFH and SFH -> image) and is a useful frozen feature for downstream
regression.

This module is deliberately self-contained (it does NOT use the FourM token
pipeline): the two encoders, the two losses, retrieval metrics, the z=0 data
loader, and a training loop. `sim/train_contrastive.py` is the CLI that drives
it; the notebook imports the same functions so the two passes stay identical.

Two configs are provided via `make_model` / `CONFIGS`:

  baseline  — faithful port of ../gal-SFH/align-representation.ipynb
              (ConvMixer patch-20, raw SFH, no augmentation, random split,
               checkpoint = last epoch).
  improved  — the same backbone with the training fixes Tutorial 3 motivates:
              fixed IQR image norm + standardized SFH, rotation/flip image
              augmentation (face-on galaxies are orientation-invariant, but the
              SFH is not — a valid label-preserving augmentation), a finer
              image patch, cosine LR with warmup, and model selection by
              validation Recall@1 instead of the last epoch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- #
# Image encoder — ConvMixer (same family as the reference notebook)
# ----------------------------------------------------------------------------- #
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class ConvMixer(nn.Module):
    """Simple, fast conv backbone for the (8,128,128) face-on image.

    A patch-embed stem (Conv stride=patch_size) followed by `depth` residual
    depthwise+pointwise blocks, then global-average-pool to a vector. With a
    large patch_size this is extremely cheap; a smaller patch_size keeps more
    spatial detail (the `improved` config uses patch_size=16 instead of 20).
    """

    def __init__(self, dim, depth, channels=8, kernel_size=3, patch_size=20, n_out=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size, bias=False),
            nn.GELU(),
            nn.BatchNorm2d(dim),
        )
        for _ in range(depth):
            self.net.append(nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(dim),
                )),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(dim),
            ))
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(dim, 1024),
            nn.GELU(),
            nn.Linear(1024, n_out),
        )

    def forward(self, x):
        return self.projection(self.net(x))


# ----------------------------------------------------------------------------- #
# SFH encoder — transformer over the 24 time bins with attention pooling
# ----------------------------------------------------------------------------- #
class SFHTimeEncoding(nn.Module):
    """Learnable per-bin time embeddings (the SFH is fixed-length, 24 bins)."""

    def __init__(self, d_emb, T):
        super().__init__()
        self.time_emb = nn.Parameter(torch.randn(1, T, d_emb) * 0.02)

    def forward(self, x):
        return self.time_emb.expand(x.size(0), -1, -1)


class SFHTransformerAttnEncoder(nn.Module):
    """Embed each (value, time-bin), run a TransformerEncoder, attention-pool to
    a single vector, then a small MLP head. Same architecture as the reference."""

    def __init__(self, emb=128, T=24, nhead=2, num_layers=4,
                 ff_dim=256, dropout=0.1, n_out=128):
        super().__init__()
        self.emb = emb
        self.T = T
        self.embedding_mag = nn.Linear(1, emb)
        self.time_emb = SFHTimeEncoding(emb, T)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb, nhead=nhead, dim_feedforward=ff_dim, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(emb))
        self.query = nn.Parameter(torch.randn(emb))
        self.attn_pool = nn.MultiheadAttention(
            embed_dim=emb, num_heads=nhead, dropout=0.0, batch_first=True)
        self.projection = nn.Sequential(
            nn.Linear(emb, emb * 4), nn.GELU(), nn.Linear(emb * 4, n_out))

    def forward(self, x):
        B, T = x.shape
        assert T == self.T, f"Input length {T} != expected {self.T}"
        h = self.embedding_mag(x[..., None]) + self.time_emb(x)
        h = self.transformer(h)
        q = self.query.unsqueeze(0).expand(B, 1, self.emb)
        pooled, _ = self.attn_pool(q, h, h)
        return self.projection(pooled.squeeze(1))


# ----------------------------------------------------------------------------- #
# CLIP wrapper
# ----------------------------------------------------------------------------- #
class GalaxySFHCLIP(nn.Module):
    """Two encoders + linear projection heads to a shared `enc_dim` sphere."""

    def __init__(self, enc_dim=128, temperature=10.0,
                 transformer_kwargs=None, conv_kwargs=None):
        super().__init__()
        transformer_kwargs = transformer_kwargs or {"emb": 256, "T": 24, "nhead": 2, "n_out": 128}
        conv_kwargs = conv_kwargs or {"dim": 32, "depth": 16, "channels": 8,
                                      "kernel_size": 3, "patch_size": 20, "n_out": 128}
        self.enc_dim = enc_dim
        # Learnable temperature (log-space) and bias, as in SigLIP.
        self.logit_scale = nn.Parameter(torch.tensor(math.log(temperature)))
        self.logit_bias = nn.Parameter(torch.tensor(-10.0))
        self.sfh_encoder = SFHTransformerAttnEncoder(**transformer_kwargs)
        self.image_encoder = ConvMixer(**conv_kwargs)
        self.sfh_projection = nn.Linear(transformer_kwargs["n_out"], enc_dim)
        self.image_projection = nn.Linear(conv_kwargs["n_out"], enc_dim)

    def image_embeddings_with_projection(self, x_img):
        x = self.image_projection(self.image_encoder(x_img))
        return x / x.norm(dim=-1, keepdim=True)

    def sfh_embeddings_with_projection(self, x_sfh):
        x = self.sfh_projection(self.sfh_encoder(x_sfh))
        return x / x.norm(dim=-1, keepdim=True)

    def forward(self, x_img, x_sfh):
        # Keep temperature in a sane range (exp(4.6052) ~ 100).
        self.logit_scale.data.clamp_(0, 4.6052)
        return (self.image_embeddings_with_projection(x_img),
                self.sfh_embeddings_with_projection(x_sfh))


# ----------------------------------------------------------------------------- #
# Losses
# ----------------------------------------------------------------------------- #
def clip_loss(image_embeddings, sfh_embeddings, temperature=1.0):
    """Softmax (InfoNCE) CLIP loss with the symmetric soft-target trick."""
    log_softmax = nn.LogSoftmax(dim=1)
    logits = (sfh_embeddings @ image_embeddings.T) / temperature
    images_similarity = image_embeddings @ image_embeddings.T
    sfh_similarity = sfh_embeddings @ sfh_embeddings.T
    targets = F.softmax((images_similarity + sfh_similarity) / (2 * temperature), dim=-1)
    images_loss = (-targets.T * log_softmax(logits.T)).sum(1)
    sfh_loss = (-targets * log_softmax(logits)).sum(1)
    return (images_loss + sfh_loss) / 2.0


def sigmoid_loss(image_embeds, sfh_embeds, logit_scale=1.0, logit_bias=-10.0):
    """SigLIP sigmoid loss (https://arxiv.org/abs/2303.15343). Pairwise; the
    diagonal are positives, off-diagonal negatives, so it needs no large batch
    of in-batch negatives to be well-behaved."""
    bs = sfh_embeds.shape[0]
    labels = 2 * torch.eye(bs, device=sfh_embeds.device) - torch.ones((bs, bs), device=sfh_embeds.device)
    logits = (sfh_embeds @ image_embeds.t()) * logit_scale + logit_bias
    return -torch.mean(F.logsigmoid(labels * logits.float()))


def compute_loss(image_emb, sfh_emb, model, loss_type):
    if loss_type == "sigmoid":
        return sigmoid_loss(image_emb, sfh_emb,
                            model.logit_scale.exp(), model.logit_bias).mean()
    elif loss_type == "softmax":
        return clip_loss(image_emb, sfh_emb, model.logit_scale.exp()).mean()
    raise ValueError("loss_type must be 'sigmoid' or 'softmax'")


# ----------------------------------------------------------------------------- #
# Retrieval metrics
# ----------------------------------------------------------------------------- #
@torch.no_grad()
def encode_all(model, imgs, sfhs, device, batch_size=256):
    """Return L2-normalized (image_emb, sfh_emb) for the whole array."""
    model.eval()
    img_embs, sfh_embs = [], []
    for i in range(0, len(imgs), batch_size):
        xi = imgs[i:i + batch_size].to(device)
        xs = sfhs[i:i + batch_size].to(device)
        img_embs.append(model.image_embeddings_with_projection(xi).cpu())
        sfh_embs.append(model.sfh_embeddings_with_projection(xs).cpu())
    return torch.cat(img_embs), torch.cat(sfh_embs)


def _retrieval_one_direction(query_emb, key_emb, ks=(1, 5, 10)):
    """For each query row, rank all keys by cosine sim; the correct key is the
    diagonal index. Returns Recall@k and median rank (1-based)."""
    sims = query_emb @ key_emb.T                      # (N, N), already normalized
    N = sims.shape[0]
    # rank of the true (diagonal) match = #keys scoring strictly higher + 1
    true_score = sims.diag().unsqueeze(1)
    ranks = (sims > true_score).sum(dim=1) + 1        # 1-based
    out = {f"recall@{k}": float((ranks <= k).float().mean()) for k in ks}
    out["median_rank"] = float(ranks.float().median())
    out["mean_rank"] = float(ranks.float().mean())
    out["N"] = N
    return out


def retrieval_metrics(img_emb, sfh_emb, ks=(1, 5, 10)):
    """Both retrieval directions. 'i2s' = use image to retrieve SFH."""
    return {
        "i2s": _retrieval_one_direction(img_emb, sfh_emb, ks),
        "s2i": _retrieval_one_direction(sfh_emb, img_emb, ks),
    }


# ----------------------------------------------------------------------------- #
# Data
# ----------------------------------------------------------------------------- #
@dataclass
class CLIPData:
    img_train: torch.Tensor
    sfh_train: torch.Tensor
    img_val: torch.Tensor
    sfh_val: torch.Tensor
    sfh_mean: float
    sfh_std: float
    img_norm: dict
    train_idx: np.ndarray
    val_idx: np.ndarray


def load_clip_data(npz_path, img_norm, sfh_floor=-3.0, standardize_sfh=True,
                   val_frac=0.1, seed=0) -> CLIPData:
    """Load the z=0 .npz, normalize the 8-band image with the FIXED IQR stats
    (shared with the rest of the tutorial), regularize + optionally standardize
    the SFH, and make a reproducible train/val split.

    SFH input is channel 1 of the (N,2,24) array (log10 SFR per look-back-time
    bin); channel 0 is the time grid and is identical for every galaxy.
    """
    d = np.load(npz_path)
    imgs = d["star_faceon"].astype(np.float32)                 # (N,8,128,128)
    sfh = d["sfh"][:, 1, :].astype(np.float32).copy()          # (N,24) log SFR
    sfh[sfh < sfh_floor] = sfh_floor                           # regularize quenched tail

    median, iqr = float(img_norm["median"]), float(img_norm["iqr"])
    imgs = (imgs - median) / iqr

    N = len(imgs)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    n_val = int(val_frac * N)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    # Standardize the SFH using TRAIN statistics only (no val leakage).
    if standardize_sfh:
        sfh_mean = float(sfh[train_idx].mean())
        sfh_std = float(sfh[train_idx].std())
    else:
        sfh_mean, sfh_std = 0.0, 1.0
    sfh_n = (sfh - sfh_mean) / sfh_std

    t = lambda a: torch.from_numpy(a).float()
    return CLIPData(
        img_train=t(imgs[train_idx]), sfh_train=t(sfh_n[train_idx]),
        img_val=t(imgs[val_idx]), sfh_val=t(sfh_n[val_idx]),
        sfh_mean=sfh_mean, sfh_std=sfh_std, img_norm={"median": median, "iqr": iqr},
        train_idx=train_idx, val_idx=val_idx,
    )


def augment_images(x):
    """Label-preserving augmentation for FACE-ON images: a random multiple-of-90
    rotation + random flips. A face-on galaxy has no preferred in-plane
    orientation, so this does not change its SFH — it only tells the image
    encoder to be orientation-invariant, which sharply reduces overfitting on a
    few-thousand-galaxy set. Applied to the (B,8,128,128) batch on-device."""
    k = int(torch.randint(0, 4, (1,)).item())
    if k:
        x = torch.rot90(x, k, dims=(2, 3))
    if torch.rand(1).item() < 0.5:
        x = torch.flip(x, dims=(2,))
    if torch.rand(1).item() < 0.5:
        x = torch.flip(x, dims=(3,))
    return x


# ----------------------------------------------------------------------------- #
# Model configs
# ----------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    name: str
    # data
    standardize_sfh: bool = True
    augment: bool = True
    # model
    enc_dim: int = 128
    temperature: float = 10.0
    conv_kwargs: dict = field(default_factory=lambda: {
        "dim": 32, "depth": 16, "channels": 8, "kernel_size": 3,
        "patch_size": 20, "n_out": 128})
    transformer_kwargs: dict = field(default_factory=lambda: {
        "emb": 256, "T": 24, "nhead": 2, "n_out": 128})
    # optim
    loss_type: str = "sigmoid"
    epochs: int = 100
    lr: float = 5e-4
    weight_decay: float = 0.01
    batch_size: int = 32
    warmup_epochs: int = 0
    select_by: str = "loss"          # "loss" (last/best val loss) or "recall@1"


CONFIGS = {
    # Faithful port of ../gal-SFH/align-representation.ipynb.
    "baseline": TrainConfig(
        name="baseline",
        standardize_sfh=False, augment=False,
        conv_kwargs={"dim": 32, "depth": 16, "channels": 8, "kernel_size": 3,
                     "patch_size": 20, "n_out": 128},
        loss_type="sigmoid", epochs=100, lr=5e-4, weight_decay=0.01,
        batch_size=32, warmup_epochs=0, select_by="loss",
    ),
    # Same backbone, training fixes motivated in the notebook.
    "improved": TrainConfig(
        name="improved",
        standardize_sfh=True, augment=True,
        conv_kwargs={"dim": 64, "depth": 16, "channels": 8, "kernel_size": 5,
                     "patch_size": 16, "n_out": 128},
        transformer_kwargs={"emb": 256, "T": 24, "nhead": 4, "n_out": 128},
        loss_type="sigmoid", epochs=200, lr=5e-4, weight_decay=0.05,
        batch_size=64, warmup_epochs=10, select_by="recall@1",
    ),
}


def make_model(cfg: TrainConfig) -> GalaxySFHCLIP:
    return GalaxySFHCLIP(enc_dim=cfg.enc_dim, temperature=cfg.temperature,
                         transformer_kwargs=dict(cfg.transformer_kwargs),
                         conv_kwargs=dict(cfg.conv_kwargs))


# ----------------------------------------------------------------------------- #
# Training
# ----------------------------------------------------------------------------- #
def _lr_lambda(epoch, warmup, total):
    if warmup and epoch < warmup:
        return (epoch + 1) / warmup
    # cosine from 1 -> ~0 over the post-warmup epochs
    p = (epoch - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


def train_clip(model, data: CLIPData, cfg: TrainConfig, device, verbose=True,
               log_every=10):
    """Train one CLIP model. Returns (history, best_state_dict, best_metrics).

    Model selection: `select_by="loss"` keeps the lowest-val-loss state;
    `select_by="recall@1"` keeps the state with the highest mean cross-modal
    Recall@1 on the val set (the metric we actually care about)."""
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda e: _lr_lambda(e, cfg.warmup_epochs, cfg.epochs))

    n_train = len(data.img_train)
    history = {"train_loss": [], "val_loss": [], "val_recall@1": [],
               "val_recall@5": [], "logit_scale": []}
    best_state = None
    best_score = -math.inf      # higher-is-better internal score
    best_metrics = None

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        perm = torch.randperm(n_train)
        tr_loss, nb = 0.0, 0
        for i in range(0, n_train, cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            xi = data.img_train[idx].to(device)
            xs = data.sfh_train[idx].to(device)
            if cfg.augment:
                xi = augment_images(xi)
            img_emb, sfh_emb = model(xi, xs)
            loss = compute_loss(img_emb, sfh_emb, model, cfg.loss_type)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item(); nb += 1
        tr_loss /= nb
        sched.step()

        # ---- validation ----
        model.eval()
        with torch.no_grad():
            vi = data.img_val.to(device)
            vs = data.sfh_val.to(device)
            ie, se = model(vi, vs)
            val_loss = compute_loss(ie, se, model, cfg.loss_type).item()
        img_emb, sfh_emb = encode_all(model, data.img_val, data.sfh_val, device)
        rm = retrieval_metrics(img_emb, sfh_emb)
        mean_r1 = 0.5 * (rm["i2s"]["recall@1"] + rm["s2i"]["recall@1"])
        mean_r5 = 0.5 * (rm["i2s"]["recall@5"] + rm["s2i"]["recall@5"])

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_recall@1"].append(mean_r1)
        history["val_recall@5"].append(mean_r5)
        history["logit_scale"].append(float(model.logit_scale.exp().item()))

        score = mean_r1 if cfg.select_by == "recall@1" else -val_loss
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = {"epoch": epoch, "val_loss": val_loss,
                            "val_recall@1": mean_r1, "val_recall@5": mean_r5,
                            "retrieval": rm}

        if verbose and (epoch % log_every == 0 or epoch == 1 or epoch == cfg.epochs):
            print(f"[{cfg.name}] ep {epoch:03d} train={tr_loss:.4f} "
                  f"val={val_loss:.4f} R@1={mean_r1:.3f} R@5={mean_r5:.3f} "
                  f"lr={opt.param_groups[0]['lr']:.2e}")

    if best_state is None:                # epochs==0 guard
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return history, best_state, best_metrics
