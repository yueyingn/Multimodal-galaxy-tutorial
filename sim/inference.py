"""Minimal any-to-any inference helpers for the trained FourM transformer.

This is a trimmed, self-contained version of the prediction path in the full
project's `sim/downstream.py`. Given a batch of pre-tokenized modalities, we put
the *encoder* (conditioning) modalities in fully-visible slots and the *decoder*
(target) modalities in masked slots, run one forward pass, and read off the
predicted target tokens (argmax) — or, for scalars, the softmax-weighted expected
value over the codebook (which respects the ordinal bin structure).
"""

import torch

__all__ = [
    "build_mod_dict", "run_inference", "run_inference_scalar_ev",
    "run_inference_samples",
]


def build_mod_dict(tokens_batch: dict, encoder_keys: list, decoder_keys: list,
                   device: str) -> dict:
    """Assemble the FourM mod_dict for inference.

    Encoder modalities: their true tokens are visible (input_mask=0).
    Decoder modalities: tokens hidden and predicted (input_mask=1, target_mask=0).
    """
    mod_dict = {}
    for key, tokens in tokens_batch.items():
        B, N = tokens.shape
        t = tokens.long().to(device)
        if key in encoder_keys:
            mod_dict[key] = {
                "tensor":                 t,
                "input_mask":             torch.zeros(B, N, dtype=torch.bool, device=device),
                "target_mask":            torch.ones( B, N, dtype=torch.bool, device=device),
                "decoder_attention_mask": torch.zeros(B, N, dtype=torch.bool, device=device),
            }
        elif key in decoder_keys:
            mod_dict[key] = {
                "tensor":                 torch.zeros(B, N, dtype=torch.long, device=device),
                "input_mask":             torch.ones( B, N, dtype=torch.bool, device=device),
                "target_mask":            torch.zeros(B, N, dtype=torch.bool, device=device),
                "decoder_attention_mask": torch.zeros(B, N, dtype=torch.bool, device=device),
            }
    return mod_dict


@torch.no_grad()
def _forward_logits(model, batch, encoder_keys, decoder_keys, device):
    """Shared forward path: returns the per-modality decoder logits dict."""
    mod_dict = build_mod_dict(batch, encoder_keys, decoder_keys, device)
    n_enc = sum(mod_dict[k]["tensor"].shape[1] for k in encoder_keys)
    n_dec = sum(mod_dict[k]["tensor"].shape[1] for k in decoder_keys)

    enc_emb_dict = {m: model.encoder_embeddings[m](d)
                    for m, d in mod_dict.items() if m in model.encoder_embeddings}
    enc_tokens, enc_emb, enc_mask, _ = model.forward_mask_encoder(enc_emb_dict, n_enc)

    dec_emb_dict = {m: model.decoder_embeddings[m].forward_embed(d)
                    for m, d in mod_dict.items() if m in model.decoder_embeddings}
    dec_tokens, dec_emb, _, _, dec_attn_mask, dec_mod_mask = \
        model.forward_mask_decoder(dec_emb_dict, n_dec)

    x = model.forward_encoder(enc_tokens + enc_emb, encoder_mask=enc_mask)
    context = model.decoder_proj_context(x) + enc_emb
    y = model.forward_decoder(dec_tokens + dec_emb, context,
                              encoder_mask=enc_mask, decoder_attention_mask=dec_attn_mask)

    dec_only = {k: v for k, v in dec_emb_dict.items() if k in decoder_keys}
    return model.forward_logits(y, dec_only, dec_mod_mask, return_all_logits=False)


@torch.no_grad()
def run_inference(model, all_tokens: dict, encoder_keys: list, decoder_keys: list,
                  batch_size: int = 64, device: str = "cpu",
                  decoder_vocab_caps: dict | None = None) -> dict:
    """Argmax prediction of every decoder modality's tokens, given the encoder
    modalities. Returns {modality_key: LongTensor[N, n_tokens]}.

    decoder_vocab_caps: optional {key: cap} to clip argmax to the first `cap`
    logits — needed when the model's output vocab is wider than the codec's
    codebook (e.g. scalar model-vocab 4096 vs. codec 1024).
    """
    N = next(iter(all_tokens.values())).shape[0]
    preds = {k: [] for k in decoder_keys}
    caps = decoder_vocab_caps or {}
    for start in range(0, N, batch_size):
        batch = {k: v[start:start + batch_size] for k, v in all_tokens.items()
                 if k in encoder_keys or k in decoder_keys}
        B = next(iter(batch.values())).shape[0]
        logits = _forward_logits(model, batch, encoder_keys, decoder_keys, device)
        for mod in decoder_keys:
            N_mod = batch[mod].shape[1]
            if mod in logits and logits[mod].numel() > 0:
                l3 = logits[mod].reshape(B, N_mod, -1)
                cap = caps.get(mod)
                if cap is not None and cap < l3.shape[-1]:
                    l3 = l3[..., :cap]
                pred = l3.argmax(dim=-1)
            else:
                pred = torch.zeros(B, N_mod, dtype=torch.long)
            preds[mod].append(pred.cpu())
    return {k: torch.cat(v, dim=0) for k, v in preds.items()}


@torch.no_grad()
def run_inference_scalar_ev(model, all_tokens: dict, codec, target_key: str,
                            encoder_keys: list, batch_size: int = 64,
                            device: str = "cpu", return_std: bool = False):
    """Predict a single-token scalar as the softmax-weighted expected value over
    the codec's codebook (smoother than argmax; respects ordinal bins).

    Returns a 1-D numpy array of length N (the posterior mean). If
    ``return_std=True``, returns ``(mean, std)`` where ``std`` is the posterior
    standard deviation in physical units — sqrt(E[x²] − E[x]²) under the
    predicted softmax over codebook values. That spread is a principled 1σ
    uncertainty: it widens for galaxies whose scalar the image poorly
    constrains and shrinks where the model is confident.
    """
    V = codec.quantizer.codebook_size
    codebook_vals = codec.quantizer.decode(torch.arange(V, device=device)).to(device).float()
    N = next(iter(all_tokens.values())).shape[0]
    means, stds = [], []
    for start in range(0, N, batch_size):
        batch = {k: v[start:start + batch_size] for k, v in all_tokens.items()
                 if k in encoder_keys or k == target_key}
        B = next(iter(batch.values())).shape[0]
        logits = _forward_logits(model, batch, encoder_keys, [target_key], device)
        l = logits[target_key].reshape(B, 1, -1).squeeze(1)         # (B, V_model)
        probs = torch.softmax(l, dim=-1)
        if probs.shape[-1] != V:
            probs = probs[:, :V]
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        mean = (probs * codebook_vals[None, :]).sum(dim=-1)
        means.append(mean.cpu())
        if return_std:
            ex2 = (probs * codebook_vals[None, :] ** 2).sum(dim=-1)
            stds.append((ex2 - mean ** 2).clamp_min(0).sqrt().cpu())
    mean = torch.cat(means, dim=0).numpy()
    if return_std:
        return mean, torch.cat(stds, dim=0).numpy()
    return mean


@torch.no_grad()
def run_inference_samples(model, all_tokens: dict, encoder_keys: list,
                          decoder_keys: list, n_samples: int = 64,
                          temperature: float = 1.0, batch_size: int = 64,
                          device: str = "cpu", seed: int = 0,
                          decoder_vocab_caps: dict | None = None) -> dict:
    """Draw token samples from the model's per-token posterior for each decoder
    modality, given the encoder modalities.

    Instead of the single argmax token sequence, this samples each token from
    its softmax (scaled by ``temperature``). Decoding the resulting samples and
    taking per-point quantiles gives a **predictive uncertainty band** for
    sequence modalities (SFH, density profiles) — the spread reflects how much
    the conditioning genuinely pins down the target.

    Returns {modality_key: LongTensor[n_samples, N, n_tokens]}.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    N = next(iter(all_tokens.values())).shape[0]
    caps = decoder_vocab_caps or {}
    # preallocate per modality
    out = {k: None for k in decoder_keys}
    for start in range(0, N, batch_size):
        batch = {k: v[start:start + batch_size] for k, v in all_tokens.items()
                 if k in encoder_keys or k in decoder_keys}
        B = next(iter(batch.values())).shape[0]
        logits = _forward_logits(model, batch, encoder_keys, decoder_keys, device)
        for mod in decoder_keys:
            N_mod = batch[mod].shape[1]
            l3 = logits[mod].reshape(B, N_mod, -1)
            cap = caps.get(mod)
            if cap is not None and cap < l3.shape[-1]:
                l3 = l3[..., :cap]
            probs = torch.softmax(l3 / max(temperature, 1e-6), dim=-1)  # (B, N_mod, V)
            flat = probs.reshape(B * N_mod, -1)
            samp = torch.multinomial(flat, n_samples, replacement=True, generator=g)
            samp = samp.reshape(B, N_mod, n_samples).permute(2, 0, 1)   # (S, B, N_mod)
            out[mod] = samp.cpu() if out[mod] is None else torch.cat([out[mod], samp.cpu()], dim=1)
    return out
