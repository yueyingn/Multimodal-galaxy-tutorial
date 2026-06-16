"""Build the multi-redshift tutorial dataset (TNG-100-small).

Reads the per-shard TNG-100 source
    /home/yueyingn/turbo/gal-4M-train/train-data/TNG-100/Snap{snap}_{idx:03d}.npz
and writes, into
    /home/yueyingn/turbo/gal-4M-train/train-data/TNG-100-small/
one reduced .npz per redshift:
    Snap72.npz Snap78.npz Snap84.npz Snap91.npz Snap99.npz   (TRAIN rows)
plus a single held-out, in-distribution validation set spanning all redshifts:
    val.npz                                                   (~N_VAL_PER_Z per z)
plus the train-set image normalization constants:
    img_norm.json   {median, q25, q75, iqr}

Reductions vs the raw shards (identical schema to the original Snap99_z0.npz):
  * drop the 3 extra star projections (*_xy/_yz/_xz) and all gas images
    (gas_faceon/_xy/_yz/_xz) — only the face-on star image is in scope;
  * store star_faceon as float16 (verified: fp16 round-trip rms error is
    ~74x below the image codec's own reconstruction error → negligible);
  * everything else keeps its native float32.

Train/val split: per redshift, N_VAL_PER_Z halos are drawn with a fixed RNG
(fair, uniform — NOT the mass-sorted head/tail that biased the old subset) and
routed to val.npz; the rest go to that redshift's Snap{snap}.npz. So train and
val are disjoint (no leakage) and val is in-distribution across all 5 redshifts.
"""

import glob
import json
import os
import time

import numpy as np

SRC = "/home/yueyingn/turbo/gal-4M-train/train-data/TNG-100"
DST = "/home/yueyingn/turbo/gal-4M-train/train-data/TNG-100-small"
SNAPS = [72, 78, 84, 91, 99]            # z = 0.40, 0.30, 0.20, 0.10, 0.00
N_VAL_PER_Z = 100                       # held-out val galaxies per redshift
SEED = 1234
NORM_IMGS_PER_Z = 250                   # train images sampled per z for IQR norm

DROP_SUFFIXES = ("_xy", "_yz", "_xz")
DROP_KEYS = ("gas_faceon",)
FP16_KEYS = ("star_faceon",)            # the only field we down-cast


def kept_keys(d0):
    return [k for k in d0.files
            if not k.endswith(DROP_SUFFIXES) and k not in DROP_KEYS]


def out_dtype(key, src_dtype):
    return np.float16 if key in FP16_KEYS else src_dtype


def main():
    os.makedirs(DST, exist_ok=True)
    rng = np.random.default_rng(SEED)

    val_acc = {}          # key -> list of per-snap val arrays
    norm_pixels = []      # sampled train star images (fp32) for IQR norm
    grand_train = 0
    grand_val = 0
    t0 = time.time()

    for snap in SNAPS:
        files = sorted(glob.glob(os.path.join(SRC, f"Snap{snap}_*.npz")))
        if not files:
            raise FileNotFoundError(f"No shards for Snap{snap} in {SRC}")

        # pass 1: schema + counts
        with np.load(files[0]) as d0:
            keys = kept_keys(d0)
            specs = {k: (d0[k].shape, d0[k].dtype) for k in keys}
        counts = []
        for f in files:
            with np.load(f) as d:
                counts.append(int(d[keys[0]].shape[0]))
        total = int(sum(counts))

        # fair val/train index split for this redshift
        n_val = min(N_VAL_PER_Z, total // 4)
        perm = rng.permutation(total)
        val_idx = np.sort(perm[:n_val])
        is_val = np.zeros(total, dtype=bool)
        is_val[val_idx] = True
        n_train = total - n_val

        # preallocate train + val outputs
        train_out, val_out = {}, {}
        for k in keys:
            shp, dt = specs[k]
            odt = out_dtype(k, dt)
            train_out[k] = np.empty((n_train,) + tuple(shp[1:]), dtype=odt)
            val_out[k] = np.empty((n_val,) + tuple(shp[1:]), dtype=odt)

        # fill, streaming shard by shard
        g_off = 0   # running global row index for this redshift
        t_off = 0   # running train write offset
        v_off = 0   # running val write offset
        for f in files:
            with np.load(f) as d:
                n_i = int(d[keys[0]].shape[0])
                vmask_i = is_val[g_off:g_off + n_i]
                tsel = ~vmask_i
                vsel = vmask_i
                nt_i = int(tsel.sum())
                nv_i = int(vsel.sum())
                for k in keys:
                    arr = d[k]
                    odt = out_dtype(k, specs[k][1])
                    if nt_i:
                        train_out[k][t_off:t_off + nt_i] = arr[tsel].astype(odt, copy=False)
                    if nv_i:
                        val_out[k][v_off:v_off + nv_i] = arr[vsel].astype(odt, copy=False)
                t_off += nt_i
                v_off += nv_i
                g_off += n_i

        # sample some train images for norm (from the assembled fp16 train array)
        take = min(NORM_IMGS_PER_Z, n_train)
        sidx = rng.choice(n_train, size=take, replace=False)
        norm_pixels.append(train_out["star_faceon"][sidx].astype(np.float32))

        # write this redshift's TRAIN file
        out_path = os.path.join(DST, f"Snap{snap}.npz")
        np.savez(out_path, **train_out)
        sz = os.path.getsize(out_path) / 1e9
        print(f"  Snap{snap}: total={total} train={n_train} val={n_val} "
              f"-> {out_path} ({sz:.2f} GB)  [{time.time()-t0:.0f}s]", flush=True)

        for k in keys:
            val_acc.setdefault(k, []).append(val_out[k])
        grand_train += n_train
        grand_val += n_val
        del train_out  # free ~1 GB before next redshift

    # concatenate val across redshifts and write
    val_final = {k: np.concatenate(v, axis=0) for k, v in val_acc.items()}
    val_path = os.path.join(DST, "val.npz")
    np.savez(val_path, **val_final)
    # report val redshift composition
    sf = val_final["scale_factor"]
    print(f"\n  val.npz: {grand_val} galaxies across "
          f"{len(np.unique(np.round(sf,4)))} redshifts -> {val_path}", flush=True)

    # global IQR image normalization from the train sample
    pix = np.concatenate([p.reshape(-1) for p in norm_pixels])
    median = float(np.median(pix))
    q25 = float(np.percentile(pix, 25))
    q75 = float(np.percentile(pix, 75))
    norm = {"median": median, "q25": q25, "q75": q75, "iqr": q75 - q25,
            "n_images_sampled": int(sum(p.shape[0] for p in norm_pixels)),
            "note": "global IQR norm over a fair train-image sample across all redshifts"}
    with open(os.path.join(DST, "img_norm.json"), "w") as fh:
        json.dump(norm, fh, indent=2)
    print(f"  img_norm.json: median={median:.4f} IQR={q75-q25:.4f}", flush=True)

    print(f"\nDone in {time.time()-t0:.0f}s. "
          f"train={grand_train}  val={grand_val}  -> {DST}", flush=True)


if __name__ == "__main__":
    main()
