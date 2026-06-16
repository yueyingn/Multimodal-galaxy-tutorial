"""Data & checkpoint resolution for the Gal4M tutorial.

The validation data and trained checkpoints are published on the Hugging Face
Hub so the notebooks run without access to the original simulation volume.
Everything lives in a single repo that mirrors this working directory: a
``data/`` folder and a ``checkpoints/`` folder.

The helpers are **local-first** — they return an existing local file untouched
and only download from the Hub when it is missing.  Override the repo or the
local directories with environment variables:

    GAL4M_HF_REPO          - the Hub repo id (holds data/ + checkpoints/)
    GAL4M_DATA_DIR         - local data directory        (default: data)
    GAL4M_CHECKPOINTS_DIR  - local checkpoints directory (default: checkpoints)
"""

import os

HF_REPO = os.environ.get("GAL4M_HF_REPO", "yueyingn/multimodal-galaxy-tutorial")

DATA_DIR = os.environ.get("GAL4M_DATA_DIR", "data")
CHECKPOINTS_DIR = os.environ.get("GAL4M_CHECKPOINTS_DIR", "checkpoints")


def _fetch(repo_relpath, local_path):
    """Return ``local_path``, downloading ``repo_relpath`` from the Hub if absent."""
    if not os.path.exists(local_path):
        from huggingface_hub import hf_hub_download

        # local_dir="." so the repo's data/<f> or checkpoints/<f> lands at the
        # matching local path.
        hf_hub_download(repo_id=HF_REPO, filename=repo_relpath, local_dir=".")
    return local_path


def data_path(name):
    """Local path to a data file (downloads ``data/<name>`` from the Hub if absent)."""
    return _fetch(f"data/{name}", os.path.join(DATA_DIR, name))


def checkpoint_path(name):
    """Local path to a checkpoint (downloads ``checkpoints/<name>`` if absent)."""
    return _fetch(f"checkpoints/{name}", os.path.join(CHECKPOINTS_DIR, name))


def download_data(**kwargs):
    """Download the whole ``data/`` folder from the Hub into ``./data``."""
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=HF_REPO, allow_patterns=["data/*"], local_dir=".", **kwargs)


def download_checkpoints(**kwargs):
    """Download the whole ``checkpoints/`` folder from the Hub into ``./checkpoints``."""
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=HF_REPO, allow_patterns=["checkpoints/*"], local_dir=".", **kwargs)
