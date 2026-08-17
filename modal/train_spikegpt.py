"""Modal launcher for SpikeGPT-style language-model training probes."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Literal
from urllib.request import urlretrieve

import modal

APP_NAME = "myelin-spikegpt"
REMOTE_ROOT = "/root/myelin"
VENV_PY = "/opt/venv/bin/python"
DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)

GPU = "H100"

# Replicate the exact uv-locked env (torch 2.13 nightly cu130 + Triton) as
# modal/experiments.py, so the Triton kernels and torch.compile match a local run.
# Extras: cuda (Triton), tracking (wandb), tokenization (datasets for streaming).
# `myelin` is a PRIVATE git dep, so sync steps get git + a GitHub token (Modal
# secret "github-token"); we inject it via git insteadOf and rm ~/.gitconfig in the
# same RUN so the token is never baked into a layer.
_GITHUB_SECRET = modal.Secret.from_name("github-token")
_SYNC_EXTRAS = "--extra cuda --extra tracking --extra tokenization"


def _authed_sync(sync_flags: str) -> str:
    auth = (
        'git config --global url."https://x-access-token:${GITHUB_TOKEN}@github.com/".insteadOf '
        '"https://github.com/"'
    )
    return (
        f"cd {REMOTE_ROOT} && {auth} && uv sync --frozen {_SYNC_EXTRAS} {sync_flags}; "
        "rc=$?; rm -f /root/.gitconfig; exit $rc"
    )


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("uv")
    .env({"UV_PROJECT_ENVIRONMENT": "/opt/venv"})
    .add_local_file("pyproject.toml", f"{REMOTE_ROOT}/pyproject.toml", copy=True)
    .add_local_file("uv.lock", f"{REMOTE_ROOT}/uv.lock", copy=True)
    .add_local_file(".python-version", f"{REMOTE_ROOT}/.python-version", copy=True)
    .add_local_file("README.md", f"{REMOTE_ROOT}/README.md", copy=True)
    .run_commands(_authed_sync("--no-install-project"), secrets=[_GITHUB_SECRET])
    .add_local_dir("src", f"{REMOTE_ROOT}/src", copy=True)
    .add_local_dir("examples", f"{REMOTE_ROOT}/examples", copy=True)
    .run_commands(_authed_sync(""), secrets=[_GITHUB_SECRET])
)

app = modal.App(APP_NAME, image=image)

# Persistent cache for tokenized .bin corpora, so re-runs skip re-tokenization.
corpora = modal.Volume.from_name("myelin-corpora", create_if_missing=True)
CORPUS_DIR = "/corpora"

# Streaming BPE datasets selectable via --dataset (resolved by prepare_token_corpus).
BPE_DATASETS = ("fineweb-edu", "openwebtext")


def _ensure_tiny_shakespeare() -> Path:
    data_dir = Path(REMOTE_ROOT) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "tinyshakespeare.txt"
    if not path.exists():
        urlretrieve(DATA_URL, path)
    return path


def _prepare_bpe_corpus(dataset: str, hf_config: str | None, max_tokens: int) -> Path:
    """Stream + BPE-tokenize a slice of a HuggingFace dataset to a cached .bin.

    Delegates to examples/prepare_token_corpus.py (the same --dataset presets used
    locally). The corpus lands on the persistent ``myelin-corpora`` volume so a
    later run with the same (dataset, config, size) reuses it instead of paying to
    re-tokenize. FineWeb-Edu / OpenWebText are BPE (GPT-NeoX, 50277 vocab), so the
    model must train with --vocab bpe (handled by the caller).
    """
    tag = f"{dataset}_{hf_config or 'default'}_{max_tokens}"
    out = Path(CORPUS_DIR) / f"{tag}.bin"
    if out.with_suffix(".json").exists():
        print(f"corpus cached, skipping tokenization: {out}", flush=True)
        return out
    command = [
        VENV_PY,
        f"{REMOTE_ROOT}/examples/prepare_token_corpus.py",
        "--dataset",
        dataset,
        "--vocab",
        "bpe",
        "--max-tokens",
        str(max_tokens),
        "--output",
        str(out),
    ]
    if hf_config:
        command.extend(["--hf-config", hf_config])
    print(f"tokenizing {dataset} ({hf_config or 'default'}) -> {out}", flush=True)
    subprocess.run(command, cwd=REMOTE_ROOT, check=True)
    corpora.commit()  # persist the .bin + sidecar for future runs
    return out


def _run_spikegpt(
    *,
    dataset: str,
    hf_config: str | None,
    max_tokens: int,
    val_holdout_tokens: int,
    preset: str,
    context_length: int,
    layers: int,
    embedding: int,
    batch: int,
    steps: int,
    compile_policy: str,
    lr: float,
    lr_final: float | None,
    warmup_steps: int,
    dropout: float,
    weight_decay: float,
    grad_clip: float,
    log_every: int,
    eval_every: int,
    eval_batches: int,
    activation_checkpointing: bool,
    wandb: bool,
    wandb_project: str,
    wandb_run_name: str | None,
    seed: int = 0,
    spiking: bool = True,
    checkpoint: bool = False,
) -> None:
    command = [
        VENV_PY,
        f"{REMOTE_ROOT}/examples/train_tiny_spikegpt.py",
        "--device",
        "cuda",
        "--compile",
        compile_policy,
        "--matmul-precision",
        "high",
        "--amp",  # match the grid explicitly (also the trainer default)
        "bf16",
    ]
    # Data source: a BPE .bin (tokenized from a streaming HF dataset) or the
    # byte-level tiny-shakespeare fallback.
    if dataset in BPE_DATASETS:
        corpus = _prepare_bpe_corpus(dataset, hf_config, max_tokens)
        command += [
            "--train-bin",
            str(corpus),
            "--vocab",
            "bpe",
            "--val-holdout-tokens",
            str(val_holdout_tokens),
        ]
    elif dataset == "shakespeare":
        command += ["--text-file", str(_ensure_tiny_shakespeare()), "--vocab", "byte"]
    else:
        raise ValueError(
            f"unknown dataset {dataset!r}; expected shakespeare or one of {BPE_DATASETS}"
        )
    command += [
        "--preset",
        preset,
        "--context-length",
        str(context_length),
        "--layers",
        str(layers),
        "--embedding",
        str(embedding),
        "--model-type",
        "rwkv",
        "--batch",
        str(batch),
        "--steps",
        str(steps),
        "--lr",
        str(lr),
        "--lr-schedule",  # match the grid (cosine to lr_final); also the default
        "cosine",
        "--warmup-steps",
        str(warmup_steps),
        "--dropout",
        str(dropout),
        "--weight-decay",
        str(weight_decay),
        "--grad-clip",
        str(grad_clip),
        "--seed",
        str(seed),
        "--log-every",
        str(log_every),
        "--eval-every",
        str(eval_every),
        # Grid eval spec: strided over a 2M-token cap of the val holdout — NOT the
        # legacy --eval-batches, so val/loss is exactly comparable to the 5090 runs.
        "--val-eval",
        "strided",
        "--val-eval-tokens",
        "2000000",
        "--sample-tokens",
        "16",
    ]
    if lr_final is not None:
        command += ["--lr-final", str(lr_final)]
    if not spiking:  # non-spiking (continuous RWKV) baseline arm
        command.append("--no-spiking")
    if checkpoint and wandb_run_name:
        ckdir = f"{CORPUS_DIR}/ckpt"
        command += ["--checkpoint-out", f"{ckdir}/{wandb_run_name}.ckpt",
                    "--checkpoint-every", "5000",
                    "--best-checkpoint-out", f"{ckdir}/{wandb_run_name}.best.pt"]
    if activation_checkpointing:
        command.append("--activation-checkpointing")
    if wandb:
        command.extend(["--wandb", "--wandb-project", wandb_project])
        if wandb_run_name is not None:
            command.extend(["--wandb-run-name", wandb_run_name])
    # Pin the W&B entity so the run lands in pitheta/<project> (our shared workspace),
    # not the API key's default personal entity. WANDB_API_KEY comes from the
    # wandb-secret attached to the run function.
    env = {**os.environ, "WANDB_ENTITY": "pitheta"}
    subprocess.run(command, cwd=REMOTE_ROOT, check=True, env=env)
    if checkpoint:
        corpora.commit()  # persist checkpoints written to the mounted volume


@app.function(cpu=16, memory=32 * 1024, timeout=24 * 60 * 60, volumes={CORPUS_DIR: corpora})
def prepare_corpus(
    dataset: str = "fineweb-edu",
    hf_config: str = "sample-100BT",
    max_tokens: int = 25_000_000_000,
) -> str:
    """Stream + tokenize a corpus to the persistent volume on a CPU-only container,
    so training runs with the same (dataset, hf_config, max_tokens) hit the cache:

        modal run --detach modal/train_spikegpt.py::prepare_corpus \\
            --dataset fineweb-edu --hf-config sample-100BT --max-tokens 25000000000
    """
    return str(_prepare_bpe_corpus(dataset, hf_config, max_tokens))


@app.function(
    gpu=GPU,
    timeout=24 * 60 * 60,
    volumes={CORPUS_DIR: corpora},
    secrets=[modal.Secret.from_name("wandb-secret")],  # WANDB_API_KEY for logging
)
def run_h100(
    *,
    dataset: str = "fineweb-edu",
    hf_config: str | None = "sample-10BT",
    max_tokens: int = 600_000_000,
    val_holdout_tokens: int = 5_000_000,
    preset: str = "custom",
    context_length: int = 1024,
    layers: int = 12,
    embedding: int = 512,
    batch: int = 16,
    steps: int = 20000,
    compile_policy: str = "regional",
    lr: float = 6.0e-4,
    lr_final: float | None = 6.0e-5,
    warmup_steps: int = 200,
    dropout: float = 0.0,
    weight_decay: float = 0.01,
    grad_clip: float = 1.0,
    log_every: int = 100,
    eval_every: int = 500,
    eval_batches: int = 16,
    activation_checkpointing: bool = False,
    wandb: bool = False,
    wandb_project: str = "myelin",
    wandb_run_name: str | None = None,
    seed: int = 0,
    spiking: bool = True,
    checkpoint: bool = False,
) -> None:
    """Run a single-GPU SpikeGPT training job on H100."""

    _run_spikegpt(
        dataset=dataset,
        hf_config=hf_config,
        max_tokens=max_tokens,
        val_holdout_tokens=val_holdout_tokens,
        preset=preset,
        context_length=context_length,
        layers=layers,
        embedding=embedding,
        batch=batch,
        steps=steps,
        compile_policy=compile_policy,
        lr=lr,
        lr_final=lr_final,
        warmup_steps=warmup_steps,
        dropout=dropout,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
        log_every=log_every,
        eval_every=eval_every,
        eval_batches=eval_batches,
        activation_checkpointing=activation_checkpointing,
        wandb=wandb,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name or (f"modal-spikegpt-{dataset}" if wandb else None),
        seed=seed,
        spiking=spiking,
        checkpoint=checkpoint,
    )


@app.local_entrypoint()
def main(
    target: Literal["h100"] = "h100",
    dataset: str = "fineweb-edu",
    hf_config: str = "sample-10BT",
    max_tokens: int = 600_000_000,
    val_holdout_tokens: int = 5_000_000,
    preset: str = "custom",
    context_length: int = 1024,
    layers: int = 12,
    embedding: int = 512,
    batch: int = 16,
    steps: int = 20000,
    compile_policy: str = "regional",
    lr: float = 6.0e-4,
    lr_final: float = 6.0e-5,
    warmup_steps: int = 200,
    dropout: float = 0.0,
    weight_decay: float = 0.01,
    grad_clip: float = 1.0,
    log_every: int = 100,
    eval_every: int = 500,
    eval_batches: int = 16,
    activation_checkpointing: bool = False,
    wandb: bool = False,
    wandb_project: str = "myelin",
    wandb_run_name: str | None = None,
    seed: int = 0,
    spiking: bool = True,
    checkpoint: bool = False,
) -> None:
    """Launch a single-GPU Modal SpikeGPT training job.

    Default trains on FineWeb-Edu (streamed + BPE-tokenized, cached to a volume).
    Use --dataset openwebtext for OpenWebText, or --dataset shakespeare for the
    tiny byte-level smoke. --hf-config default streams the full FineWeb-Edu
    (~1.3T tokens); sample-10BT/100BT/350BT are the official random subsets.
    """

    kwargs = {
        "dataset": dataset,
        "hf_config": hf_config,
        "max_tokens": max_tokens,
        "val_holdout_tokens": val_holdout_tokens,
        "preset": preset,
        "context_length": context_length,
        "layers": layers,
        "embedding": embedding,
        "batch": batch,
        "steps": steps,
        "compile_policy": compile_policy,
        "lr": lr,
        "lr_final": lr_final,
        "warmup_steps": warmup_steps,
        "dropout": dropout,
        "weight_decay": weight_decay,
        "grad_clip": grad_clip,
        "log_every": log_every,
        "eval_every": eval_every,
        "eval_batches": eval_batches,
        "activation_checkpointing": activation_checkpointing,
        "wandb": wandb,
        "wandb_project": wandb_project,
        "wandb_run_name": wandb_run_name,
        "seed": seed,
        "spiking": spiking,
        "checkpoint": checkpoint,
    }
    if target == "h100":
        run_h100.remote(**kwargs)
        return
    raise ValueError("target must be 'h100'")
