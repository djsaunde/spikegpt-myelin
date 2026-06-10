"""Modal launcher for SpikeGPT-style language-model training probes."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Literal
from urllib.request import urlretrieve

import modal

APP_NAME = "myelin-spikegpt"
REMOTE_ROOT = "/root/myelin"
DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)

GPU = "H100"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=1.24.4",
        "torch",
        "triton",
        "wandb",
    )
    .add_local_dir(
        ".",
        remote_path=REMOTE_ROOT,
        ignore=[
            ".git",
            ".venv",
            ".pytest_cache",
            ".ruff_cache",
            "data",
            "dist",
            "runs",
            "wandb",
        ],
    )
)

app = modal.App(APP_NAME, image=image)


def _ensure_tiny_shakespeare() -> Path:
    data_dir = Path(REMOTE_ROOT) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "tinyshakespeare.txt"
    if not path.exists():
        urlretrieve(DATA_URL, path)
    return path


def _run_spikegpt(
    *,
    context_length: int,
    layers: int,
    embedding: int,
    batch: int,
    steps: int,
    compile_policy: str,
    lr: float,
    dropout: float,
    grad_clip: float,
    log_every: int,
    eval_every: int,
    eval_batches: int,
    activation_checkpointing: bool,
    wandb: bool,
    wandb_project: str,
    wandb_run_name: str | None,
) -> None:
    text_file = _ensure_tiny_shakespeare()
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{REMOTE_ROOT}/src:{REMOTE_ROOT}/examples"
    command = [
        sys.executable,
        f"{REMOTE_ROOT}/examples/train_tiny_spikegpt.py",
        "--device",
        "cuda",
        "--compile",
        compile_policy,
        "--matmul-precision",
        "high",
        "--text-file",
        str(text_file),
        "--vocab",
        "byte",
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
        "--dropout",
        str(dropout),
        "--grad-clip",
        str(grad_clip),
        "--log-every",
        str(log_every),
        "--eval-every",
        str(eval_every),
        "--eval-batches",
        str(eval_batches),
        "--sample-tokens",
        "16",
    ]
    if activation_checkpointing:
        command.append("--activation-checkpointing")
    if wandb:
        command.extend(["--wandb", "--wandb-project", wandb_project])
        if wandb_run_name is not None:
            command.extend(["--wandb-run-name", wandb_run_name])
    subprocess.run(command, cwd=REMOTE_ROOT, env=env, check=True)


@app.function(gpu=GPU, timeout=4 * 60 * 60)
def run_h100(
    *,
    context_length: int = 1024,
    layers: int = 12,
    embedding: int = 512,
    batch: int = 1,
    steps: int = 20,
    compile_policy: str = "regional",
    lr: float = 6.0e-4,
    dropout: float = 0.03,
    grad_clip: float = 1.0,
    log_every: int = 10,
    eval_every: int = 10,
    eval_batches: int = 1,
    activation_checkpointing: bool = False,
    wandb: bool = False,
    wandb_project: str = "myelin",
) -> None:
    """Run a single-GPU SpikeGPT probe on H100."""

    _run_spikegpt(
        context_length=context_length,
        layers=layers,
        embedding=embedding,
        batch=batch,
        steps=steps,
        compile_policy=compile_policy,
        lr=lr,
        dropout=dropout,
        grad_clip=grad_clip,
        log_every=log_every,
        eval_every=eval_every,
        eval_batches=eval_batches,
        activation_checkpointing=activation_checkpointing,
        wandb=wandb,
        wandb_project=wandb_project,
        wandb_run_name="modal-spikegpt-h100" if wandb else None,
    )


@app.local_entrypoint()
def main(
    target: Literal["h100"] = "h100",
    context_length: int = 1024,
    layers: int = 12,
    embedding: int = 512,
    batch: int = 1,
    steps: int = 20,
    compile_policy: str = "regional",
    lr: float = 6.0e-4,
    dropout: float = 0.03,
    grad_clip: float = 1.0,
    log_every: int = 10,
    eval_every: int = 10,
    eval_batches: int = 1,
    activation_checkpointing: bool = False,
    wandb: bool = False,
    wandb_project: str = "myelin",
) -> None:
    """Launch a single-GPU Modal SpikeGPT training probe."""

    kwargs = {
        "context_length": context_length,
        "layers": layers,
        "embedding": embedding,
        "batch": batch,
        "steps": steps,
        "compile_policy": compile_policy,
        "lr": lr,
        "dropout": dropout,
        "grad_clip": grad_clip,
        "log_every": log_every,
        "eval_every": eval_every,
        "eval_batches": eval_batches,
        "activation_checkpointing": activation_checkpointing,
        "wandb": wandb,
        "wandb_project": wandb_project,
    }
    if target == "h100":
        run_h100.remote(**kwargs)
        return
    raise ValueError("target must be 'h100'")
