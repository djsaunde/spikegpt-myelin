"""Utilities shared by example scripts."""

from __future__ import annotations

import argparse
from typing import Any, cast

import torch

DEFAULT_GRAD_CLIP = 0.1
COMPILE_POLICIES = ("auto", "on", "off")
ENCODINGS = ("repeat", "poisson", "latency")
MATMUL_PRECISIONS = ("highest", "high", "medium")


def print_model_summary(model: torch.nn.Module) -> None:
    total_params = 0
    trainable_params = 0
    print("| Parameter | Shape | Trainable | Count |", flush=True)
    print("|---|---:|---:|---:|", flush=True)
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        total_params += count
        if parameter.requires_grad:
            trainable_params += count
        shape = "x".join(str(dim) for dim in parameter.shape)
        print(f"| {name} | {shape} | {parameter.requires_grad} | {count} |", flush=True)

    print()
    print(f"total_params={total_params}", flush=True)
    print(f"trainable_params={trainable_params}", flush=True)


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    to_local = getattr(tensor, "to_local", None)
    if to_local is None:
        return tensor
    return cast(torch.Tensor, to_local())


def _tensor_memory_mb(tensor: torch.Tensor) -> float:
    return tensor.numel() * tensor.element_size() / (1024 * 1024)


def _local_tensor_memory_mb(tensor: torch.Tensor) -> float:
    return _tensor_memory_mb(_local_tensor(tensor))


def print_resident_memory_summary(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    parameter_mb = sum(_tensor_memory_mb(parameter) for parameter in model.parameters())
    local_parameter_mb = sum(_local_tensor_memory_mb(parameter) for parameter in model.parameters())
    trainable_parameter_mb = sum(
        _tensor_memory_mb(parameter) for parameter in model.parameters() if parameter.requires_grad
    )
    local_trainable_parameter_mb = sum(
        _local_tensor_memory_mb(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print(f"parameter_memory_mb={parameter_mb:.3f}", flush=True)
    print(f"local_parameter_memory_mb={local_parameter_mb:.3f}", flush=True)
    print(f"trainable_parameter_memory_mb={trainable_parameter_mb:.3f}", flush=True)
    print(f"local_trainable_parameter_memory_mb={local_trainable_parameter_mb:.3f}", flush=True)

    if optimizer is None:
        return
    optimizer_state_mb = 0.0
    local_optimizer_state_mb = 0.0
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                optimizer_state_mb += _tensor_memory_mb(value)
                local_optimizer_state_mb += _local_tensor_memory_mb(value)
    print(f"optimizer_state_memory_mb={optimizer_state_mb:.3f}", flush=True)
    print(f"local_optimizer_state_memory_mb={local_optimizer_state_mb:.3f}", flush=True)


def print_step_time_summary(step_times: list[float]) -> None:
    if step_times:
        average_step_ms = sum(step_times) / len(step_times) * 1000
        print(f"average_step_ms={average_step_ms:.3f}", flush=True)
    if len(step_times) > 1:
        post_warmup_step_ms = sum(step_times[1:]) / len(step_times[1:]) * 1000
        print(f"post_warmup_average_step_ms={post_warmup_step_ms:.3f}", flush=True)
    if len(step_times) > 2:
        steady_state_step_ms = sum(step_times[2:]) / len(step_times[2:]) * 1000
        print(f"steady_state_average_step_ms={steady_state_step_ms:.3f}", flush=True)


def reset_cuda_peak_memory(device: str) -> None:
    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)
        torch.cuda.reset_peak_memory_stats(torch_device)


def cuda_peak_memory_mb(device: str) -> float | None:
    torch_device = torch.device(device)
    if torch_device.type != "cuda":
        return None
    torch.cuda.synchronize(torch_device)
    return torch.cuda.max_memory_allocated(torch_device) / (1024 * 1024)


def print_cuda_peak_memory_summary(device: str) -> None:
    peak_mb = cuda_peak_memory_mb(device)
    if peak_mb is not None:
        print(f"peak_cuda_memory_mb={peak_mb:.3f}", flush=True)


def add_wandb_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="myelin")
    parser.add_argument("--wandb-run-name")


def add_matmul_precision_arg(parser: argparse.ArgumentParser, *, default: str = "highest") -> None:
    parser.add_argument(
        "--matmul-precision",
        choices=MATMUL_PRECISIONS,
        default=default,
        help="float32 matmul precision for CUDA examples; use high to enable TF32 where available",
    )


def configure_matmul_precision(precision: str) -> None:
    torch.set_float32_matmul_precision(precision)


def add_compile_policy_arg(
    parser: argparse.ArgumentParser,
    *,
    extra_policies: tuple[str, ...] = (),
) -> None:
    choices = (*COMPILE_POLICIES, *extra_policies)
    extra_help = ""
    if extra_policies:
        extra_help = f" Additional policies for this example: {', '.join(extra_policies)}."
    parser.add_argument(
        "--compile",
        nargs="?",
        const="on",
        choices=choices,
        default="auto",
        help=(
            "compile policy: auto compiles CUDA training by default, on always compiles, "
            "off disables. Bare --compile is shorthand for --compile on." + extra_help
        ),
    )


def resolve_compile_policy(policy: str, device: str) -> bool:
    if policy == "on":
        return True
    if policy == "off":
        return False
    if policy == "auto":
        return torch.device(device).type == "cuda" and getattr(torch, "compile", None) is not None
    msg = f"unsupported compile policy: {policy}"
    raise ValueError(msg)


def compile_training_model(model: torch.nn.Module, enabled: bool) -> torch.nn.Module:
    if not enabled:
        return model
    return cast(torch.nn.Module, torch.compile(model, mode="reduce-overhead", fullgraph=True))


def add_grad_clip_arg(
    parser: argparse.ArgumentParser,
    *,
    default: float = DEFAULT_GRAD_CLIP,
) -> None:
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=default,
        help="clip gradient norm before optimizer step; use 0 to disable",
    )


def add_surrogate_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--surrogate-slope", type=float, default=5.0)
    parser.add_argument(
        "--smooth-forward",
        action="store_true",
        help="use smooth surrogate values in the forward pass instead of binary spikes",
    )


def add_encoding_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--encoding", choices=ENCODINGS, default="poisson")


def encode_time_series(values: torch.Tensor, timesteps: int, encoding: str) -> torch.Tensor:
    repeated = values.unsqueeze(0).expand(timesteps, *values.shape).contiguous()
    if encoding == "repeat":
        return repeated
    if encoding == "poisson":
        return (torch.rand_like(repeated) < repeated).to(dtype=values.dtype)
    if encoding == "latency":
        time = torch.arange(timesteps, device=values.device)
        spike_time = ((1.0 - values).clamp(0.0, 1.0) * (timesteps - 1)).round().to(torch.long)
        spikes = time.view(timesteps, *([1] * values.ndim)) == spike_time.unsqueeze(0)
        return (spikes & (values.unsqueeze(0) > 0)).to(dtype=values.dtype)
    msg = f"unsupported encoding: {encoding}"
    raise ValueError(msg)


def clip_gradients(model: torch.nn.Module, max_norm: float | None) -> float | None:
    if max_norm is None or max_norm <= 0:
        return None
    return float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm))


def init_wandb(
    *,
    enabled: bool,
    project: str,
    run_name: str | None,
    config: dict[str, Any],
):
    if not enabled:
        return None
    try:
        import wandb
    except ModuleNotFoundError as exc:
        msg = "Weights & Biases support requires `uv sync --extra tracking`."
        raise ModuleNotFoundError(msg) from exc
    run = wandb.init(project=project, name=run_name, config=config)
    run.define_metric("*", step_metric="trainer/step")
    return run


def log_wandb(run: Any, metrics: dict[str, float | int], *, step: int | None = None) -> None:
    if run is not None:
        payload = dict(metrics)
        if step is not None:
            payload["trainer/step"] = step
        run.log(payload)


def finish_wandb(run: Any) -> None:
    if run is not None:
        run.finish()
