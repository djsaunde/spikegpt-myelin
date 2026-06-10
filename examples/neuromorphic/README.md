# SpikeGPT → SpiNNaker2 (neuromorphic) prototypes

First step toward deploying SpikeGPT on SpiNNaker2: validate the spiking (LIF)
parts in the **free Brian2 simulator** — no hardware, no private SDK. See
`benchmarks/results/spikegpt_neuromorphic_feasibility.md` for the full mapping +
memory study.

## Status

✅ `py-spinnaker2` Brian2 backend installed and running (no hardware).
✅ `spinnaker2_lif_prototype.py` — a SpikeGPT `SpikingSequenceLIF` channel maps to
   a SpiNNaker2 `lif` neuron; spike trains match **exactly** across a drive sweep
   **except at the v == threshold boundary** (SpikeGPT fires at `v >= θ`, the
   SpiNNaker2 `lif` at `v > θ` — measure-zero for real continuous inputs; nudge θ
   by −ε to match). Result: **19/21 neurons exact**, the 2 misses are exactly the
   boundary drives.

⬜ Next: feed a *time-varying* current (a real att/ffn output sequence) via a
   `Projection` from the upstream Linear (the deployment dataflow), then the WKV /
   LayerNorm as custom ARM-core C (gated behind hardware/SDK access — see below).

## Environment setup (the part that isn't obvious)

The full toolchain is partly gated; the **Brian2 simulator path works with public
packages only**, but the dependency chain took reverse-engineering:

```bash
# 1. Clone WITHOUT --recursive: the recursive submodule (s2-sim2lab-app) is the
#    PRIVATE, hardware-gated SDK and will fail. We don't need it for Brian2.
git clone --depth 1 https://gitlab.com/spinnaker2/py-spinnaker2.git

# 2. py-spinnaker2 needs Python 3.12 (its gen-2 SpiNNMan2 pins >=3.12,<3.13).
uv venv --python 3.12 .venv && source .venv/bin/activate

# 3. Install the gen-2 stack. SpiNNMan2 (the gen-2 comms layer, public on GitLab)
#    provides spinnman.utilities.address_utils that py-spinnaker2 needs; it pulls
#    numpy>=2, so use a numpy-2-compatible brian2 (>=2.10). The rest are plain deps.
uv pip install "git+https://gitlab.com/spinnaker2/SpiNNMan2.git@v0.2.3b" \
    "brian2>=2.10" scipy tqdm matplotlib pyyaml sortedcontainers

# 4. Run via PYTHONPATH (skip `pip install .` — its build wants the private libs).
PYTHONPATH=<path>/py-spinnaker2/src python spinnaker2_lif_prototype.py
```

Notes / gotchas found:
- `--recursive` clone fails on the private `s2-sim2lab-app` submodule — clone plain.
- `pip install -e .` fails (build wants the private `libs/`); use `PYTHONPATH` instead.
- The package imports the gen-2 `spinnman` namespace from **SpiNNMan2** (not the
  gen-1 `SpiNNMan` on PyPI). Python **3.12** is mandatory for SpiNNMan2.
- The Brian2 backend supports only the **LIF family** (`lif`, `lif_curr_exp`, …),
  **not custom neuron models** — so the WKV custom-C core can't be simulated for
  free; validate WKV numerics off-chip against the PyTorch/Triton reference.

## SpikeGPT LIF → SpiNNaker2 `lif` parameter mapping

| SpikeGPT (`SpikingSequenceLIF`) | SpiNNaker2 `lif` | value (tau=2, thr=1) |
|---|---|---|
| `decay = 1 - 1/tau` | `alpha_decay` | 0.5 |
| `threshold` | `threshold` (note `>` vs `>=`) | 1.0 |
| `reset` | `v_reset` + `reset=reset_to_v_reset` | 0.0 |
| input current `x[t]` | `i_offset` (const) / `I_syn[t]` (synaptic) | per-token |
