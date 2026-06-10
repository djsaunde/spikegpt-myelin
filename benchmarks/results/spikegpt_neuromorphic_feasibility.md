# SpikeGPT → SpiNNaker2: neuromorphic feasibility + hardware-graph mapping

What it would take to deploy a trained SpikeGPT on SpiNNaker2 neuromorphic
hardware, and the concrete blockers. Tool: `examples/spikegpt_neuromorphic_feasibility.py`.

## Toolchain reality (researched June 2026)

- **`py-spinnaker2`** (GitLab, v0.7.0, actively maintained) is the stack: PyNN-style
  `Population` / `Projection` / `Network`. Built-in neuron models: the `lif` family,
  Izhikevich, a few specialized; an MLA module wraps the int8 MAC accelerator.
- **Free Brian2 simulator** exists (no board) but runs **only the standard LIF
  family — not custom neuron models.** Even installing the full toolchain needs the
  **private** `s2-sim2lab-app` SDK, gated behind hardware access (email SpiNNcloud /
  go through the NSF THOR system at UTSA).
- **Custom continuous compute on the Cortex-M4F PEs IS the intended mechanism** (a
  "neuron model" = precompiled C with arbitrary math), so the WKV/LayerNorm/Linears
  *can* run on the ARM cores — but it's hand-written C against that private, "under
  substantial restructuring" SDK, **with no free-simulator coverage**.
- **Prior art = EGRU** (arXiv:2312.09084): the first LM on SpiNNaker2 — an
  event-based GRU, **all compute as FP32 C on the ARM cores**, 95%-pruned sparse-CSR
  weights, **batch-1, memory-bound**; 0.065 J/inf vs 1.19 J on A100. No transformer/
  RWKV/SpikeGPT has been ported. EGRU is the working template for our task.

## Hardware-graph decomposition — 41M model (12L/512d, enwik8 byte vocab)

| component | params | % | SpiNNaker2 target |
|---|---:|---:|---|
| ffn_proj | 28.3M | 68.8% | continuous-ARM |
| att.kvr_proj | 9.4M | 22.9% | continuous-ARM |
| att.output | 3.1M | 7.6% | continuous-ARM |
| embedding + head | 0.26M | 0.6% | continuous-ARM |
| **total** | **41.2M** | 100% | **continuous-ARM (100%)** |

Plus 2·n_layer = 24 **weightless** LIF populations (the only true event-driven
part) and the WKV / LayerNorm / token-shift / residual adds (continuous-ARM).

**The headline structural finding, in hardware terms:** under the canonical
`Linear→LIF` placement, **100% of the weights are continuous-ARM** — the projections
read the continuous residual, so they run as ordinary FP32/MAC compute on the ARM
cores (the EGRU pattern), and **SpiNNaker2's signature event-driven sparse-synapse
fabric is essentially unused** (only the weightless LIF thresholding touches it).
This is the energy analysis (`spikegpt_216m_energy.md`) restated as a placement
map: the deployment is "continuous RNN on ARM cores + cheap LIF," not a spiking
network. Switching to `spike_input` (`LIF→Linear`) moves **91.7%** of the weights
(att.kvr 22.9% + ffn input parts 68.8%) onto the spike-synaptic fabric — at the
measured ~10% accuracy cost.

## Per-PE memory feasibility (128 KB/PE, 152 PEs/chip, 96 KB/PE usable for weights)

| weight format | 41M model | 216M model |
|---|---|---|
| fp32 dense | 165 MB → **11 chips** | 861 MB → **58 chips** |
| int8 dense | 41 MB → **2.8 chips** | 215 MB → **14 chips** |
| int8, 90%-pruned CSR | 12 MB → **0.8 chips** | 65 MB → **4.3 chips** |

**Weight storage is the binding constraint, exactly as EGRU found.** Even at int8,
the 41M model is a ~3-chip, batch-1 deployment; fitting a single chip needs ~90%
pruning (→ retraining + an accuracy hit, TBD). The 216M is firmly multi-chip
(14 chips int8). EGRU hit this wall with a *smaller* model.

## What it would take (path, mirroring EGRU)

1. **Access** (weeks of lead time) — SpiNNcloud academic / THOR-UTSA / EBRAINS;
   required even for the full simulator, since the SDK is private.
2. LIF activations → built-in `lif` populations (prototype in the free Brian2 sim).
3. WKV + LayerNorm + Linears → **custom FP32 C PE apps**, sparse-CSR weights to fit
   128 KB/PE, activations broadcast over the NoC (EGRU's recipe).
4. Stitch as one `Network`; validate LIF in Brian2 + WKV numerics off-chip vs the
   PyTorch reference (the custom WKV core has **no free-sim coverage**).

## Biggest blockers

- **Access to the private, in-flux C SDK** — not the math — is the real wall; the
  free simulator can't run the custom WKV core.
- **Memory / batch-1**: multi-chip, weight-storage-bound; dense int8 = ~3 chips
  (41M) / ~14 (216M). Pruning to fit fewer chips needs retraining.
- **The spiking fabric is unused by the faithful architecture.** To actually exploit
  SpiNNaker2's event-driven sparsity you must adopt `spike_input` (or an EGRU-like
  event-native sequence mixer), which costs accuracy — the central research tension.
- WKV numerical stability in fixed/float on M4F (max-subtraction trick, tight memory).

## Verdict

Mappable in principle — EGRU is near-proof — but a research-grade effort gated on
private-SDK/hardware access, landing a multi-chip batch-1 deployment whose energy
win (for the faithful model) is "low-power chip running continuous compute," not
event-driven sparsity. The honest neuromorphic payoff requires an event-native
redesign. Artifacts here (the hardware graph + memory budget + WKV reference
numerics) are what an access-gated build would consume.
