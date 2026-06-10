# Project: Deploy SpikeGPT on SpiNNaker2 (neuromorphic inference)

**Handoff doc for a follow-up agent.** Goal: take a trained `myelin` SpikeGPT
language model and run *inference* on SpiNNaker2 neuromorphic hardware (or a
faithful simulator), closing the gap between today's primitive-level hardware
export and a full-model deployment.

Status: **scoping done, not started.** The crux is the WKV time-mixing, which is
not a spiking operation. Read "The central tension" before estimating effort.

---

## Why this is interesting (and the central tension)

SpikeGPT was designed to be neuromorphic-friendly, and most of it maps cleanly:

- **LIF activations** (`SpikingSequenceLIF`) → the native primitive of every
  neuromorphic chip (membrane integrate, threshold, reset, leak).
- **Binary spikes between layers** → event-driven spike routing.
- **Sparsity** (~19% fire rate at our settings) → silent neurons cost ~zero
  energy on event-driven HW. This is the whole efficiency premise.
- **A `Linear` fed by spikes** → synaptic weight-accumulation (the array these
  chips are built around).

**The tension: the WKV time-mixing (RWKV linear attention) is NOT spiking.** It
is a continuous-valued, exp-weighted running average with a learned decay and a
*division* for normalization (see `weighted_key_value_loop` in
`src/myelin/language.py`):

```
wkv_t = (Σ_{i<t} exp(w·(t-1-i)+k_i)·v_i + exp(u+k_t)·v_t)
        / (Σ_{i<t} exp(w·(t-1-i)+k_i)   + exp(u+k_t))
```

It needs `exp`, real-valued accumulators, and division — none of which
specialized neuromorphic cores (e.g. Loihi's microcoded neurons) do well. So
SpikeGPT spikes the *activations* but its *sequence mixer* is ordinary continuous
arithmetic. **That is the hard part of this project.**

## Target hardware: SpiNNaker2 (not Loihi)

- **SpiNNaker2** is *digital* neuromorphic — each node pairs spiking accelerators
  with general **ARM cores**. So it can run the continuous WKV on the ARM cores
  *and* the event-driven LIF in the same fabric. That makes it the only realistic
  target for the *whole* model. Energy savings come from the sparse spike routing,
  even though the WKV runs as regular compute.
- **Loihi 2 is out**: its specialized cores don't fit the WKV's exp-normalization,
  and Intel deprecated much of the Lava stack. The repo already **removed** the
  Lava/Loihi codepath for this reason.

This is **inference-only**: train on GPU (surrogate-gradient BPTT — already done),
then deploy the forward pass. There is a trained checkpoint to use:
`runs/spikegpt_enwik8_12L512d_fast.best.pt` (12L/512d, ~41M, enwik8 byte-level,
**strided test BPC ~1.235**; `runs/` is gitignored — regenerate or copy it over).

---

## Current state of the repo's hardware export (what exists today)

`src/myelin/hardware.py` (~1130 lines) is a **primitive-level** export, fully
decoupled from SpikeGPT. It handles exactly one thing: a generic **`LinearLIF`**
block (`src/myelin/modules.py:585` — a dense `Linear` synapse → `LIF` neuron, used
by the MNIST-style examples, **not** by SpikeGPT).

What it does, end to end:
- `export_linear_lif_module(module)` (`hardware.py:493`) — pulls
  `module.synapse.weight` + `module.unroll.cell.params` (a `LIFParams`) → float
  `DenseLIFHardwareExport`.
- `quantize_dense_lif_export(..., num_bits=8)` (`:663`) — fixed-point weights.
- `plan_dense_lif_placement(...)` (`:435`) — neurons/core, incoming synapses/core,
  core count.
- `export_spinnaker2_dense_lif_manifest(...)` (`:575`) → `SpiNNaker2DenseLIFManifest`.
- `export_linear_lif_hardware_bundle(...)` (`:514`) — writes all artifacts.

That is the genuinely-clean mapping (quantized weight-accumulation synapses + LIF
neurons placed on cores) and a good proof-of-concept.

**What it does NOT handle (the gaps):**
1. **The WKV — zero support.** No `wkv`/`time_mix`/`rwkv`/`attention` anywhere in
   `hardware.py`. The continuous recurrence has no schema, no quantization, no
   placement. It must be hand-written on the ARM cores; nothing exists.
2. **SpikeGPT's own layers are the wrong shape.** `language.py` uses `LinearLIF`
   **0 times** — it has `SpikeTimeMix` (`:763`, WKV+LIF), `SpikeChannelMix`
   (`:876`), and `SpikingSequenceLIF` (`:612`). The exporter expects
   `module.synapse.weight` / `module.unroll.cell.params`, which these don't
   expose. So even the *spiking* parts aren't directly exportable.
3. **No model assembly** — export is per-layer; no SpikeGPT → multi-core graph
   (residuals, layernorms, routing, recurrence-state handoff, token-as-time wiring).
4. **The continuous bits** — `spike_embedding`, layernorms, the head, the WKV
   normalization — none spike-mappable, none exported.

---

## Work breakdown (the gap-closing project)

Roughly in dependency order. Treat as a research/engineering effort, not a swap.

1. **Decompose the model into a hardware graph.** Walk `SpikeLanguageModel` and
   classify every op as `spike-synaptic` (Linear-fed-by-spikes + LIF →
   neuromorphic cores), `continuous` (WKV, layernorms, embeddings, head → ARM
   cores), or `routing/state`. Output: a per-block placement graph. *Accept:* a
   serialized graph that round-trips and names every op's target.

2. **Exporter for `SpikingSequenceLIF`.** Adapt the `LinearLIF` path (or add a
   sibling) so SpikeGPT's actual spike layers export to the dense-LIF schema
   (weights from the upstream `Linear` in `SpikeChannelMix`/`SpikeTimeMix`, LIF
   params from the layer). *Accept:* a SpikeGPT block's LIF maps to a
   `SpiNNaker2DenseLIFManifest` and dequantized inference matches the GPU layer
   within quantization tolerance.

3. **WKV on the ARM cores.** Implement the exp-normalized recurrence as
   fixed/float ARM-core code (the per-(b,c) running num/den/max form in
   `weighted_key_value_loop` is the reference; the production kernel is
   `src/myelin/wkv_triton.py`). Decide precision (the WKV is numerically
   sensitive — fp32 on GPU; on ARM, evaluate fixed-point vs float). *Accept:*
   ARM-side WKV matches the GPU `weighted_key_value` within tolerance on real
   activations.

4. **Quantization-aware path.** 8-bit weights will cost BPC. Add a QAT or
   post-training-quant pass and measure the BPC hit vs the 1.235 GPU baseline
   (use `examples/evaluate_spikegpt_checkpoint.py`, default full-context strided
   eval). *Accept:* a measured quantized BPC + a decision on whether it's
   acceptable / needs QAT retraining.

5. **End-to-end simulator first, then hardware.** Validate the whole pipeline on
   a SpiNNaker2 simulator (py-spinnaker2 / the SpiNNcloud toolchain) before real
   silicon — token-as-time wiring, recurrence-state persistence across tokens,
   routing. *Accept:* generates coherent text matching the GPU model's outputs
   (within quant noise) on the simulator.

6. **(Optional) Energy/throughput characterization** vs the GPU baseline — the
   actual point of neuromorphic. Note the WKV+Linears on ARM cores won't get the
   event-sparsity win, so report *partial* savings honestly.

---

## Risks / open questions / honest caveats

- **The WKV is the crux.** If ARM-core WKV is too slow / imprecise, the whole
  value proposition weakens. Consider as a *separate* research direction an
  architecture whose sequence-mixer is itself spike/event-based (then the chip's
  sparsity actually pays off end-to-end).
- **Quantization → BPC.** Unknown hit; could need QAT retraining (GPU).
- **Scale & partitioning.** ~41M params across many cores; SpiNNaker2 toolchain is
  research-grade. Budget integration time.
- **Energy claim is partial.** Only the spiking parts get the event-driven win;
  the continuous WKV + dense Linears don't. Report this honestly — it's the
  standard overclaim in "SNN LLM on neuromorphic HW" work.
- **Ecosystem.** Confirm current SpiNNaker2 access + toolchain (SpiNNcloud) before
  committing; Loihi/Lava is deprecated and not an option.

## Where to start (orientation for the next agent)

- Architecture: `src/myelin/language.py` — `SpikeLanguageModel`, `SpikeGPTBlock`,
  `SpikeTimeMix` (WKV), `SpikeChannelMix`, `SpikingSequenceLIF`,
  `weighted_key_value` / `weighted_key_value_loop`.
- WKV kernel (reference math): `src/myelin/wkv_triton.py`.
- Existing export: `src/myelin/hardware.py` + `src/myelin/benchmarks/hardware_export.py`.
- Generic spike block the export currently targets: `LinearLIF` in
  `src/myelin/modules.py:585`.
- Trained model + eval: `runs/spikegpt_enwik8_12L512d_fast.best.pt`,
  `examples/evaluate_spikegpt_checkpoint.py`.
- Conceptual writeup of the mapping (this doc's source): the LIF/WKV split and the
  "spiking model with non-spiking attention" framing.

**First concrete deliverable:** task 1 (the hardware-graph decomposition) + task 2
(a `SpikingSequenceLIF` exporter) — they're the smallest steps that prove the
spike path on SpikeGPT's *own* layers and surface the WKV boundary explicitly.
