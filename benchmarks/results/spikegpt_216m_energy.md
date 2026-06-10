# SpikeGPT 216M — energy analysis: does the spiking buy the efficiency it claims?

The SNN pitch is energy: a binary spike turns a spike-fed `Linear`'s
multiply-accumulate (MAC) into a multiply-free accumulate (AC), so SpikeGPT's
paper (Zhu et al., arXiv:2302.13939, §3.7 / Table 1) claims **~20–32× fewer
operations** vs a dense transformer. This checks whether that holds for our 216M.

Tool: `examples/spikegpt_energy.py` (per-token MAC count by component × the
45 nm energy figures the paper itself uses: `E_MAC=4.5 pJ`, `E_AC=0.9 pJ`).

## Per-token cost (gpt2-216m: 18L/768d, vocab 50277)

matmul MACs/token = **176.6 M** = token-mix 42.5 M + channel-mix 95.6 M + **head 38.6 M**.

| scenario | energy/token (µJ, 45 nm) | vs dense |
|---|---:|---:|
| dense ANN (all MAC) | 794.84 | 1.00× |
| SpikeGPT Table-1 **claim** (projections as AC × r=0.34) | 215.99 | **3.68×** |
| **as-built = canonical SpikeGPT** (Linear→LIF; all MAC) | 794.84 | **1.00×** |

## Findings

**1. The AC energy win is not implemented — by canonical SpikeGPT *or* by us.**
The paper's Table 1 prices the projection matmuls as firing-rate-scaled accumulate
ops (`E_AC·R̂·FLOPs`), which requires a *spike train as each `Linear`'s input*
(LIF→Linear). But the canonical SpikeGPT `model.py` places the LIF **after** each
sub-block and adds spikes to a **continuous float residual** (`x = x + lif(att(ln(x)))`)
— so every `Linear` reads continuous floats and is an ordinary **MAC**. Our
reproduction does exactly the same. **So we faithfully match canonical SpikeGPT;
the AC conversion the energy table assumes is a property of neither the released
code nor a faithful repro.** As-built energy = dense ANN = **1.00× (no saving).**

**2. Even the idealized claim is only ~3.7× for our model — not 32×.** If we
*grant* the paper's AC accounting (the middle row), our 216M still only reaches
**3.68×**, because (a) our measured firing rate **r≈0.34** is ~2× the paper's
R̂=0.15 (per-projection ratio `E_MAC/(r·E_AC)` = 14.7× vs their 33.3×), and (b) the
**50 277-vocab BPE head is a large MAC term (38.6 M) that stays continuous and
dominates** the idealized energy. The paper's 32× was measured on the 46M
*byte-level* model (vocab 256 → negligible head, R̂=0.15); neither condition holds
for a real BPE LM.

**3. What the spiking *does* buy:** the inter-layer activations are binary and
sparse, which is cheap to **route/store** on event-driven hardware — but the
*compute* (projections + the continuous WKV's exp/division) is dense MAC. The WKV
is correctly never claimed as AC (the paper leaves it at `E_MAC`, conceding part
of the network is non-spiking).

## Honest framing

> Our reproduction faithfully matches canonical SpikeGPT's `Linear→LIF` placement;
> consequently — like canonical SpikeGPT itself — its matmuls are MAC, not AC.
> SpikeGPT's headline energy advantage is predicated on an AC conversion that
> neither the canonical code nor a faithful reproduction actually performs, and
> which for a real BPE-vocab model would in any case be diluted to ~3.7× (not 32×)
> by the higher firing rate and the large LM head.

Paired with the convergence ablation (the continuous twin trains ~5–7× faster to
better PPL, `spikegpt_216m_wikitext.md` / the ablation run), the picture is that
on GPU the spiking layers are **mostly cost, little realized benefit**. The win is
only available on hypothetical event-driven hardware *after* restructuring to
LIF→Linear (which would change the architecture and require retraining).

## Empirical confirmation: the energy-realizing placement costs the most accuracy

The energy win is only available *after* restructuring to `LIF→Linear` (spikes
feed the projections). We built that variant (`config.spike_input`) and ran a
3-way A/B (Modal `arch_ablation_ab`, 6L/512d, enwik8 byte-level, 4k steps, same
recipe). Val BPC (lower = better):

| variant | Val BPC @ 4k | vs continuous | block firing rate |
|---|---:|---:|---:|
| continuous (no spikes) | **1.467** | — | ~0 |
| spiking (canonical `Linear→LIF`) | 1.597 | +0.130 (+8.9%) | 0.21 |
| spike-input (`LIF→Linear`, hardware-faithful) | 1.618 | +0.151 (+10.3%) | 0.32 |

The placement whose matmuls are genuinely spike-driven AC (`spike-input`) is the
**worst** for accuracy — worse even than canonical spiking, which already trails
the continuous model. And its firing rate is *higher* (0.32 vs 0.21), shrinking
even the realizable AC saving. So SpikeGPT's efficiency and quality pull in
opposite directions: making the energy win real costs ~10% BPC on top of an
architecture that, as-built, doesn't realize the win at all.

## Caveats
- The headline factor varies by source: 20× ops (abstract) / 32.2× (v5 Table 1) /
  "5× energy" (press). It is an **operations** projection, not measured energy.
- The 45 nm `E_MAC`/`E_AC` are the paper's; absolute µJ scale with the node, but
  the *ratios* (the point here) do not.
- Firing rate r=0.34 is the training-log block-spike rate; a corpus-level
  measurement via `analyze_spikegpt_spikes.py` would refine it (doesn't change the
  conclusion — the as-built row is rate-independent at 1.00×).

Sources: arXiv:2302.13939 §3.7 + Table 1; ridgerchu/SpikeGPT `src/model.py`
(`Block.forward`, `RWKV_TimeMix.forward`, `RWKV_ChannelMix.forward`); Horowitz
ISSCC 2014 (energy constants).
