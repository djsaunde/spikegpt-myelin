# Speedrun leaderboard — 46M (12L/512d) enwik8, wall-clock to target val BPC

Steady-state ms/step excludes the compile/warmup ramp. wall-to-target is steady-state seconds to reach the target BPC (proxy for the full run).

| label | target | steady ms/step | wall-to-target | final BPC @ steps |
|---|---:|---:|---|---|
| baseline-tuned | 1.7 | 202.7 | 201.1s @ step 1000 | 1.4733 @ 2500 |
| +ctxwarmup | 1.7 | 203.1 | 146.3s @ step 750 | 1.4652 @ 2500 |
| +ctxwarmup+lr3e3 | 1.7 | 196.8 | 140.4s @ step 750 | 1.4645 @ 2500 |
| +ctx128+lr3e3 | 1.7 | 200.0 | 143.6s @ step 750 | 1.4723 @ 2500 |
| base@1.50 | 1.5 | 197.7 | 387.4s @ step 1950 | 1.4408 @ 3000 |
| ctxwarm+lr3e3@1.50 | 1.5 | 205.6 | 443.9s @ step 1950 | 1.4394 @ 3000 |
| lr3e3-static@1.50 | 1.5 | 425.0 | 892.5s @ step 2100 | 1.4504 @ 3000 |
| wsd@1.50 | 1.5 | 387.5 | 962.4s @ step 2400 | 1.4257 @ 3000 |

## Findings (honest)

**Metric correction.** "Wall-clock to an *intermediate* target" (1.70/1.50) is the
wrong objective — it penalizes warmup-stable-decay (WSD), which stays high then
drops sharply, and rewards front-loaded tricks. The right speedrun metric is
**val BPC at a fixed step budget** (better quality in the same budget = faster to
the real target). Steps/BPC are deterministic; wall-clock *timing* was noisy this
session (later runs reported ~2x ms/step — environmental GPU degradation), so
judge on steps + BPC.

**WSD > cosine (the win).** At 3000 steps WSD reaches **1.4257 BPC vs cosine's
1.4408** — a real quality-at-budget improvement to the tuned recipe.

**ctx-warmup is NOT a compiled-path win.** It reaches the meaningful 1.50 target at
the *same* step (1950) as baseline — no convergence advantage — and requires
`dynamic=True` shape-polymorphic compile (slower kernels than static). Its −30% at
the *easy* 1.70 target was front-loaded (bigger early batch from constant-tokens
shorter-ctx converges faster to easy targets only). The −8–15% in
`ctx_warmup_5090.md` was measured **eager**; under compilation the dynamic-shape
penalty negates it. So ctx-warmup helps eager / easy-target runs but not the
compiled run to a real quality bar.

**Rejected:** lr 3e-3 static (worse convergence: 1.4504), ctx-128 (too-short early
context hurts), Muon and RWKV-7-swap (earlier experiments).
