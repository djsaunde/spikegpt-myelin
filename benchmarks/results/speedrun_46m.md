# Speedrun leaderboard — 46M (12L/512d) enwik8, wall-clock to target val BPC

Steady-state ms/step excludes the compile/warmup ramp. wall-to-target is steady-state seconds to reach the target BPC (proxy for the full run).

| label | target | steady ms/step | wall-to-target | final BPC @ steps |
|---|---:|---:|---|---|
| baseline-tuned | 1.7 | 202.7 | 201.1s @ step 1000 | 1.4733 @ 2500 |
| +ctxwarmup | 1.7 | 203.1 | 146.3s @ step 750 | 1.4652 @ 2500 |
| +ctxwarmup+lr3e3 | 1.7 | 196.8 | 140.4s @ step 750 | 1.4645 @ 2500 |
| +ctx128+lr3e3 | 1.7 | 200.0 | 143.6s @ step 750 | 1.4723 @ 2500 |
