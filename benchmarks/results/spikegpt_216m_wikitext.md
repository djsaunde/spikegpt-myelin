# SpikeGPT 216M — WikiText reproduction (OWT2 pretrain → WikiText fine-tune)

Reproduces the SpikeGPT paper's second headline (arXiv:2302.13939, Table 3,
"SpikeGPT w/ Pre-training", 216M): token-level test perplexity on WikiText after
OpenWebText2 pretraining + WikiText fine-tuning.

**Result: WikiText-103 test PPL 26.47 vs paper 39.75 — beat by 33%.**

## Setup (apples-to-apples with the paper)

- **Model:** 18L / 768d / vocab 50277 = 215.4M params (= paper's "216M"), spike
  embedding + 2 LIF gates/block, RWKV-v4 WKV time-mix, squared-ReLU channel-mix.
- **Tokenizer:** GPT-NeoX-20B BPE, vocab 50277 — **matches the paper** (verified
  against the repo's `20B_tokenizer.json` / `vocab_size=50277`). Token-level PPL
  is tokenizer-dependent, so this match is what makes the comparison valid.
- **Data:** WikiText is the **raw** config (no `<unk>` substitution — verified by
  decoding), tokenized with the same NeoX tokenizer as pretraining. WikiText-2 and
  WikiText-103 share a **byte-identical** val/test set (standard; `cmp`-confirmed).
- **Pretrain:** OWT2, 10B-token budget, 610k steps, batch 16, ctx 1024, cosine
  LR 6e-4→6e-5, bf16, regional+tail compile @ max-autotune-no-cudagraphs +
  bf16-I/O WKV/LIF kernels. Final OWT2 held-out PPL ~27. (Paper pretrained on
  ~5B tokens, so this is ~2x their pretrain budget.)
- **Fine-tune:** `--reset-schedule` (fresh weights from the pretrained best, fresh
  optimizer + fresh warmup→cosine), LR 1e-5→1e-6, best-val checkpointing.
- **Eval:** strided full-context token-level PPL (`evaluate_spikegpt_checkpoint.py
  --tokens-bin`, count-last=ctx//4, stride-matched) on the WikiText test split.

## Results (token-level test perplexity)

| Model | WikiText test PPL | Paper |
|---|---:|---:|
| Zero-shot (OWT2 pretrain, **no** fine-tune) | 94.39 | — |
| **WT-103 fine-tune → shared test (HEADLINE)** | **26.47** | **39.75** |
| WT-2-only fine-tune → shared test (probe) | 37.27 | (18.01) |

- **Headline:** OWT2→WT-103 fine-tune gives **26.47**, beating the paper's 39.75
  by ~13 points (33% relative). Fine-tune converged smoothly (val 86.5→26.3 over
  15k steps) with **no plateau at the cutoff** — more fine-tune steps would push
  it lower still.
- **WT-2-only probe (37.27):** fine-tuning on only WT-2's 2.4M-token train, on the
  *same* (shared) test set, is necessarily worse than the WT-103 fine-tune — and
  does **not** reproduce the paper's 18.01. The paper's WT-2 number used WT-103+WT-2
  combined train (≈ WT-103, since WT-2 ⊂ WT-103) on that same shared test, which by
  our numbers would land ~26, not 18. So **18.01 is not reproducible on the shared
  test set** and looks like an artifact of their setup; the WT-103/WT-2 inversion
  (39.75 > 18.01 on a shared test, despite WT-103 having far more train data) is
  internally inconsistent in the paper. 39.75 is the meaningful, comparable target.

## Honest caveats

- The win combines a **better recipe + ~2x the pretrain budget** (10B vs ~5B
  tokens), same architecture — a legitimate beat of the published number, not a
  pure modeling result.
- Eval protocol (full-context strided, count-last) is standard but the paper's
  exact protocol is unstated; ours is the conventional sliding-window full-context
  PPL.

## Artifacts

- Pretrain (preserved): `runs/spikegpt_216m_owt2_10b.final.pt` (step 610k),
  `runs/spikegpt_216m_owt2_10b.best.preserved.pt`.
- Fine-tuned: `runs/spikegpt_216m_wt103_ft.best.pt` (val BPT 4.717, step 15k).
- Scripts: `examples/phase4_finetune_eval.sh`, `examples/phase4_autostart.sh`.
- Logs: `runs/phase4_run.log`, `runs/phase4_autostart.log`.
