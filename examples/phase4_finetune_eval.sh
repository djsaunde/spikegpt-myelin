#!/usr/bin/env bash
# Phase 4: fine-tune the OWT2-pretrained 216M SpikeGPT on WikiText, then report
# token-level test perplexity vs the paper (SpikeGPT Table 3, "With Pre-training":
# WT-103 test 39.75, WT-2 test 18.01; both fine-tuned, token-level, NeoX BPE).
#
# IMPORTANT protocol facts (verified against arXiv:2302.13939 + ridgerchu/SpikeGPT
# + our own data):
#   * Tokenizer MATCHES: paper uses GPT-NeoX-20B BPE (vocab 50277); so do we ->
#     token-level PPL is directly comparable.
#   * WikiText-103 and WikiText-2 share a BYTE-IDENTICAL test set (and val set) --
#     confirmed: `cmp data/wikitext103/test.bin data/wikitext2/test.bin` == equal.
#     The paper's WT-103 (39.75) and WT-2 (18.01) numbers are on the SAME test set
#     with ~the same fine-tune corpus (their WT-2 row fine-tunes on WT-103+WT-2,
#     and WT-2 train is a subset of WT-103 train), so the 2.2x split is internally
#     inconsistent -- 39.75 looks undertrained. THE REAL TARGET IS WT-103 = 39.75.
#     We produce ONE honest number (WT-103 fine-tune -> shared test); we cannot
#     reproduce the 39.75/18.01 split because the test set is identical.
#   * Paper fine-tune LR is ~3e-6 (repo README: "smaller LR than pre-training to
#     avoid catastrophic forgetting"; pretrain was 4e-4). We use a slightly higher
#     but still conservative 1e-5->1e-6 and rely on best-val checkpointing; SWEEP
#     {3e-6, 1e-5, 3e-5} at run time if the first pass under/over-fits.
#   * Our OWT2 pretrain budget (10B tokens) is ~2x the paper's (~5B).
#
# Fine-tune uses --reset-schedule: load only the model weights from the
# pretrained checkpoint, reset previous_steps=0 (fresh warmup->cosine over
# --steps) and start with a fresh optimizer. Runs are short, so no resume
# wrapper (and --reset-schedule + a fixed pretrained --checkpoint-in would
# restart from pretrain on any resume anyway).
set -euo pipefail
cd "$(dirname "$0")/.."

PRETRAINED="${PRETRAINED:-runs/spikegpt_216m_owt2_10b.best.pt}"  # final Phase-3 best
COMMON_TRAIN=(
  --device cuda
  --reset-schedule
  --val-holdout-tokens 0          # 0 => use --val-bin (else a train tail is held out)
  --batch 16
  --weight-decay 0.1 --grad-clip 1.0 --dropout 0.0
  --amp bf16 --matmul-precision high
  --compile regional --compile-tail --compile-mode max-autotune-no-cudagraphs --compile-warmup
  --wandb --wandb-project myelin
)

if [[ ! -f "$PRETRAINED" ]]; then
  echo "ERROR: pretrained checkpoint not found: $PRETRAINED" >&2
  echo "Set PRETRAINED=<path to Phase-3 best> and re-run." >&2
  exit 1
fi

# --- STEP 0: zero-shot baseline (de-risk) -------------------------------------
# Eval the pretrained (NOT fine-tuned) checkpoint on the WikiText test set BEFORE
# fine-tuning. Three purposes: (1) the "OWT2 pretrain, no fine-tune" baseline so we
# can measure the fine-tune delta; (2) validates the standalone eval harness on the
# real 216M BPE model before we depend on it; (3) the catastrophic-forgetting
# tripwire -- if a fine-tune's first val is WORSE than this, the LR is too hot.
echo "=== STEP 0: zero-shot WikiText test PPL (pretrained, no fine-tune) ==="
uv run python examples/evaluate_spikegpt_checkpoint.py "$PRETRAINED" \
  --tokens-bin data/wikitext103/test.bin --device cuda --batch 16

# --- HEADLINE: WikiText-103 fine-tune (118.7M train tokens; ~7.2k steps/epoch) -
# This is the real target (vs paper 39.75). Conservative LR + best-val checkpoint.
uv run python examples/train_tiny_spikegpt.py "${COMMON_TRAIN[@]}" \
  --checkpoint-in "$PRETRAINED" \
  --train-bin data/wikitext103/train.bin \
  --val-bin data/wikitext103/validation.bin \
  --steps 15000 --lr 1e-5 --lr-final 1e-6 --warmup-steps 200 --lr-schedule cosine \
  --log-every 200 --eval-every 500 --val-eval-tokens 250000 \
  --checkpoint-out runs/spikegpt_216m_wt103_ft.pt \
  --best-checkpoint-out runs/spikegpt_216m_wt103_ft.best.pt --checkpoint-every 2000 \
  --wandb-run-name wt103_216m_ft

# --- SECONDARY: WikiText-2-only fine-tune (2.4M train tokens; overfits fast) ---
# Data-scaling probe. NOTE the test set is identical to WT-103's, so on less train
# data this will be WORSE than the WT-103 run, not better -- it does NOT reproduce
# the paper's 18.01 (their WT-2 row fine-tuned on WT-103+WT-2 combined). Kept to
# show the effect of train-set size on the same eval.
uv run python examples/train_tiny_spikegpt.py "${COMMON_TRAIN[@]}" \
  --checkpoint-in "$PRETRAINED" \
  --train-bin data/wikitext2/train.bin \
  --val-bin data/wikitext2/validation.bin \
  --steps 800 --lr 1e-5 --lr-final 1e-6 --warmup-steps 50 --lr-schedule cosine \
  --log-every 25 --eval-every 50 --val-eval-tokens 250000 \
  --checkpoint-out runs/spikegpt_216m_wt2_ft.pt \
  --best-checkpoint-out runs/spikegpt_216m_wt2_ft.best.pt --checkpoint-every 200 \
  --wandb-run-name wt2_216m_ft

# --- Token-level test perplexity (strided, full-context) vs paper -------------
# Both eval on the SAME (shared) WikiText test set; difference is only the model.
echo "=== WT-103 fine-tune -> shared test PPL (HEADLINE; paper WT-103: 39.75) ==="
uv run python examples/evaluate_spikegpt_checkpoint.py runs/spikegpt_216m_wt103_ft.best.pt \
  --tokens-bin data/wikitext103/test.bin --device cuda --batch 16

echo "=== WT-2-only fine-tune -> shared test PPL (probe; paper WT-2: 18.01) ==="
uv run python examples/evaluate_spikegpt_checkpoint.py runs/spikegpt_216m_wt2_ft.best.pt \
  --tokens-bin data/wikitext2/test.bin --device cuda --batch 16
