"""Streamlit playground for trained SpikeGPT checkpoints.

Run:  uv run --extra app streamlit run examples/app/spikegpt_playground.py

Only curated, known-good checkpoints from model_registry.json are offered (the
runs/ directory is full of scratch checkpoints we never want to expose). Config
and parameter counts are read from the checkpoint itself; the registry only
curates which models appear and their measured BPC.
"""

from __future__ import annotations

import html
import json
import math
import re
import time
from pathlib import Path
from typing import cast

import streamlit as st
import torch

from spikegpt import (
    BPEVocabulary,
    ByteVocabulary,
    CharacterVocabulary,
    SamplingMode,
    collect_spike_statistics,
    load_spike_language_checkpoint,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = Path(__file__).resolve().parent / "model_registry.json"


@st.cache_data
def load_registry() -> list[dict]:
    """Curated entries whose checkpoint file actually exists, in registry order."""
    entries = json.loads(REGISTRY_PATH.read_text())
    return [e for e in entries if (REPO_ROOT / e["path"]).exists()]


@st.cache_resource(show_spinner="Loading checkpoint…")
def load_model(rel_path: str, device: str):
    checkpoint = load_spike_language_checkpoint(REPO_ROOT / rel_path, map_location=device)
    model = checkpoint.model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    return model, checkpoint.vocabulary, checkpoint.metadata, model.config, n_params


def surprisal_bits(model, ids: torch.Tensor, prompt_len: int) -> list[tuple[int, float]]:
    """Per-token (token_id, bits) for the generated continuation under the model."""
    with torch.no_grad():
        logits = model(ids)
    logp = torch.log_softmax(logits[0].float(), dim=-1)  # [T, V]
    out: list[tuple[int, float]] = []
    for pos in range(prompt_len, ids.shape[1]):
        tok = int(ids[0, pos])
        bits = -logp[pos - 1, tok].item() / math.log(2.0)
        out.append((tok, bits))
    return out


def byte_glyph(tok: int) -> str:
    if tok == 10:
        return "⏎\n"
    if 32 <= tok < 127:
        return html.escape(chr(tok))
    return "·"


def token_glyph(tok: int, vocab) -> str:
    """Display glyph for one token, vocab-aware: a raw byte for byte models, the
    decoded character/sub-word piece for char/BPE models (so the surprisal heatmap
    is readable for the 216M BPE models, not just the byte-level ones)."""
    if isinstance(vocab, ByteVocabulary):
        return byte_glyph(tok)
    piece = vocab.decode(torch.tensor([tok]))
    if piece == "":
        return "∅"
    return html.escape(piece).replace("\n", "⏎\n")


def detokenize_wikitext(text: str) -> str:
    """Reverse WikiText's Moses/PTB surface format for readable display.

    WikiText stores punctuation space-separated (" ,"), splits contractions
    (" 's", " n't"), and encodes hyphens / decimals / thousands with the @-@ /
    @.@ / @,@ sentinels (e.g. "Blu @-@ ray", "22 @.@ 5", "1 @,@ 000"). This
    undoes the common, unambiguous cases; double quotes are left as-is (open vs
    close is ambiguous)."""
    text = text.replace(" @-@ ", "-").replace("@-@", "-")
    text = text.replace(" @.@ ", ".").replace("@.@", ".")
    text = text.replace(" @,@ ", ",").replace("@,@", ",")
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)  # no space before sentence punctuation
    text = re.sub(r"\s+'(s|re|ve|ll|d|m)\b", r"'\1", text)  # possessives / contractions
    text = re.sub(r"\s+n't\b", "n't", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)  # no space after opening bracket
    text = re.sub(r"\s+([)\]])", r"\1", text)  # no space before closing bracket
    return _pair_double_quotes(text)


def _pair_double_quotes(text: str) -> str:
    """Attach spaced double quotes by alternating open/close (the Moses /
    sacremoses / NLTK detokenizer heuristic): the 1st/3rd/… quote is opening
    (drop the space after it), the 2nd/4th/… is closing (drop the space before).
    Mis-pairs only on unbalanced quotes; graceful either way."""
    parts = text.split('"')
    if len(parts) == 1:
        return text
    out = parts[0]
    for i, part in enumerate(parts[1:], start=1):
        # odd quote = opening (drop the space after); even = closing (drop the space before)
        out = out + '"' + part.lstrip(" ") if i % 2 == 1 else out.rstrip(" ") + '"' + part
    return out


def bits_color(bits: float, vmax: float = 8.0) -> str:
    # green (confident, low bits) -> red (surprised, high bits)
    f = max(0.0, min(1.0, bits / vmax))
    r, g = int(40 + 200 * f), int(180 - 120 * f)
    return f"background-color: rgba({r},{g},60,0.30)"


st.set_page_config(page_title="SpikeGPT playground", layout="wide")
st.title("⚡ SpikeGPT playground")

registry = load_registry()
if not registry:
    st.error(f"No known-good checkpoints found. Checked {REGISTRY_PATH} against {REPO_ROOT}/runs.")
    st.stop()

with st.sidebar:
    st.header("Model")
    labels = [f"{e['name']}  —  {e['metric']}" for e in registry]
    idx = st.selectbox("Checkpoint", range(len(registry)), format_func=lambda i: labels[i])
    entry = registry[idx]
    cuda_ok = torch.cuda.is_available()
    # Default to the GPU when one is present; only offer CPU as a fallback (and
    # as the only option when no CUDA device exists).
    device = st.radio(
        "Device",
        ["cuda", "cpu"] if cuda_ok else ["cpu"],
        horizontal=True,
        help="Defaults to the GPU when available. CPU is the fallback (and the only "
        "option with no CUDA device); long generations are much slower on it.",
    )

    model, vocab, metadata, config, n_params = load_model(entry["path"], device)

    st.header("Sampling")
    ctx_len = config.context_length
    max_new = st.slider("Max new tokens", 8, ctx_len, min(128, ctx_len), 8)
    sampling = st.radio("Strategy", ["multinomial", "greedy"], horizontal=True)
    temperature_disabled = sampling == "greedy"
    temperature = st.slider("Temperature", 0.1, 2.0, 0.9, 0.05, disabled=temperature_disabled)
    top_k = st.slider("Top-k", 1, 256, 40, 1, disabled=temperature_disabled)
    seed = st.number_input("Seed", value=0, step=1)

    st.header("Display")
    render_entities = st.checkbox(
        "Render Wikipedia entities",
        value=True,
        help="enwik8 stores quotes/brackets as HTML entities (&quot;, &lt;, …). "
        "Decode them to readable characters in the generation panel. The per-byte "
        "surprisal heatmap below always shows the raw bytes the model produced.",
    )
    if isinstance(vocab, BPEVocabulary):
        detok_wikitext = st.checkbox(
            "De-tokenize WikiText (readable punctuation)",
            value=True,
            help="WikiText stores punctuation space-separated (' ,' ' .'), splits "
            'contractions (" \'s", " n\'t"), and uses @-@/@.@/@,@ for hyphens/decimals/'
            "thousands. Clean those up in the generation panel for readability. The "
            "surprisal heatmap below always shows the raw tokens the model produced.",
        )
    else:
        detok_wikitext = False

c1, c2, c3, c4 = st.columns(4)
c1.metric("Dataset", entry["dataset"])
c2.metric("Quality", entry["metric"].split("(")[0].strip())
c3.metric("Config", f"{config.n_layer}L · {config.n_embd}d · ctx {config.context_length}")
c4.metric("Params", f"{n_params / 1e6:.1f}M")
st.caption(entry["blurb"] + f"  ·  vocab={vocab.size}  ·  type={config.model_type}")

if "shakespeare" in entry["dataset"].lower():
    default_prompt = "ROMEO:"
elif isinstance(vocab, BPEVocabulary):
    default_prompt = "The history of"  # NB: no trailing space (see below)
else:
    default_prompt = "The "
prompt = st.text_area("Prompt", value=default_prompt, height=90)

if st.button("Generate", type="primary"):
    if not prompt:
        st.warning("Enter a prompt.")
        st.stop()
    # BPE tokenizers attach a space to the *following* word (" the" is one token),
    # so a prompt ending in a bare space is an out-of-distribution state that
    # derails generation into junk. Strip trailing spaces for BPE models.
    if isinstance(vocab, BPEVocabulary) and prompt != prompt.rstrip(" "):
        prompt = prompt.rstrip(" ")
        st.caption(
            "ℹ️ Trailing space removed: BPE attaches spaces to the next token, so a "
            "prompt ending in a space produces garbage. Generating from the trimmed prompt."
        )
    prompt_ids = vocab.encode(prompt).unsqueeze(0).to(device)
    torch_device = torch.device(device)
    plen = prompt_ids.shape[1]

    def _sync() -> None:
        if torch_device.type == "cuda":
            torch.cuda.synchronize()

    gen_kwargs = {
        "temperature": float(temperature),
        "top_k": int(top_k),
        "sampling": cast(SamplingMode, sampling),
    }

    st.subheader("Generation")
    gen_box = st.empty()

    def render_generation(continuation: str) -> None:
        # Optionally decode enwik8's HTML entities (&quot;, &lt;, …) for readability.
        shown_prompt = html.unescape(prompt) if render_entities else prompt
        shown_cont = html.unescape(continuation) if render_entities else continuation
        # Optionally undo WikiText's Moses surface format (spaced punctuation, @-@ …).
        if detok_wikitext:
            shown_prompt = detokenize_wikitext(shown_prompt)
            shown_cont = detokenize_wikitext(shown_cont)
        gen_box.markdown(
            f"<div style='font-family:monospace;white-space:pre-wrap;border:1px solid #ddd;"
            f"padding:10px;border-radius:6px'>"
            f"<span style='color:#888'>{html.escape(shown_prompt)}</span>"
            f"<span style='font-weight:600'>{html.escape(shown_cont)}</span>▌</div>",
            unsafe_allow_html=True,
        )

    render_generation("")  # show the prompt immediately

    # Warm the CUDA kernels with a throwaway token so TTFT / throughput are not
    # skewed by one-time compilation.
    torch.manual_seed(int(seed))
    next(model.generate_stream(prompt_ids, max_new_tokens=1, **gen_kwargs), None)

    # Stream the generation, updating the panel as tokens arrive. Keep tokens on
    # the device and only sync/decode at render time (deferred GPU->CPU copy lets
    # the decode loop run without a per-token host stall), and throttle the UI to
    # ~16 fps so re-rendering never bottlenecks the stream. Re-seed so the shown
    # output is reproducible.
    torch.manual_seed(int(seed))
    token_tensors: list[torch.Tensor] = []  # each [1, 1] on device
    ttft = 0.0
    render_interval = 0.06
    last_render = 0.0
    _sync()
    start = time.perf_counter()
    for i, token in enumerate(
        model.generate_stream(prompt_ids, max_new_tokens=int(max_new), **gen_kwargs)
    ):
        if i == 0:
            _sync()
            ttft = time.perf_counter() - start
        token_tensors.append(token)
        now = time.perf_counter()
        if now - last_render >= render_interval:
            ids_so_far = torch.cat(token_tensors, dim=1)[0].cpu()
            render_generation(vocab.decode(ids_so_far))
            last_render = now
    _sync()
    gen_time = time.perf_counter() - start
    tokens_per_s = len(token_tensors) / gen_time if gen_time > 0 else 0.0

    generated = torch.cat(token_tensors, dim=1)
    out = torch.cat([prompt_ids, generated], dim=1)
    render_generation(vocab.decode(generated[0].cpu()))  # final full render

    # The post-generation analyses call model.forward, which is capped at the
    # context length, so score the last context_length tokens when a long
    # generation overflows the window (no-op for shorter samples).
    view = out[:, -ctx_len:]
    view_plen = max(0, plen - (out.shape[1] - view.shape[1]))

    p1, p2, p3 = st.columns(3)
    p1.metric(
        "Time to first token",
        f"{ttft * 1000:.0f} ms",
        help="Prompt prefill + first decoded token, on the selected device.",
    )
    # Byte-level models step one byte per token; char-level one char. Name the
    # unit precisely (a multi-byte UTF-8 char is several byte-tokens, so for a
    # byte model "chars/s" would be wrong); fall back to the generic "tokens".
    if isinstance(vocab, ByteVocabulary):
        unit = "bytes"
    elif isinstance(vocab, CharacterVocabulary):
        unit = "chars"
    else:
        unit = "tokens"
    p2.metric(
        "Throughput",
        f"{tokens_per_s:.1f} {unit}/s",
        help=f"{len(token_tensors)} {unit} / {gen_time:.2f}s end-to-end on {device}.",
    )
    p3.metric("Total generation", f"{gen_time:.2f} s")

    # SNN-distinctive readout: spike statistics over the generated sequence.
    stats = collect_spike_statistics(
        model, view[0].cpu(), context_length=view.shape[1], batch_size=1, max_windows=1
    )
    pops = stats.populations
    block_pops = [p for k, p in pops.items() if k != "embedding"]
    mean_block = sum(p.density for p in block_pops) / len(block_pops) if block_pops else 0.0
    dead_total = sum(p.dead_count for p in block_pops)
    sat_total = sum(p.saturated_count for p in block_pops)
    block_neurons = sum(p.num_channels for p in block_pops)
    dead_sat_pct = (dead_total + sat_total) / block_neurons * 100 if block_neurons else 0.0
    s1, s2, s3 = st.columns(3)
    emb = pops["embedding"].density if "embedding" in pops else float("nan")
    s1.metric("Embedding spike rate", f"{emb * 100:.1f}%")
    s2.metric(
        "Mean block spike rate",
        f"{mean_block * 100:.1f}%",
        help="Fraction of neurons firing — sparsity is the point of an SNN.",
    )
    s3.metric(
        "Dead / saturated / total",
        f"{dead_total} / {sat_total} / {block_neurons:,}",
        f"{dead_sat_pct:.1f}% of block neurons",
        delta_color="off",
        help="Block neurons that never fired / fired on (nearly) every token "
        f"**in this {view.shape[1]}-token sample** / total block neurons "
        f"(2 LIF populations × {config.n_layer} layers × {config.n_embd} channels). "
        "On a short sample a dead neuron usually just had few chances to fire; "
        "run examples/analyze_spikegpt_spikes.py over a corpus for the true "
        "dead-neuron count (~0.1% for the ctx-3072 model).",
    )

    with st.expander("Spiking analysis — per-layer, per-position, neuron health"):
        profile = stats.per_layer_profile()
        if profile:
            st.caption("Firing density by depth (fraction of neurons firing per step)")
            st.bar_chart(
                {
                    "time (lif1)": [t for _, t, _ in profile],
                    "channel (lif2)": [c for _, _, c in profile],
                }
            )
        st.caption("Firing density by token position in the sequence")
        total_ch = sum(p.num_channels for p in pops.values())
        mean_pos = [
            sum(p.per_position_rate[t].item() * p.num_channels for p in pops.values()) / total_ch
            for t in range(stats.context_length)
        ]
        st.line_chart(mean_pos)
        st.caption("Per-population density and neuron health")
        st.table(
            {
                "population": list(pops),
                "neurons": [p.num_channels for p in pops.values()],
                "density %": [f"{p.density * 100:.1f}" for p in pops.values()],
                "dead": [f"{p.dead_count} ({p.dead_fraction * 100:.1f}%)" for p in pops.values()],
                "saturated": [
                    f"{p.saturated_count} ({p.saturated_fraction * 100:.1f}%)"
                    for p in pops.values()
                ],
            }
        )

    # per-token surprisal heatmap over the continuation
    unit_singular = unit[:-1] if unit.endswith("s") else unit
    st.subheader(f"Per-{unit_singular} surprisal (bits) — green = confident, red = surprised")
    pieces = []
    for tok, bits in surprisal_bits(model, view, view_plen):
        pieces.append(
            f"<span title='{bits:.2f} bits' style='{bits_color(bits)};padding:1px'>"
            f"{token_glyph(tok, vocab)}</span>"
        )
    st.markdown(
        "<div style='font-family:monospace;white-space:pre-wrap;line-height:1.8'>"
        + "".join(pieces)
        + "</div>",
        unsafe_allow_html=True,
    )
    bits_only = [b for _, b in surprisal_bits(model, view, view_plen)]
    if bits_only:
        bits_name = "BPC" if unit in ("bytes", "chars") else "BPT"
        st.caption(
            f"mean {sum(bits_only) / len(bits_only):.2f} bits/{unit_singular} over the "
            f"{len(bits_only)} generated {unit} (≈ {bits_name} of this sample)"
        )
