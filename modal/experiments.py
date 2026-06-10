"""Modal harness for GPU experiments while the local 5090 is busy.

Targets RTX-PRO-6000 (Blackwell, sm_120) — the same compute capability as our
local RTX 5090 — so step-time / throughput numbers are directly comparable. The
image replicates our exact uv environment (torch 2.13 nightly cu130 + Triton),
so the Triton WKV/LIF kernels and torch.compile behave identically to local.

Run:
  uv run --extra cloud modal run modal/experiments.py::gpu_info
  uv run --extra cloud modal run modal/experiments.py::compile_mode_ab
"""

from __future__ import annotations

import subprocess

import modal

APP_NAME = "myelin-experiments"
REMOTE = "/root/myelin"
VENV_PY = "/opt/venv/bin/python"
GPU = "RTX-PRO-6000"  # sm_120, same architecture as the local RTX 5090

# Dep layer (cached unless pyproject/uv.lock change), then code, then install project.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("uv")
    .env({"UV_PROJECT_ENVIRONMENT": "/opt/venv"})
    .add_local_file("pyproject.toml", f"{REMOTE}/pyproject.toml", copy=True)
    .add_local_file("uv.lock", f"{REMOTE}/uv.lock", copy=True)
    .add_local_file(".python-version", f"{REMOTE}/.python-version", copy=True)
    .add_local_file("README.md", f"{REMOTE}/README.md", copy=True)
    .run_commands(f"cd {REMOTE} && uv sync --frozen --extra cuda --no-install-project")
    .add_local_dir("src", f"{REMOTE}/src", copy=True)
    .add_local_dir("examples", f"{REMOTE}/examples", copy=True)
    .add_local_file("data/enwik8", f"{REMOTE}/data/enwik8", copy=True)
    .run_commands(f"cd {REMOTE} && uv sync --frozen --extra cuda")
)

app = modal.App(APP_NAME, image=image)

# Persistent storage for downloadable artifacts (profiler traces, etc.).
traces = modal.Volume.from_name("myelin-traces", create_if_missing=True)


def _run(code: str) -> None:
    """Run a python snippet in the replicated uv venv and stream its output."""
    subprocess.run([VENV_PY, "-c", code], cwd=REMOTE, check=True)


@app.function(gpu=GPU, timeout=20 * 60)
def gpu_info() -> None:
    """Smoke test: confirm the GPU is sm_120 and the nightly stack imports."""
    _run(
        "import torch, triton\n"
        "print('torch', torch.__version__, '| triton', triton.__version__)\n"
        "print('device', torch.cuda.get_device_name())\n"
        "print('capability', torch.cuda.get_device_capability())\n"
        "print('arch_list', torch.cuda.get_arch_list())\n"
        "from spikegpt.wkv_triton import weighted_key_value_triton\n"
        "import torch as t\n"
        "k=t.randn(2,32,128,device='cuda'); v=t.randn(2,32,128,device='cuda')\n"
        "td=t.randn(128,device='cuda'); tf=t.randn(128,device='cuda')\n"
        "y=weighted_key_value_triton(k,v,td,tf); t.cuda.synchronize()\n"
        "print('triton WKV ok, out', tuple(y.shape))"
    )


@app.function(gpu=GPU, timeout=30 * 60)
def lif_bf16_bench() -> None:
    """Validate + benchmark the prototype bf16-I/O LIF vs the fp32-I/O kernel."""
    _run(
        "import time, torch\n"
        "from spikegpt.language import SpikingSequenceLIF\n"
        "from myelin.neurons import LIFParams, LIFState\n"
        "from myelin.triton.lif_bf16 import surrogate_lif_bf16io\n"
        "dev='cuda'; B,T,C=64,1024,512\n"
        "# --- parity: feed bf16-rounded inputs to both; isolates the bf16-storage effect\n"
        "torch.manual_seed(0)\n"
        "xb=torch.randn(B,T,C,device=dev,dtype=torch.bfloat16)\n"
        "xf=xb.float().clone().requires_grad_(True)\n"
        "ref=SpikingSequenceLIF(tau=2.0,threshold=1.0,surrogate_slope=2.0,fused=False).to(dev)\n"
        "sp_ref=ref(xf); gy=torch.randn_like(sp_ref); sp_ref.backward(gy)\n"
        "xp=xb.clone().requires_grad_(True)\n"
        "cur=xp.movedim(1,0).contiguous()\n"
        "init=LIFState(membrane=torch.zeros(B,C,device=dev,dtype=torch.bfloat16))\n"
        "sp_p=surrogate_lif_bf16io(cur,init,LIFParams(tau_mem=2.0,threshold=1.0,reset=0.0)).movedim(0,1)\n"
        "sp_p.backward(gy.bfloat16())\n"
        "spike_match=(sp_p.float()==sp_ref).float().mean().item()\n"
        "gref=xf.grad; gp=xp.grad.float()\n"
        "rel=((gp-gref).abs().max()/(gref.abs().max()+1e-9)).item()\n"
        "print(f'parity: spikes match {spike_match*100:.2f}%  grad rel-err {rel:.3e}')\n"
        "# --- bench fwd+bwd: bf16-I/O prototype vs fp32-I/O production kernel\n"
        "def bench(make, x):\n"
        "    for _ in range(5): _do(make,x)\n"
        "    torch.cuda.synchronize(); t0=time.perf_counter()\n"
        "    for _ in range(30): _do(make,x)\n"
        "    torch.cuda.synchronize(); return (time.perf_counter()-t0)/30*1e3\n"
        "def _do(make,x):\n"
        "    xi=x.clone().requires_grad_(True); s=make(xi); s.sum().backward()\n"
        "fp32=SpikingSequenceLIF(tau=2.0,threshold=1.0,surrogate_slope=2.0,fused=True).to(dev)\n"
        "x32=torch.randn(B,T,C,device=dev)\n"
        "def proto(xi):\n"
        "    c=xi.movedim(1,0).contiguous()\n"
        "    i=LIFState(membrane=torch.zeros(B,C,device=dev,dtype=xi.dtype))\n"
        "    return surrogate_lif_bf16io(c,i,LIFParams(tau_mem=2.0,threshold=1.0,reset=0.0))\n"
        "ms32=bench(lambda xi: fp32(xi), x32)\n"
        "msbf=bench(proto, xb)\n"
        "print(f'LIF fwd+bwd  fp32-I/O {ms32:.3f} ms  |  "
        "bf16-I/O {msbf:.3f} ms  ({ms32/msbf:.2f}x)')"
    )


@app.function(gpu=GPU, timeout=30 * 60)
def wkv_io_cast_cost() -> None:
    """Scope the fp32-cast overhead in the *real* (Triton) WKV path on GPU.

    On CUDA the model uses ``weighted_key_value_triton`` (a [B,T,C]-native fused
    kernel — no transposes). But its wrapper materializes fp32 copies via
    ``key.float().contiguous()`` / ``value.float().contiguous()`` (fwd) plus three
    more on backward (grad_y, k, v) before the kernel reads them at 4 bytes/elem.
    A bf16-I/O WKV kernel (load bf16, cast to fp32 in-register, fp32 recurrence,
    store bf16 — same trick that gave the LIF ~1.5x) would drop those. This sizes
    the addressable cost: full WKV fwd+bwd vs. just the fp32 cast copies.
    """
    _run(
        "import time, torch\n"
        "from spikegpt.wkv_triton import weighted_key_value_triton\n"
        "dev='cuda'; B,T,C=16,1024,768  # 216M run dims\n"
        "torch.manual_seed(0)\n"
        "k=torch.randn(B,T,C,device=dev,dtype=torch.bfloat16)\n"
        "v=torch.randn(B,T,C,device=dev,dtype=torch.bfloat16)\n"
        "td=torch.randn(C,device=dev); tf=torch.randn(C,device=dev)\n"
        "gy=torch.randn(B,T,C,device=dev,dtype=torch.bfloat16)\n"
        "def bench(fn,n=50):\n"
        "    for _ in range(5): fn()\n"
        "    torch.cuda.synchronize(); t0=time.perf_counter()\n"
        "    for _ in range(n): fn()\n"
        "    torch.cuda.synchronize(); return (time.perf_counter()-t0)/n*1e3\n"
        "def full():\n"
        "    ki=k.clone().requires_grad_(True); vi=v.clone().requires_grad_(True)\n"
        "    out=weighted_key_value_triton(ki,vi,td,tf); out.backward(gy)\n"
        "def cast_only():\n"
        "    # the addressable copies: fwd k,v upcast + bwd grad_y,k,v upcast (5 total)\n"
        "    a=k.float().contiguous(); b=v.float().contiguous()\n"
        "    c=gy.float().contiguous(); d=k.float().contiguous(); e=v.float().contiguous()\n"
        "    return a,b,c,d,e\n"
        "ms_full=bench(full); ms_cast=bench(cast_only)\n"
        "print(f'WKV(Triton) fwd+bwd {ms_full:.3f} ms  |  fp32 cast copies {ms_cast:.3f} ms')\n"
        "print(f'addressable fp32 casts = {ms_cast/ms_full*100:.1f}% of WKV  '\n"
        "      f'(a bf16-I/O kernel also halves the kernel input bandwidth on top)')"
    )


@app.function(gpu=GPU, timeout=30 * 60)
def wkv_bf16io_validate() -> None:
    """Validate + benchmark the bf16-I/O WKV vs the fp32-I/O production kernel.

    Confirms (a) the output and all four grads (gk, gv, gtime_decay, gtime_first)
    are bit-identical to weighted_key_value_triton on bf16 inputs (fp32 in-register
    + fp32 replay scratch => no precision change), and (b) the fwd+bwd wall-time
    saved by dropping the .float().contiguous() materializations and halving the
    kernel's input bandwidth. At the 216M dims.
    """
    _run(
        "import time, torch\n"
        "from spikegpt.wkv_triton import weighted_key_value_triton\n"
        "from spikegpt.wkv_bf16 import weighted_key_value_triton_bf16io\n"
        "dev='cuda'; B,T,C=16,1024,768  # 216M run dims\n"
        "torch.manual_seed(0)\n"
        "k0=torch.randn(B,T,C,device=dev,dtype=torch.bfloat16)\n"
        "v0=torch.randn(B,T,C,device=dev,dtype=torch.bfloat16)\n"
        "td=torch.randn(C,device=dev,requires_grad=True)\n"
        "tf=torch.randn(C,device=dev,requires_grad=True)\n"
        "gy=torch.randn(B,T,C,device=dev,dtype=torch.bfloat16)\n"
        "def run(fn):\n"
        "    k=k0.clone().requires_grad_(True); v=v0.clone().requires_grad_(True)\n"
        "    td2=td.detach().clone().requires_grad_(True)\n"
        "    tf2=tf.detach().clone().requires_grad_(True)\n"
        "    out=fn(k,v,td2,tf2); out.backward(gy)\n"
        "    return out, k.grad, v.grad, td2.grad, tf2.grad\n"
        "o_ref,gk_ref,gv_ref,gtd_ref,gtf_ref=run(weighted_key_value_triton)\n"
        "o_bf,gk_bf,gv_bf,gtd_bf,gtf_bf=run(weighted_key_value_triton_bf16io)\n"
        "def cmp(name,a,b):\n"
        "    ex=torch.equal(a,b)\n"
        "    md=(a.float()-b.float()).abs().max().item()\n"
        "    print(f'  {name:10s} exact={ex}  max|d|={md:.3e}')\n"
        "print('(a) bf16-I/O vs fp32-I/O kernel (bit-parity):')\n"
        "cmp('out',o_ref,o_bf); cmp('grad_k',gk_ref,gk_bf); cmp('grad_v',gv_ref,gv_bf)\n"
        "cmp('grad_td',gtd_ref,gtd_bf); cmp('grad_tf',gtf_ref,gtf_bf)\n"
        "def bench(fn,n=50):\n"
        "    for _ in range(5): fn()\n"
        "    torch.cuda.synchronize(); t0=time.perf_counter()\n"
        "    for _ in range(n): fn()\n"
        "    torch.cuda.synchronize(); return (time.perf_counter()-t0)/n*1e3\n"
        "ms_ref=bench(lambda: run(weighted_key_value_triton))\n"
        "ms_bf=bench(lambda: run(weighted_key_value_triton_bf16io))\n"
        "print(f'(b) WKV fwd+bwd  fp32-I/O {ms_ref:.3f} ms  |  "
        "bf16-I/O {ms_bf:.3f} ms  ({ms_ref/ms_bf:.2f}x, -{(1-ms_bf/ms_ref)*100:.1f}%)')"
    )


@app.function(gpu=GPU, timeout=30 * 60)
def wkv_step_ab() -> None:
    """End-to-end 216M step: bf16-I/O WKV vs fp32-I/O WKV (eager, model dims).

    The op-level win is -32%; this measures what fraction of a real fwd+bwd+opt
    step that buys, by toggling the kernel the model dispatches to. Eager (the
    WKV is an opaque custom op, so compile doesn't change its cost; eager just
    inflates the non-WKV share, making this a conservative lower bound on the
    compiled step win the live run will see).
    """
    _run(
        "import time, torch\n"
        "import spikegpt.language as L\n"
        "from spikegpt.language import SpikeGPTConfig, SpikeLanguageModel\n"
        "from spikegpt.wkv_triton import weighted_key_value_triton\n"
        "from spikegpt.wkv_bf16 import weighted_key_value_triton_bf16io\n"
        "torch.set_float32_matmul_precision('high')\n"
        "dev='cuda'; B,T=16,1024\n"
        "cfg=SpikeGPTConfig(vocab_size=50277,context_length=T,n_layer=18,n_embd=768,dropout=0.0)\n"
        "torch.manual_seed(0); model=SpikeLanguageModel(cfg).to(dev)\n"
        "opt=torch.optim.AdamW(model.parameters(),lr=1e-4,fused=True)\n"
        "ids=torch.randint(0,50277,(B,T+1),device=dev); inp,tgt=ids[:,:T],ids[:,1:]\n"
        "def step():\n"
        "    opt.zero_grad(set_to_none=True)\n"
        "    with torch.autocast('cuda',dtype=torch.bfloat16):\n"
        "        loss,_=model(inp,tgt)\n"
        "    loss.backward(); opt.step()\n"
        "def bench():\n"
        "    for _ in range(5): step()\n"
        "    torch.cuda.synchronize(); ts=[]\n"
        "    for _ in range(20):\n"
        "        torch.cuda.synchronize(); t0=time.perf_counter(); step()\n"
        "        torch.cuda.synchronize(); ts.append((time.perf_counter()-t0)*1e3)\n"
        "    ts.sort(); return ts[len(ts)//2]\n"
        "# current default dispatches to bf16-I/O under autocast; force each in turn\n"
        "L.weighted_key_value_triton_bf16io=weighted_key_value_triton_bf16io\n"
        "ms_bf=bench()\n"
        "L.weighted_key_value_triton_bf16io=weighted_key_value_triton  # force fp32-I/O\n"
        "ms_f32=bench()\n"
        "print(f'216M eager step  fp32-I/O WKV {ms_f32:.2f} ms  |  "
        "bf16-I/O WKV {ms_bf:.2f} ms  ({ms_f32/ms_bf:.3f}x, -{(1-ms_bf/ms_f32)*100:.1f}%)')"
    )


@app.function(gpu=GPU, timeout=60 * 60)
def arch_ablation_ab() -> None:
    """3-way architecture A/B on enwik8 (byte-level): spiking vs continuous vs
    spike-input, same model/data/recipe. Answers what the spiking placement costs:

    * spiking      — canonical Linear->LIF (spikes added to a continuous residual).
    * continuous   — LIF gates removed (vanilla RWKV-v4; the upper bound on quality).
    * spike-input  — LIF->Linear (projections consume spikes; the AC-capable,
                     'hardware-faithful' placement that could earn the energy win).

    Small 6L/512d, 4k steps each; prints the last few val-eval rows per variant so
    the convergence ordering is directly readable (byte-level => Val BPC).
    """
    import subprocess

    common = [
        VENV_PY,
        "examples/train_tiny_spikegpt.py",
        "--device",
        "cuda",
        "--vocab",
        "byte",
        "--text-file",
        "data/enwik8",
        "--preset",
        "custom",
        "--layers",
        "6",
        "--embedding",
        "512",
        "--context-length",
        "1024",
        "--batch",
        "16",
        "--steps",
        "4000",
        "--lr",
        "1e-3",
        "--lr-final",
        "1e-4",
        "--warmup-steps",
        "200",
        "--amp",
        "bf16",
        "--matmul-precision",
        "high",
        "--compile",
        "regional",
        "--compile-mode",
        "max-autotune-no-cudagraphs",
        "--compile-warmup",
        "--eval-every",
        "500",
        "--eval-batches",
        "16",
        "--log-every",
        "1000",
    ]
    variants = [
        ("spiking (canonical Linear->LIF)", []),
        ("continuous (no spikes)", ["--no-spiking"]),
        ("spike-input (LIF->Linear)", ["--spike-input"]),
    ]
    for name, flags in variants:
        print(f"\n========== {name} ==========", flush=True)
        out = subprocess.run(common + flags, cwd=REMOTE, capture_output=True, text=True)
        rows = [ln for ln in out.stdout.splitlines() if ln.startswith("| ") and ln.count("|") >= 6]
        val_rows = [ln for ln in rows if ln.split("|")[3].strip()]  # Val Loss column non-empty
        if val_rows:
            print("Step | TrainLoss | ValLoss | ValBPC | ValPPL ... (last 4 val evals):")
            print("\n".join(val_rows[-4:]))
        else:
            print("no val rows parsed; stdout tail:\n", out.stdout[-1500:])
        if out.returncode != 0:
            print("returncode", out.returncode, "STDERR tail:\n", out.stderr[-1500:])


@app.function(gpu=GPU, timeout=50 * 60)
def ctx3072_sweep() -> None:
    """Batch sweep at ctx 3072 (12L/512d) — find tok/s + peak GB for the ctx-3072 repro.

    Peak GB also tells us the local 32GB ceiling (RTX-PRO-6000 here is 96GB).
    """
    _run(
        "import time, torch\n"
        "from spikegpt.language import SpikeGPTConfig, SpikeLanguageModel\n"
        "torch.set_float32_matmul_precision('high')\n"
        "dev='cuda'; T=3072\n"
        "cfg=SpikeGPTConfig(vocab_size=256,context_length=T,n_layer=12,n_embd=512,dropout=0.03)\n"
        "print(f'{torch.cuda.get_device_name()} | 12L/512d ctx{T} bf16 regional-compile')\n"
        "print('| batch | step ms | tok/s | peak GB | fits 32GB? |')\n"
        "print('|--:|--:|--:|--:|--|')\n"
        "for B in [64,72,80,88]:\n"
        "    torch.manual_seed(0); torch._dynamo.reset()\n"
        "    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()\n"
        "    try:\n"
        "        m=SpikeLanguageModel(cfg).to(dev).train()\n"
        "        for i,blk in enumerate(m.blocks): m.blocks[i]=torch.compile(blk)\n"
        "        opt=torch.optim.AdamW(m.parameters(),lr=1e-4)\n"
        "        ids=torch.randint(0,256,(B,T+1),device=dev); inp,tgt=ids[:,:T],ids[:,1:]\n"
        "        ts=[]\n"
        "        for i in range(10):\n"
        "            torch.cuda.synchronize(); t0=time.perf_counter()\n"
        "            opt.zero_grad(set_to_none=True)\n"
        "            with torch.autocast('cuda',dtype=torch.bfloat16): loss,_=m(inp,tgt)\n"
        "            loss.backward(); opt.step(); torch.cuda.synchronize()\n"
        "            if i>=4: ts.append((time.perf_counter()-t0)*1e3)\n"
        "        ts.sort(); ms=ts[len(ts)//2]; peak=torch.cuda.max_memory_allocated()/1e9\n"
        "        fits='yes' if peak<31 else 'NO'\n"
        "        print(f'| {B} | {ms:.0f} | {B * T / ms * 1000:,.0f} | "
        "{peak:.1f} | {fits} |', flush=True)\n"
        "        del m,opt\n"
        "    except torch.cuda.OutOfMemoryError:\n"
        "        print(f'| {B} | OOM | | >96 | NO |',flush=True); break\n"
    )


@app.function(gpu=GPU, timeout=90 * 60)
def ctx3072_lr_probe() -> None:
    """Light LR probe for the ctx-3072 / batch-72 repro: fixed-LR arms, ~1800 steps.

    Reuses the production trainer (--lr == --lr-final → constant LR). Brackets the
    tuned 2e-3 (batch-64/ctx-1024) against 4e-3, since batch-72/ctx-3072 averages
    ~3.4x more tokens/step (less gradient noise → likely tolerates a higher LR).
    """
    import io
    import os
    import urllib.request
    import zipfile

    if not os.path.exists("/tmp/enwik8"):
        data = urllib.request.urlopen("http://mattmahoney.net/dc/enwik8.zip", timeout=180).read()
        with open("/tmp/enwik8", "wb") as f:
            f.write(zipfile.ZipFile(io.BytesIO(data)).read("enwik8")[:60_000_000])
    for lr in ("2e-3", "4e-3"):
        print(f"=== ctx3072 batch72 fixed LR {lr} ===", flush=True)
        subprocess.run(
            [
                VENV_PY,
                f"{REMOTE}/examples/train_tiny_spikegpt.py",
                "--device",
                "cuda",
                "--text-file",
                "/tmp/enwik8",
                "--vocab",
                "byte",
                "--val-fraction",
                "0.05",
                "--context-length",
                "3072",
                "--layers",
                "12",
                "--embedding",
                "512",
                "--batch",
                "72",
                "--steps",
                "1800",
                "--lr",
                lr,
                "--lr-final",
                lr,
                "--warmup-steps",
                "200",
                "--weight-decay",
                "0.1",
                "--dropout",
                "0.03",
                "--compile",
                "regional",
                "--compile-mode",
                "default",
                "--compile-warmup",
                "--log-every",
                "100",
                "--eval-every",
                "600",
            ],
            cwd=REMOTE,
            check=True,
        )


@app.function(gpu=GPU, timeout=40 * 60)
def compile_scope_ab() -> None:
    """A/B regional (per-block) vs full-model torch.compile on the 12L step."""
    _run(
        "import time, torch\n"
        "from spikegpt.language import SpikeGPTConfig, SpikeLanguageModel\n"
        "torch.set_float32_matmul_precision('high')\n"
        "dev='cuda'; B,T=64,1024\n"
        "cfg=SpikeGPTConfig(vocab_size=256,context_length=T,n_layer=12,n_embd=512,dropout=0.03)\n"
        "def bench(scope):\n"
        "    torch.manual_seed(0); torch._dynamo.reset()\n"
        "    m=SpikeLanguageModel(cfg).to(dev).train()\n"
        "    if scope=='regional':\n"
        "        for i,blk in enumerate(m.blocks): m.blocks[i]=torch.compile(blk)\n"
        "        run=m\n"
        "    else:\n"
        "        run=torch.compile(m)\n"
        "    opt=torch.optim.AdamW(m.parameters(),lr=1e-4,fused=True)\n"
        "    ids=torch.randint(0,256,(B,T+1),device=dev); inp,tgt=ids[:,:T],ids[:,1:]\n"
        "    ts=[]\n"
        "    for i in range(14):\n"
        "        torch.cuda.synchronize(); t0=time.perf_counter()\n"
        "        opt.zero_grad(set_to_none=True)\n"
        "        with torch.autocast('cuda',dtype=torch.bfloat16): loss,_=run(inp,tgt)\n"
        "        loss.backward(); opt.step(); torch.cuda.synchronize()\n"
        "        if i>=4: ts.append((time.perf_counter()-t0)*1e3)\n"
        "    ts.sort(); ms=ts[len(ts)//2]; peak=torch.cuda.max_memory_allocated()/1e9\n"
        "    del m,opt; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()\n"
        "    return ms,peak,float(loss)\n"
        "print('12L/512d ctx1024 B64 bf16 default-mode:')\n"
        "print('| scope | step ms | peak GB | loss |'); print('|---|--:|--:|--:|')\n"
        "for scope in ['regional','full']:\n"
        "    try:\n"
        "        ms,peak,l=bench(scope)\n"
        "        print(f'| {scope} | {ms:.1f} | {peak:.2f} | {l:.3f} |',flush=True)\n"
        "    except Exception as e:\n"
        "        print(f'| {scope} | ERR {type(e).__name__}: {str(e)[:60]} | | |',flush=True)\n"
    )


@app.function(gpu=GPU, timeout=20 * 60)
def kernel_autotune() -> None:
    """Cheap occupancy A/B: WKV BLOCK and LIF block_size at the real shapes (fwd+bwd)."""
    _run(
        "import time, torch\n"
        "import spikegpt.wkv_triton as W\n"
        "from myelin.triton.lif_bf16 import surrogate_lif_bf16io\n"
        "from myelin.neurons import LIFParams, LIFState\n"
        "dev='cuda'; B,T,C=64,1024,512\n"
        "def timed(fn,n=40):\n"
        "    for _ in range(8): fn()\n"
        "    torch.cuda.synchronize(); t0=time.perf_counter()\n"
        "    for _ in range(n): fn()\n"
        "    torch.cuda.synchronize(); return (time.perf_counter()-t0)/n*1e3\n"
        "k=torch.randn(B,T,C,device=dev); v=torch.randn(B,T,C,device=dev)\n"
        "td=torch.randn(C,device=dev); tf=torch.randn(C,device=dev)\n"
        "def wkv():\n"
        "    a=[x.clone().requires_grad_(True) for x in (k,v,td,tf)]\n"
        "    W.weighted_key_value_triton(*a).sum().backward()\n"
        "print('WKV fwd+bwd (B64 T1024 C512):')\n"
        "for blk in [32,64,128]:\n"
        "    W._BLOCK=blk\n"
        "    print(f'  BLOCK={blk}: {timed(wkv):.3f} ms', flush=True)\n"
        "W._BLOCK=64\n"
        "cur=torch.randn(T,B,C,device=dev,dtype=torch.bfloat16)\n"
        "par=LIFParams(tau_mem=2.0,threshold=1.0,reset=0.0)\n"
        "def lif(bs):\n"
        "    def f():\n"
        "        xi=cur.clone().requires_grad_(True)\n"
        "        init=LIFState(membrane=torch.zeros(B,C,device=dev,dtype=torch.bfloat16))\n"
        "        surrogate_lif_bf16io(xi,init,par,block_size=bs).sum().backward()\n"
        "    return f\n"
        "print('LIF bf16-I/O fwd+bwd (T1024 B64 C512):')\n"
        "for bs in [128,256,512]:\n"
        "    print(f'  block_size={bs}: {timed(lif(bs)):.3f} ms', flush=True)\n"
    )


@app.function(gpu=GPU, timeout=20 * 60)
def integrated_bf16_smoke() -> None:
    """Confirm the integrated model takes the bf16-I/O LIF path end-to-end."""
    _run(
        "import torch\n"
        "from unittest.mock import patch\n"
        "import myelin.triton.lif_bf16 as m\n"
        "from spikegpt.language import SpikeGPTConfig, SpikeLanguageModel\n"
        "cfg=SpikeGPTConfig(vocab_size=256,context_length=256,n_layer=2,n_embd=128,dropout=0.0)\n"
        "model=SpikeLanguageModel(cfg).to('cuda').train()\n"
        "ids=torch.randint(0,256,(4,257),device='cuda')\n"
        "calls={'n':0}; orig=m.surrogate_lif_bf16io\n"
        "def spy(*a,**k): calls['n']+=1; return orig(*a,**k)\n"
        "with patch.object(m,'surrogate_lif_bf16io',spy), "
        "patch('spikegpt.language.surrogate_lif_bf16io',spy):\n"
        "    with torch.autocast('cuda',dtype=torch.bfloat16):\n"
        "        loss,_=model(ids[:,:256],ids[:,1:])\n"
        "    loss.backward()\n"
        "g=sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)\n"
        'print(f\'loss={float(loss):.3f}  grad_sum={g:.1f}  bf16-LIF calls={calls["n"]} '
        "(expect {cfg.n_layer*2})')"
    )


@app.function(gpu=GPU, timeout=60 * 60)
def lif_convergence_ab() -> None:
    """Train SpikeGPT twice (fp32 LIF vs bf16-I/O LIF) and compare val curves."""
    import io
    import os
    import urllib.request
    import zipfile

    if not os.path.exists("/tmp/enwik8"):
        data = urllib.request.urlopen("http://mattmahoney.net/dc/enwik8.zip", timeout=180).read()
        raw = zipfile.ZipFile(io.BytesIO(data)).read("enwik8")[:40_000_000]
        with open("/tmp/enwik8", "wb") as f:
            f.write(raw)
    for lif in ("fp32", "bf16"):
        subprocess.run(
            [
                VENV_PY,
                f"{REMOTE}/examples/lif_convergence_ab.py",
                "--text-file",
                "/tmp/enwik8",
                "--lif",
                lif,
                "--steps",
                "4000",
            ],
            cwd=REMOTE,
            check=True,
        )


@app.function(gpu=GPU, timeout=40 * 60)
def compile_mode_ab() -> None:
    """A/B torch.compile modes on the 12L step (the open throughput question)."""
    _run(
        "import time, torch\n"
        "from spikegpt.language import SpikeGPTConfig, SpikeLanguageModel\n"
        "torch.set_float32_matmul_precision('high')\n"
        "dev='cuda'; B,T=64,1024\n"
        "cfg=SpikeGPTConfig(vocab_size=256,context_length=T,n_layer=12,n_embd=512,dropout=0.03)\n"
        "def bench(mode):\n"
        "    torch.manual_seed(0); torch._dynamo.reset()\n"
        "    m=SpikeLanguageModel(cfg).to(dev).train()\n"
        "    kw = {} if mode=='default' else {'mode': mode}\n"
        "    for i,blk in enumerate(m.blocks):\n"
        "        m.blocks[i]=torch.compile(blk, **kw)\n"
        "    opt=torch.optim.AdamW(m.parameters(),lr=1e-4)\n"
        "    ids=torch.randint(0,256,(B,T+1),device=dev); inp,tgt=ids[:,:T],ids[:,1:]\n"
        "    ts=[]\n"
        "    for i in range(14):\n"
        "        torch.cuda.synchronize(); t0=time.perf_counter()\n"
        "        opt.zero_grad(set_to_none=True)\n"
        "        with torch.autocast('cuda',dtype=torch.bfloat16): loss,_=m(inp,tgt)\n"
        "        loss.backward(); opt.step(); torch.cuda.synchronize()\n"
        "        if i>=4: ts.append((time.perf_counter()-t0)*1e3)\n"
        "    ts.sort(); ms=ts[len(ts)//2]; peak=torch.cuda.max_memory_allocated()/1e9\n"
        "    del m,opt; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()\n"
        "    return ms,peak\n"
        "print(torch.cuda.get_device_name(),'| 12L/512d ctx1024 B64 bf16 regional-compile')\n"
        "print('| mode | step ms | tok/s | peak GB |'); print('|---|--:|--:|--:|')\n"
        "for mode in ['default','reduce-overhead','max-autotune-no-cudagraphs']:\n"
        "    try:\n"
        "        ms,peak=bench(mode)\n"
        "        print(f'| {mode} | {ms:.1f} | {B*T/ms*1000:,.0f} | {peak:.2f} |',flush=True)\n"
        "    except Exception as e:\n"
        "        print(f'| {mode} | ERR {type(e).__name__}: {str(e)[:50]} | | |',flush=True)"
    )


@app.function(gpu=GPU, timeout=40 * 60)
def wkv_throughput() -> None:
    """Deciding experiment: chunked/parallel matmul WKV vs the production Triton
    kernel at real shapes (C=512, T=1024/3072, B=12/24/64), bf16 then fp32."""
    for dtype in ("bf16", "fp32"):
        subprocess.run(
            [
                VENV_PY,
                "-m",
                "spikegpt.benchmarks.wkv_throughput",
                "--device",
                "cuda",
                "--batches",
                "12",
                "24",
                "64",
                "--channels",
                "512",
                "--timesteps",
                "1024",
                "3072",
                "--chunk-sizes",
                "32",
                "64",
                "128",
                "256",
                "--dtype",
                dtype,
                "--repeats",
                "20",
            ],
            cwd=REMOTE,
            check=True,
        )


@app.function(gpu=GPU, timeout=30 * 60, volumes={"/traces": traces})
def mfu_216m() -> None:
    """Accurate MFU + profiler op-breakdown for the 216M training step (sm_120,
    same arch as the local 5090, so the MFU% is representative). The Chrome trace
    is written to the persistent ``myelin-traces`` volume for download."""
    subprocess.run(
        [
            VENV_PY,
            "-m",
            "spikegpt.benchmarks.spikegpt_mfu",
            "--device",
            "cuda",
            "--preset",
            "gpt2-216m",
            "--batch",
            "16",
            "--amp",
            "bf16",
            "--compile",
            "regional",
            "--trace",
            "--trace-out",
            "/traces/spikegpt_216m_trace.json",
        ],
        cwd=REMOTE,
        check=True,
    )
    traces.commit()


@app.function(gpu=GPU, timeout=40 * 60)
def mfu_compile_ab() -> None:
    """A/B compile scope x mode for the 216M step: does max-autotune (and/or full
    compile of the head/CE tail) beat the default regional compile?"""
    mode = "max-autotune-no-cudagraphs"
    for scope, tail in (("regional", False), ("regional", True), ("full", False)):
        label = f"{scope}{'+tail' if tail else ''}"
        print(f"\n========== compile={label} mode={mode} ==========", flush=True)
        cmd = [
            VENV_PY,
            "-m",
            "spikegpt.benchmarks.spikegpt_mfu",
            "--device",
            "cuda",
            "--preset",
            "gpt2-216m",
            "--batch",
            "16",
            "--amp",
            "bf16",
            "--compile",
            scope,
            "--compile-mode",
            mode,
        ]
        if tail:
            cmd.append("--compile-tail")
        subprocess.run(cmd, cwd=REMOTE, check=True)
