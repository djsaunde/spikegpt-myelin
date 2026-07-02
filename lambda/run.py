#!/usr/bin/env python3
"""CLI for the Lambda Cloud experiment harness (VM counterpart to ``modal run``).

Stdlib-only; run with the system interpreter. Set ``LAMBDA_API_KEY`` first
(https://cloud.lambda.ai/api-keys). Usage walkthrough: ../ONBOARDING.md and
./README.md. Quick tour:

  python3 lambda/run.py types                 # instance types + capacity
  python3 lambda/run.py keys --add            # register your SSH pubkey (once)
  python3 lambda/run.py smoke                 # launch -> verify GPU/WKV -> terminate
  python3 lambda/run.py run -- <command>      # run anything in the synced env
  python3 lambda/run.py launch/provision/exec/fetch/terminate   # persistent-instance flow

One-shot smoke/run/train LAUNCH and TERMINATE on exit (even on error/Ctrl-C)
unless given --keep or --instance-id.
"""

from __future__ import annotations

import argparse
import base64
import sys

import harness as H

SMOKE_PY = """
import torch, triton
print('torch', torch.__version__, '| triton', triton.__version__)
print('device', torch.cuda.get_device_name())
print('capability', torch.cuda.get_device_capability())
from spikegpt.wkv_triton import weighted_key_value_triton
k = torch.randn(2, 32, 128, device='cuda'); v = torch.randn(2, 32, 128, device='cuda')
td = torch.randn(128, device='cuda'); tf = torch.randn(128, device='cuda')
y = weighted_key_value_triton(k, v, td, tf); torch.cuda.synchronize()
print('triton WKV ok, out', tuple(y.shape))
"""


def _py_c(src: str) -> str:
    """Quote-safe ``python -c`` (base64 payload survives the ssh/shell layers)."""
    b64 = base64.b64encode(src.encode()).decode()
    return f"python -c \"import base64; exec(base64.b64decode('{b64}').decode())\""


def _resolve_ip(client: H.LambdaClient, id_or_ip: str) -> str:
    if id_or_ip.count(".") == 3 and all(p.isdigit() for p in id_or_ip.split(".")):
        return id_or_ip
    inst = client.get_instance(id_or_ip)
    if inst.status != "active" or not inst.ip:
        raise H.LambdaError(f"instance {id_or_ip} is {inst.status} (ip {inst.ip})")
    return inst.ip


# --- read-only / lifecycle -------------------------------------------------- #
def cmd_types(a: argparse.Namespace) -> int:
    rows = []
    for name, info in sorted(H.LambdaClient().instance_types().items()):
        if a.gpu and a.gpu not in name:
            continue
        price = info["instance_type"].get("price_cents_per_hour")
        regions = [r["name"] for r in info["regions_with_capacity_available"]]
        rows.append((name, f"${price/100:.2f}/hr" if price is not None else "?",
                     ", ".join(regions) or "(no capacity)"))
    if not rows:
        print("no matching instance types")
        return 0
    w = max(len(r[0]) for r in rows)
    for name, price, cap in rows:
        print(f"{name:<{w}}  {price:>10}  {cap}")
    return 0


def cmd_keys(a: argparse.Namespace) -> int:
    client = H.LambdaClient()
    if a.add:
        print(f"registered/using ssh key: {client.ensure_ssh_key(a.key + '.pub')}")
    for k in client.list_ssh_keys():
        print(f"{k['name']}\t{' '.join(k['public_key'].split()[:2])[:50]}...")
    return 0


def cmd_ls(a: argparse.Namespace) -> int:
    instances = H.LambdaClient().list_instances()
    if not instances:
        print("no running instances")
        return 0
    for i in instances:
        cols = [i.id, f"{i.status:<10}", i.instance_type, i.region, i.ip or "-", i.name or ""]
        print("\t".join(str(c) for c in cols))
    return 0


def cmd_launch(a: argparse.Namespace) -> int:
    client = H.LambdaClient()
    region = client.pick_region(a.type, a.region)
    key_name = client.ensure_ssh_key(a.key + ".pub")
    ids = client.launch(instance_type=a.type, region=region, ssh_key_names=[key_name], name=a.name)
    try:  # a hung 'booting' instance bills silently — kill it if it never boots
        inst = client.wait_active(ids[0])
    except H.LambdaError:
        print(f"startup failed — terminating {ids[0]}", file=sys.stderr)
        try:
            client.terminate(ids)
        except Exception as e:
            print(f"WARNING: terminate failed: {e} — TERMINATE MANUALLY: {ids[0]}", file=sys.stderr)
        raise
    print(f"{inst.id}\t{inst.ip}")
    r = "lambda/run.py"
    print(f"\nnext:\n  python3 {r} provision {inst.id}\n"
          f"  python3 {r} exec {inst.id} -- python -m spikegpt.benchmarks.spikegpt_mfu\n"
          f"  python3 {r} terminate {inst.id}   # when done — stops billing", file=sys.stderr)
    return 0


def cmd_terminate(a: argparse.Namespace) -> int:
    client = H.LambdaClient()
    ids = [i.id for i in client.list_instances()] if a.all else a.ids
    if not ids:
        print("nothing to terminate (pass instance ids or --all)", file=sys.stderr)
        return 0 if a.all else 2
    for t in client.terminate(ids):
        print(f"terminated {t['id']}")
    return 0


def cmd_provision(a: argparse.Namespace) -> int:
    client = H.LambdaClient()
    ip = _resolve_ip(client, a.instance)
    H.wait_for_ssh(ip, a.key)
    H.rsync_up(ip, key_path=a.key)
    token = H.github_token()
    if not token:
        print("warning: no GitHub token — private myelin clone will fail", file=sys.stderr)
    H.ssh_run(ip, H.provision_script(token), a.key)
    print("provisioned")
    return 0


def cmd_exec(a: argparse.Namespace) -> int:
    ip = _resolve_ip(H.LambdaClient(), a.instance)
    return H.ssh_run(ip, H.uv_run(" ".join(a.command), a.extra), a.key, check=False)


def cmd_fetch(a: argparse.Namespace) -> int:
    ip = _resolve_ip(H.LambdaClient(), a.instance)
    remote = a.remote if a.remote.startswith("/") else f"{H.REMOTE_ROOT}/{a.remote}"
    H.rsync_down(ip, remote, a.local, a.key)
    return 0


# --- one-shot: launch -> provision -> run -> fetch -> terminate ------------- #
def _oneshot(a: argparse.Namespace, command: str) -> int:
    with H.Session(H.LambdaClient(), instance_type=a.type, region=a.region,
                   instance_id=a.instance_id, name=a.name, keep=a.keep, key_path=a.key) as s:
        if not a.instance_id or a.provision:
            s.provision()
        rc = s.run(command, a.extra)
        if a.fetch:
            s.fetch(f"{H.REMOTE_ROOT}/runs/", "runs/")
        return rc


def cmd_smoke(a: argparse.Namespace) -> int:
    return _oneshot(a, _py_c(SMOKE_PY))


def cmd_run(a: argparse.Namespace) -> int:
    return _oneshot(a, " ".join(a.command))


def cmd_train(a: argparse.Namespace) -> int:
    base = [
        "python", "examples/train_tiny_spikegpt.py", "--device", "cuda", "--vocab", "byte",
        "--text-file", "data/enwik8", "--layers", str(a.layers), "--embedding", str(a.embedding),
        "--context-length", str(a.context_length), "--batch", str(a.batch), "--steps", str(a.steps),
        "--lr", str(a.lr), "--amp", "bf16", "--matmul-precision", "high", "--compile", "regional",
        "--eval-every", str(a.eval_every), "--log-every", str(a.log_every),
    ]
    return _oneshot(a, " ".join(base + a.command))


# --- argparse --------------------------------------------------------------- #
def _lifecycle(p: argparse.ArgumentParser) -> None:
    """Flags shared by the one-shot experiment commands."""
    p.add_argument("--type", default=H.DEFAULT_INSTANCE_TYPE)
    p.add_argument("--region", default=None, help="preferred region (else first with capacity)")
    p.add_argument("--name", default="spikegpt-experiment")
    p.add_argument("--key", default=H.DEFAULT_SSH_KEY, help="SSH private key path")
    p.add_argument("--extra", default="cuda", help="uv extra to run under")
    p.add_argument("--keep", action="store_true", help="do NOT terminate on exit")
    p.add_argument("--instance-id", default=None,
                   help="reuse an instance (implies --keep; skips provision unless --provision)")
    p.add_argument("--provision", action="store_true", help="force provision when reusing")
    p.add_argument("--no-fetch", dest="fetch", action="store_false", help="skip pulling runs/ back")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lambda/run.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("types", help="list instance types + capacity")
    t.add_argument("--gpu", default=None, help="substring filter, e.g. h100")
    t.set_defaults(func=cmd_types)

    k = sub.add_parser("keys", help="list/register SSH keys")
    k.add_argument("--add", action="store_true", help="register local pubkey with Lambda")
    k.add_argument("--key", default=H.DEFAULT_SSH_KEY)
    k.set_defaults(func=cmd_keys)

    sub.add_parser("ls", help="list running instances").set_defaults(func=cmd_ls)

    la = sub.add_parser("launch", help="launch a persistent instance")
    la.add_argument("--type", default=H.DEFAULT_INSTANCE_TYPE)
    la.add_argument("--region", default=None)
    la.add_argument("--name", default="spikegpt-dev")
    la.add_argument("--key", default=H.DEFAULT_SSH_KEY)
    la.set_defaults(func=cmd_launch)

    pr = sub.add_parser("provision", help="rsync repo + uv sync on an instance")
    pr.add_argument("instance", help="instance id or ip")
    pr.add_argument("--key", default=H.DEFAULT_SSH_KEY)
    pr.set_defaults(func=cmd_provision)

    ex = sub.add_parser("exec", help="run a command on an existing instance")
    ex.add_argument("instance", help="instance id or ip")
    ex.add_argument("--key", default=H.DEFAULT_SSH_KEY)
    ex.add_argument("--extra", default="cuda")
    ex.add_argument("command", nargs=argparse.REMAINDER, help="-- <command...>")
    ex.set_defaults(func=cmd_exec)

    fe = sub.add_parser("fetch", help="rsync an artifact down (remote relative to repo root)")
    fe.add_argument("instance", help="instance id or ip")
    fe.add_argument("remote")
    fe.add_argument("local")
    fe.add_argument("--key", default=H.DEFAULT_SSH_KEY)
    fe.set_defaults(func=cmd_fetch)

    tm = sub.add_parser("terminate", help="terminate instances (stops billing)")
    tm.add_argument("ids", nargs="*", help="instance ids")
    tm.add_argument("--all", action="store_true", help="terminate ALL running instances")
    tm.set_defaults(func=cmd_terminate)

    sm = sub.add_parser("smoke", help="one-shot GPU/env smoke test")
    _lifecycle(sm)
    sm.set_defaults(func=cmd_smoke)

    rn = sub.add_parser("run", help="one-shot: launch -> provision -> run cmd -> fetch -> kill")
    _lifecycle(rn)
    rn.add_argument("command", nargs=argparse.REMAINDER, help="-- <command...>")
    rn.set_defaults(func=cmd_run)

    tr = sub.add_parser("train", help="one-shot SpikeGPT training run on enwik8")
    _lifecycle(tr)
    tr.add_argument("--layers", type=int, default=12)
    tr.add_argument("--embedding", type=int, default=512)
    tr.add_argument("--context-length", type=int, default=1024)
    tr.add_argument("--batch", type=int, default=64)
    tr.add_argument("--steps", type=int, default=4000)
    tr.add_argument("--lr", default="2e-3")
    tr.add_argument("--eval-every", type=int, default=500)
    tr.add_argument("--log-every", type=int, default=100)
    tr.add_argument("command", nargs=argparse.REMAINDER, help="-- <extra train flags...>")
    tr.set_defaults(func=cmd_train)
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    if getattr(a, "command", None) and a.command and a.command[0] == "--":
        a.command = a.command[1:]  # argparse REMAINDER keeps a leading "--"
    try:
        return a.func(a)
    except H.LambdaError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
