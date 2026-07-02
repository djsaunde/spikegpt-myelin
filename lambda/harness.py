"""Lambda Cloud harness for GPU experiments (a VM-based counterpart to ``modal/``).

Modal is serverless; Lambda Cloud is raw GPU VMs, so this owns the whole
lifecycle: launch an instance via the Cloud API, wait for SSH, replicate our
exact frozen uv env, rsync the repo + data up, run a command, fetch artifacts,
and — critically for a credit-billed platform — *always* terminate instances it
launched, even on error/Ctrl-C. Stdlib-only (urllib + ssh/rsync) so it runs
locally without the project's CUDA env. Auth: ``LAMBDA_API_KEY``. See ``run.py``.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Literal

API_BASE = "https://cloud.lambdalabs.com/api/v1"
# Cloudflare bot-fight 1010-blocks urllib's default UA; present a browser-ish one.
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
DEFAULT_INSTANCE_TYPE = "gpu_1x_h100_sxm5"  # ~$4.29/hr; gpu_1x_h100_pcie is cheaper/slower
SSH_USER = "ubuntu"
REMOTE_ROOT = f"/home/{SSH_USER}/spikegpt-myelin"
DEFAULT_SSH_KEY = os.path.expanduser("~/.ssh/id_ed25519")
CUDA_COMPAT_DIR = "/usr/local/cuda-13.0/compat"  # cu130 forward-compat libs (see provision_script)

# rsync excludes — caches/envs/outputs. NOTE: data/ is deliberately kept (data/enwik8
# is gitignored but experiments need it; the instance link is fast).
RSYNC_EXCLUDES = (
    ".git", ".venv", ".pytest_cache", ".ruff_cache", ".mypy_cache", "__pycache__",
    "torch_compile_debug", "wandb", "runs", "dist", "build", "*.egg-info",
)
_SSH_OPTS = [
    "-o", "StrictHostKeyChecking=accept-new", "-o", "UserKnownHostsFile=/dev/null",
    "-o", "LogLevel=ERROR", "-o", "ServerAliveInterval=30",
]


class LambdaError(RuntimeError):
    """A Lambda Cloud API or SSH/rsync failure, surfaced with the API message."""


def _log(msg: str) -> None:
    print(f"[lambda] {msg}", file=sys.stderr, flush=True)


@dataclass
class Instance:
    id: str
    name: str | None
    status: str
    ip: str | None
    instance_type: str | None
    region: str | None

    @classmethod
    def from_json(cls, d: dict) -> Instance:
        return cls(
            id=d["id"], name=d.get("name"), status=d.get("status", "unknown"),
            ip=d.get("ip"), instance_type=(d.get("instance_type") or {}).get("name"),
            region=(d.get("region") or {}).get("name"),
        )


class LambdaClient:
    """urllib client for the Lambda Cloud API (HTTP Basic: key as username)."""

    def __init__(self, api_key: str | None = None):
        self.api_key = (
            api_key or os.environ.get("LAMBDA_API_KEY") or os.environ.get("LAMBDA_CLOUD_API_KEY")
        )
        if not self.api_key:
            raise LambdaError(
                "No Lambda API key. Set LAMBDA_API_KEY (get one at "
                "https://cloud.lambda.ai/api-keys)."
            )
        self._auth = "Basic " + base64.b64encode(f"{self.api_key}:".encode()).decode()

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(f"{API_BASE}{path}", data=data, method=method)
        req.add_header("Authorization", self._auth)
        req.add_header("User-Agent", _USER_AGENT)
        req.add_header("Accept", "application/json")
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")
            try:
                err = json.loads(detail).get("error", {})
                msg = f"{err.get('code', e.code)}: {err.get('message', detail)}"
                if err.get("suggestion"):
                    msg += f" ({err['suggestion']})"
            except Exception:
                msg = f"{e.code}: {detail}"
            raise LambdaError(f"{method} {path} failed — {msg}") from None
        except urllib.error.URLError as e:
            raise LambdaError(f"{method} {path} failed — {e.reason}") from None

    def instance_types(self) -> dict:
        return self._request("GET", "/instance-types")["data"]

    def list_instances(self) -> list[Instance]:
        return [Instance.from_json(d) for d in self._request("GET", "/instances")["data"]]

    def get_instance(self, instance_id: str) -> Instance:
        return Instance.from_json(self._request("GET", f"/instances/{instance_id}")["data"])

    def list_ssh_keys(self) -> list[dict]:
        return self._request("GET", "/ssh-keys")["data"]

    def add_ssh_key(self, name: str, public_key: str) -> dict:
        return self._request("POST", "/ssh-keys", {"name": name, "public_key": public_key.strip()})[
            "data"
        ]

    def launch(self, *, instance_type: str, region: str, ssh_key_names: list[str],
               name: str | None = None, quantity: int = 1) -> list[str]:
        body = {"region_name": region, "instance_type_name": instance_type,
                "ssh_key_names": ssh_key_names, "quantity": quantity}
        if name:
            body["name"] = name
        return self._request("POST", "/instance-operations/launch", body)["data"]["instance_ids"]

    def terminate(self, instance_ids: list[str]) -> list[dict]:
        return self._request(
            "POST", "/instance-operations/terminate", {"instance_ids": instance_ids}
        )["data"]["terminated_instances"]

    def pick_region(self, instance_type: str, prefer: str | None = None) -> str:
        """A region with available capacity for ``instance_type`` (prefer if given)."""
        types = self.instance_types()
        if instance_type not in types:
            raise LambdaError(
                f"Unknown instance type {instance_type!r}. Available: {', '.join(sorted(types))}"
            )
        regions = [r["name"] for r in types[instance_type]["regions_with_capacity_available"]]
        if not regions:
            raise LambdaError(f"No capacity for {instance_type} anywhere right now (try `types`).")
        if prefer and prefer not in regions:
            raise LambdaError(f"No {instance_type} in {prefer}. Available: {', '.join(regions)}")
        return prefer or regions[0]

    def ensure_ssh_key(self, public_key_path: str = DEFAULT_SSH_KEY + ".pub") -> str:
        """Register the local pubkey with Lambda if absent; return its name.

        Matched on key material (ignoring comment) so an existing key is reused.
        """
        with open(public_key_path) as f:
            pub = f.read().strip()
        body = " ".join(pub.split()[:2])
        for k in self.list_ssh_keys():
            if " ".join(k["public_key"].split()[:2]) == body:
                return k["name"]
        name = f"spikegpt-{os.uname().nodename.split('.')[0]}"[:60]
        self.add_ssh_key(name, pub)
        return name

    def wait_active(self, instance_id: str, timeout: float = 420.0) -> Instance:
        """Poll until the instance is ``active`` with a public IP, or time out."""
        deadline, last = time.time() + timeout, ""
        while time.time() < deadline:
            inst = self.get_instance(instance_id)
            if inst.status != last:
                _log(f"instance {instance_id} -> {inst.status}")
                last = inst.status
            if inst.status == "active" and inst.ip:
                return inst
            if inst.status in ("terminated", "error", "unhealthy"):
                raise LambdaError(f"instance {instance_id} entered status {inst.status}")
            time.sleep(10)
        raise LambdaError(f"instance {instance_id} not active after {timeout:.0f}s")


# --------------------------------------------------------------------------- #
# SSH / rsync                                                                  #
# --------------------------------------------------------------------------- #
def _ssh_base(key_path: str) -> list[str]:
    return ["ssh", "-i", key_path, *_SSH_OPTS]


def wait_for_ssh(ip: str, key_path: str = DEFAULT_SSH_KEY, timeout: float = 300.0) -> None:
    """Block until the instance accepts SSH (it boots a bit after going active)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = subprocess.run(
            [*_ssh_base(key_path), "-o", "ConnectTimeout=10", f"{SSH_USER}@{ip}", "true"],
            capture_output=True,
        )
        if r.returncode == 0:
            _log(f"ssh to {ip} ready")
            return
        time.sleep(8)
    raise LambdaError(f"ssh to {ip} not ready after {timeout:.0f}s")


def ssh_run(ip: str, command: str, key_path: str = DEFAULT_SSH_KEY, check: bool = True) -> int:
    """Run a shell command on the instance, streaming output live."""
    rc = subprocess.run([*_ssh_base(key_path), f"{SSH_USER}@{ip}", command]).returncode
    if check and rc != 0:
        raise LambdaError(f"remote command failed (exit {rc}): {command[:80]}")
    return rc


def rsync_up(ip: str, local: str = ".", remote: str = REMOTE_ROOT,
             key_path: str = DEFAULT_SSH_KEY) -> None:
    """rsync the repo up (excluding caches/envs/outputs; data/ is included)."""
    ssh = " ".join(_ssh_base(key_path))
    excludes = [f"--exclude={p}" for p in RSYNC_EXCLUDES]
    _log(f"rsync up -> {ip}:{remote}")
    rc = subprocess.run(
        ["rsync", "-az", "--delete", "-e", ssh, *excludes,
         f"{local.rstrip('/')}/", f"{SSH_USER}@{ip}:{remote}/"]
    ).returncode
    if rc != 0:
        raise LambdaError(f"rsync up failed (exit {rc})")


def rsync_down(ip: str, remote: str, local: str, key_path: str = DEFAULT_SSH_KEY) -> None:
    """rsync an artifact down (best-effort; missing remote paths just warn)."""
    ssh = " ".join(_ssh_base(key_path))
    os.makedirs(os.path.dirname(local) or ".", exist_ok=True)
    _log(f"rsync down {ip}:{remote} -> {local}")
    if subprocess.run(["rsync", "-az", "-e", ssh, f"{SSH_USER}@{ip}:{remote}", local]).returncode:
        _log("rsync down failed — artifact may not exist")


# --------------------------------------------------------------------------- #
# Provisioning + remote-run builders                                           #
# --------------------------------------------------------------------------- #
# uv installs to ~/.local/bin; put the CUDA-13 compat libs first on the loader
# path (harmless when absent), and cd into the repo for every remote command.
_ENV_PREFIX = (
    'export PATH="$HOME/.local/bin:$PATH"; '
    f'export LD_LIBRARY_PATH="{CUDA_COMPAT_DIR}:${{LD_LIBRARY_PATH:-}}"; '
    f"cd {REMOTE_ROOT} && "
)


def github_token() -> str | None:
    """GitHub token for cloning the private ``myelin`` dep: env, then ``gh auth token``."""
    for var in ("GH_TOKEN", "GITHUB_TOKEN"):
        if os.environ.get(var):
            return os.environ[var].strip()
    try:
        r = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except FileNotFoundError:
        pass
    return None


def provision_script(gh_token: str | None = None) -> str:
    """Install uv + sync the frozen CUDA env, with two Lambda-specific fixups.

    (1) ``myelin`` is private: when ``gh_token`` is given, git clones it via an
    ``insteadOf`` rewrite in a throwaway ``GIT_CONFIG_GLOBAL`` deleted right after
    the sync (never touches the instance's real ~/.gitconfig).
    (2) cu130 torch needs an R580+ driver but Lambda ships CUDA-12.8, so install
    NVIDIA's ``cuda-compat-13-0`` (datacenter-GPU forward compat) — idempotent.
    """
    compat = (
        f"if [ ! -d {CUDA_COMPAT_DIR} ]; then "
        "distro=$(. /etc/os-release; echo $ID$VERSION_ID | tr -d .); "
        "curl -fsSL -o /tmp/cuda-keyring.deb "
        "https://developer.download.nvidia.com/compute/cuda/repos/$distro/x86_64/"
        "cuda-keyring_1.1-1_all.deb; "
        "sudo dpkg -i /tmp/cuda-keyring.deb; sudo apt-get -qq update; "
        "sudo apt-get -qq install -y cuda-compat-13-0; fi; "
    )
    head = (
        "set -uo pipefail; "
        'export PATH="$HOME/.local/bin:$PATH"; '
        "command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh; "
        + compat + f"cd {REMOTE_ROOT}; "
    )
    if gh_token:
        rewrite = f"https://x-access-token:{gh_token}@github.com/"
        return (
            head
            + 'export GIT_CONFIG_GLOBAL="$HOME/.gitconfig.lambda"; '
            + f'git config --file "$GIT_CONFIG_GLOBAL" url."{rewrite}".insteadOf '
              '"https://github.com/"; '
            + 'uv sync --frozen --extra cuda; rc=$?; rm -f "$GIT_CONFIG_GLOBAL"; exit $rc'
        )
    return head + "uv sync --frozen --extra cuda"


def uv_run(command: str, extra: str = "cuda") -> str:
    """Wrap a command to run inside the synced uv env on the instance."""
    return _ENV_PREFIX + f"uv run --extra {extra} {command}"


# --------------------------------------------------------------------------- #
# Managed-instance context: guaranteed teardown of anything we launch          #
# --------------------------------------------------------------------------- #
class Session:
    """Own an instance for a ``with`` block.

    With ``instance_id`` it attaches to an existing instance and never terminates
    it; otherwise it launches one and — unless ``keep`` — terminates it on exit
    (including on exception/KeyboardInterrupt, and on a failed startup). This is
    the guardrail against runaway credit spend.
    """

    def __init__(self, client: LambdaClient, *, instance_type: str = DEFAULT_INSTANCE_TYPE,
                 region: str | None = None, instance_id: str | None = None,
                 name: str | None = None, keep: bool = False, key_path: str = DEFAULT_SSH_KEY):
        self.client = client
        self.instance_type = instance_type
        self.region = region
        self.instance_id = instance_id
        self.name = name
        self.keep = keep or instance_id is not None
        self.key_path = key_path
        self._launched = instance_id is None
        self.instance: Instance | None = None

    def _terminate(self, why: str) -> None:
        if not (self._launched and not self.keep and self.instance_id):
            return
        _log(f"{why} — terminating {self.instance_id}")
        try:
            self.client.terminate([self.instance_id])
            _log(f"terminated {self.instance_id}")
        except Exception as e:
            _log(f"WARNING: terminate failed: {e} — TERMINATE MANUALLY: {self.instance_id}")

    def __enter__(self) -> Session:
        if self.instance_id is None:
            region = self.client.pick_region(self.instance_type, self.region)
            key_name = self.client.ensure_ssh_key(self.key_path + ".pub")
            _log(f"launching {self.instance_type} in {region} (key {key_name})")
            self.instance_id = self.client.launch(
                instance_type=self.instance_type, region=region,
                ssh_key_names=[key_name], name=self.name or "spikegpt-experiment",
            )[0]
            _log(f"launched {self.instance_id}")
        # __exit__ does NOT run if __enter__ raises, so terminate here on a hung
        # boot/SSH timeout — otherwise the launched instance would leak and bill.
        try:
            self.instance = self.client.wait_active(self.instance_id)
            wait_for_ssh(self.ip, self.key_path)
        except BaseException:
            self._terminate("startup failed")
            raise
        return self

    def __exit__(self, exc_type, exc, tb) -> Literal[False]:
        if self._launched and not self.keep:
            self._terminate("done")
        elif self.instance_id:
            ip = self.instance.ip if self.instance else "?"
            _log(f"leaving {self.instance_id} running (ip {ip})")
        return False  # propagate exceptions

    @property
    def ip(self) -> str:
        if not self.instance or not self.instance.ip:
            raise LambdaError("instance has no IP yet")
        return self.instance.ip

    def provision(self) -> None:
        rsync_up(self.ip, key_path=self.key_path)
        token = github_token()
        if not token:
            _log("WARNING: no GitHub token — the private myelin clone will fail")
        _log("provisioning env (uv sync --frozen --extra cuda) — first run downloads torch nightly")
        ssh_run(self.ip, provision_script(token), self.key_path)

    def run(self, command: str, extra: str = "cuda") -> int:
        return ssh_run(self.ip, uv_run(command, extra), self.key_path, check=False)

    def fetch(self, remote: str, local: str) -> None:
        rsync_down(self.ip, remote, local, self.key_path)
