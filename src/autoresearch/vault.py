from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import os
import secrets
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator
from urllib import error as urlerror
from urllib import request as urlrequest

import numpy as np
import pandas as pd

from . import metrics
from .contract import Contract
from .scoring import Candidate, FeatureMatrix
from .utils import (
    atomic_write_json,
    atomic_write_bytes,
    chain_hash,
    ensure_dir,
    json_safe,
    now_iso,
    read_json,
    read_jsonl,
    sha256_bytes,
    sha256_obj,
)

MAGIC = b"ARSEAL1"


class VaultError(Exception):
    pass


class PermissionDenied(PermissionError):
    pass


class BudgetExhausted(VaultError):
    pass


# ---------------------------------------------------------------------------
# OS-enforced sealing (spec §2: "sealing must be physical, not conventional")
# ---------------------------------------------------------------------------
# A sealed blob and its key carry an OS deny-read lock, so an accidental read
# from the search process (pd.read_parquet, open, a debugger) fails with a real
# PermissionError raised by the operating system, not by this library. Only the
# label service lifts the lock, reads, and restores it. Against a deliberate
# act by the same OS account this is a speed bump, not a wall: production
# deployments run the label service under its own account that owns the key.

_OS_USER: str | None = None


def _os_user() -> str:
    global _OS_USER
    if _OS_USER is None:
        try:
            _OS_USER = subprocess.run(["whoami"], capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            _OS_USER = os.environ.get("USERNAME", "")
    return _OS_USER


def os_seal(path: str | Path) -> None:
    """Deny read access to ``path`` for the current OS account."""
    p = str(path)
    if os.name == "nt":
        subprocess.run(["icacls", p, "/deny", f"{_os_user()}:(RD)"], capture_output=True, check=True)
    else:
        os.chmod(p, 0)


def os_unseal(path: str | Path) -> None:
    p = str(path)
    if os.name == "nt":
        subprocess.run(["icacls", p, "/remove:d", _os_user()], capture_output=True, check=True)
    else:
        os.chmod(p, 0o600)


def is_os_sealed(path: str | Path) -> bool:
    """True when the OS refuses this process a read of ``path``."""
    try:
        with open(path, "rb") as fh:
            fh.read(1)
        return False
    except PermissionError:
        return True


@contextlib.contextmanager
def os_unsealed(*paths: str | Path) -> Iterator[None]:
    """Lift the OS lock for the duration of a read, then restore it."""
    lifted: list[str | Path] = []
    try:
        for p in paths:
            if Path(p).exists():
                os_unseal(p)
                lifted.append(p)
        yield
    finally:
        for p in lifted:
            os_seal(p)


# ---------------------------------------------------------------------------
# Authenticated encryption (encrypt-then-MAC, HMAC-SHA256 stream)
# ---------------------------------------------------------------------------

def _keys(master: bytes) -> tuple[bytes, bytes]:
    enc = hmac.new(master, b"autoresearch-enc", hashlib.sha256).digest()
    mac = hmac.new(master, b"autoresearch-mac", hashlib.sha256).digest()
    return enc, mac


def _keystream(key: bytes, nonce: bytes, n: int) -> bytes:
    out = bytearray()
    counter = 0
    while len(out) < n:
        out.extend(hmac.new(key, nonce + counter.to_bytes(8, "big"), hashlib.sha256).digest())
        counter += 1
    return bytes(out[:n])


def encrypt_bytes(data: bytes, master: bytes) -> bytes:
    enc_key, mac_key = _keys(master)
    nonce = secrets.token_bytes(16)
    stream = _keystream(enc_key, nonce, len(data))
    cipher = bytes(a ^ b for a, b in zip(data, stream))
    tag = hmac.new(mac_key, nonce + cipher, hashlib.sha256).digest()
    return MAGIC + nonce + tag + cipher


def decrypt_bytes(blob: bytes, master: bytes) -> bytes:
    if not blob.startswith(MAGIC):
        raise VaultError("Not an AutoResearch sealed blob")
    body = blob[len(MAGIC):]
    nonce, tag, cipher = body[:16], body[16:48], body[48:]
    enc_key, mac_key = _keys(master)
    expected = hmac.new(mac_key, nonce + cipher, hashlib.sha256).digest()
    if not hmac.compare_digest(tag, expected):
        raise VaultError("Sealed blob failed authentication (tampered or wrong key)")
    stream = _keystream(enc_key, nonce, len(cipher))
    return bytes(a ^ b for a, b in zip(cipher, stream))


# ---------------------------------------------------------------------------
# Vault
# ---------------------------------------------------------------------------

class Vault:
    """Content-addressed custody with roles, capabilities, and a hash-chained ledger."""

    ROLE_CAPS = {
        "discovery": {"search", "benchmark", "interpret"},
        "sealed": {"confirm"},
        "benchmark_only": {"benchmark", "interpret"},
        "support_only": {"interpret"},
    }

    def __init__(self, root: str | Path):
        self.root = Path(root)
        ensure_dir(self.root)
        self.ledger_path = self.root / "ledger.jsonl"
        self.keys_dir = ensure_dir(self.root / "keys")

    # ------------------------------------------------------------------
    # Ledger
    # ------------------------------------------------------------------
    def ledger_entries(self) -> list[dict]:
        return read_jsonl(self.ledger_path)

    def _append_ledger(self, payload: dict[str, Any]) -> dict:
        entries = self.ledger_entries()
        prev = entries[-1]["hash"] if entries else "GENESIS"
        body = dict(payload)
        body["ts"] = now_iso()
        body["prev_hash"] = prev
        body["hash"] = chain_hash(prev, body)
        from .utils import append_jsonl

        append_jsonl(self.ledger_path, body)
        return body

    def verify_ledger(self) -> tuple[bool, str | None, int]:
        entries = self.ledger_entries()
        prev = "GENESIS"
        for i, entry in enumerate(entries):
            body = dict(entry)
            claimed = body.pop("hash", None)
            if claimed is None:
                return False, f"entry {i} has no hash", len(entries)
            expected = chain_hash(prev, body)
            if claimed != expected:
                return False, f"entry {i} hash mismatch", len(entries)
            prev = claimed
        return True, None, len(entries)

    def ledger_head(self) -> str:
        entries = self.ledger_entries()
        return entries[-1]["hash"] if entries else "GENESIS"

    # ------------------------------------------------------------------
    # Ingest and read
    # ------------------------------------------------------------------
    def make_key(self, dataset_hash: str) -> Path:
        path = self.keys_dir / f"{dataset_hash}.key"
        if not path.exists():
            atomic_write_bytes(path, secrets.token_bytes(32))
            os_seal(path)
        return path

    @staticmethod
    def load_key(path: str | Path) -> bytes:
        with os_unsealed(path):
            return Path(path).read_bytes()

    def ingest_dataframe(
        self,
        df: pd.DataFrame,
        role: str,
        name: str,
        source: str = "unknown",
        license: str = "unknown",
        sealed: bool = False,
        outcome_column: str | None = None,
        group_column: str | None = None,
        twin: bool = True,
        budget: int = 0,
        key_path: str | Path | None = None,
    ) -> dict[str, Any]:
        if role not in self.ROLE_CAPS:
            raise VaultError(f"Unknown role {role!r}; choose from {sorted(self.ROLE_CAPS)}")
        if sealed and not key_path:
            raise VaultError("Sealed ingest requires an encryption key path")
        # Content address is computed over a canonical parquet serialisation.
        parquet = df.to_parquet(index=False)
        dataset_hash = sha256_bytes(parquet)
        ddir = ensure_dir(self.root / dataset_hash)
        meta_path = ddir / "meta.json"
        if meta_path.exists():
            existing = read_json(meta_path)
            if existing.get("role") != role:
                raise VaultError(
                    f"Dataset {dataset_hash} is already held as role {existing.get('role')!r}; "
                    f"refusing to relabel the same content as {role!r}"
                )
            return existing

        if sealed:
            key = self.load_key(key_path)
            atomic_write_bytes(ddir / "sealed.enc", encrypt_bytes(parquet, key))
            os_seal(ddir / "sealed.enc")
            data_file = "sealed.enc"
        else:
            atomic_write_bytes(ddir / "data.parquet", parquet)
            data_file = "data.parquet"

        if twin:
            twin_df = self._make_twin(df, outcome_column, group_column)
            twin_df.to_parquet(ddir / "twin.parquet", index=False)

        group_ids = []
        if group_column and group_column in df.columns:
            group_ids = sorted(map(str, df[group_column].dropna().unique().tolist()))
        n_groups = len(group_ids) if group_ids else len(df)
        prevalence = None
        if outcome_column and outcome_column in df.columns:
            vals = pd.to_numeric(df[outcome_column], errors="coerce").dropna()
            if len(vals) and set(vals.unique()).issubset({0, 1}):
                prevalence = float(vals.mean())

        meta = {
            "dataset_hash": dataset_hash,
            "name": name,
            "role": role,
            "source": source,
            "license": license,
            "created_at": now_iso(),
            "sealed": bool(sealed),
            "encryption": "HMAC-SHA256 stream + HMAC-SHA256 tag" if sealed else "none",
            "os_sealed": bool(sealed),
            "data_file": data_file,
            "twin_file": "twin.parquet" if twin else None,
            "n_rows": int(len(df)),
            "n_groups": int(n_groups),
            "columns": [str(c) for c in df.columns],
            "dtypes": {str(c): str(t) for c, t in df.dtypes.items()},
            "outcome_column": outcome_column,
            "group_column": group_column,
            "group_ids_hash": sha256_obj(group_ids) if group_ids else None,
            "prevalence": prevalence if role != "sealed" else None,
            "capabilities": sorted(self.ROLE_CAPS[role]),
            "budget_total": int(budget) if role == "sealed" else None,
            "budget_remaining": int(budget) if role == "sealed" else None,
            "parquet_sha256": dataset_hash,
        }
        atomic_write_json(meta_path, meta)
        atomic_write_json(ddir / "budget.json", {"total": int(budget), "remaining": int(budget)}) if role == "sealed" else None
        self._append_ledger(
            {
                "type": "ingest",
                "dataset": dataset_hash,
                "name": name,
                "role": role,
                "sealed": bool(sealed),
                "n_rows": int(len(df)),
                "actor": "vault",
            }
        )
        return meta

    def ingest(
        self,
        path: str | Path,
        role: str,
        name: str | None = None,
        source: str | None = None,
        license: str = "unknown",
        sealed: bool = False,
        outcome_column: str | None = None,
        group_column: str | None = None,
        twin: bool = True,
        budget: int = 0,
        key_path: str | Path | None = None,
    ) -> dict[str, Any]:
        from .contract import read_table

        df = read_table(path)
        return self.ingest_dataframe(
            df,
            role=role,
            name=name or Path(path).stem,
            source=source or str(path),
            license=license,
            sealed=sealed,
            outcome_column=outcome_column,
            group_column=group_column,
            twin=twin,
            budget=budget,
            key_path=key_path,
        )

    def meta(self, dataset_hash: str) -> dict[str, Any]:
        meta_path = self.root / dataset_hash / "meta.json"
        if not meta_path.exists():
            raise VaultError(f"Unknown dataset {dataset_hash}")
        return read_json(meta_path)

    def _check_capability(self, meta: dict, actor: str, capability: str) -> None:
        role = meta["role"]
        caps = self.ROLE_CAPS.get(role, set())
        if capability not in caps:
            raise PermissionDenied(f"role {role!r} does not grant capability {capability!r}")
        if role == "sealed" and capability != "confirm":
            raise PermissionDenied("sealed datasets grant only the confirm capability")
        if meta.get("sealed") and actor != "label_service":
            raise PermissionDenied(
                "sealed datasets may only be opened by the label service; direct reads are refused"
            )

    def read(self, dataset_hash: str, actor: str = "search", capability: str = "search", purpose: str = "") -> pd.DataFrame:
        meta = self.meta(dataset_hash)
        try:
            self._check_capability(meta, actor, capability)
        except PermissionDenied:
            self._append_ledger(
                {
                    "type": "read_denied",
                    "dataset": dataset_hash,
                    "actor": actor,
                    "capability": capability,
                    "purpose": purpose,
                }
            )
            raise
        if meta["sealed"]:
            raise PermissionDenied("Direct read of a sealed dataset is not permitted; use the label service")
        df = pd.read_parquet(self.root / dataset_hash / meta["data_file"])
        self._append_ledger(
            {"type": "read", "dataset": dataset_hash, "actor": actor, "capability": capability, "purpose": purpose}
        )
        return df

    def twin(self, dataset_hash: str) -> pd.DataFrame:
        meta = self.meta(dataset_hash)
        twin_file = meta.get("twin_file")
        if not twin_file:
            raise VaultError("No synthetic twin exists for this dataset")
        self._append_ledger({"type": "twin_read", "dataset": dataset_hash, "actor": "debug"})
        return pd.read_parquet(self.root / dataset_hash / twin_file)

    def decrypt_sealed(self, dataset_hash: str, key_path: str | Path, actor: str = "label_service") -> pd.DataFrame:
        meta = self.meta(dataset_hash)
        if not meta.get("sealed"):
            return self.read(dataset_hash, actor=actor, capability="confirm")
        if actor != "label_service":
            self._append_ledger({"type": "unseal_denied", "dataset": dataset_hash, "actor": actor})
            raise PermissionDenied("Only the label service may decrypt sealed data")
        key = self.load_key(key_path)
        blob_path = self.root / dataset_hash / "sealed.enc"
        with os_unsealed(blob_path):
            blob = blob_path.read_bytes()
        parquet = decrypt_bytes(blob, key)
        self._append_ledger({"type": "unseal", "dataset": dataset_hash, "actor": actor})
        import io

        return pd.read_parquet(io.BytesIO(parquet))

    # ------------------------------------------------------------------
    def _make_twin(self, df: pd.DataFrame, outcome_column: str | None, group_column: str | None) -> pd.DataFrame:
        twin = df.copy()
        if outcome_column and outcome_column in twin.columns:
            if group_column and group_column in twin.columns:
                rng = np.random.default_rng(seed=42)
                for g in twin[group_column].dropna().unique():
                    idx = twin.index[twin[group_column] == g]
                    vals = twin.loc[idx, outcome_column].to_numpy()
                    twin.loc[idx, outcome_column] = rng.permutation(vals)
            else:
                twin[outcome_column] = np.random.default_rng(seed=42).permutation(twin[outcome_column].to_numpy())
        # Structure-preserving enough for pipeline debugging: features and group
        # structure are real, outcomes are a within-group permutation (null).
        return twin


# ---------------------------------------------------------------------------
# Sealed label service (in-process)
# ---------------------------------------------------------------------------

class SealedLabelService:
    """Holds sealed labels. Returns only aggregates; never rows, labels or features."""

    def __init__(
        self,
        vault: Vault,
        dataset_hash: str,
        contract: Contract,
        key_path: str | Path,
        budget: int | None = None,
    ):
        self.vault = vault
        self.dataset_hash = dataset_hash
        self.contract = contract
        self._df = vault.decrypt_sealed(dataset_hash, key_path)
        self._meta = vault.meta(dataset_hash)
        self._lock = threading.Lock()
        budget_path = vault.root / dataset_hash / "budget.json"
        if budget_path.exists():
            b = read_json(budget_path)
            self._budget_total = int(b.get("total", 0))
            self._budget_remaining = int(b.get("remaining", self._budget_total))
        else:
            self._budget_total = int(budget if budget is not None else self._meta.get("budget_total") or 0)
            self._budget_remaining = self._budget_total
            atomic_write_json(budget_path, {"total": self._budget_total, "remaining": self._budget_remaining})
        self._calls: list[dict[str, Any]] = []

    def meta(self) -> dict[str, Any]:
        y = pd.to_numeric(self._df[self.contract.outcome_column], errors="raise").to_numpy()
        groups = self._df[self.contract.group_column].astype(str).to_numpy() if self.contract.group_column else np.array(["__all__"] * len(self._df))
        return {
            "dataset_hash": self.dataset_hash,
            "name": self._meta.get("name"),
            "role": "sealed",
            "sealed": True,
            "n_rows": int(len(self._df)),
            "n_groups": int(len(set(groups))),
            "columns": [str(c) for c in self._df.columns],
            "outcome_column": self.contract.outcome_column,
            "group_column": self.contract.group_column,
            "group_ids_hash": self._meta.get("group_ids_hash"),
            "sealed_at": self._meta.get("created_at"),
            "prevalence": float(np.mean(y)),
            "budget_total": int(self._budget_total),
            "budget_remaining": int(self._budget_remaining),
            "mode": "in_process_label_service",
        }

    def _persist_budget(self) -> None:
        atomic_write_json(
            self.vault.root / self.dataset_hash / "budget.json",
            {"total": int(self._budget_total), "remaining": int(self._budget_remaining)},
        )

    def confirm(
        self,
        candidate: dict[str, Any] | Candidate,
        purpose: str = "primary confirmation",
        cost: int = 1,
        detail: str = "full",
        seed: int = 0,
        bootstrap: int = 1000,
    ) -> dict[str, Any]:
        with self._lock:
            cand = candidate if isinstance(candidate, Candidate) else Candidate.from_dict(candidate)
            if self._budget_remaining < cost:
                self.vault._append_ledger(
                    {
                        "type": "confirm_denied",
                        "dataset": self.dataset_hash,
                        "actor": "label_service",
                        "candidate": cand.id,
                        "purpose": purpose,
                        "budget_remaining": self._budget_remaining,
                    }
                )
                raise BudgetExhausted(
                    f"Unsealing budget exhausted: {self._budget_remaining} remaining, {cost} requested"
                )
            fm = FeatureMatrix(
                self._df,
                self.contract,
                fit=False,
                fit_stats=cand.params.get("fit_stats", {}),
            )
            s_cand = fm.score_candidate(cand)
            s_base = fm.score_candidate(self.contract.baseline_candidate())
            y = self._df[self.contract.outcome_column].to_numpy().astype(int)
            groups = (
                self._df[self.contract.group_column].astype(str).to_numpy()
                if self.contract.group_column
                else np.array(["__all__"] * len(self._df))
            )
            m = metrics.compute_metrics(y, groups, s_cand, s_base, top_k=self.contract.top_k)
            result: dict[str, Any] = {
                "dataset_hash": self.dataset_hash,
                "candidate_id": cand.id,
                "purpose": purpose,
                "n": m["n"],
                "n_groups": m["n_groups"],
                "n_groups_evaluable": m["n_groups_evaluable"],
                "prevalence": m["prevalence"],
                "auc_candidate": m["auc_candidate"],
                "auc_baseline": m["auc_baseline"],
                "delta": m["delta_auc"],
                "point_delta": m["delta_auc"],
                "top_k_hit_rate": m.get("top_k_hit_rate"),
                "top_k_enrichment": m.get("top_k_enrichment"),
                "frac_improved": m.get("frac_improved"),
                "mean_pct_rank": m.get("mean_pct_rank"),
                "rank_coef": m.get("rank_coef"),
                "detail": detail,
            }
            if detail == "full":
                boot = metrics.group_bootstrap_deltas(y, groups, s_cand, s_base, n_boot=bootstrap, seed=seed)
                summ = metrics.bootstrap_summary(boot)
                result.update(
                    {
                        "delta": summ["delta"] if np.isfinite(summ["delta"]) else m["delta_auc"],
                        "ci_low": summ["ci_low"],
                        "ci_high": summ["ci_high"],
                        "se": summ["se"],
                        "p_one_sided": summ["p_one_sided"],
                        "p_two_sided": summ["p_two_sided"],
                        "n_boot": summ["n_boot"],
                    }
                )
                per_group = metrics.per_group_delta(y, groups, s_cand, s_base)
                result["per_group_deltas"] = per_group
                vals = np.array(list(per_group.values()))
                result["stability_confirmation"] = float(np.mean(vals > 0)) if len(vals) else float("nan")
                result["leave_one_group_out"] = {
                    "min_delta": float(vals.min()) if len(vals) else None,
                    "max_delta": float(vals.max()) if len(vals) else None,
                    "n_groups": int(len(vals)),
                }
            self._budget_remaining -= cost
            self._persist_budget()
            entry = {
                "type": "confirm",
                "dataset": self.dataset_hash,
                "actor": "label_service",
                "candidate": cand.id,
                "purpose": purpose,
                "cost": int(cost),
                "budget_remaining": int(self._budget_remaining),
                "delta": float(result["delta"]) if result.get("delta") is not None else None,
                "n": int(result["n"]),
            }
            self.vault._append_ledger(entry)
            result["budget_remaining"] = int(self._budget_remaining)
            result["access_entry_hash"] = self.vault.ledger_head()
            self._calls.append(entry)
            return json_safe(result)

    def access_log(self) -> list[dict[str, Any]]:
        return list(self._calls)


# ---------------------------------------------------------------------------
# Label service across a process boundary (spec §2)
# ---------------------------------------------------------------------------
# The search process never holds the key, never opens the sealed blob, and
# never sees a row. It sends a candidate specification over HTTP with a bearer
# token issued by the service process and receives an aggregate back. Each call
# decrements the budget the service persists in the vault ledger.

SERVICE_MODE = "separate_process_label_service"


class _LabelHandler(BaseHTTPRequestHandler):
    service: SealedLabelService
    token: str

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def _send(self, code: int, obj: Any) -> None:
        body = json.dumps(json_safe(obj)).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        return hmac.compare_digest(self.headers.get("Authorization", ""), f"Bearer {self.token}")

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send(200, {"ok": True})
            return
        if not self._authorized():
            self._send(401, {"error": "unauthorized"})
            return
        if self.path == "/meta":
            meta = self.service.meta()
            meta["mode"] = SERVICE_MODE
            meta["service_pid"] = os.getpid()
            self._send(200, meta)
            return
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if not self._authorized():
            self._send(401, {"error": "unauthorized"})
            return
        if self.path != "/confirm":
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            result = self.service.confirm(
                payload["candidate"],
                purpose=payload.get("purpose", "confirmation"),
                cost=int(payload.get("cost", 1)),
                detail=payload.get("detail", "full"),
                seed=int(payload.get("seed", 0)),
                bootstrap=int(payload.get("bootstrap", 1000)),
            )
            self._send(200, result)
        except BudgetExhausted as exc:
            self._send(429, {"error": "budget_exhausted", "message": str(exc)})
        except Exception as exc:  # reported to the client, never swallowed
            self._send(400, {"error": type(exc).__name__, "message": str(exc)})


def label_service_main(
    vault_root: str | Path,
    dataset_hash: str,
    contract_path: str | Path,
    key_path: str | Path,
    budget: int | None = None,
    host: str = "127.0.0.1",
) -> int:
    """Entry point of the label-service process.

    Prints one JSON line ``{"ready": true, "port": ..., "token": ...}`` on
    stdout and serves until its stdin closes, so it cannot outlive the process
    that started it.
    """
    from .contract import load_contract

    vault = Vault(vault_root)
    contract = load_contract(contract_path)
    service = SealedLabelService(vault, dataset_hash, contract, key_path, budget=budget)
    token = secrets.token_urlsafe(32)
    handler = type("LabelHandler", (_LabelHandler,), {"service": service, "token": token})
    httpd = ThreadingHTTPServer((host, 0), handler)

    def _watch_parent() -> None:
        try:
            sys.stdin.read()
        finally:
            httpd.shutdown()

    threading.Thread(target=_watch_parent, daemon=True).start()
    ready = {"ready": True, "port": httpd.server_address[1], "token": token, "pid": os.getpid()}
    print(json.dumps(ready), flush=True)
    httpd.serve_forever()
    httpd.server_close()
    return 0


class SealedClient:
    """Search-side client. Sends candidate specifications; receives aggregates."""

    def __init__(self, base_url: str, token: str, timeout: float = 300.0):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def _request(self, method: str, path: str, payload: dict | None = None) -> dict:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urlrequest.Request(self.base_url + path, data=data, method=method)
        req.add_header("Authorization", f"Bearer {self.token}")
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urlrequest.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urlerror.HTTPError as exc:
            body = json.loads(exc.read().decode("utf-8") or "{}")
            if exc.code == 429:
                raise BudgetExhausted(body.get("message", "unsealing budget exhausted")) from None
            if exc.code == 401:
                raise PermissionDenied("label service rejected the token") from None
            raise VaultError(f"label service error {body.get('error')}: {body.get('message')}") from None

    def meta(self) -> dict:
        return self._request("GET", "/meta")

    def confirm(
        self,
        candidate: dict[str, Any] | Candidate,
        purpose: str = "confirmation",
        cost: int = 1,
        detail: str = "full",
        seed: int = 0,
        bootstrap: int = 1000,
    ) -> dict:
        spec = candidate.to_dict() if isinstance(candidate, Candidate) else candidate
        payload = {
            "candidate": spec,
            "purpose": purpose,
            "cost": cost,
            "detail": detail,
            "seed": seed,
            "bootstrap": bootstrap,
        }
        return self._request("POST", "/confirm", payload)


class LabelServiceProcess:
    """Start the label service in its own process; yield a client to it.

    The caller hands the key location to the child and keeps nothing else: the
    returned client carries only a URL and a token.
    """

    def __init__(
        self,
        vault_root: str | Path,
        dataset_hash: str,
        contract_path: str | Path,
        key_path: str | Path,
        budget: int | None = None,
        startup_timeout: float = 120.0,
    ):
        self.args = [
            sys.executable, "-m", "autoresearch", "label-service",
            "--vault-root", str(vault_root),
            "--dataset", dataset_hash,
            "--contract", str(contract_path),
            "--key", str(key_path),
        ]
        if budget is not None:
            self.args += ["--budget", str(int(budget))]
        self.startup_timeout = startup_timeout
        self.proc: subprocess.Popen | None = None
        self.pid: int | None = None

    def __enter__(self) -> SealedClient:
        env = os.environ.copy()
        pkg_root = str(Path(__file__).resolve().parents[1])
        env["PYTHONPATH"] = pkg_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        self.proc = subprocess.Popen(
            self.args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        ready: dict[str, Any] = {}

        def _read_ready() -> None:
            line = self.proc.stdout.readline() if self.proc and self.proc.stdout else ""
            if line.strip():
                ready.update(json.loads(line))

        reader = threading.Thread(target=_read_ready, daemon=True)
        reader.start()
        reader.join(self.startup_timeout)
        if not ready.get("ready"):
            self._stop()
            err = self.proc.stderr.read() if self.proc.stderr else ""
            raise VaultError(f"label service failed to start: {err.strip()[-2000:]}")
        self.pid = int(ready["pid"])
        return SealedClient(f"http://127.0.0.1:{ready['port']}", ready["token"])

    def _stop(self) -> None:
        if self.proc is None:
            return
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()

    def __exit__(self, *exc: Any) -> None:
        self._stop()
