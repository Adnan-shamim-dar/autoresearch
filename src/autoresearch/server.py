from __future__ import annotations

import json
import mimetypes
import secrets
import threading
import time
import traceback
import webbrowser
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from .certificate import verify_certificate
from .contract import ContractError, load_contract
from .registry import Registry
from .utils import json_safe, now_iso, read_json, sha256_file
from .vault import Vault

WEB_DIR = Path(__file__).with_name("web")
MAX_LIVE_JOBS = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_api(obj: Any) -> Any:
    return json_safe(obj)


def _read_optional(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return read_json(path)
    except Exception:
        return None


def _fmt_hash(value: str | None, n: int = 12) -> str:
    return (value or "")[:n]


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------


class AppState:
    def __init__(self, root: Path):
        self.root = root
        self._lock = threading.Lock()
        self._contract = None
        self._power_cache: dict[str, Any] = {}
        self.live_lock = threading.Lock()
        self.live_jobs: dict[str, "LiveJob"] = {}
        self.live_order: list[str] = []

    # -- project files ----------------------------------------------------
    @property
    def demo_info(self) -> dict[str, Any] | None:
        return _read_optional(self.root / "data" / "demo_info.json")

    def contract(self):
        with self._lock:
            if self._contract is None:
                info = self.demo_info
                if not info:
                    raise FileNotFoundError("Demo substrate not initialized. Run `autoresearch demo --reset`.")
                self._contract = load_contract(info["contract_path"], info["discovery_path"])
            return self._contract

    # -- summaries --------------------------------------------------------
    def vault_summary(self) -> dict[str, Any]:
        try:
            vault = Vault(self.root / "vault")
            ok, error, n = vault.verify_ledger()
            head = vault.ledger_head()
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "entries": 0, "head": "unknown"}
        info = self.demo_info or {}
        meta = info.get("confirmation_meta") or {}
        budget = _read_optional(self.root / "vault" / str(meta.get("dataset_hash")) / "budget.json") or {}
        return {
            "ok": ok,
            "error": error,
            "entries": n,
            "head": head,
            "sealed": {
                "dataset_hash": meta.get("dataset_hash"),
                "name": meta.get("name"),
                "n_rows": meta.get("n_rows"),
                "n_groups": meta.get("n_groups"),
                "budget_total": budget.get("total", meta.get("budget_total")),
                "budget_remaining": budget.get("remaining", meta.get("budget_remaining")),
                "encryption": meta.get("encryption"),
            },
        }

    def registry_summary(self) -> dict[str, Any]:
        try:
            registry = Registry(self.root / "state")
            ok, error, n = registry.verify()
            summary = registry.summary()
            summary["ok"] = ok
            summary["error"] = error
            summary["entries"] = n
            return summary
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def runs(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for cert_path in sorted((self.root / "runs").glob("*/certificate.json")):
            cert = _read_optional(cert_path) or {}
            run_dir = cert_path.parent
            run = _read_optional(run_dir / "run.json") or {}
            ev = cert.get("evidence") or {}
            out.append(
                {
                    "run_id": cert.get("run_id") or run_dir.name,
                    "dir": run_dir.name,
                    "tier": cert.get("tier"),
                    "status": cert.get("status"),
                    "verdict": cert.get("verdict"),
                    "claim_type": cert.get("claim_type"),
                    "claim": cert.get("claim"),
                    "certificate_hash": cert.get("certificate_hash"),
                    "contract_hash": cert.get("contract_hash"),
                    "seed": cert.get("seed"),
                    "created_at": run.get("created_at"),
                    "mode": run.get("mode"),
                    "delta": ev.get("delta"),
                    "ci_low": ev.get("ci_low"),
                    "ci_high": ev.get("ci_high"),
                    "n_confirmation": ev.get("n_confirmation"),
                    "n_groups_confirmation": ev.get("n_groups_confirmation"),
                    "tier_oc": cert.get("gate_oc_at_this_effect"),
                    "has_markdown": (run_dir / "certificate.md").exists(),
                }
            )
        out.sort(key=lambda r: (r.get("created_at") or "", r.get("run_id") or ""), reverse=True)
        return out

    def run_detail(self, run_id: str) -> dict[str, Any]:
        # Accept either the run directory name or the certificate run id.
        if not run_id or run_id in (".", "..") or Path(run_id).name != run_id:
            raise FileNotFoundError(f"Unknown run: {run_id}")
        candidates = [self.root / "runs" / run_id]
        if not candidates[0].exists():
            for p in (self.root / "runs").glob("*/certificate.json"):
                cert = _read_optional(p) or {}
                if cert.get("run_id") == run_id:
                    candidates = [p.parent]
                    break
        run_dir = candidates[0]
        if not run_dir.exists():
            raise FileNotFoundError(f"Unknown run: {run_id}")
        cert = _read_optional(run_dir / "certificate.json")
        if cert is None:
            raise FileNotFoundError(f"Run has no certificate: {run_id}")
        events = []
        ev_path = run_dir / "events.jsonl"
        if ev_path.exists():
            try:
                with open(ev_path, "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            events.append(json.loads(line))
            except Exception:
                events = []
        run = _read_optional(run_dir / "run.json") or {}
        files = []
        for p in sorted(run_dir.glob("*")):
            if p.is_file():
                files.append({"name": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p)})
        markdown = ""
        md = run_dir / "certificate.md"
        if md.exists():
            markdown = md.read_text(encoding="utf-8")
        return {
            "run_id": cert.get("run_id"),
            "dir": run_dir.name,
            "certificate": cert,
            "run": run,
            "gate": _read_optional(run_dir / "gate.json"),
            "power": _read_optional(run_dir / "power.json"),
            "selection_null": _read_optional(run_dir / "selection_null.json"),
            "adversary": _read_optional(run_dir / "adversary.json"),
            "confirmation": _read_optional(run_dir / "confirmation.json"),
            "selection": _read_optional(run_dir / "selection.json"),
            "validation": _read_optional(run_dir / "validation.json"),
            "budget": _read_optional(run_dir / "budget.json"),
            "events": events[-2500:],
            "files": files,
            "markdown": markdown,
        }

    def verify_run(self, run_id: str | None, cert_obj: dict | None) -> dict[str, Any]:
        if cert_obj is None:
            if not run_id:
                raise ValueError("provide run_id or certificate")
            detail = self.run_detail(run_id)
            cert_obj = detail["certificate"]
            run_dir = self.root / "runs" / detail["dir"]
        else:
            run_dir = None
        key_path = self.root / "state" / "certificate.key"
        key = key_path.read_bytes() if key_path.exists() else None
        result = verify_certificate(cert_obj, key)
        manifest = None
        if run_dir is not None:
            run = _read_optional(run_dir / "run.json") or {}
            files = run.get("files") or {}
            bad = []
            for name, expected in files.items():
                fp = run_dir / name
                actual = sha256_file(fp) if fp.exists() else None
                if actual != expected:
                    bad.append({"file": name, "expected": expected, "actual": actual})
            manifest = {"ok": not bad, "checked": len(files), "mismatches": bad}
        registered = None
        try:
            registry = Registry(self.root / "state")
            registered = any(
                e.get("contract_hash") == cert_obj.get("contract_hash")
                for e in registry.entries("contract_registered")
            )
        except Exception:
            pass
        return {"certificate": result, "manifest": manifest, "contract_registered": registered}

    # -- power ------------------------------------------------------------
    def power(self, params: dict[str, Any]) -> dict[str, Any]:
        from .power import assess

        contract = self.contract()
        info = self.demo_info or {}
        meta = info.get("confirmation_meta") or {}
        defaults = {
            "confirmation_n": meta.get("n_rows") or contract.power.confirmation_n,
            "confirmation_groups": meta.get("n_groups") or contract.power.confirmation_groups,
            "prevalence": info.get("prevalence_confirmation") or meta.get("prevalence"),
            "k_candidates": contract.power.k_candidates,
            "effect": contract.decision.smallest_effect_that_matters,
            "alpha": contract.decision.alpha,
            "target_power": contract.decision.target_power,
        }
        chosen = {**defaults, **{k: v for k, v in params.items() if v not in (None, "")}}
        cache_key = json.dumps(_safe_api(chosen), sort_keys=True)
        with self._lock:
            if cache_key in self._power_cache:
                return {"assessment": self._power_cache[cache_key], "defaults": defaults, "cached": True}
        assessment = assess(
            contract,
            contract.data,
            confirmation_n=int(chosen["confirmation_n"]) if chosen.get("confirmation_n") else None,
            confirmation_groups=int(chosen["confirmation_groups"]) if chosen.get("confirmation_groups") else None,
            confirmation_prevalence=float(chosen["prevalence"]) if chosen.get("prevalence") is not None else None,
            k_candidates=int(chosen["k_candidates"]) if chosen.get("k_candidates") else None,
            alpha=float(chosen["alpha"]) if chosen.get("alpha") is not None else None,
            target_power=float(chosen["target_power"]) if chosen.get("target_power") is not None else None,
            effect=float(chosen["effect"]) if chosen.get("effect") is not None else None,
            bootstrap=300,
            seed=12345,
        )
        payload = assessment.to_dict()
        with self._lock:
            if len(self._power_cache) > 48:
                self._power_cache.clear()
            self._power_cache[cache_key] = payload
        return {"assessment": payload, "defaults": defaults, "cached": False}

    # -- live runs --------------------------------------------------------
    def start_live(self, params: dict[str, Any]) -> dict[str, Any]:
        with self.live_lock:
            for jid in reversed(self.live_order):
                job = self.live_jobs.get(jid)
                if job is not None and job.status in ("queued", "running"):
                    return {"ok": False, "busy": True, "job": job.summary()}
            live_id = f"live-{time.strftime('%H%M%S')}-{secrets.token_hex(3)}"
            job = LiveJob(
                live_id=live_id,
                workspace=self.root / "live" / live_id,
                params=params,
            )
            self.live_jobs[live_id] = job
            self.live_order.append(live_id)
            while len(self.live_order) > MAX_LIVE_JOBS:
                old = self.live_order.pop(0)
                self.live_jobs.pop(old, None)
            job.start()
            return {"ok": True, "job": job.summary()}

    def live(self, live_id: str) -> "LiveJob":
        job = self.live_jobs.get(live_id)
        if job is None:
            raise FileNotFoundError(f"Unknown live run: {live_id}")
        return job


@dataclass
class LiveJob:
    live_id: str
    workspace: Path
    params: dict[str, Any]
    status: str = "queued"
    events: list[dict[str, Any]] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None
    started_at: str = field(default_factory=now_iso)
    finished_at: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def start(self) -> None:
        t = threading.Thread(target=self._run, name=f"live-{self.live_id}", daemon=True)
        t.start()

    def _append(self, ev: dict[str, Any]) -> None:
        with self._lock:
            self.events.append(ev)

    def snapshot_events(self, start: int = 0) -> list[dict[str, Any]]:
        with self._lock:
            return list(self.events[start:])

    def _run(self) -> None:
        from .demo import run_demo, write_demo_files

        self.status = "running"
        try:
            p = self.params
            n_patients = int(p.get("patients", 300))
            n_groups = int(p.get("confirmation_groups", 100))
            if n_groups >= n_patients - 8:
                n_groups = max(20, n_patients // 3)
            self._append(
                {
                    "seq": 0,
                    "ts": now_iso(),
                    "stage": "substrate",
                    "status": "start",
                    "message": f"Building a sandboxed synthetic substrate: {n_patients} patients, {n_groups} sealed confirmation groups.",
                    "data": {},
                }
            )
            write_demo_files(
                self.workspace,
                seed=int(p.get("seed", 20261114)),
                reset=True,
                n_patients=n_patients,
                n_confirmation_groups=n_groups,
            )
            info = _read_optional(self.workspace / "data" / "demo_info.json") or {}
            self._append(
                {
                    "seq": 1,
                    "ts": now_iso(),
                    "stage": "substrate",
                    "status": "ok",
                    "message": (
                        f"Substrate ready: {info.get('n_discovery', '?')} discovery rows / "
                        f"{info.get('n_confirmation', '?')} sealed rows; contract {str(info.get('contract_hash', ''))[:12]}."
                    ),
                    "data": {"contract_hash": info.get("contract_hash")},
                }
            )
            result = run_demo(
                self.workspace,
                n_permutations=int(p.get("permutations", 19)),
                candidate_budget=int(p.get("candidates", 72)),
                generations=int(p.get("generations", 2)),
                event_sink=self._append,
            )
            cert = result["certificate"]
            self.result = {
                "run_id": result["run_id"],
                "tier": cert.get("tier"),
                "status": cert.get("status"),
                "verdict": cert.get("verdict"),
                "claim": cert.get("claim"),
                "certificate_hash": cert.get("certificate_hash"),
                "certificate": cert,
                "power": result.get("power"),
                "run_dir": str(result["run_dir"]),
                "info": info,
            }
            self.status = "done"
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.status = "error"
            self._append(
                {
                    "seq": len(self.events) + 1,
                    "ts": now_iso(),
                    "stage": "error",
                    "status": "error",
                    "message": self.error,
                    "data": {},
                }
            )
        finally:
            self.finished_at = now_iso()

    def summary(self) -> dict[str, Any]:
        with self._lock:
            event_count = len(self.events)
        return {
            "live_id": self.live_id,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "event_count": event_count,
            "params": self.params,
            "error": self.error,
            "result": self.result,
        }


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    server_version = "AutoResearch/0.1"
    protocol_version = "HTTP/1.1"
    app: AppState

    def log_message(self, fmt: str, *args: Any) -> None:
        if self.path.startswith("/api/live") and self.path.endswith("/stream"):
            return
        print(f"  {self.command} {self.path} -> {args[1] if len(args) > 1 else ''}", flush=True)

    # -- response helpers -------------------------------------------------
    def _headers(self, code: int, content_type: str, length: int, extra: dict[str, str] | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-store")
        if extra:
            for k, v in extra.items():
                self.send_header(k, v)
        self.end_headers()

    def _send_json(self, code: int, obj: Any) -> None:
        body = json.dumps(_safe_api(obj), ensure_ascii=False, allow_nan=False).encode("utf-8")
        self._headers(code, "application/json; charset=utf-8", len(body))
        self.wfile.write(body)

    def _send_text(self, code: int, text: str, content_type: str = "text/plain; charset=utf-8") -> None:
        body = text.encode("utf-8")
        self._headers(code, content_type, len(body))
        self.wfile.write(body)

    def _error(self, code: int, message: str, extra: dict[str, Any] | None = None) -> None:
        self._send_json(code, {"ok": False, "error": message, **(extra or {})})

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            obj = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            raise ValueError(f"invalid JSON body: {exc}") from exc
        if not isinstance(obj, dict):
            raise ValueError("JSON body must be an object")
        return obj

    # -- routing ----------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        query = parse_qs(parsed.query)
        try:
            if path in ("/", "/index.html"):
                self._serve_static("index.html")
                return
            if path.startswith("/assets/"):
                self._serve_static(path[len("/assets/") :])
                return
            if path == "/favicon.svg":
                self._serve_static("favicon.svg")
                return
            if path == "/api/status":
                self._send_json(200, self._status())
                return
            if path == "/api/runs":
                self._send_json(200, {"ok": True, "runs": self.app.runs()})
                return
            if path.startswith("/api/runs/"):
                self._send_json(200, {"ok": True, **self.app.run_detail(path.split("/", 3)[3])})
                return
            if path == "/api/power":
                params: dict[str, Any] = {}
                for key in (
                    "confirmation_n",
                    "confirmation_groups",
                    "prevalence",
                    "k_candidates",
                    "effect",
                    "alpha",
                    "target_power",
                ):
                    if key in query and query[key][0] != "":
                        params[key] = query[key][0]
                self._send_json(200, {"ok": True, **self.app.power(params)})
                return
            if path == "/api/atlas":
                self._send_json(200, self._atlas())
                return
            if path == "/api/contract":
                self._send_json(200, self._contract_payload())
                return
            if path == "/api/calibration":
                cal = _read_optional(self.app.root / "calibration" / "oc_table.json")
                leak = _read_optional(self.app.root / "calibration" / "feedback_leak.json")
                if leak:
                    leak = {k: v for k, v in leak.items() if k != "rows"}
                self._send_json(200, {"ok": True, "table": cal, "feedback_leak": leak})
                return
            if path == "/api/ledger":
                self._send_json(200, {"ok": True, "vault": self.app.vault_summary(), "registry": self.app.registry_summary()})
                return
            if path == "/api/live/current":
                active = None
                for jid in reversed(self.app.live_order):
                    job = self.app.live_jobs.get(jid)
                    if job and job.status in ("queued", "running"):
                        active = job.summary()
                        break
                self._send_json(200, {"ok": True, "active": active})
                return
            if path.startswith("/api/live/") and path.endswith("/stream"):
                live_id = path.split("/")[3]
                self._stream_live(live_id, int(query.get("from", ["0"])[0] or 0))
                return
            if path.startswith("/api/live/") and path.endswith("/certificate"):
                live_id = path.split("/")[3]
                job = self.app.live(live_id)
                if not job.result:
                    self._error(404, "live run has no certificate yet")
                    return
                self._send_json(200, job.result["certificate"])
                return
            if path.startswith("/api/live/"):
                live_id = path.split("/")[3]
                self._send_json(200, {"ok": True, "job": self.app.live(live_id).summary()})
                return
            if path.startswith("/api/file/"):
                parts = path.split("/", 4)
                if len(parts) < 5:
                    self._error(400, "bad file path")
                    return
                self._serve_run_file(parts[3], parts[4])
                return
            if path == "/README.md" or path == "/docs/SPEC_COVERAGE.md":
                p = self.app.root / path.lstrip("/")
                if p.exists():
                    self._send_text(200, p.read_text(encoding="utf-8"), "text/markdown; charset=utf-8")
                else:
                    self._error(404, "document not found")
                return
            self._error(404, "not found")
        except FileNotFoundError as exc:
            self._error(404, str(exc))
        except ContractError as exc:
            self._error(422, exc.message, {"reason_code": exc.reason_code, "details": exc.details})
        except ValueError as exc:
            self._error(400, str(exc))
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            return
        except Exception as exc:
            traceback.print_exc()
            self._error(500, f"{type(exc).__name__}: {exc}")

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            body = self._body()
            if path == "/api/verify":
                result = self.app.verify_run(body.get("run_id"), body.get("certificate"))
                self._send_json(200, {"ok": True, **result})
                return
            if path == "/api/live/start":
                params = {
                    "seed": int(body.get("seed", 20261114)),
                    "patients": int(body.get("patients", 300)),
                    "confirmation_groups": int(body.get("confirmation_groups", 100)),
                    "candidates": int(body.get("candidates", 72)),
                    "generations": int(body.get("generations", 2)),
                    "permutations": int(body.get("permutations", 19)),
                }
                if not (40 <= params["patients"] <= 900):
                    self._error(400, "patients must be between 40 and 900")
                    return
                if not (8 <= params["confirmation_groups"] < params["patients"]):
                    self._error(400, "confirmation groups must be between 8 and patients-1")
                    return
                if not (8 <= params["candidates"] <= 300):
                    self._error(400, "candidates must be between 8 and 300")
                    return
                if not (1 <= params["generations"] <= 8):
                    self._error(400, "generations must be between 1 and 8")
                    return
                if not (3 <= params["permutations"] <= 299):
                    self._error(400, "permutations must be between 3 and 299")
                    return
                self._send_json(202, self.app.start_live(params))
                return
            if path == "/api/power":
                self._send_json(200, {"ok": True, **self.app.power(body)})
                return
            self._error(404, "not found")
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            return
        except ValueError as exc:
            self._error(400, str(exc))
        except Exception as exc:
            traceback.print_exc()
            self._error(500, f"{type(exc).__name__}: {exc}")

    # -- payload builders -------------------------------------------------
    def _status(self) -> dict[str, Any]:
        info = self.app.demo_info
        runs = self.app.runs()
        latest = runs[0] if runs else None
        cal = _read_optional(self.app.root / "calibration" / "oc_table.json") or {}
        generated = bool(info)
        return {
            "ok": True,
            "initialized": generated,
            "project": str(self.app.root),
            "demo": {
                "contract_hash": info.get("contract_hash") if info else None,
                "n_discovery": info.get("n_discovery") if info else None,
                "n_confirmation": info.get("n_confirmation") if info else None,
                "prevalence_discovery": info.get("prevalence_discovery") if info else None,
                "prevalence_confirmation": info.get("prevalence_confirmation") if info else None,
                "baseline_auc_discovery": info.get("baseline_auc_discovery") if info else None,
                "generator": info.get("generator") if info else None,
                "substrate": (
                    "real data: Wisconsin Diagnostic Breast Cancer (no groups, signs pre-committed)"
                    if info.get("domain") == "wdbc" else "synthetic TESLA/CEDAR-shaped fixture"
                ) if info else None,
                "domain": (info.get("domain") or "immunotherapy") if info else None,
            },
            "vault": self.app.vault_summary(),
            "registry": self.app.registry_summary(),
            "latest": latest,
            "runs_count": len(runs),
            "certificates": runs,
            "calibration": {
                "exists": bool(cal),
                "generated_at": cal.get("generated_at"),
                "rows": cal.get("rows", []),
                "reps_per_effect": (cal.get("parameters") or {}).get("reps_per_effect"),
                "workers": (cal.get("parameters") or {}).get("workers"),
                "elapsed_seconds": (cal.get("parameters") or {}).get("elapsed_seconds"),
                "fpr_alpha": cal.get("fpr_alpha"),
                "fpr_decision": cal.get("fpr_decision"),
            },
            "limits": {
                "demo": "Synthetic fixture. Not clinical evidence and not the real TESLA/CEDAR data.",
                "calibration": "Synthetic planted-signal calibration demonstrates the gate; it is not real-data instrument validation.",
                "label_service": (
                    "Label service runs as a separate process; sealed files carry an OS read lock. "
                    "A dedicated OS account for the service is a deployment step."
                ),
            },
        }

    def _atlas(self) -> dict[str, Any]:
        d = Path(__file__).with_name("domains") / "immunotherapy"
        catalog = _read_optional(d / "catalog.json") or {}
        modalities = _read_optional(d / "modalities.json") or {}
        opportunities = _read_optional(d / "opportunities.json") or {}
        brief = ""
        if (d / "brief.md").exists():
            brief = (d / "brief.md").read_text(encoding="utf-8")
        return {
            "ok": True,
            "catalog": catalog,
            "modalities": modalities,
            "opportunities": opportunities,
            "brief": brief,
        }

    def _contract_payload(self) -> dict[str, Any]:
        from .budget import compute_budget

        contract = self.app.contract()
        info = self.app.demo_info or {}
        meta = info.get("confirmation_meta") or {}
        budget = _read_optional(self.app.root / "vault" / str(meta.get("dataset_hash")) / "budget.json") or {}
        budget_info = compute_budget(contract, contract.data).to_dict()
        return {
            "ok": True,
            "contract_hash": contract.hash,
            "yaml": contract.canonical_yaml(),
            "raw": contract.to_dict(),
            "features": [f.to_dict() for f in contract.features],
            "decision": {
                "action": contract.decision.action,
                "smallest_effect_that_matters": contract.decision.smallest_effect_that_matters,
                "alpha": contract.decision.alpha,
                "target_power": contract.decision.target_power,
                "max_runs": contract.decision.max_runs,
            },
            "validation": contract.validation.to_dict() if contract.validation else None,
            "budget": budget_info,
            "discovery": {
                "n_rows": info.get("n_discovery"),
                "n_groups": (info.get("discovery_meta") or {}).get("n_groups"),
                "prevalence": info.get("prevalence_discovery"),
                "baseline_auc": info.get("baseline_auc_discovery"),
            },
            "sealed": {
                **meta,
                "budget_total": budget.get("total", meta.get("budget_total")),
                "budget_remaining": budget.get("remaining", meta.get("budget_remaining")),
            },
            "vault": self.app.vault_summary(),
            "registry": self.app.registry_summary(),
        }

    def _serve_run_file(self, run_id: str, filename: str) -> None:
        detail = self.app.run_detail(run_id)
        run_dir = self.app.root / "runs" / detail["dir"]
        name = Path(filename).name
        if name != filename or not name:
            self._error(400, "invalid file name")
            return
        p = run_dir / name
        if not p.exists() or not p.is_file():
            self._error(404, "file not found")
            return
        if p.suffix == ".json":
            self._send_text(200, p.read_text(encoding="utf-8"), "application/json; charset=utf-8")
        elif p.suffix == ".md":
            self._send_text(200, p.read_text(encoding="utf-8"), "text/markdown; charset=utf-8")
        else:
            data = p.read_bytes()
            self._headers(200, "application/octet-stream", len(data), {"Content-Disposition": f'attachment; filename="{name}"'})
            self.wfile.write(data)

    def _serve_static(self, name: str, prefix: str = "") -> None:
        base = WEB_DIR / prefix
        p = (base / name).resolve()
        try:
            p.relative_to(base.resolve())
        except ValueError:
            self._error(400, "invalid path")
            return
        if not p.exists() or not p.is_file():
            self._error(404, "asset not found")
            return
        ctype = mimetypes.guess_type(str(p))[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript", "application/json"):
            ctype += "; charset=utf-8"
        data = p.read_bytes()
        self._headers(200, ctype, len(data))
        self.wfile.write(data)

    def _stream_live(self, live_id: str, start: int = 0) -> None:
        job = self.app.live(live_id)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        sent = max(0, start)
        idle = 0
        try:
            while True:
                events = job.snapshot_events(sent)
                for ev in events:
                    seq = ev.get("seq", sent)
                    payload = json.dumps(_safe_api(ev), ensure_ascii=False, allow_nan=False)
                    self.wfile.write(f"id: {seq}\nevent: stage\ndata: {payload}\n\n".encode("utf-8"))
                    sent += 1
                self.wfile.flush()
                if job.status in ("done", "error"):
                    tail = job.snapshot_events(sent)
                    for ev in tail:
                        payload = json.dumps(_safe_api(ev), ensure_ascii=False, allow_nan=False)
                        self.wfile.write(f"event: stage\ndata: {payload}\n\n".encode("utf-8"))
                    final = json.dumps(_safe_api(job.summary()), ensure_ascii=False, allow_nan=False)
                    self.wfile.write(f"event: done\ndata: {final}\n\n".encode("utf-8"))
                    self.wfile.flush()
                    break
                idle += 1
                if idle % 50 == 0:
                    self.wfile.write(b": heartbeat\n\n")
                    self.wfile.flush()
                time.sleep(0.2)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            return


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def serve(
    root: str | Path,
    host: str = "127.0.0.1",
    port: int = 8787,
    open_browser: bool = True,
    live_patients: int = 420,
) -> int:
    root = Path(root).expanduser().resolve()
    if not WEB_DIR.exists():
        print(f"Web assets missing: {WEB_DIR}")
        return 2
    app = AppState(root)
    handler = type("AutoResearchHandler", (Handler,), {"app": app})
    httpd = ThreadingHTTPServer((host, port), handler)
    httpd.daemon_threads = True
    url = f"http://{host}:{port}/"
    print("")
    print("  AutoResearch console")
    print(f"  {url}")
    print(f"  root: {root}")
    print("  Ctrl+C to stop.")
    print("")
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        httpd.server_close()
    return 0
