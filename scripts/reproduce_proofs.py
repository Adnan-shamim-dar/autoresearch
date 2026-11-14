"""One-command reproduction of AutoResearch's published evidence.

    python scripts/reproduce_proofs.py            # about 5 minutes
    python scripts/reproduce_proofs.py --full     # also rebuilds both OC tables (about an hour)

Each check recomputes a published number from scratch in a fresh temporary
directory and compares it with the value committed in this repository. The
script writes ``reproduction_report.json``; send that file back to report a
reproduction or a failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

RPP_URL = "https://raw.githubusercontent.com/CenterForOpenScience/rpp/master/data_allformats/RPPdataConverted.xlsx"
RPP_SHA256 = "d260682aec9af5f8f7aae149fa3e017d64c53e456e9d1b11bddbf86121015e76"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def check_unit_tests(_work: Path) -> dict[str, Any]:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-m", "not integration"],
        cwd=REPO, capture_output=True, text=True,
    )
    lines = [ln for ln in proc.stdout.splitlines() if " passed" in ln or " failed" in ln or " error" in ln]
    tail = lines[-1].strip("= ") if lines else (proc.stdout + proc.stderr)[-400:]
    return {"ok": proc.returncode == 0, "observed": tail, "expected": "all passed"}


def check_backtest(work: Path) -> dict[str, Any]:
    from autoresearch.backtest import run_backtest, load_rpp

    data = work / "RPPdataConverted.xlsx"
    urllib.request.urlretrieve(RPP_URL, data)  # noqa: S310 - fixed https URL
    digest = _sha256(data)
    if digest != RPP_SHA256:
        return {"ok": False, "observed": {"data_sha256": digest}, "expected": {"data_sha256": RPP_SHA256}}
    observed = run_backtest(load_rpp(data))
    published = json.loads((REPO / "docs" / "backtest" / "results.json").read_text(encoding="utf-8"))
    keys = ("n", "accepted", "refused", "accepted_replicated", "refused_replicated")
    obs = {r: {k: observed[r][k] for k in keys} for r in ("primary", "secondary")}
    exp = {r: {k: published[r][k] for k in keys} for r in ("primary", "secondary")}
    return {"ok": obs == exp, "observed": obs, "expected": exp}


def check_wdbc(work: Path) -> dict[str, Any]:
    from autoresearch.cli import main as cli

    root = work / "wdbc"
    if cli(["--root", str(root), "init", "--domain", "wdbc"]) != 0:
        return {"ok": False, "observed": "init failed", "expected": "init ok"}
    cli(["--root", str(root), "demo", "--skip-economics"])
    certs = sorted(root.glob("runs/*/certificate.json"))
    if not certs:
        return {"ok": False, "observed": "no certificate", "expected": "refusal with sign test"}
    cert = json.loads(certs[-1].read_text(encoding="utf-8"))
    signs = (cert.get("evidence") or {}).get("sign_commitments") or {}
    obs = {"status": cert.get("status"), "signs_held": signs.get("agree"), "n_signs": signs.get("n_signs")}
    exp = {"status": "refused", "signs_held": 9, "n_signs": 10}
    verified = cli(["--root", str(root), "verify", "--run", certs[-1].parent.name]) == 0
    obs["manifest_and_signature_verify"] = verified
    exp["manifest_and_signature_verify"] = True
    return {"ok": obs == exp, "observed": obs, "expected": exp}


def check_bench_sample(_work: Path) -> dict[str, Any]:
    """Re-run 20 benchmark tasks for the three fast reference agents and compare row by row."""
    from autoresearch.bench import run_task

    published = json.loads((REPO / "bench" / "results.json").read_text(encoding="utf-8"))
    salt = published["salt"]
    keys = ("task_id", "agent", "claimed", "outcome")
    want = {(r["agent"], r["task_id"]): {k: r[k] for k in keys} for r in published["rows"]}
    obs, exp = [], []
    for agent in ("naive_search", "holdout_search", "iterating_agent"):
        for i in range(20):
            row = run_task((salt, i, agent, None))
            obs.append({k: row[k] for k in keys})
            exp.append(want[(agent, row["task_id"])])
    mismatches = [o for o, e in zip(obs, exp) if o != e]
    return {"ok": not mismatches, "observed": f"{len(obs) - len(mismatches)}/{len(obs)} rows match",
            "expected": f"{len(exp)}/{len(exp)} rows match", "mismatches": mismatches[:5]}


def _oc_rows(payload: dict[str, Any]) -> list[tuple[float, float, float]]:
    return [(r["effect"], round(r["detect"], 3), round(r["detect_confirmed"], 3)) for r in payload["rows"]]


def check_oc_synthetic(work: Path) -> dict[str, Any]:
    from autoresearch.calibration import run_calibration

    published = json.loads((REPO / "calibration" / "oc_table.json").read_text(encoding="utf-8"))
    p = published["parameters"]
    observed = run_calibration(work, reps=p["reps_per_effect"], effects=tuple(p["effects"]), candidates=p["candidates"],
                               permutations=p["permutations"], patients=p["patients"],
                               confirmation_groups=p["confirmation_groups"])
    return {"ok": _oc_rows(observed) == _oc_rows(published), "observed": _oc_rows(observed), "expected": _oc_rows(published)}


def check_oc_real(work: Path) -> dict[str, Any]:
    from autoresearch.calibration_real import run_real_calibration

    published = json.loads((REPO / "calibration" / "oc_table_real.json").read_text(encoding="utf-8"))
    p = published["parameters"]
    observed = run_real_calibration(work, reps=p["reps_per_effect"], effects=tuple(p["effects"]),
                                    candidates=p["candidates"], permutations=p["permutations"])
    return {"ok": _oc_rows(observed) == _oc_rows(published), "observed": _oc_rows(observed), "expected": _oc_rows(published)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--full", action="store_true", help="also rebuild both OC tables (slow)")
    parser.add_argument("--report", default="reproduction_report.json")
    args = parser.parse_args()

    checks: list[tuple[str, Callable[[Path], dict[str, Any]]]] = [
        ("unit_tests", check_unit_tests),
        ("replication_backtest_rpp", check_backtest),
        ("wdbc_real_data_refusal_and_signs", check_wdbc),
        ("benchmark_sample", check_bench_sample),
    ]
    if args.full:
        checks += [("oc_table_synthetic", check_oc_synthetic), ("oc_table_real_data", check_oc_real)]

    results: dict[str, Any] = {}
    for name, fn in checks:
        print(f"[....] {name}", flush=True)
        with tempfile.TemporaryDirectory(prefix=f"ar-{name}-") as tmp:
            try:
                results[name] = fn(Path(tmp))
            except Exception as exc:  # report, don't hide
                results[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        print(f"[{'PASS' if results[name]['ok'] else 'FAIL'}] {name}", flush=True)

    report = {
        "schema": "autoresearch.reproduction_report/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_head": _git_head(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "all_passed": all(r["ok"] for r in results.values()),
        "checks": results,
    }
    Path(args.report).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\n{'All checks passed' if report['all_passed'] else 'Some checks FAILED'}; report: {args.report}")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
