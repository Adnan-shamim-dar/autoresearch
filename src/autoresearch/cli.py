from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# Windows consoles default to a legacy code page. The product speaks UTF-8.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from .certificate import load_or_create_key, verify_certificate
from .contract import ContractError, load_contract
from .utils import fmt_num, fmt_pct, read_json, sha256_file


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------

def _root(args: argparse.Namespace) -> Path:
    return Path(getattr(args, "root", ".") or ".").expanduser().resolve()


def _print_json(obj: Any) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def _rule(title: str = "") -> None:
    print("")
    if title:
        print(title)
        print("-" * len(title))
    else:
        print("-" * 64)


def _demo_info(root: Path) -> dict[str, Any] | None:
    p = root / "data" / "demo_info.json"
    if not p.exists():
        return None
    try:
        return read_json(p)
    except Exception:
        return None


def _demo_paths(root: Path, info: dict[str, Any] | None = None) -> dict[str, Path]:
    info = info or {}
    return {
        "contract": root / "examples" / "immunotherapy" / "contract.yaml",
        "discovery": root / "data" / "immunotherapy_discovery.csv",
        "confirmation": root / "data" / "immunotherapy_confirmation.csv",
        "key": root / "vault" / "keys" / "demo-confirmation.key",
        "vault": root / "vault",
        "state": root / "state",
        "runs": root / "runs",
    }


def _latest_demo_run(root: Path) -> Path | None:
    info = _demo_info(root)
    if not info:
        return None
    run_id = f"ar-{str(info.get('contract_hash', ''))[:8]}-12345"
    p = root / "runs" / run_id
    return p if p.exists() else None


def _resolve_run(root: Path, run: str) -> Path:
    p = Path(run)
    if p.is_absolute() or (p.exists() and p.is_dir()):
        return p.resolve()
    candidate = root / "runs" / run
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Run {run!r} not found under {root / 'runs'}")


def _read_certificate(run_dir: Path) -> dict[str, Any]:
    p = run_dir / "certificate.json"
    if not p.exists():
        raise FileNotFoundError(f"No certificate.json in {run_dir}")
    return read_json(p)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_init(args: argparse.Namespace) -> int:
    from .demo import write_demo_files

    root = _root(args)
    print(f"Building the synthetic immunotherapy substrate under {root} ...")
    info = write_demo_files(
        root,
        seed=args.seed,
        reset=True,
        n_patients=args.patients,
        n_confirmation_groups=args.confirm_groups,
    )
    print(f"  discovery rows : {info['n_discovery']}  (prevalence {fmt_pct(info['prevalence_discovery'])})")
    print(f"  sealed rows    : {info['n_confirmation']}  (prevalence {fmt_pct(info['prevalence_confirmation'])})")
    print(f"  baseline AUC   : {fmt_num(info['baseline_auc_discovery'])}")
    print(f"  contract hash  : {info['contract_hash']}")
    print(f"  vault          : {root / 'vault'}")
    print(f"  registry       : {root / 'state'}")
    print("\nNext: autoresearch demo")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from .demo import build_economics_refusal, run_demo, write_demo_files

    root = _root(args)
    info = _demo_info(root)
    if args.reset or not info:
        if info:
            print("--reset requested: replacing the demo substrate (vault, registry, runs).")
        info = write_demo_files(
            root,
            seed=args.seed,
            reset=True,
            n_patients=args.patients,
            n_confirmation_groups=args.confirm_groups,
        )
    t0 = time.time()
    last_stage = {"stage": None}

    def sink(ev: dict[str, Any]) -> None:
        stage = ev.get("stage")
        if args.verbose and stage != last_stage["stage"]:
            print(f"  [{ev.get('seq'):>3}] {stage:<11} {ev.get('message')}")
            last_stage["stage"] = stage

    print("Running the discovery engine against the sealed confirmation set ...")
    result = run_demo(
        root,
        n_permutations=args.permutations,
        candidate_budget=args.candidates,
        generations=args.generations,
        event_sink=sink,
    )
    cert = result["certificate"]
    elapsed = time.time() - t0
    _rule("Certificate")
    print(f"  run       : {result['run_id']}")
    print(f"  tier      : {cert.get('tier') or cert.get('status')}")
    print(f"  verdict   : {cert.get('verdict')}")
    print(f"  hash      : {cert.get('certificate_hash')}")
    print(f"  elapsed   : {elapsed:.1f}s")
    print("")
    print(cert.get("claim", ""))
    ev = cert.get("evidence") or {}
    sp = cert.get("search_price") or {}
    _rule("Search price")
    print(f"  proposed candidates     : {sp.get('proposed')}")
    print(f"  effective independent   : {sp.get('effective_independent')}")
    print(f"  selection-null p        : {(sp.get('selection_null') or {}).get('p_value')}")
    print(f"  confirmation units      : {ev.get('n_confirmation')} across {ev.get('n_groups_confirmation')} groups")
    print(f"  confirmation delta      : {fmt_num(ev.get('delta'), 4)}  (95% CI {fmt_num(ev.get('ci_low'), 4)} to {fmt_num(ev.get('ci_high'), 4)})")
    print(f"  confirmation stability  : {fmt_pct(ev.get('confirmation_stability'))}")

    if not args.skip_economics:
        try:
            econ = build_economics_refusal(root)
            ec = econ["certificate"]
            _rule("Economics refusal (first-class output)")
            print(f"  run     : {econ['run_id']}")
            print(f"  verdict : {ec.get('verdict')} / {ec.get('status')}")
            print(f"  {ec.get('claim')}")
        except Exception as exc:  # pragma: no cover - demo convenience
            print(f"  economics refusal failed: {exc}")

    print("")
    print(f"Artifacts: {result['run_dir']}")
    print(f"Certificate Markdown: {Path(result['run_dir']) / 'certificate.md'}")
    if args.json:
        _rule("Certificate JSON")
        _print_json(cert)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from .engine import Engine, RunConfig
    from .vault import SERVICE_MODE, LabelServiceProcess, Vault

    root = _root(args)
    if args.demo:
        info = _demo_info(root)
        if not info:
            raise FileNotFoundError("No demo substrate. Run `autoresearch init` first.")
        contract_path = root / "examples" / "immunotherapy" / "contract.yaml"
        discovery_path = root / "data" / "immunotherapy_discovery.csv"
        key_path = root / "vault" / "keys" / "demo-confirmation.key"
        dataset_hash = info["confirmation_meta"]["dataset_hash"]
        run_dir = root / "runs" / f"ar-{str(info['contract_hash'])[:8]}-{args.seed}"
        domain_brief = _domain_file("brief.md")
        limitations = [
            "Substrate is a synthetic TESLA/CEDAR-style demo, not the real TESLA or CEDAR data.",
            "The gate's operating-characteristics table is calibrated on the synthetic substrate; treat it as a demonstration.",
        ]
    else:
        if not args.contract or not args.discovery:
            raise SystemExit("--contract and --discovery are required unless --demo is used")
        contract_path = Path(args.contract).expanduser().resolve()
        discovery_path = Path(args.discovery).expanduser().resolve()
        if not args.dataset or not args.vault_root or not args.key:
            raise SystemExit("generic runs require --dataset, --vault-root and --key (or use --demo)")
        dataset_hash = args.dataset
        key_path = Path(args.key).expanduser().resolve()
        run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else (
            root / "runs" / f"ar-{load_contract(contract_path, discovery_path).hash[:8]}-{args.seed}"
        )
        domain_brief = ""
        limitations = []

    vault = Vault(args.vault_root or (root / "vault"))
    calibration = []
    cal_path = root / "calibration" / "oc_table.json"
    if cal_path.exists():
        try:
            calibration = read_json(cal_path).get("rows", [])
        except Exception:
            calibration = []

    def sink(ev: dict[str, Any]) -> None:
        if args.verbose:
            print(f"  [{ev.get('seq'):>3}] {ev.get('stage'):<11} {ev.get('status'):<10} {ev.get('message')}")

    service_proc = LabelServiceProcess(
        vault.root, dataset_hash, contract_path, key_path, budget=vault.meta(dataset_hash).get("budget_total")
    )
    cfg = RunConfig(
        contract_path=contract_path,
        discovery_path=discovery_path,
        run_dir=run_dir,
        vault_root=vault.root,
        registry_root=root / "state",
        sealed=None,
        sealed_dataset_hash=dataset_hash,
        run_seed=args.seed,
        n_permutations=args.permutations,
        n_splits=args.splits,
        candidate_budget=args.candidates,
        generations=args.generations,
        domain_brief=domain_brief,
        calibration_table=calibration,
        key_path=root / "state" / "certificate.key",
        event_sink=sink,
        mode=SERVICE_MODE,
        limitations=limitations,
    )
    with service_proc as client:
        cfg.sealed = client
        result = Engine(cfg).run()
    cert = result.certificate
    print(f"run {result.run_id}: tier={cert.get('tier') or cert.get('status')} verdict={cert.get('verdict')}")
    print(cert.get("claim", ""))
    print(f"certificate: {result.run_dir / 'certificate.json'}")
    if args.json:
        _print_json(cert)
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    contract_path = Path(args.contract).expanduser().resolve()
    data_path = Path(args.data).expanduser().resolve()
    try:
        contract = load_contract(contract_path, data_path)
    except ContractError as exc:
        print(f"INVALID [{exc.reason_code}] {exc.message}")
        if exc.details:
            _print_json(exc.details)
        return 2
    report = contract.validation.to_dict() if contract.validation else {}
    if args.json:
        _print_json(report)
    else:
        print(f"VALID  contract {contract.hash}")
        print(f"  rows       : {report.get('stats', {}).get('n_rows')}")
        print(f"  groups     : {report.get('stats', {}).get('n_groups')}")
        print(f"  prevalence : {fmt_pct(report.get('stats', {}).get('prevalence'))}")
        print(f"  baseline   : {contract.baseline.name} (AUC {fmt_num(report.get('stats', {}).get('baseline_auc'))})")
        for c in report.get("checks", []):
            print(f"  [{'ok' if c.get('ok') else 'FAIL'}] {c.get('name')}: {c.get('detail')}")
        for w in report.get("warnings", []):
            print(f"  [warn] {w}")
    return 0


def cmd_power(args: argparse.Namespace) -> int:
    from .power import assess, assess_continuous

    root = _root(args)
    if args.demo == "economics" or args.economics:
        result = assess_continuous(
            n_per_group=args.n_per_group,
            sd=args.sd,
            effect=args.effect or 0.10,
            alpha=args.alpha,
            target_power=args.target_power,
        )
        if args.out:
            Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        if args.json:
            _print_json(result)
        else:
            print("Economics power assessment (two-sample continuous outcome)")
            print(f"  verdict          : {result['verdict']}")
            print(f"  MDE              : {result['mde']:.3f}")
            print(f"  power            : {result['power']:.3f}")
            print(f"  required / arm   : {result['required_n_per_group']}")
            print(f"  prescription     : {result['prescription']}")
        return 0

    contract_path = Path(args.contract).expanduser().resolve() if args.contract else root / "examples" / "immunotherapy" / "contract.yaml"
    data_path = Path(args.data).expanduser().resolve() if args.data else root / "data" / "immunotherapy_discovery.csv"
    contract = load_contract(contract_path, data_path)
    info = _demo_info(root) or {}
    meta = info.get("confirmation_meta") or {}
    n_conf = args.confirmation_n or meta.get("n_rows") or contract.power.confirmation_n
    g_conf = args.confirmation_groups or meta.get("n_groups") or contract.power.confirmation_groups
    prev = args.prevalence if args.prevalence is not None else meta.get("prevalence")
    power = assess(
        contract,
        contract.data,
        confirmation_n=int(n_conf) if n_conf else None,
        confirmation_groups=int(g_conf) if g_conf else None,
        confirmation_prevalence=float(prev) if prev is not None else None,
        k_candidates=args.k or contract.power.k_candidates,
        alpha=args.alpha,
        target_power=args.target_power,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )
    if args.out:
        Path(args.out).write_text(json.dumps(power.to_dict(), indent=2), encoding="utf-8")
    if args.json:
        _print_json(power.to_dict())
        return 0
    print(f"VERDICT: {power.verdict}")
    print(f"  effect that matters     : {power.effect_that_matters:.3f}")
    print(f"  MDE plain               : {power.mde_plain:.3f}")
    print(f"  MDE after search K={power.k_candidates:<4}: {power.mde_search:.3f}")
    print(f"  confirmation rows/groups: {power.confirmation_n} / {power.confirmation_groups}")
    print(f"  baseline AUC            : {power.baseline_auc:.3f}")
    print(f"  max K answerable        : {power.max_k_answerable}")
    print(f"  required rows (search)  : {power.required_n_search}")
    print("")
    print(power.prescription)
    _rule("Design advisor (ranked)")
    for d in power.designs:
        power_txt = fmt_pct(d.get("power_at_current_n")) if d.get("power_at_current_n") is not None else "n/a"
        print(f"  - {d['name']}: power={power_txt}; required_n={d.get('required_n')}; required_groups={d.get('required_groups')}")
        print(f"      establishes: {d.get('establishes')}")
    if power.warnings:
        _rule("Warnings")
        for w in power.warnings:
            print(f"  - {w}")
    return 0


def cmd_economics(args: argparse.Namespace) -> int:
    from .demo import build_economics_refusal

    root = _root(args)
    result = build_economics_refusal(root)
    cert = result["certificate"]
    if args.json:
        _print_json(cert)
    else:
        print(f"run {result['run_id']}: {cert.get('status')} / {cert.get('verdict')}")
        print(cert.get("claim"))
        print(f"certificate: {Path(result['run_dir']) / 'certificate.json'}")
    return 0


def cmd_certificates(args: argparse.Namespace) -> int:
    root = _root(args)
    rows = []
    for p in sorted((root / "runs").glob("*/certificate.json")):
        try:
            cert = read_json(p)
        except Exception:
            continue
        rows.append(
            {
                "run_id": cert.get("run_id"),
                "tier": cert.get("tier"),
                "status": cert.get("status"),
                "verdict": cert.get("verdict"),
                "claim_type": cert.get("claim_type"),
                "contract_hash": cert.get("contract_hash"),
                "certificate_hash": cert.get("certificate_hash"),
                "claim": cert.get("claim"),
                "path": str(p.parent),
            }
        )
    if args.json:
        _print_json(rows)
        return 0
    if not rows:
        print("No certificates yet. Run `autoresearch demo`.")
        return 0
    for r in rows:
        print(f"{r['run_id']:<24} {str(r['tier'] or r['status']):<12} {r['verdict']:<14} {r['certificate_hash'][:12]}  {r['claim'][:90]}")
    print(f"\n{len(rows)} certificate(s). Denominator: {len({r['contract_hash'] for r in rows})} contract(s).")
    return 0


def cmd_certificate(args: argparse.Namespace) -> int:
    root = _root(args)
    run_dir = _resolve_run(root, args.run)
    cert = _read_certificate(run_dir)
    if args.markdown:
        print((run_dir / "certificate.md").read_text(encoding="utf-8"))
    else:
        _print_json(cert)
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    root = _root(args)
    run_dir = _resolve_run(root, args.run) if args.run else None
    cert_path = Path(args.certificate).expanduser().resolve() if args.certificate else (run_dir / "certificate.json")
    cert = read_json(cert_path)
    key_path = Path(args.key).expanduser().resolve() if args.key else root / "state" / "certificate.key"
    key = key_path.read_bytes() if key_path.exists() else None
    result = verify_certificate(cert, key)
    manifest = None
    if run_dir:
        run_json = run_dir / "run.json"
        if run_json.exists():
            rj = read_json(run_json)
            files = rj.get("files") or {}
            bad = []
            for name, expected in files.items():
                fp = (run_dir / name) if not Path(name).is_absolute() else Path(name)
                actual = sha256_file(fp) if fp.exists() else None
                if actual != expected:
                    bad.append({"file": name, "expected": expected, "actual": actual})
            manifest = {"ok": not bad, "checked": len(files), "mismatches": bad}
    if args.json:
        _print_json({"certificate": result, "manifest": manifest, "path": str(cert_path)})
    else:
        hash_ok = "certificate_hash_mismatch" not in result["reasons"]
        sig_ok = not ({"signature_mismatch", "key_fingerprint_mismatch"} & set(result["reasons"]))
        print(f"certificate hash check : {'OK' if hash_ok else 'FAILED'}")
        print(f"signature              : {('OK' if sig_ok else 'FAILED') if key else 'not checked (no key)'}")
        print(f"certificate hash       : {result['certificate_hash']}")
        print(f"key fingerprint        : {result.get('key_fingerprint')}")
        if manifest:
            print(f"artifact manifest      : {'OK' if manifest['ok'] else 'FAILED'} ({manifest['checked']} files)")
            for m in manifest["mismatches"]:
                print(f"  mismatch: {m['file']}")
        if not result["valid"]:
            print("reasons: " + ", ".join(result["reasons"]))
    if not result["valid"]:
        return 2
    if manifest is not None and not manifest["ok"]:
        return 3
    return 0


def cmd_reproduce(args: argparse.Namespace) -> int:
    root = _root(args)
    run_dir = _resolve_run(root, args.run)
    cert = _read_certificate(run_dir)
    key = (root / "state" / "certificate.key").read_bytes()
    check = verify_certificate(cert, key)
    if not check["valid"]:
        print("Cannot reproduce: certificate signature/hash does not verify.")
        _print_json(check)
        return 2
    print(f"Certificate {cert.get('certificate_hash')} verifies.")
    if not args.fresh:
        print("Use --fresh to rebuild the synthetic substrate and re-run the registered procedure.")
        print("The fresh run compares certificate hashes field by field; no sealed budget is reused.")
        return 0

    run_json = read_json(run_dir / "run.json") if (run_dir / "run.json").exists() else {}
    cfg = run_json.get("config") or {}
    info = _demo_info(root) or {}
    gen = info.get("generator") or {}
    temp = Path(tempfile.mkdtemp(prefix="autoresearch-repro-"))
    print(f"Fresh substrate: {temp}")
    try:
        from .demo import run_demo, write_demo_files

        write_demo_files(
            temp,
            seed=int(gen.get("seed", 20261114)),
            reset=True,
            n_patients=int(gen.get("n_patients", 480)),
            n_confirmation_groups=int(gen.get("n_confirmation_groups", 160)),
        )
        cal = root / "calibration" / "oc_table.json"
        if cal.exists():
            (temp / "calibration").mkdir(parents=True, exist_ok=True)
            shutil.copy2(cal, temp / "calibration" / "oc_table.json")
        t0 = time.time()
        result = run_demo(
            temp,
            n_permutations=int(cfg.get("n_permutations", args.permutations)),
            candidate_budget=int(cfg.get("candidate_budget", args.candidates)),
            generations=int(cfg.get("generations", args.generations)),
        )
        new_cert = result["certificate"]
        same_hash = new_cert.get("certificate_hash") == cert.get("certificate_hash")
        same_claim = new_cert.get("claim") == cert.get("claim")
        print(f"Fresh run complete in {time.time() - t0:.1f}s")
        print(f"  original hash : {cert.get('certificate_hash')}")
        print(f"  fresh hash    : {new_cert.get('certificate_hash')}")
        print(f"  hash match    : {'YES' if same_hash else 'NO'}")
        print(f"  claim match   : {'YES' if same_claim else 'NO'}")
        print(f"  fresh tier    : {new_cert.get('tier')}/{new_cert.get('verdict')}")
        if not (same_hash and same_claim):
            ev_old = cert.get("evidence") or {}
            ev_new = new_cert.get("evidence") or {}
            keys = ["discovery_delta", "delta", "ci_low", "ci_high", "p_one_sided", "stability_confirmation"]
            for k in keys:
                if ev_old.get(k) != ev_new.get(k):
                    print(f"  differs: {k}: {ev_old.get(k)} -> {ev_new.get(k)}")
            if not args.keep:
                pass
            return 1
    finally:
        if args.keep:
            print(f"Kept: {temp}")
        else:
            shutil.rmtree(temp, ignore_errors=True)
    return 0


def cmd_ledger(args: argparse.Namespace) -> int:
    from .registry import Registry
    from .vault import Vault

    root = _root(args)
    vault = Vault(root / "vault")
    registry = Registry(root / "state")
    vok, verr, vn = vault.verify_ledger()
    rok, rerr, rn = registry.verify()
    out = {
        "vault": {"ok": vok, "error": verr, "entries": vn, "head": vault.ledger_head()},
        "registry": {"ok": rok, "error": rerr, "entries": rn, "head": registry.entries()[-1]["hash"] if registry.entries() else "GENESIS"},
        "summary": registry.summary(),
    }
    if args.json:
        _print_json(out)
    else:
        print(f"vault ledger    : {'OK' if vok else 'BROKEN'} ({vn} entries)")
        if verr:
            print(f"  {verr}")
        print(f"registry ledger : {'OK' if rok else 'BROKEN'} ({rn} entries)")
        if rerr:
            print(f"  {rerr}")
        s = out["summary"]
        print(f"contracts       : {s['contracts_registered']} registered (denominator)")
        print(f"predictions     : {s['predictions_committed']}")
        print(f"certificates    : {s['certificates_issued']} {s['tiers']}")
        for ds, st in s["alpha"].items():
            print(f"alpha {ds[:16]:<16}: {st['spent']:.4f} / {st['allocated']:.4f}")
    return 0 if (vok and rok) else 2


def cmd_datasets(args: argparse.Namespace) -> int:
    domain = _domain_dir()
    catalog = read_json(domain / "catalog.json")
    modalities = read_json(domain / "modalities.json")
    opportunities = read_json(domain / "opportunities.json")
    if args.json:
        _print_json({"catalog": catalog, "modalities": modalities, "opportunities": opportunities})
        return 0
    print(f"{catalog['domain']}  ({len(catalog['datasets'])} datasets, {len(modalities['modalities'])} modalities)")
    print(catalog.get("disclaimer", ""))
    _rule("Datasets")
    for d in catalog["datasets"]:
        print(f"  {d['id']:<20} {d['access_tier']:<9} {d['label_status']:<12} {d['role']:<12} {d['name']}")
    _rule("Modalities")
    for m in modalities["modalities"]:
        margin = m.get("gross_margin")
        margin_s = fmt_pct(margin) if margin is not None else "n/a"
        print(f"  {m['id']:<22} {m['class']:<24} margin={margin_s:<7} price=${m.get('price_usd') or 'n/a'}")
    _rule("Opportunities")
    for o in opportunities["opportunities"]:
        print(f"  #{o['rank']} {o['name']}  score={o.get('risk_adjusted_score')}")
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    from .calibration import publish_oc_table, run_calibration

    root = _root(args)
    if args.publish_only:
        result = publish_oc_table(root)
    else:
        result = run_calibration(
            root,
            reps=args.reps,
            effects=tuple(args.effects),
            candidates=args.candidates,
            permutations=args.permutations,
            workers=args.workers,
        )
    if args.json:
        _print_json(result)
    else:
        print(f"Calibration written to {root / 'calibration' / 'oc_table.json'}")
        print(f"Human-readable table: {root / 'calibration' / 'oc_table.md'}")
        print(f"{'effect':>8} {'n':>4} {'true effect':>12} {'promoted':>9} {'confirmed':>10} {'fpr':>7}")
        for row in result["rows"]:
            print(
                f"{row['effect']:>8.3f} {row['n']:>4} {row['mean_true_effect']:>12.3f} "
                f"{row['detect']:>9.3f} {row['detect_confirmed']:>10.3f} {row['fpr']:>7.3f}"
            )
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .server import serve

    return serve(
        root=_root(args),
        host=args.host,
        port=args.port,
        open_browser=not args.no_open,
        live_patients=args.live_patients,
    )


def cmd_leak_test(args: argparse.Namespace) -> int:
    from .feedback_leak import publish, run_experiment

    root = _root(args)
    print(f"Feedback-leak test: {args.replicates} pure-null replicates per channel ...")
    result = run_experiment(replicates=args.replicates)
    path = publish(root, result)
    for ch in ("leaky", "guarded"):
        a = result["summary"][ch]
        print(f"  {ch:<8} FPR {a['fpr']:.2f} ({a['false_positives']}/{a['n']}, 95% CI {a['fpr_ci_low']:.2f}-{a['fpr_ci_high']:.2f})")
    print(f"Published: {path}")
    return 0


def cmd_label_service(args: argparse.Namespace) -> int:
    from .vault import label_service_main

    return label_service_main(
        vault_root=args.vault_root,
        dataset_hash=args.dataset,
        contract_path=args.contract,
        key_path=args.key,
        budget=args.budget,
    )


def cmd_mcp(args: argparse.Namespace) -> int:
    from .mcp_server import main as mcp_main

    return mcp_main(["--root", str(_root(args))])


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _domain_dir() -> Path:
    return Path(__file__).parent / "domains" / "immunotherapy"


def _domain_file(name: str) -> str:
    p = _domain_dir() / name
    return p.read_text(encoding="utf-8") if p.exists() else ""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="autoresearch",
        description="Discovery with a receipt: a claim certificate, or a first-class refusal.",
    )
    p.add_argument("--root", default=".", help="Project root (default: current directory)")
    sub = p.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="Build the synthetic immunotherapy demo substrate")
    init.add_argument("--seed", type=int, default=20261114)
    init.add_argument("--patients", type=int, default=480)
    init.add_argument("--confirm-groups", type=int, default=160)
    init.set_defaults(func=cmd_init)

    demo = sub.add_parser("demo", help="Run the end-to-end demo (confirmed claim + economics refusal)")
    demo.add_argument("--reset", action="store_true", help="Rebuild the substrate and reset vault/registry first")
    demo.add_argument("--seed", type=int, default=20261114)
    demo.add_argument("--patients", type=int, default=480)
    demo.add_argument("--confirm-groups", type=int, default=160)
    demo.add_argument("--candidates", type=int, default=150)
    demo.add_argument("--generations", type=int, default=4)
    demo.add_argument("--permutations", type=int, default=79)
    demo.add_argument("--skip-economics", action="store_true")
    demo.add_argument("--verbose", action="store_true")
    demo.add_argument("--json", action="store_true")
    demo.set_defaults(func=cmd_demo)

    run = sub.add_parser("run", help="Run the engine (demo or a generic contract)")
    run.add_argument("--demo", action="store_true")
    run.add_argument("--contract")
    run.add_argument("--discovery")
    run.add_argument("--vault-root")
    run.add_argument("--dataset")
    run.add_argument("--key")
    run.add_argument("--run-dir")
    run.add_argument("--seed", type=int, default=12345)
    run.add_argument("--candidates", type=int, default=None)
    run.add_argument("--generations", type=int, default=None)
    run.add_argument("--permutations", type=int, default=99)
    run.add_argument("--splits", type=int, default=5)
    run.add_argument("--verbose", action="store_true")
    run.add_argument("--json", action="store_true")
    run.set_defaults(func=cmd_run)

    validate = sub.add_parser("validate", help="Validate a contract against data")
    validate.add_argument("--contract", required=True)
    validate.add_argument("--data", required=True)
    validate.add_argument("--json", action="store_true")
    validate.set_defaults(func=cmd_validate)

    power = sub.add_parser("power", help="Price the search before proposing anything")
    power.add_argument("--contract")
    power.add_argument("--data")
    power.add_argument("--confirmation-n", type=int)
    power.add_argument("--confirmation-groups", type=int)
    power.add_argument("--prevalence", type=float)
    power.add_argument("--k", type=int)
    power.add_argument("--alpha", type=float, default=0.05)
    power.add_argument("--target-power", type=float, default=0.8)
    power.add_argument("--bootstrap", type=int, default=300)
    power.add_argument("--seed", type=int, default=12345)
    power.add_argument("--out")
    power.add_argument("--demo", choices=["immunotherapy", "economics"])
    power.add_argument("--economics", action="store_true")
    power.add_argument("--n-per-group", type=int, default=1)
    power.add_argument("--sd", type=float, default=0.12)
    power.add_argument("--effect", type=float)
    power.add_argument("--json", action="store_true")
    power.set_defaults(func=cmd_power)

    econ = sub.add_parser("economics", help="Emit the economics refusal certificate")
    econ.add_argument("--json", action="store_true")
    econ.set_defaults(func=cmd_economics)

    certs = sub.add_parser("certificates", help="List issued certificates")
    certs.add_argument("--json", action="store_true")
    certs.set_defaults(func=cmd_certificates)

    cert = sub.add_parser("certificate", help="Show a certificate")
    cert.add_argument("--run", required=True)
    cert.add_argument("--markdown", action="store_true")
    cert.set_defaults(func=cmd_certificate)

    verify = sub.add_parser("verify", help="Verify a certificate and its artifact manifest")
    verify.add_argument("--run")
    verify.add_argument("--certificate")
    verify.add_argument("--key")
    verify.add_argument("--json", action="store_true")
    verify.set_defaults(func=cmd_verify)

    repro = sub.add_parser("reproduce", help="Verify, or rebuild and re-run for byte-identical artifacts")
    repro.add_argument("--run", required=True)
    repro.add_argument("--fresh", action="store_true")
    repro.add_argument("--keep", action="store_true")
    repro.add_argument("--candidates", type=int, default=150)
    repro.add_argument("--generations", type=int, default=4)
    repro.add_argument("--permutations", type=int, default=79)
    repro.set_defaults(func=cmd_reproduce)

    ledger = sub.add_parser("ledger", help="Verify the vault and registry hash chains")
    ledger.add_argument("--json", action="store_true")
    ledger.set_defaults(func=cmd_ledger)

    datasets = sub.add_parser("datasets", help="Show the immunotherapy domain atlas")
    datasets.add_argument("--json", action="store_true")
    datasets.set_defaults(func=cmd_datasets)

    cal = sub.add_parser("calibrate", help="Run the calibration harness and publish the OC table")
    cal.add_argument("--reps", type=int, default=20)
    cal.add_argument("--effects", type=float, nargs="+", default=[0.0, 0.03, 0.06, 0.10, 0.15])
    cal.add_argument("--candidates", type=int, default=32)
    cal.add_argument("--permutations", type=int, default=39)
    cal.add_argument("--workers", type=int, default=0)
    cal.add_argument("--publish-only", action="store_true", help="Re-summarize an existing table; do not re-run")
    cal.add_argument("--json", action="store_true")
    cal.set_defaults(func=cmd_calibrate)

    serve = sub.add_parser("serve", help="Serve the web console")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8787)
    serve.add_argument("--no-open", action="store_true")
    serve.add_argument("--live-patients", type=int, default=420)
    serve.set_defaults(func=cmd_serve)

    mcp = sub.add_parser("mcp", help="Run the MCP stdio server")
    # Also accept --root after the subcommand (MCP client configs commonly do).
    mcp.add_argument("--root", dest="root", default=argparse.SUPPRESS, help="Project root")
    mcp.set_defaults(func=cmd_mcp)

    leak = sub.add_parser("leak-test", help="Run the spec §5 feedback-leak test and publish the result")
    leak.add_argument("--replicates", type=int, default=100)
    leak.set_defaults(func=cmd_leak_test)

    ls = sub.add_parser(
        "label-service",
        help="Run the sealed label service as its own process (started by the engine; not for direct use)",
    )
    ls.add_argument("--vault-root", required=True)
    ls.add_argument("--dataset", required=True)
    ls.add_argument("--contract", required=True)
    ls.add_argument("--key", required=True)
    ls.add_argument("--budget", type=int, default=None)
    ls.set_defaults(func=cmd_label_service)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except ContractError as exc:
        print(f"CONTRACT ERROR [{exc.reason_code}] {exc.message}", file=sys.stderr)
        if exc.details:
            print(json.dumps(exc.details, indent=2, ensure_ascii=False), file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"FILE ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        if os.environ.get("AUTORESEARCH_DEBUG"):
            raise
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
