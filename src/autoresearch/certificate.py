from __future__ import annotations

import hmac
import hashlib
from pathlib import Path
from typing import Any

from .utils import (
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
    canonical_json,
    ensure_dir,
    json_safe,
    sha256_obj,
)


def load_or_create_key(path: str | Path) -> bytes:
    p = Path(path)
    ensure_dir(p.parent)
    if p.exists():
        return p.read_bytes()
    import secrets

    key = secrets.token_bytes(32)
    atomic_write_bytes(p, key)
    return key


def key_fingerprint(key: bytes) -> str:
    return hashlib.sha256(key).hexdigest()[:16]


def _body(cert: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in cert.items() if k not in ("certificate_hash", "signature")}


def sign_certificate(cert: dict[str, Any], key: bytes) -> dict[str, Any]:
    body = _body(cert)
    cert_hash = sha256_obj(body)
    sig = hmac.new(key, canonical_json(body).encode("utf-8"), hashlib.sha256).hexdigest()
    out = dict(cert)
    out["certificate_hash"] = cert_hash
    out["signature"] = {
        "alg": "HMAC-SHA256",
        "key_fingerprint": key_fingerprint(key),
        "value": sig,
    }
    return json_safe(out)


def verify_certificate(cert: dict[str, Any], key: bytes | None = None) -> dict[str, Any]:
    reasons: list[str] = []
    body = _body(cert)
    expected_hash = sha256_obj(body)
    if cert.get("certificate_hash") != expected_hash:
        reasons.append("certificate_hash_mismatch")
    sig = cert.get("signature") or {}
    if key is not None:
        expected_sig = hmac.new(key, canonical_json(body).encode("utf-8"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(str(sig.get("value", "")), expected_sig):
            reasons.append("signature_mismatch")
        if sig.get("key_fingerprint") != key_fingerprint(key):
            reasons.append("key_fingerprint_mismatch")
    return {
        "valid": len(reasons) == 0,
        "reasons": reasons,
        "certificate_hash": expected_hash,
        "signed": bool(sig),
        "key_fingerprint": sig.get("key_fingerprint"),
        "checked_signature": key is not None,
    }


def build_certificate(
    *,
    contract,
    run_id: str,
    seed: int,
    claim: str,
    claim_type: str,
    status: str,
    tier: str | None,
    verdict: str,
    provenance: dict[str, Any],
    search_price: dict[str, Any],
    gate_oc: dict[str, Any] | None,
    adversary: list[dict[str, Any]],
    commitments: dict[str, Any],
    interpretability_cost: dict[str, Any],
    refusals: list[Any],
    reproduce: str,
    evidence: dict[str, Any],
    limitations: list[str] | None = None,
    expires: str | None = None,
    key: bytes | None = None,
) -> dict[str, Any]:
    cert: dict[str, Any] = {
        "schema": "autoresearch.certificate/v1",
        "claim": claim,
        "claim_type": claim_type,
        "status": status,
        "tier": tier,
        "verdict": verdict,
        "contract_hash": contract.hash,
        "contract_version": contract.contract_version,
        "unit": contract.unit,
        "group": contract.group,
        "outcome": contract.outcome,
        "baseline": contract.baseline.name,
        "smallest_effect_that_matters": contract.decision.smallest_effect_that_matters,
        "decision_action": contract.decision.action,
        "run_id": run_id,
        "seed": int(seed),
        "provenance": provenance,
        "search_price": search_price,
        "gate_oc_at_this_effect": gate_oc,
        "adversary": adversary,
        "commitments": commitments,
        "interpretability_cost": interpretability_cost,
        "refusals": refusals,
        "reproduce": reproduce,
        "evidence": evidence,
        "limitations": limitations or [],
        "expires": expires,
    }
    if key is not None:
        return sign_certificate(cert, key)
    return json_safe(cert)


def save_certificate(cert: dict[str, Any], run_dir: str | Path, key: bytes | None = None) -> tuple[Path, Path]:
    d = ensure_dir(run_dir)
    if key is not None and "signature" not in cert:
        cert = sign_certificate(cert, key)
    jpath = d / "certificate.json"
    mpath = d / "certificate.md"
    atomic_write_json(jpath, cert)
    atomic_write_text(mpath, render_markdown(cert))
    atomic_write_json(d / "certificate_manifest.json", {
        "certificate_hash": cert.get("certificate_hash"),
        "files": {"certificate.json": "certificate.json", "certificate.md": "certificate.md"},
    })
    return jpath, mpath


def render_markdown(cert: dict[str, Any]) -> str:
    tier = cert.get("tier") or cert.get("status") or "unknown"
    lines = []
    lines.append(f"# AutoResearch certificate — {str(tier).upper()}")
    lines.append("")
    lines.append(f"> {cert.get('claim', '')}")
    lines.append("")
    lines.append(f"- **Status:** {cert.get('status')} / **Tier:** {cert.get('tier')} / **Verdict:** {cert.get('verdict')}")
    lines.append(f"- **Claim type:** {cert.get('claim_type')}")
    lines.append(f"- **Contract hash:** `{cert.get('contract_hash')}`")
    lines.append(f"- **Run:** `{cert.get('run_id')}`  **Seed:** {cert.get('seed')}")
    lines.append(f"- **Decision action:** {cert.get('decision_action')}")
    lines.append(f"- **Smallest effect that matters:** {cert.get('smallest_effect_that_matters')}")
    lines.append("")

    sp = cert.get("search_price") or {}
    lines.append("## Search price")
    lines.append("")
    lines.append("| Proposed | Effective independent | Alpha allocated | Alpha spent | Selection-null p | Null max-stat p95 |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    lines.append(
        f"| {sp.get('proposed', 'n/a')} | {sp.get('effective_independent', 'n/a')} | "
        f"{sp.get('alpha_allocated', 'n/a')} | {sp.get('alpha_spent', 'n/a')} | "
        f"{sp.get('selection_null', {}).get('p_value', 'n/a')} | "
        f"{sp.get('selection_null', {}).get('max_stat_p95', 'n/a')} |"
    )
    lines.append("")

    ev = cert.get("evidence") or {}
    lines.append("## Evidence")
    lines.append("")
    lines.append(f"- Confirmation units: {ev.get('n_confirmation')} across {ev.get('n_groups_confirmation')} groups")
    lines.append(f"- Delta: {ev.get('delta')} (95% CI {ev.get('ci_low')} to {ev.get('ci_high')})")
    lines.append(f"- Candidate: `{ev.get('candidate', {}).get('id', 'n/a')}` — {ev.get('candidate', {}).get('mechanism', '')}")
    lines.append(f"- Baseline: {ev.get('baseline')}")
    lines.append("")

    oc = cert.get("gate_oc_at_this_effect")
    if oc:
        lines.append("## Operating characteristics at this effect")
        lines.append("")
        lines.append("| Effect | Detection rate | False-positive rate | Source |")
        lines.append("|---:|---:|---:|---|")
        lines.append(f"| {oc.get('effect')} | {oc.get('detect')} | {oc.get('fpr')} | {oc.get('source')} |")
        lines.append("")

    adv = cert.get("adversary") or []
    if adv:
        lines.append("## Adversary")
        lines.append("")
        lines.append("| Attack | Result | p | Note |")
        lines.append("|---|---|---:|---|")
        for a in adv:
            lines.append(f"| {a.get('attack')} | {a.get('result')} | {a.get('p_value', 'n/a')} | {str(a.get('note', ''))[:140]} |")
        lines.append("")

    ref = cert.get("refusals") or []
    if ref:
        lines.append("## Refusals and limitations")
        lines.append("")
        for r in ref:
            if isinstance(r, dict):
                lines.append(f"- **{r.get('reason_code', 'refusal')}:** {r.get('message', '')}")
            else:
                lines.append(f"- {r}")
        lines.append("")

    lim = cert.get("limitations") or []
    if lim:
        lines.append("## Scope")
        lines.append("")
        for l in lim:
            lines.append(f"- {l}")
        lines.append("")

    lines.append("## Reproduce")
    lines.append("")
    lines.append("```")
    lines.append(str(cert.get("reproduce", "")))
    lines.append("```")
    lines.append("")
    lines.append(f"Certificate hash: `{cert.get('certificate_hash', 'unsigned')}`")
    sig = cert.get("signature") or {}
    if sig:
        lines.append(f"Signature: `{sig.get('alg')}` key `{sig.get('key_fingerprint')}`")
    lines.append("")
    return "\n".join(lines)
