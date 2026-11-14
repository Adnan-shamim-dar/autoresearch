"""Certificate signing, verification, and tamper detection."""

from __future__ import annotations

import copy

import pytest

from autoresearch.certificate import (
    key_fingerprint,
    load_or_create_key,
    render_markdown,
    save_certificate,
    sign_certificate,
    verify_certificate,
)
from autoresearch.utils import read_json


@pytest.fixture()
def key(tmp_path):
    return load_or_create_key(tmp_path / "state" / "certificate.key")


@pytest.fixture()
def cert():
    return {
        "schema": "autoresearch.certificate/v1",
        "claim": "Rule beats baseline by 0.05",
        "claim_type": "prediction_rule",
        "status": "issued",
        "tier": "confirmed",
        "verdict": "affirmed",
        "evidence": {"delta": 0.05, "ci_low": 0.02, "ci_high": 0.08},
        "search_price": {"proposed": 40, "effective_independent": 12},
        "reproduce": "python -m autoresearch reproduce --run ar-x-1",
    }


def test_key_is_created_once_and_reused(tmp_path):
    path = tmp_path / "k" / "certificate.key"
    first = load_or_create_key(path)
    second = load_or_create_key(path)
    assert len(first) == 32
    assert first == second


def test_signed_certificate_verifies(cert, key):
    signed = sign_certificate(cert, key)
    result = verify_certificate(signed, key)
    assert result["valid"], result["reasons"]
    assert result["checked_signature"]
    assert signed["signature"]["key_fingerprint"] == key_fingerprint(key)


def test_signing_does_not_mutate_input(cert, key):
    before = copy.deepcopy(cert)
    sign_certificate(cert, key)
    assert cert == before


def test_tampered_claim_fails_hash_and_signature(cert, key):
    signed = sign_certificate(cert, key)
    signed["evidence"]["delta"] = 0.50
    result = verify_certificate(signed, key)
    assert not result["valid"]
    assert "certificate_hash_mismatch" in result["reasons"]
    assert "signature_mismatch" in result["reasons"]


def test_rehashed_forgery_fails_signature(cert, key):
    """A forger who recomputes the hash still cannot produce the HMAC."""
    signed = sign_certificate(cert, key)
    signed["tier"] = "confirmed"
    signed["verdict"] = "affirmed"
    signed["claim"] = "Something stronger"
    forged = sign_certificate(signed, b"x" * 32)
    forged["signature"] = signed["signature"]
    result = verify_certificate(forged, key)
    assert "certificate_hash_mismatch" not in result["reasons"]
    assert "signature_mismatch" in result["reasons"]
    assert not result["valid"]


def test_wrong_key_is_rejected(cert, key):
    signed = sign_certificate(cert, key)
    result = verify_certificate(signed, b"y" * 32)
    assert not result["valid"]
    assert "key_fingerprint_mismatch" in result["reasons"]


def test_without_key_only_hash_is_checked(cert, key):
    signed = sign_certificate(cert, key)
    assert verify_certificate(signed)["valid"]
    signed["claim"] = "edited"
    result = verify_certificate(signed)
    assert not result["valid"]
    assert result["reasons"] == ["certificate_hash_mismatch"]


def test_save_writes_json_markdown_and_manifest(cert, key, tmp_path):
    jpath, mpath = save_certificate(cert, tmp_path / "run", key)
    saved = read_json(jpath)
    assert verify_certificate(saved, key)["valid"]
    manifest = read_json(tmp_path / "run" / "certificate_manifest.json")
    assert manifest["certificate_hash"] == saved["certificate_hash"]
    md = mpath.read_text(encoding="utf-8")
    assert "CONFIRMED" in md
    assert saved["certificate_hash"] in md


def test_markdown_renders_refusal_without_optional_sections():
    md = render_markdown({"status": "refused", "claim": "Refused: underpowered", "refusals": ["power"]})
    assert "REFUSED" in md
    assert "## Refusals and limitations" in md
    assert "unsigned" in md
