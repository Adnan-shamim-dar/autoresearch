"""The MCP stdio server speaks JSON-RPC and exposes the agent tools (spec §11)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

SRC = str(Path(__file__).resolve().parents[1] / "src")


def _session(root, messages):
    env = dict(os.environ, PYTHONPATH=SRC)
    proc = subprocess.run(
        [sys.executable, "-m", "autoresearch", "--root", str(root), "mcp"],
        input="\n".join(json.dumps(m) for m in messages) + "\n",
        capture_output=True, text=True, env=env, timeout=120,
    )
    return [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]


def test_tools_are_listed_and_certificate_check_works(tmp_path):
    from autoresearch.certificate import sign_certificate

    key_path = tmp_path / "k.key"
    key_path.write_bytes(b"k" * 32)
    cert = sign_certificate({"claim": "x", "tier": "confirmed"}, b"k" * 32)
    out = _session(tmp_path, [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "autoresearch_check_certificate", "arguments": {"certificate": cert, "key_path": str(key_path)}}},
    ])
    by_id = {m["id"]: m for m in out}
    names = {t["name"] for t in by_id[2]["result"]["tools"]}
    assert {"autoresearch_verify", "autoresearch_check_certificate", "autoresearch_power"} <= names
    verify_schema = next(t for t in by_id[2]["result"]["tools"] if t["name"] == "autoresearch_verify")["inputSchema"]
    assert set(verify_schema["required"]) == {"contract_path", "data_path", "candidate"}
    checked = json.loads(by_id[3]["result"]["content"][0]["text"])
    assert checked["valid"] is True
