from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

from .certificate import verify_certificate
from .contract import ContractError, load_contract
from .utils import read_json

PROTOCOL_VERSION = "2024-11-05"
_ROOT = "."
SERVER_NAME = "autoresearch"
SERVER_VERSION = "0.1.0"


def _log(*args: Any) -> None:
    print(*args, file=sys.stderr, flush=True)


def _domain_dir() -> Path:
    return Path(__file__).parent / "domains" / "immunotherapy"


def _json_text(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, default=str)


def _tool(name: str, title: str, description: str, schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "title": title,
        "description": description,
        "inputSchema": schema,
    }


TOOLS = [
    _tool(
        "autoresearch_power",
        "Price a search",
        "Run the AutoResearch power oracle. Returns MDE before and after pricing K candidates, a verdict, a prescription, and ranked alternative designs. This is the day-one tool: it needs no search and no sealed labels.",
        {
            "type": "object",
            "properties": {
                "contract_path": {"type": "string", "description": "Path to contract YAML"},
                "data_path": {"type": "string", "description": "Path to discovery CSV/Parquet"},
                "confirmation_n": {"type": "integer", "description": "Rows in the sealed confirmation set"},
                "confirmation_groups": {"type": "integer", "description": "Groups in the sealed confirmation set"},
                "confirmation_prevalence": {"type": "number"},
                "k_candidates": {"type": "integer", "default": 100},
                "alpha": {"type": "number", "default": 0.05},
                "target_power": {"type": "number", "default": 0.8},
            },
            "required": ["contract_path", "data_path"],
        },
    ),
    _tool(
        "autoresearch_validate",
        "Validate a contract",
        "Check a contract against real data: types, group structure, missingness, metric computability, leakage scan and split compatibility. Returns named reasons, never a bare stack trace.",
        {
            "type": "object",
            "properties": {
                "contract_path": {"type": "string"},
                "data_path": {"type": "string"},
            },
            "required": ["contract_path", "data_path"],
        },
    ),
    _tool(
        "autoresearch_verify",
        "Certify a candidate",
        "verify(contract, data, candidate) -> certificate. Bring a candidate rule you already have; it is priced at K=1, "
        "checked on discovery data, confirmed once on the sealed set through the label service, attacked by the "
        "adversary, and returned as a signed certificate or a refusal. Spends one alpha unit of the sealed dataset's lifetime budget.",
        {
            "type": "object",
            "properties": {
                "contract_path": {"type": "string"},
                "data_path": {"type": "string", "description": "Discovery CSV/Parquet"},
                "candidate": {
                    "type": "object",
                    "description": "{weights: {feature: weight}, transforms?: {feature: rank|log_rank|z}, mechanism?: one sentence}",
                },
                "root": {"type": "string", "description": "Project root holding vault/ and state/ (default: server root)"},
                "sealed_dataset": {"type": "string", "description": "Sealed dataset hash (default: demo substrate)"},
                "vault_key": {"type": "string", "description": "Key path handed to the label-service process"},
                "seed": {"type": "integer", "default": 12345},
            },
            "required": ["contract_path", "data_path", "candidate"],
        },
    ),
    _tool(
        "autoresearch_check_certificate",
        "Check a certificate",
        "Check a claim certificate's hash and HMAC signature. A reviewer or another agent can call this without trusting the issuer.",
        {
            "type": "object",
            "properties": {
                "certificate": {"type": "object", "description": "Certificate JSON object"},
                "certificate_path": {"type": "string"},
                "key_path": {"type": "string", "description": "Optional HMAC key for signature verification"},
            },
        },
    ),
    _tool(
        "autoresearch_catalog",
        "Immunotherapy atlas",
        "Return the dataset catalog, modality economics, and ranked opportunities for the immunotherapy domain application.",
        {
            "type": "object",
            "properties": {
                "section": {"type": "string", "enum": ["all", "datasets", "modalities", "opportunities"], "default": "all"},
            },
        },
    ),
    _tool(
        "autoresearch_run",
        "Run the demo engine",
        "Run the end-to-end discovery demo on a scratch synthetic substrate. Returns the certificate or refusal, search price, and artifact paths. This is proof the gate can say yes; it is not a clinical result.",
        {
            "type": "object",
            "properties": {
                "root": {"type": "string", "description": "Scratch directory (default: a new temp directory)"},
                "candidate_budget": {"type": "integer", "default": 60},
                "generations": {"type": "integer", "default": 2},
                "permutations": {"type": "integer", "default": 29},
            },
        },
    ),
]


def tool_power(args: dict[str, Any]) -> dict[str, Any]:
    from .power import assess

    contract = load_contract(args["contract_path"], args["data_path"])
    power = assess(
        contract,
        contract.data,
        confirmation_n=args.get("confirmation_n"),
        confirmation_groups=args.get("confirmation_groups"),
        confirmation_prevalence=args.get("confirmation_prevalence"),
        k_candidates=int(args.get("k_candidates", contract.power.k_candidates)),
        alpha=float(args.get("alpha", contract.decision.alpha)),
        target_power=float(args.get("target_power", contract.decision.target_power)),
    )
    return power.to_dict()


def tool_validate(args: dict[str, Any]) -> dict[str, Any]:
    contract = load_contract(args["contract_path"], args["data_path"])
    return contract.validation.to_dict() if contract.validation else {"ok": True}


def tool_verify(args: dict[str, Any]) -> dict[str, Any]:
    from .api import verify
    from .registry import Registry

    root = args.get("root") or _ROOT
    detail = {k: v for k, v in args.items() if k != "candidate"}
    try:
        cert = _verify_call(verify, args, root)
    except Exception as exc:
        Registry(Path(root) / "state").log_operator_decision("mcp:autoresearch_verify", detail, f"error {type(exc).__name__}")
        raise
    Registry(Path(root) / "state").log_operator_decision("mcp:autoresearch_verify", detail, "ok")
    return cert


def _verify_call(verify, args: dict[str, Any], root: str) -> dict[str, Any]:
    return verify(
        args["contract_path"],
        args["data_path"],
        args["candidate"],
        root=root,
        sealed_dataset=args.get("sealed_dataset"),
        vault_key=args.get("vault_key"),
        seed=int(args.get("seed", 12345)),
    )


def tool_check_certificate(args: dict[str, Any]) -> dict[str, Any]:
    if args.get("certificate"):
        cert = args["certificate"]
    elif args.get("certificate_path"):
        cert = read_json(args["certificate_path"])
    else:
        raise ValueError("provide certificate or certificate_path")
    key = None
    if args.get("key_path"):
        key = Path(args["key_path"]).read_bytes()
    return verify_certificate(cert, key)


def tool_catalog(args: dict[str, Any]) -> dict[str, Any]:
    section = args.get("section", "all")
    d = _domain_dir()
    out: dict[str, Any] = {"domain": "immunotherapy", "disclaimer": "Analysis of public sources; synthetic demo is not clinical evidence."}
    if section in ("all", "datasets"):
        out["catalog"] = read_json(d / "catalog.json")
    if section in ("all", "modalities"):
        out["modalities"] = read_json(d / "modalities.json")
    if section in ("all", "opportunities"):
        out["opportunities"] = read_json(d / "opportunities.json")
    if (d / "brief.md").exists():
        out["brief"] = (d / "brief.md").read_text(encoding="utf-8")[:8000]
    return out


def tool_run(args: dict[str, Any]) -> dict[str, Any]:
    import tempfile

    from .demo import run_demo, write_demo_files

    root = Path(args.get("root") or tempfile.mkdtemp(prefix="autoresearch-mcp-"))
    write_demo_files(root, seed=20261114, reset=True, n_patients=360, n_confirmation_groups=120)
    result = run_demo(
        root,
        n_permutations=int(args.get("permutations", 29)),
        candidate_budget=int(args.get("candidate_budget", 60)),
        generations=int(args.get("generations", 2)),
    )
    cert = result["certificate"]
    return {
        "run_id": result["run_id"],
        "run_dir": result["run_dir"],
        "tier": cert.get("tier"),
        "status": cert.get("status"),
        "verdict": cert.get("verdict"),
        "claim": cert.get("claim"),
        "certificate_hash": cert.get("certificate_hash"),
        "search_price": cert.get("search_price"),
        "evidence": cert.get("evidence"),
        "certificate": cert,
    }


TOOL_HANDLERS = {
    "autoresearch_power": tool_power,
    "autoresearch_validate": tool_validate,
    "autoresearch_verify": tool_verify,
    "autoresearch_check_certificate": tool_check_certificate,
    "autoresearch_catalog": tool_catalog,
    "autoresearch_run": tool_run,
}


def resources_list() -> dict[str, Any]:
    d = _domain_dir()
    return {
        "resources": [
            {
                "uri": "autoresearch://catalog/immunotherapy",
                "name": "Immunotherapy dataset catalog",
                "description": "Datasets, labels, access terms and contract readiness.",
                "mimeType": "application/json",
            },
            {
                "uri": "autoresearch://modalities/immunotherapy",
                "name": "Immunotherapy modality economics",
                "description": "Modality status, price, turnaround and reported gross margins.",
                "mimeType": "application/json",
            },
            {
                "uri": "autoresearch://opportunities/immunotherapy",
                "name": "Immunotherapy opportunities",
                "description": "Ranked commercial opportunities derived from the domain report.",
                "mimeType": "application/json",
            },
            {
                "uri": "autoresearch://calibration/current",
                "name": "Gate operating characteristics",
                "description": "Published synthetic planted-signal OC table. A demonstration of the instrument, not real-data validation.",
                "mimeType": "application/json",
            },
        ],
        "templates": [],
    }


def resource_read(uri: str, root: Path) -> dict[str, Any]:
    d = _domain_dir()
    mapping = {
        "autoresearch://catalog/immunotherapy": d / "catalog.json",
        "autoresearch://modalities/immunotherapy": d / "modalities.json",
        "autoresearch://opportunities/immunotherapy": d / "opportunities.json",
        "autoresearch://calibration/current": root / "calibration" / "oc_table.json",
    }
    if uri not in mapping:
        raise ValueError(f"unknown resource: {uri}")
    p = mapping[uri]
    if not p.exists():
        raise FileNotFoundError(str(p))
    text = p.read_text(encoding="utf-8")
    return {
        "contents": [
            {
                "uri": uri,
                "mimeType": "application/json",
                "text": text,
            }
        ]
    }


def handle(msg: dict[str, Any], root: Path) -> dict[str, Any] | None:
    mid = msg.get("id")
    method = msg.get("method")
    params = msg.get("params") or {}
    if method is None:
        return None
    if "id" not in msg:
        return None  # notification
    try:
        if method == "initialize":
            result = {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}, "resources": {}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            }
        elif method == "ping":
            result = {}
        elif method == "tools/list":
            result = {"tools": TOOLS}
        elif method == "tools/call":
            name = params.get("name")
            arguments = params.get("arguments") or {}
            if name not in TOOL_HANDLERS:
                raise ValueError(f"unknown tool: {name}")
            payload = TOOL_HANDLERS[name](arguments)
            result = {
                "content": [{"type": "text", "text": _json_text(payload)}],
                "isError": False,
            }
        elif method == "resources/list":
            result = resources_list()
        elif method == "resources/read":
            result = resource_read(str(params.get("uri", "")), root)
        elif method == "resources/templates/list":
            result = {"resourceTemplates": []}
        elif method in ("shutdown", "exit"):
            result = {}
        else:
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"method not found: {method}"}}
        return {"jsonrpc": "2.0", "id": mid, "result": result}
    except ContractError as exc:
        return {
            "jsonrpc": "2.0",
            "id": mid,
            "result": {
                "content": [{"type": "text", "text": _json_text({"ok": False, "error": exc.to_dict()})}],
                "isError": True,
            },
        }
    except Exception as exc:
        _log(traceback.format_exc())
        return {
            "jsonrpc": "2.0",
            "id": mid,
            "result": {
                "content": [{"type": "text", "text": _json_text({"ok": False, "error": f"{type(exc).__name__}: {exc}"})}],
                "isError": True,
            },
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="autoresearch mcp", description="AutoResearch MCP stdio server")
    parser.add_argument("--root", default=".")
    args = parser.parse_args(argv)
    root = Path(args.root).expanduser().resolve()
    global _ROOT
    _ROOT = str(root)
    for stream in (sys.stdin, sys.stdout):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    _log(f"AutoResearch MCP server ready (root={root})")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as exc:
            resp = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": f"parse error: {exc}"}}
        else:
            if isinstance(msg, list):
                responses = [handle(m, root) for m in msg]
                resp = [r for r in responses if r is not None]
            else:
                resp = handle(msg, root)
        if resp is None:
            continue
        sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
        sys.stdout.flush()
        if isinstance(msg, dict) and msg.get("method") in ("shutdown", "exit"):
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
