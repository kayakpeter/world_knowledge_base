"""
Open Brain MCP client — thin HTTP wrapper for search_thoughts.

POSTs JSON-RPC 2.0 to the open-brain-mcp Supabase edge function and returns
parsed thought dicts.  Returns [] on any error (no exceptions propagate).

Designed to be called from main.py to populate GEO_THOUGHTS and CRACK_HISTORIES.
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# ─── Config ──────────────────────────────────────────────────────────────────

_SUPABASE_URL = "https://sqhcdhfkinkukduvggtz.supabase.co"
_MCP_ENDPOINT = f"{_SUPABASE_URL}/functions/v1/open-brain-mcp"
_BRAIN_KEY_ENV_FILE = "/media/peter/fast-storage/.env"

# ─── Key loading ─────────────────────────────────────────────────────────────

def _load_brain_key() -> str:
    """Return BRAIN_KEY from env var or fallback .env file."""
    key = os.environ.get("BRAIN_KEY", "")
    if key:
        return key
    env_path = Path(_BRAIN_KEY_ENV_FILE)
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("BRAIN_KEY="):
                return line.split("=", 1)[1].strip()
    return ""


# ─── Text parser ─────────────────────────────────────────────────────────────

# Each block is separated by "---" lines; within a block:
#   [N] similarity=X.XXX | ISO8601_TIMESTAMP
#   meta: {JSON_OBJECT}
#   content text...
_BLOCK_HEADER = re.compile(
    r"\[(\d+)\]\s+similarity=([\d.]+)\s+\|\s+([\d\-T:.+Z]+)"
)


def _parse_thought_text(text: str) -> list[dict[str, Any]]:
    """
    Parse the MCP text response into a list of thought dicts.

    Each dict has keys:
        similarity  : float
        created_at  : str (ISO 8601)
        metadata    : dict  (parsed from "meta: {...}" line; {} on bad JSON)
        content     : str

    Returns [] if no thoughts found.
    """
    # Split on separator lines
    blocks = re.split(r"\n---+\n", text)
    thoughts: list[dict[str, Any]] = []

    for block in blocks:
        block = block.strip()
        if not block:
            continue
        header_match = _BLOCK_HEADER.search(block)
        if not header_match:
            continue

        similarity = float(header_match.group(2))
        created_at = header_match.group(3)

        # Everything after the header line
        after_header = block[header_match.end():].strip()

        # Extract meta JSON from "meta: {...}" line
        metadata: dict = {}
        content_lines: list[str] = []
        for line in after_header.splitlines():
            if line.startswith("meta:"):
                raw_json = line[len("meta:"):].strip()
                try:
                    metadata = json.loads(raw_json)
                except (json.JSONDecodeError, ValueError):
                    metadata = {}
            else:
                content_lines.append(line)

        content = "\n".join(content_lines).strip()
        thoughts.append(
            {
                "similarity": similarity,
                "created_at": created_at,
                "metadata": metadata,
                "content": content,
            }
        )

    return thoughts


# ─── Public API ──────────────────────────────────────────────────────────────

def search_thoughts(
    query: str,
    filter_agent: str | None = None,
    filter_subsystem: str | None = None,
    match_count: int = 10,
    match_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """
    Search Open Brain for thoughts matching *query*.

    Returns list of thought dicts (see _parse_thought_text).
    Returns [] on any error.
    """
    key = _load_brain_key()
    if not key:
        logger.warning("search_thoughts: BRAIN_KEY not found — returning []")
        return []

    arguments: dict[str, Any] = {
        "query": query,
        "match_count": match_count,
        "match_threshold": match_threshold,
    }
    if filter_agent:
        arguments["filter_agent"] = filter_agent
    if filter_subsystem:
        arguments["filter_subsystem"] = filter_subsystem

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "search_thoughts",
            "arguments": arguments,
        },
    }

    try:
        response = httpx.post(
            _MCP_ENDPOINT,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "x-brain-key": key,
            },
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()
        text = data["result"]["content"][0]["text"]
        return _parse_thought_text(text)
    except Exception as exc:
        logger.error(f"search_thoughts failed: {exc}")
        return []
