"""
Geopolitical Override Manager

Allows agents to manually force HMM regime state for countries where
real-time geopolitical events have outpaced slow-moving macro indicators.

Override file: data/build/overrides/active_overrides.json
Survives KB rebuilds. phase7_build_report merges overrides into hmm_states.json.
regime_multiplier_exporter reads override_state when present and not expired.

Usage (CLI):
    python3 processing/geo_override.py set "South Korea" S2_Crisis \
        --reason "KOSPI circuit breaker; THAAD redeployment" \
        --set-by hephaestus --expires 2026-03-18

    python3 processing/geo_override.py list
    python3 processing/geo_override.py clear "South Korea"
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

VALID_STATES = {"S0_Tranquil", "S1_Turbulent", "S2_Crisis"}
DEFAULT_EXPIRY_DAYS = 7

OVERRIDES_DIR = Path(__file__).parent.parent / "data" / "build" / "overrides"
OVERRIDES_FILE = OVERRIDES_DIR / "active_overrides.json"


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _load() -> dict[str, dict]:
    if not OVERRIDES_FILE.exists():
        return {}
    return json.loads(OVERRIDES_FILE.read_text())


def _save(overrides: dict[str, dict]) -> None:
    OVERRIDES_DIR.mkdir(parents=True, exist_ok=True)
    OVERRIDES_FILE.write_text(json.dumps(overrides, indent=2, default=str))


def is_expired(entry: dict) -> bool:
    """Return True if this override entry has passed its expiry timestamp."""
    exp_str = entry.get("override_expires")
    if not exp_str:
        return True  # no expiry = treat as expired (safety default)
    try:
        exp = datetime.fromisoformat(exp_str.replace("Z", "+00:00"))
        return _now_utc() > exp
    except ValueError:
        logger.warning("Unparseable override_expires '%s' — treating as expired", exp_str)
        return True


def get_active_overrides() -> dict[str, dict]:
    """Return only non-expired overrides, logging warnings for any that have expired."""
    overrides = _load()
    active = {}
    for country, entry in overrides.items():
        if is_expired(entry):
            logger.warning(
                "Override for '%s' (state=%s, set by %s on %s) HAS EXPIRED — "
                "falling back to HMM state. Renew or clear it.",
                country,
                entry.get("override_state"),
                entry.get("override_set_by"),
                entry.get("override_set_at"),
            )
        else:
            active[country] = entry
    return active


def merge_into_states(hmm_states: dict[str, dict]) -> dict[str, dict]:
    """
    Merge active overrides into an hmm_states dict (from phase7_build_report).

    For each active override, injects override_* fields into the country entry.
    Expired overrides are included as null fields so the exporter knows to skip them.
    """
    overrides = _load()
    for country, entry in overrides.items():
        if country not in hmm_states:
            hmm_states[country] = {}
        if is_expired(entry):
            hmm_states[country].update({
                "override_state": None,
                "override_reason": None,
                "override_set_by": entry.get("override_set_by"),
                "override_set_at": entry.get("override_set_at"),
                "override_expires": entry.get("override_expires"),
            })
        else:
            hmm_states[country].update({
                "override_state": entry["override_state"],
                "override_reason": entry["override_reason"],
                "override_set_by": entry.get("override_set_by"),
                "override_set_at": entry.get("override_set_at"),
                "override_expires": entry.get("override_expires"),
            })
    return hmm_states


def set_override(
    country: str,
    state: str,
    reason: str,
    set_by: str,
    expires: datetime | None = None,
) -> None:
    if state not in VALID_STATES:
        raise ValueError(f"Invalid state '{state}'. Must be one of {VALID_STATES}")
    if not reason:
        raise ValueError("override_reason is required")

    if expires is None:
        expires = _now_utc() + timedelta(days=DEFAULT_EXPIRY_DAYS)

    overrides = _load()
    overrides[country] = {
        "override_state": state,
        "override_reason": reason,
        "override_set_by": set_by,
        "override_set_at": _now_utc().isoformat(),
        "override_expires": expires.isoformat(),
    }
    _save(overrides)
    logger.info(
        "Override set: %s → %s (expires %s, set by %s)",
        country, state, expires.isoformat()[:10], set_by,
    )


def clear_override(country: str) -> bool:
    overrides = _load()
    if country not in overrides:
        return False
    del overrides[country]
    _save(overrides)
    logger.info("Override cleared for '%s'", country)
    return True


def list_overrides() -> list[dict]:
    overrides = _load()
    rows = []
    now = _now_utc()
    for country, entry in sorted(overrides.items()):
        exp_str = entry.get("override_expires", "")
        try:
            exp = datetime.fromisoformat(exp_str.replace("Z", "+00:00"))
            days_left = (exp - now).days
            expired = now > exp
        except ValueError:
            days_left = -1
            expired = True
        rows.append({
            "country": country,
            "state": entry.get("override_state"),
            "reason": entry.get("override_reason", "")[:60],
            "set_by": entry.get("override_set_by"),
            "expires": exp_str[:10] if exp_str else "?",
            "days_left": days_left,
            "expired": expired,
        })
    return rows


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _cli() -> None:
    parser = argparse.ArgumentParser(description="Geopolitical HMM override manager")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_set = sub.add_parser("set", help="Set an override for a country")
    p_set.add_argument("country", help="Country name (must match COUNTRIES list)")
    p_set.add_argument("state", choices=sorted(VALID_STATES), help="Override state")
    p_set.add_argument("--reason", required=True, help="Reason for override")
    p_set.add_argument("--set-by", default="hephaestus", help="Agent setting the override")
    p_set.add_argument("--expires", default=None,
                       help="Expiry date YYYY-MM-DD (default: +7 days)")

    sub.add_parser("list", help="List all overrides")

    p_clear = sub.add_parser("clear", help="Clear an override")
    p_clear.add_argument("country", help="Country name")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.cmd == "set":
        expires = None
        if args.expires:
            expires = datetime.fromisoformat(args.expires).replace(tzinfo=timezone.utc)
        set_override(
            country=args.country,
            state=args.state,
            reason=args.reason,
            set_by=args.set_by,
            expires=expires,
        )
        print(f"✓ Override set: {args.country} → {args.state}")

    elif args.cmd == "list":
        rows = list_overrides()
        if not rows:
            print("No overrides set.")
            return
        print(f"{'Country':<20} {'State':<14} {'Expires':<12} {'Left':>5}  {'By':<12}  Reason")
        print("-" * 90)
        for r in rows:
            expired_tag = " [EXPIRED]" if r["expired"] else ""
            print(
                f"{r['country']:<20} {r['state']:<14} {r['expires']:<12} "
                f"{r['days_left']:>4}d  {r['set_by']:<12}  {r['reason']}{expired_tag}"
            )

    elif args.cmd == "clear":
        if clear_override(args.country):
            print(f"✓ Override cleared for '{args.country}'")
        else:
            print(f"No override found for '{args.country}'")
            sys.exit(1)


if __name__ == "__main__":
    _cli()
