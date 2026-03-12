"""
event_triggers.py — Apollo causal chain rule table + evaluator.

After each ingestion cycle, evaluate_triggers() reads all Flag nodes
from Neo4j, checks each rule, and fires AgentDirective messages.

Idempotency: activate_scenario triggers fire only if the scenario is
not already active. Alert triggers fire every time their conditions are
met (they are informational, not state-changing).
"""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

from knowledge_base.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

_SEND_PY = Path.home() / ".claude" / "agent-tools" / "send.py"

# ── Trigger rule table ─────────────────────────────────────────────────────────
# action: "activate_scenario" | "alert"
# conditions: all key-value pairs must match Flag nodes in Neo4j

TRIGGERS: list[dict] = [
    # Scenario subgraph 1: Global Oil Shock
    {
        "id": "OIL_SHOCK_ACTIVATE",
        "action": "activate_scenario",
        "scenario_id": "GLOBAL_OIL_SHOCK",
        "conditions": {
            "HORMUZ_STATUS": "EFFECTIVE_CLOSURE",
            "TANKER_WAR_STATUS": "OPERATIONAL",
        },
        "directive": {
            "to": "prometheus",
            "type": "status",
            "subject": "KG TRIGGER: GLOBAL_OIL_SHOCK scenario activated",
            "body": (
                "Hormuz closure + active tanker war confirmed in KG graph.\n"
                "Scenario GLOBAL_OIL_SHOCK is now active.\n"
                "Review energy_bonus config — current 1.6x may need upgrade."
            ),
        },
    },
    # Scenario subgraph 2: Nuclear Threshold Watch
    {
        "id": "BUSHEHR_WATCH",
        "action": "alert",
        "conditions": {
            "BUSHEHR_NPP_RISK": "CRITICAL",
            "NUCLEAR_FLAG": "False",
        },
        "directive": {
            "to": "apollo",
            "type": "task",
            "subject": "KG TRIGGER: Bushehr NPP risk CRITICAL — nuclear threshold watch",
            "body": (
                "NUCLEAR_FLAG is still False but BUSHEHR_NPP_RISK is CRITICAL.\n"
                "Any confirmed strike on Bushehr = set NUCLEAR_FLAG: True immediately.\n"
                "Halt all positions protocol must be ready."
            ),
        },
    },
    # Scenario subgraph 3: NATO-Russia Proxy War
    {
        "id": "RUSSIA_PROXY_WAR",
        "action": "activate_scenario",
        "scenario_id": "NATO_RUSSIA_PROXY_WAR",
        "conditions": {
            "RUSSIA_IRAN_MILITARY_COORDINATION": "CONFIRMED_UK_DEFMIN_LEVEL",
            "NATO_KINETIC_ENGAGEMENT": "True",
        },
        "directive": {
            "to": "apollo",
            "type": "task",
            "subject": "KG TRIGGER: NATO-Russia proxy war scenario activated",
            "body": (
                "Russia drone tactics confirmed (UK DefSec level) + NATO kinetic engagement.\n"
                "Scenario NATO_RUSSIA_PROXY_WAR is now active.\n"
                "Review CONFLICT_DURATION: PROTRACTED → INDEFINITE upgrade.\n"
                "Ceasefire probability: revise further down."
            ),
        },
    },
    # Scenario subgraph 4: Commodity Cascade at Historic Scale
    {
        "id": "COMMODITY_CASCADE_HISTORIC",
        "action": "activate_scenario",
        "scenario_id": "COMMODITY_CASCADE",
        "conditions": {
            "COMMODITY_CASCADE": "ACTIVE",
            "OIL_SUPPLY_DISRUPTION": "HISTORIC_SCALE",
        },
        "directive": {
            "to": "apollo",
            "type": "task",
            "subject": "KG TRIGGER: Commodity cascade at historic scale — EM contagion risk",
            "body": (
                "Multi-commodity disruption confirmed at HISTORIC_SCALE (IEA framing).\n"
                "Scenario COMMODITY_CASCADE activated.\n"
                "Food security + metals = second-order EM equity contagion.\n"
                "Review position sizing for EM-exposed names and commodity ETFs."
            ),
        },
    },
    # Scenario subgraph 5: Ceasefire Pathway Collapse
    {
        "id": "CEASEFIRE_COLLAPSE",
        "action": "alert",
        "conditions": {
            "QATAR_UNDER_ATTACK": "True",
            "OMAN_SALALAH_STRUCK": "True",
        },
        "directive": {
            "to": "apollo",
            "type": "task",
            "subject": "KG TRIGGER: Both ceasefire backchannels compromised",
            "body": (
                "Qatar ceasefire channel CLOSED + Oman Salalah struck.\n"
                "No viable ceasefire backchannel remains.\n"
                "Oil long thesis strongly reinforced — exit only at AGREEMENT level.\n"
                "CONFLICT_DURATION likely extends further."
            ),
        },
    },
]


def _conditions_met(conditions: dict[str, str], active_flags: dict[str, str]) -> bool:
    """Return True if all condition key-value pairs match active_flags."""
    return all(active_flags.get(k) == v for k, v in conditions.items())


def _send_directive(directive: dict, dry_run: bool = False) -> None:
    """Send an AgentDirective message via send.py. Non-fatal on failure."""
    if dry_run:
        return
    cmd = [
        sys.executable, str(_SEND_PY),
        "--from", "hermes",
        "--to", directive["to"],
        "--type", directive["type"],
        "--subject", directive["subject"],
        "--body", directive["body"],
        "--no-reply-expected",
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=10)
        logger.info("Directive sent to %s: %s", directive["to"], directive["subject"])
    except Exception as exc:
        logger.warning("Failed to send directive to %s: %s", directive["to"], exc)


def evaluate_triggers(client: Neo4jClient, dry_run: bool = False) -> list[str]:
    """
    Evaluate all trigger rules against current Neo4j flag state.

    Returns list of trigger IDs that fired (or would fire in dry_run).
    """
    active_flags = client.get_active_flags()
    fired: list[str] = []

    for trigger in TRIGGERS:
        trigger_id = trigger["id"]
        action = trigger["action"]

        if not _conditions_met(trigger["conditions"], active_flags):
            continue

        # Idempotency check for activate_scenario triggers
        if action == "activate_scenario":
            scenario_id = trigger["scenario_id"]
            if client.is_scenario_active(scenario_id):
                logger.debug("Trigger %s: scenario %s already active, skipping", trigger_id, scenario_id)
                continue

        logger.info("Trigger %s FIRES (dry_run=%s)", trigger_id, dry_run)
        fired.append(trigger_id)

        if not dry_run:
            if action == "activate_scenario":
                client.activate_scenario(trigger["scenario_id"])
            _send_directive(trigger["directive"], dry_run=False)

    return fired
