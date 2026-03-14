# strategist/event_classifier.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


# Maps infra_id → list of keywords that indicate that infrastructure is involved.
# Keywords are lowercase; matching is case-insensitive.
INFRA_KEYWORD_MAP: dict[str, list[str]] = {
    "SUEZ_CANAL":      ["suez", "suez canal"],
    "HORMUZ_STRAIT":   ["hormuz", "strait of hormuz"],
    "KHARG_ISLAND":    ["kharg", "kharg island"],
    "ABQAIQ":          ["abqaiq", "abqaiq-khurais"],
    "FUJAIRAH":        ["fujairah"],
    "RAS_LAFFAN":      ["ras laffan", "ras laffan lng", "north field"],
    "JEBEL_ALI":       ["jebel ali", "jebel ali port"],
    "BAGHDAD_EMBASSY": ["baghdad embassy", "us embassy baghdad"],
}

# Words that raise severity to CRITICAL when confirmed
_CRITICAL_KEYWORDS = [
    "strike", "struck", "attack", "attacked", "explosion", "missile",
    "destroyed", "bombed", "fire", "burning", "sunk", "hit",
]

# Words that indicate a HIGH-severity threat or disruption
_HIGH_KEYWORDS = [
    "threaten", "threatens", "threat", "closure", "close", "blockade",
    "seized", "seize", "intercept", "drone", "sabotage", "damaged", "disruption",
]


@dataclass
class TriggerEvent:
    event_text: str
    severity: str           # CRITICAL | HIGH | MEDIUM | LOW
    confirmed: bool
    infra_ids: list[str] = field(default_factory=list)
    sectors: list[str] = field(default_factory=list)  # stub: populated by expander in Phase 2


class EventClassifier:
    """
    Rule-based event classifier.
    Maps news text → TriggerEvent (severity + infrastructure IDs) without LLM call.

    Severity logic:
    - CRITICAL: confirmed=True AND at least one critical-action keyword
    - HIGH: confirmed=False OR high-threat keyword (but no critical keyword)
    - MEDIUM: no keywords matched, but event mentions a known infra node
    - LOW: no infra match, no action keywords
    """

    def classify(self, event_text: str, *, confirmed: bool) -> TriggerEvent:
        lower = event_text.lower()

        # Detect infrastructure
        infra_ids = []
        for infra_id, keywords in INFRA_KEYWORD_MAP.items():
            if any(kw in lower for kw in keywords):
                infra_ids.append(infra_id)

        # Detect severity
        has_critical_kw = any(kw in lower for kw in _CRITICAL_KEYWORDS)
        has_high_kw     = any(kw in lower for kw in _HIGH_KEYWORDS)

        if confirmed and has_critical_kw:
            severity = "CRITICAL"
        elif has_high_kw or not confirmed:
            # Threat (not confirmed) or HIGH-action keyword
            # But only escalate if there's some geopolitical content
            if has_high_kw or infra_ids:
                severity = "HIGH"
            else:
                severity = "LOW"
        elif infra_ids:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        return TriggerEvent(
            event_text=event_text,
            severity=severity,
            confirmed=confirmed,
            infra_ids=infra_ids,
        )
