"""
Crack Detection Engine — monitors micro-economic indicator combinations
for early warning of regime transitions.

This is the operational intelligence layer. It doesn't just track individual
stats — it watches for PATTERNS of deterioration across multiple sectors
simultaneously, which is how real crises develop.

The key insight: no single indicator reliably predicts recessions.
But when consumer credit, corporate margins, labor market, AND real estate
all deteriorate together, the signal becomes extremely reliable.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import polars as pl

from config.micro_stats import CRACK_PATTERNS, CrackPattern

logger = logging.getLogger(__name__)


@dataclass
class IndicatorStatus:
    """Status of a single indicator within a crack pattern."""
    stat_name: str
    current_value: Optional[float]
    threshold: float
    direction: str
    breached: bool
    description: str
    severity: float = 0.0  # how far past threshold (normalized)


@dataclass
class PatternStatus:
    """Status of a crack pattern — is it active?"""
    pattern_id: int
    pattern_name: str
    target_regime: str
    signal_type: str
    lead_time_months: int
    indicators: list[IndicatorStatus]
    active_count: int  # how many indicators are breached
    trigger_count: int  # how many need to breach
    is_active: bool
    confidence: float  # active_count / total_indicators


@dataclass
class CountryCrackReport:
    """Full crack detection report for a single country."""
    country: str
    country_iso3: str
    overall_regime: str  # "thriving", "cracks_appearing", "crisis_imminent"
    regime_confidence: float
    active_patterns: list[PatternStatus]
    inactive_patterns: list[PatternStatus]
    top_risks: list[str]  # human-readable risk descriptions
    leading_indicators_breached: int
    total_indicators_monitored: int


class CrackDetector:
    """
    Monitors micro-economic indicators for early warning signals.

    Works by evaluating each CrackPattern against current data.
    When enough indicators in a pattern breach their thresholds,
    the pattern "activates" and signals a regime transition.
    """

    def __init__(self):
        self.patterns = CRACK_PATTERNS

    def evaluate_country(
        self,
        country: str,
        country_iso3: str,
        current_values: dict[str, float],
        previous_values: Optional[dict[str, float]] = None,
    ) -> CountryCrackReport:
        """
        Evaluate all crack patterns for a single country.

        Args:
            country: Country name
            country_iso3: ISO-3 code
            current_values: Dict of stat_name → current value
            previous_values: Dict of stat_name → previous value (for delta checks)

        Returns:
            CountryCrackReport with full assessment
        """
        active_patterns: list[PatternStatus] = []
        inactive_patterns: list[PatternStatus] = []
        all_risks: list[str] = []
        total_breached = 0
        total_monitored = 0

        for pattern in self.patterns:
            status = self._evaluate_pattern(
                pattern, current_values, previous_values
            )
            total_monitored += len(status.indicators)
            total_breached += status.active_count

            if status.is_active:
                active_patterns.append(status)
                # Collect risk descriptions from breached indicators
                for ind in status.indicators:
                    if ind.breached:
                        all_risks.append(
                            f"[{pattern.name}] {ind.description} "
                            f"(current={ind.current_value}, threshold={ind.threshold})"
                        )
            else:
                inactive_patterns.append(status)

        # Determine overall regime
        overall_regime = self._determine_regime(active_patterns)
        regime_confidence = self._calculate_regime_confidence(
            active_patterns, total_breached, total_monitored
        )

        return CountryCrackReport(
            country=country,
            country_iso3=country_iso3,
            overall_regime=overall_regime,
            regime_confidence=regime_confidence,
            active_patterns=active_patterns,
            inactive_patterns=inactive_patterns,
            top_risks=all_risks[:10],  # top 10 risks
            leading_indicators_breached=total_breached,
            total_indicators_monitored=total_monitored,
        )

    def _evaluate_pattern(
        self,
        pattern: CrackPattern,
        current_values: dict[str, float],
        previous_values: Optional[dict[str, float]],
    ) -> PatternStatus:
        """Evaluate a single crack pattern against current data."""
        indicator_statuses: list[IndicatorStatus] = []
        breached_count = 0

        for ind_def in pattern.indicators:
            stat_name = ind_def["stat_name"]
            direction = ind_def["direction"]
            threshold = ind_def["threshold"]
            description = ind_def["description"]

            current = current_values.get(stat_name)
            breached = False
            severity = 0.0

            if current is not None:
                if direction == "above" and current > threshold:
                    breached = True
                    severity = (current - threshold) / max(abs(threshold), 1e-6)

                elif direction == "below" and current < threshold:
                    breached = True
                    severity = (threshold - current) / max(abs(threshold), 1e-6)

                elif direction in ("above_delta", "below_delta") and previous_values:
                    previous = previous_values.get(stat_name)
                    if previous is not None:
                        delta = ((current - previous) / max(abs(previous), 1e-6)) * 100
                        if direction == "above_delta" and delta > threshold:
                            breached = True
                            severity = (delta - threshold) / max(abs(threshold), 1e-6)
                        elif direction == "below_delta" and delta < threshold:
                            breached = True
                            severity = (threshold - delta) / max(abs(threshold), 1e-6)

            if breached:
                breached_count += 1

            indicator_statuses.append(IndicatorStatus(
                stat_name=stat_name,
                current_value=current,
                threshold=threshold,
                direction=direction,
                breached=breached,
                description=description,
                severity=min(severity, 5.0),  # cap at 5x threshold
            ))

        is_active = breached_count >= pattern.trigger_count
        confidence = breached_count / max(len(pattern.indicators), 1)

        return PatternStatus(
            pattern_id=pattern.pattern_id,
            pattern_name=pattern.name,
            target_regime=pattern.target_regime,
            signal_type=pattern.signal_type,
            lead_time_months=pattern.lead_time_months,
            indicators=indicator_statuses,
            active_count=breached_count,
            trigger_count=pattern.trigger_count,
            is_active=is_active,
            confidence=confidence,
        )

    def _determine_regime(self, active_patterns: list[PatternStatus]) -> str:
        """Determine the overall economic regime from active patterns."""
        if not active_patterns:
            return "thriving"

        # Check if any crisis-level pattern is active
        crisis_patterns = [p for p in active_patterns if p.target_regime == "crisis_imminent"]
        if crisis_patterns:
            return "crisis_imminent"

        # Check how many "cracks" patterns are active
        crack_patterns = [p for p in active_patterns if p.target_regime == "cracks_appearing"]
        if len(crack_patterns) >= 2:
            return "cracks_appearing"  # multiple sectors showing stress
        elif len(crack_patterns) == 1:
            return "cracks_appearing"

        return "thriving"

    def _calculate_regime_confidence(
        self,
        active_patterns: list[PatternStatus],
        total_breached: int,
        total_monitored: int,
    ) -> float:
        """
        Calculate confidence in the regime assessment.

        Considers: number of active patterns, severity of breaches,
        and proportion of total indicators breached.
        """
        if total_monitored == 0:
            return 0.0

        # Weighted combination of signals
        pattern_signal = min(len(active_patterns) / 3.0, 1.0)  # 3+ patterns = max
        breadth_signal = total_breached / total_monitored
        severity_signal = 0.0
        if active_patterns:
            max_severity = max(
                max(ind.severity for ind in p.indicators if ind.breached)
                for p in active_patterns
                if any(ind.breached for ind in p.indicators)
            )
            severity_signal = min(max_severity / 3.0, 1.0)

        confidence = (
            0.40 * pattern_signal
            + 0.30 * breadth_signal
            + 0.30 * severity_signal
        )
        return round(confidence, 3)

    def evaluate_all_countries(
        self,
        country_data: dict[str, dict[str, float]],
        previous_data: Optional[dict[str, dict[str, float]]] = None,
    ) -> dict[str, CountryCrackReport]:
        """
        Evaluate all countries and return sorted by risk.

        Args:
            country_data: {country_iso3: {stat_name: value}}
            previous_data: {country_iso3: {stat_name: previous_value}}
        """
        from config.settings import COUNTRIES, COUNTRY_CODES

        reports: dict[str, CountryCrackReport] = {}

        for country in COUNTRIES:
            iso3 = COUNTRY_CODES.get(country, {}).get("iso3", "")
            current = country_data.get(iso3, {})
            previous = (previous_data or {}).get(iso3)

            report = self.evaluate_country(country, iso3, current, previous)
            reports[country] = report

        return reports

    def print_report(self, report: CountryCrackReport) -> None:
        """Print a formatted crack detection report for a country."""
        regime_emoji = {
            "thriving": "🟢",
            "cracks_appearing": "🟡",
            "crisis_imminent": "🔴",
        }

        logger.info("")
        logger.info("=" * 70)
        logger.info(
            "%s %s — Regime: %s (confidence: %.0f%%)",
            regime_emoji.get(report.overall_regime, "⚪"),
            report.country,
            report.overall_regime.upper().replace("_", " "),
            report.regime_confidence * 100,
        )
        logger.info("=" * 70)
        logger.info(
            "  Indicators breached: %d / %d",
            report.leading_indicators_breached,
            report.total_indicators_monitored,
        )

        if report.active_patterns:
            logger.info("  ACTIVE CRACK PATTERNS:")
            for p in report.active_patterns:
                logger.info(
                    "    ⚠ %s — %d/%d indicators breached (lead time: %d months)",
                    p.pattern_name, p.active_count, len(p.indicators), p.lead_time_months,
                )
                for ind in p.indicators:
                    status = "✗ BREACH" if ind.breached else "✓ OK"
                    val = f"{ind.current_value:.2f}" if ind.current_value is not None else "N/A"
                    logger.info(
                        "      %s %-35s value=%s  threshold=%s",
                        status, ind.stat_name, val, ind.threshold,
                    )

        if report.top_risks:
            logger.info("  TOP RISKS:")
            for risk in report.top_risks[:5]:
                logger.info("    → %s", risk)
