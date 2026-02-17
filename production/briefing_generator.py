"""
Weekly Briefing Generator — Greenshoe Intelligence Report

Produces a complete YouTube episode package from the KB data:
  1. Structured script with speaker notes and timing
  2. Visual assets (charts, dashboards, maps) as PNG files
  3. Episode metadata (title, description, tags, thumbnail concept)
  4. Social media clips (short-form hooks for each segment)

Episode Structure (target: 18-25 minutes):
  [0:00]  COLD OPEN — The single most consequential development this week
  [1:00]  INTRO — "This is the Greenshoe Intelligence Report" + episode overview
  [2:00]  SEGMENT 1: CRACK WATCH — Which economies are showing stress
  [5:00]  SEGMENT 2: THE WIRE — This week's headlines decoded through the KB
  [10:00] SEGMENT 3: DEEP DIVE — One country's generational plan in focus
  [16:00] SEGMENT 4: THE BOARD — Resource control and competition matrix updates
  [19:00] SEGMENT 5: WHAT'S NEXT — Predictions and monitoring priorities
  [22:00] CLOSING — Call to action, next week preview

The LLM generates the actual narration script. This module provides
the data extraction, visual generation, and structural scaffolding.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ─── Episode Structure ───────────────────────────────────────────────────────

@dataclass
class ScriptSegment:
    """A single segment of the episode script."""
    segment_id: str
    title: str
    subtitle: str
    duration_minutes: float
    speaker_notes: str          # what to say (narration guide, not verbatim script)
    visual_cues: list[str]      # what to show on screen
    data_points: list[str]      # specific numbers/facts to cite
    charts: list[str]           # chart filenames to display
    transitions: str            # how to transition to next segment
    social_clip_hook: str       # 15-second hook for short-form content


@dataclass
class EpisodePackage:
    """Complete episode deliverable."""
    episode_number: int
    title: str
    subtitle: str
    date: str
    duration_estimate: float    # total minutes

    # Script
    cold_open: ScriptSegment
    intro: ScriptSegment
    segments: list[ScriptSegment]
    closing: ScriptSegment

    # Metadata
    youtube_title: str
    youtube_description: str
    tags: list[str]
    thumbnail_concept: str

    # File references
    chart_files: list[str]
    script_file: str


# ─── Visual Asset Generator ──────────────────────────────────────────────────

class VisualAssetGenerator:
    """
    Generates publication-quality charts and dashboards for video use.

    Design language: Dark background (#0a0f1a), accent colors by category,
    clean typography, minimal gridlines, data-forward.
    """

    # Greenshoe brand palette
    BG_DARK = "#0a0f1a"
    BG_CARD = "#111827"
    TEXT_PRIMARY = "#e5e7eb"
    TEXT_SECONDARY = "#9ca3af"
    ACCENT_GREEN = "#10b981"     # thriving
    ACCENT_YELLOW = "#f59e0b"    # cracks
    ACCENT_RED = "#ef4444"       # crisis
    ACCENT_BLUE = "#3b82f6"     # neutral/info
    ACCENT_PURPLE = "#8b5cf6"   # strategic
    GRID_COLOR = "#1f2937"

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_crack_dashboard(
        self,
        country_regimes: dict[str, tuple[str, float, int]],
    ) -> str:
        """
        Generate the Crack Watch dashboard — 20 countries with regime status.

        Args:
            country_regimes: {country: (regime, confidence, active_patterns)}

        Returns:
            Path to saved PNG
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig, ax = plt.subplots(figsize=(16, 9), facecolor=self.BG_DARK)
        ax.set_facecolor(self.BG_DARK)

        regime_colors = {
            "thriving": self.ACCENT_GREEN,
            "cracks_appearing": self.ACCENT_YELLOW,
            "crisis_imminent": self.ACCENT_RED,
        }

        # Sort: crisis first, then cracks, then thriving
        regime_order = {"crisis_imminent": 0, "cracks_appearing": 1, "thriving": 2}
        sorted_countries = sorted(
            country_regimes.items(),
            key=lambda x: (regime_order.get(x[1][0], 3), -x[1][1]),
        )

        countries = [c for c, _ in sorted_countries]
        confidences = [v[1] * 100 for _, v in sorted_countries]
        colors = [regime_colors.get(v[0], self.ACCENT_BLUE) for _, v in sorted_countries]
        patterns = [v[2] for _, v in sorted_countries]

        y_pos = range(len(countries))
        bars = ax.barh(y_pos, confidences, color=colors, height=0.7, alpha=0.85)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(countries, fontsize=11, color=self.TEXT_PRIMARY, fontfamily="monospace")
        ax.set_xlabel("Regime Confidence %", fontsize=12, color=self.TEXT_SECONDARY)
        ax.set_title(
            "CRACK WATCH — Global Economic Regime Assessment",
            fontsize=18, color=self.TEXT_PRIMARY, fontweight="bold", pad=20,
        )

        # Add pattern count labels
        for i, (bar, pat) in enumerate(zip(bars, patterns)):
            width = bar.get_width()
            if width > 5:
                ax.text(
                    width - 2, bar.get_y() + bar.get_height() / 2,
                    f"{pat} patterns", ha="right", va="center",
                    fontsize=9, color=self.BG_DARK, fontweight="bold",
                )

        # Legend
        legend_elements = [
            mpatches.Patch(facecolor=self.ACCENT_GREEN, label="Thriving"),
            mpatches.Patch(facecolor=self.ACCENT_YELLOW, label="Cracks Appearing"),
            mpatches.Patch(facecolor=self.ACCENT_RED, label="Crisis Imminent"),
        ]
        ax.legend(
            handles=legend_elements, loc="lower right",
            fontsize=10, facecolor=self.BG_CARD, edgecolor=self.GRID_COLOR,
            labelcolor=self.TEXT_PRIMARY,
        )

        ax.set_xlim(0, 100)
        ax.xaxis.set_major_locator(plt.MultipleLocator(20))
        ax.grid(axis="x", color=self.GRID_COLOR, linewidth=0.5, alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color(self.GRID_COLOR)
        ax.spines["left"].set_color(self.GRID_COLOR)
        ax.tick_params(axis="x", colors=self.TEXT_SECONDARY)
        ax.invert_yaxis()

        # Branding
        fig.text(
            0.99, 0.01, "GREENSHOE INTELLIGENCE",
            ha="right", va="bottom", fontsize=8, color=self.TEXT_SECONDARY,
            fontfamily="monospace", alpha=0.6,
        )

        filepath = self.output_dir / "crack_dashboard.png"
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, facecolor=self.BG_DARK, bbox_inches="tight")
        plt.close()
        return str(filepath)

    def generate_scenario_probability_chart(
        self,
        scenario_shifts: dict[str, tuple[float, float]],
    ) -> str:
        """
        Generate scenario probability shift chart — before/after arrows.

        Args:
            scenario_shifts: {scenario_name: (old_probability, new_probability)}
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(16, 9), facecolor=self.BG_DARK)
        ax.set_facecolor(self.BG_DARK)

        # Sort by absolute shift magnitude
        sorted_scenarios = sorted(
            scenario_shifts.items(),
            key=lambda x: abs(x[1][1] - x[1][0]),
            reverse=True,
        )[:12]  # top 12

        names = [s[0][:40] for s in sorted_scenarios]
        old_probs = [s[1][0] * 100 for s in sorted_scenarios]
        new_probs = [s[1][1] * 100 for s in sorted_scenarios]

        y_pos = range(len(names))

        for i, (old, new) in enumerate(zip(old_probs, new_probs)):
            color = self.ACCENT_RED if new > old else self.ACCENT_GREEN
            # Old position dot
            ax.plot(old, i, "o", color=self.TEXT_SECONDARY, markersize=8, alpha=0.5)
            # Arrow to new position
            ax.annotate(
                "", xy=(new, i), xytext=(old, i),
                arrowprops=dict(arrowstyle="->", color=color, lw=2),
            )
            # New position dot
            ax.plot(new, i, "o", color=color, markersize=10, zorder=5)
            # Delta label
            delta = new - old
            sign = "+" if delta > 0 else ""
            ax.text(
                max(old, new) + 1.5, i,
                f"{sign}{delta:.1f}%", fontsize=10, color=color,
                va="center", fontweight="bold",
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=10, color=self.TEXT_PRIMARY, fontfamily="monospace")
        ax.set_xlabel("Probability (%)", fontsize=12, color=self.TEXT_SECONDARY)
        ax.set_title(
            "SCENARIO MONITOR — Weekly Probability Shifts",
            fontsize=18, color=self.TEXT_PRIMARY, fontweight="bold", pad=20,
        )

        ax.grid(axis="x", color=self.GRID_COLOR, linewidth=0.5, alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color(self.GRID_COLOR)
        ax.spines["left"].set_color(self.GRID_COLOR)
        ax.tick_params(axis="x", colors=self.TEXT_SECONDARY)
        ax.invert_yaxis()

        fig.text(
            0.99, 0.01, "GREENSHOE INTELLIGENCE",
            ha="right", va="bottom", fontsize=8, color=self.TEXT_SECONDARY,
            fontfamily="monospace", alpha=0.6,
        )

        filepath = self.output_dir / "scenario_shifts.png"
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, facecolor=self.BG_DARK, bbox_inches="tight")
        plt.close()
        return str(filepath)

    def generate_resource_control_map(
        self,
        resources: list[tuple[str, str, str, str]],
    ) -> str:
        """
        Generate resource control matrix visualization.

        Args:
            resources: [(resource, who_controls, who_needs, note)]
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(16, 9), facecolor=self.BG_DARK)
        ax.set_facecolor(self.BG_DARK)
        ax.axis("off")

        ax.set_title(
            "THE BOARD — Critical Resource Control Matrix",
            fontsize=20, color=self.TEXT_PRIMARY, fontweight="bold",
            pad=20, loc="left",
        )

        # Table layout
        col_labels = ["Resource", "Controls", "Needs", "Strategic Note"]
        col_widths = [0.20, 0.25, 0.20, 0.35]

        # Header
        x_start = 0.02
        y = 0.88
        for j, (label, width) in enumerate(zip(col_labels, col_widths)):
            ax.text(
                x_start, y, label,
                fontsize=12, color=self.ACCENT_BLUE, fontweight="bold",
                fontfamily="monospace", transform=ax.transAxes,
            )
            x_start += width

        # Separator line — use ax.plot in axes coordinates instead
        ax.plot([0.02, 0.98], [0.86, 0.86], color=self.ACCENT_BLUE, linewidth=1,
                transform=ax.transAxes)

        # Data rows
        row_height = 0.075
        for i, (resource, controls, needs, note) in enumerate(resources[:10]):
            y = 0.82 - i * row_height
            x_start = 0.02
            row_color = self.TEXT_PRIMARY if i % 2 == 0 else self.TEXT_SECONDARY

            texts = [resource[:22], controls[:28], needs[:22], note[:42]]
            for j, (text, width) in enumerate(zip(texts, col_widths)):
                ax.text(
                    x_start, y, text,
                    fontsize=10, color=row_color,
                    fontfamily="monospace", transform=ax.transAxes,
                    verticalalignment="top",
                )
                x_start += width

        fig.text(
            0.99, 0.01, "GREENSHOE INTELLIGENCE",
            ha="right", va="bottom", fontsize=8, color=self.TEXT_SECONDARY,
            fontfamily="monospace", alpha=0.6,
        )

        filepath = self.output_dir / "resource_control.png"
        plt.savefig(filepath, dpi=150, facecolor=self.BG_DARK, bbox_inches="tight")
        plt.close()
        return str(filepath)

    def generate_generational_plan_visual(
        self,
        country: str,
        plan_name: str,
        objectives: list[tuple[str, str, str, list[str]]],
    ) -> str:
        """
        Generate visual for a country's generational plan.

        Args:
            country: Country name
            plan_name: Plan name
            objectives: [(title, timeframe, status, vulnerabilities)]
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig, ax = plt.subplots(figsize=(16, 9), facecolor=self.BG_DARK)
        ax.set_facecolor(self.BG_DARK)
        ax.axis("off")

        ax.set_title(
            f"DEEP DIVE — {country}: {plan_name}",
            fontsize=20, color=self.TEXT_PRIMARY, fontweight="bold",
            pad=20, loc="left",
        )

        status_colors = {
            "on_track": self.ACCENT_GREEN,
            "behind": self.ACCENT_YELLOW,
            "ahead": self.ACCENT_BLUE,
            "stalled": self.ACCENT_RED,
            "pivoting": self.ACCENT_PURPLE,
        }

        timeframe_x = {
            "near": 0.15,
            "medium": 0.40,
            "long": 0.65,
            "generational": 0.85,
        }

        # Timeline axis
        for tf, x in timeframe_x.items():
            label = {"near": "1-3 Years", "medium": "3-10 Years",
                      "long": "10-30 Years", "generational": "30+ Years"}[tf]
            ax.text(
                x, 0.88, label,
                fontsize=11, color=self.ACCENT_BLUE, fontweight="bold",
                ha="center", transform=ax.transAxes, fontfamily="monospace",
            )

        ax.plot([0.05, 0.95], [0.85, 0.85], color=self.GRID_COLOR,
                linewidth=2, transform=ax.transAxes)

        # Plot objectives
        for i, (title, timeframe, status, vulns) in enumerate(objectives):
            x = timeframe_x.get(timeframe, 0.5)
            y = 0.75 - i * 0.14
            color = status_colors.get(status, self.TEXT_SECONDARY)

            # Objective box
            rect = mpatches.FancyBboxPatch(
                (x - 0.12, y - 0.04), 0.24, 0.10,
                boxstyle="round,pad=0.01",
                facecolor=self.BG_CARD, edgecolor=color, linewidth=2,
                transform=ax.transAxes,
            )
            ax.add_patch(rect)

            # Title
            ax.text(
                x, y + 0.02, title[:28],
                fontsize=10, color=color, fontweight="bold",
                ha="center", transform=ax.transAxes,
            )

            # Status badge
            status_label = status.replace("_", " ").upper()
            ax.text(
                x, y - 0.02, status_label,
                fontsize=8, color=self.TEXT_SECONDARY,
                ha="center", transform=ax.transAxes, fontfamily="monospace",
            )

        fig.text(
            0.99, 0.01, "GREENSHOE INTELLIGENCE",
            ha="right", va="bottom", fontsize=8, color=self.TEXT_SECONDARY,
            fontfamily="monospace", alpha=0.6,
        )

        filepath = self.output_dir / f"deep_dive_{country.lower().replace(' ', '_')}.png"
        plt.savefig(filepath, dpi=150, facecolor=self.BG_DARK, bbox_inches="tight")
        plt.close()
        return str(filepath)

    def generate_headline_impact_chart(
        self,
        headlines: list[tuple[str, str, str, list[tuple[str, str]]]],
    ) -> str:
        """
        Generate headline-to-impact visualization.

        Args:
            headlines: [(headline, urgency, sentiment, [(country, impact)])]
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(16, 9), facecolor=self.BG_DARK)
        ax.set_facecolor(self.BG_DARK)
        ax.axis("off")

        ax.set_title(
            "THE WIRE — Headlines Decoded",
            fontsize=20, color=self.TEXT_PRIMARY, fontweight="bold",
            pad=20, loc="left",
        )

        urgency_icons = {
            "routine": "○", "notable": "◐", "urgent": "●", "critical": "◉",
        }
        sentiment_colors = {
            "positive": self.ACCENT_GREEN,
            "negative": self.ACCENT_RED,
            "neutral": self.TEXT_SECONDARY,
            "mixed": self.ACCENT_YELLOW,
        }

        y = 0.88
        for headline, urgency, sentiment, impacts in headlines[:7]:
            icon = urgency_icons.get(urgency, "?")
            color = sentiment_colors.get(sentiment, self.TEXT_SECONDARY)

            # Headline
            ax.text(
                0.03, y, f"{icon} {headline[:72]}",
                fontsize=11, color=color, fontweight="bold",
                transform=ax.transAxes,
            )

            # Impacts
            impact_text = "  →  ".join(
                f"{c} {'✅' if imp == 'advances' else '⚠️' if imp == 'threatens' else '➖'}"
                for c, imp in impacts[:4]
            )
            ax.text(
                0.05, y - 0.03, impact_text,
                fontsize=9, color=self.TEXT_SECONDARY,
                transform=ax.transAxes, fontfamily="monospace",
            )

            y -= 0.12

        fig.text(
            0.99, 0.01, "GREENSHOE INTELLIGENCE",
            ha="right", va="bottom", fontsize=8, color=self.TEXT_SECONDARY,
            fontfamily="monospace", alpha=0.6,
        )

        filepath = self.output_dir / "headline_impact.png"
        plt.savefig(filepath, dpi=150, facecolor=self.BG_DARK, bbox_inches="tight")
        plt.close()
        return str(filepath)


# ─── Briefing Script Generator ───────────────────────────────────────────────

class WeeklyBriefingGenerator:
    """
    Generates a complete weekly episode package from KB data.

    In production: feed real KB data + LLM narration generation.
    In demo: uses sample data to show the full pipeline.
    """

    def __init__(self, output_dir: Path, episode_number: int = 1):
        self.output_dir = output_dir
        self.episode_number = episode_number
        self.chart_dir = output_dir / "charts"
        self.visuals = VisualAssetGenerator(self.chart_dir)

    def generate_episode(
        self,
        week_date: str = "",
        focus_country: str = "United States",
    ) -> EpisodePackage:
        """
        Generate a complete episode package.

        This is the main entry point. In production, this pulls live KB data.
        For now, uses the demo data to demonstrate the full pipeline.
        """
        if not week_date:
            week_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        logger.info("Generating Episode #%d for week of %s", self.episode_number, week_date)

        # ── Generate visual assets ────────────────────────────────────────
        chart_files = self._generate_all_charts(focus_country)

        # ── Build script segments ─────────────────────────────────────────
        cold_open = self._build_cold_open()
        intro = self._build_intro(week_date)
        seg_crack = self._build_crack_watch()
        seg_wire = self._build_the_wire()
        seg_deep = self._build_deep_dive(focus_country)
        seg_board = self._build_the_board()
        seg_next = self._build_whats_next()
        closing = self._build_closing()

        segments = [seg_crack, seg_wire, seg_deep, seg_board, seg_next]

        # ── Episode metadata ──────────────────────────────────────────────
        total_duration = sum(s.duration_minutes for s in segments)
        total_duration += cold_open.duration_minutes + intro.duration_minutes + closing.duration_minutes

        title = self._generate_episode_title()
        description = self._generate_description(segments)
        tags = self._generate_tags()
        thumbnail = self._generate_thumbnail_concept()

        # ── Write script to markdown ──────────────────────────────────────
        script_path = self._write_script_markdown(
            cold_open, intro, segments, closing, week_date, title,
        )

        package = EpisodePackage(
            episode_number=self.episode_number,
            title=title,
            subtitle=f"Week of {week_date}",
            date=week_date,
            duration_estimate=total_duration,
            cold_open=cold_open,
            intro=intro,
            segments=segments,
            closing=closing,
            youtube_title=f"#{self.episode_number}: {title} | Greenshoe Intelligence Report",
            youtube_description=description,
            tags=tags,
            thumbnail_concept=thumbnail,
            chart_files=chart_files,
            script_file=str(script_path),
        )

        logger.info("Episode package generated:")
        logger.info("  Title: %s", package.youtube_title)
        logger.info("  Duration: ~%.0f minutes", total_duration)
        logger.info("  Charts: %d visual assets", len(chart_files))
        logger.info("  Script: %s", script_path)

        return package

    def _generate_all_charts(self, focus_country: str) -> list[str]:
        """Generate all visual assets for the episode."""
        charts = []

        # 1. Crack dashboard
        crack_data = {
            "Turkey": ("cracks_appearing", 0.87, 5),
            "Argentina": ("cracks_appearing", 0.72, 4),
            "Brazil": ("cracks_appearing", 0.35, 2),
            "Italy": ("cracks_appearing", 0.28, 1),
            "United Kingdom": ("cracks_appearing", 0.22, 1),
            "South Korea": ("thriving", 0.08, 0),
            "United States": ("thriving", 0.04, 0),
            "France": ("thriving", 0.06, 0),
            "Mexico": ("thriving", 0.12, 0),
            "China": ("cracks_appearing", 0.45, 3),
            "Japan": ("thriving", 0.10, 0),
            "Germany": ("thriving", 0.15, 0),
            "India": ("thriving", 0.01, 0),
            "Saudi Arabia": ("cracks_appearing", 0.30, 2),
            "Canada": ("thriving", 0.05, 0),
            "Indonesia": ("thriving", 0.08, 0),
            "Australia": ("thriving", 0.06, 0),
            "Poland": ("thriving", 0.03, 0),
            "Russia": ("cracks_appearing", 0.55, 3),
            "Netherlands": ("thriving", 0.02, 0),
        }
        charts.append(self.visuals.generate_crack_dashboard(crack_data))

        # 2. Scenario shifts
        scenario_shifts = {
            "Oil Settles Below $55": (0.40, 0.505),
            "USMCA Renegotiation": (0.50, 0.55),
            "USMCA Withdrawal Threat": (0.15, 0.12),
            "European Energy Shock": (0.08, 0.06),
            "Global Shipping Disruption": (0.12, 0.13),
            "US-China Tech Decoupling": (0.30, 0.31),
            "Iran Nuclear Deal": (0.25, 0.32),
            "Private Credit NPL Spike": (0.45, 0.45),
            "Italian Sovereign Debt": (0.10, 0.10),
            "China Property Collapse": (0.10, 0.10),
        }
        charts.append(self.visuals.generate_scenario_probability_chart(scenario_shifts))

        # 3. Resource control
        resources = [
            ("Semiconductors (<7nm)", "USA/NLD/JPN/KOR", "CHN excluded", "Taiwan SPOF"),
            ("Rare Earths", "CHN 87% processing", "USA/CAN/AUS", "5-10yr to build alt"),
            ("Oil (swing)", "USA/SAU/RUS", "Everyone", "Venezuela changes balance"),
            ("LNG Export", "USA/QAT/AUS", "EU/JPN/KOR", "4-5yr new capacity"),
            ("Copper (EV)", "CHL/PER/ZMB/CHN", "Everyone", "Africa = copper strategy"),
            ("Lithium", "AUS/CHL/ARG/CHN", "Everyone", "Processing bottleneck"),
            ("Food (calories)", "USA/BRA/ARG", "CHN/EGY/SAU", "Climate volatility"),
            ("Water", "CAN/BRA/RUS", "SAU/IND/MEX", "Hidden constraint"),
            ("AI Compute", "USA (NVIDIA/AMD)", "CHN developing alt", "Energy is binding"),
            ("Submarine Cables", "USA/GBR/FRA", "CHN building alt", "95% of data flows"),
        ]
        charts.append(self.visuals.generate_resource_control_map(resources))

        # 4. Deep dive country
        from processing.strategic_analysis import GENERATIONAL_PLANS
        plan = next((p for p in GENERATIONAL_PLANS if p.country == focus_country), None)
        if plan:
            objectives = [
                (obj.title, obj.timeframe, obj.current_status, obj.vulnerabilities)
                for obj in plan.objectives
            ]
            charts.append(self.visuals.generate_generational_plan_visual(
                focus_country, plan.plan_name, objectives,
            ))

        # 5. Headline impact
        headline_data = [
            ("Carney announces Chief Trade Negotiator to US", "notable", "neutral",
             [("CAN", "advances"), ("USA", "advances")]),
            ("Rubio courts Slovakia/Hungary on energy", "notable", "positive",
             [("USA", "advances"), ("RUS", "threatens"), ("DEU", "advances")]),
            ("Iran signals nuclear deal compromise", "urgent", "positive",
             [("SAU", "threatens"), ("USA", "advances"), ("IND", "advances")]),
            ("US forces board tanker (Venezuela)", "urgent", "negative",
             [("USA", "advances"), ("RUS", "threatens"), ("CHN", "threatens")]),
            ("Trump-Netanyahu align, split on endgame", "notable", "mixed",
             [("USA", "neutral"), ("ISR", "neutral"), ("IRN", "threatens")]),
            ("EU diplomat rejects US rhetoric", "notable", "negative",
             [("USA", "threatens"), ("DEU", "advances"), ("CHN", "advances")]),
            ("UK rain damages farming", "routine", "negative",
             [("GBR", "threatens")]),
        ]
        charts.append(self.visuals.generate_headline_impact_chart(headline_data))

        logger.info("Generated %d visual assets", len(charts))
        return charts

    def _build_cold_open(self) -> ScriptSegment:
        return ScriptSegment(
            segment_id="cold_open",
            title="COLD OPEN",
            subtitle="The Most Consequential Development This Week",
            duration_minutes=1.0,
            speaker_notes=(
                "Open on the Iran nuclear headline. This is the single biggest probability shift "
                "in our model this week — Oil Below $55 jumped 10.5 percentage points. That one "
                "headline cascades into Saudi fiscal math, Russian war economy sustainability, "
                "Canadian energy revenue, and the entire OPEC+ framework. If Iran returns to "
                "full production, the global energy chess board resets. Let's break it down."
            ),
            visual_cues=[
                "Show Oil Below $55 scenario probability arrow: 40% → 50.5%",
                "Flash 4 country flags affected: SAU, RUS, CAN, USA",
                "Cut to title card",
            ],
            data_points=[
                "Oil Below $55 scenario: +10.5% probability this week",
                "Iran signaling nuclear compromise for first time in 18 months",
                "Combined effect of Iran + Venezuela blockade + Rubio CEE tour",
            ],
            charts=["scenario_shifts.png"],
            transitions="'But that's just one thread in a web of developments this week. Let's start with the big picture.'",
            social_clip_hook="Iran just moved the oil probability needle by 10%. Here's what that means for 7 countries. [Thread]",
        )

    def _build_intro(self, week_date: str) -> ScriptSegment:
        return ScriptSegment(
            segment_id="intro",
            title="INTRO",
            subtitle=f"Greenshoe Intelligence Report — Week of {week_date}",
            duration_minutes=1.0,
            speaker_notes=(
                "Welcome to the Greenshoe Intelligence Report, where we decode global events through "
                "the lens of sovereign economic data, generational strategic plans, and multi-order "
                "consequence analysis. I'm Peter, and this week we're tracking 110 economic indicators "
                "across 20 countries, 50 risk scenarios, and 7 major powers' long-term strategies. "
                "Here's what moved this week."
            ),
            visual_cues=[
                "Greenshoe logo animation",
                "Quick montage of this week's charts",
                "Episode overview card with 5 segment titles",
            ],
            data_points=[
                "110 stats × 20 countries = 2,200 data points monitored",
                "50 scenarios from baseline to black swan",
                "7 generational plans tracked",
            ],
            charts=[],
            transitions="'Let's start where we always start — the crack watch.'",
            social_clip_hook="",
        )

    def _build_crack_watch(self) -> ScriptSegment:
        return ScriptSegment(
            segment_id="seg1_crack_watch",
            title="CRACK WATCH",
            subtitle="Global Economic Regime Assessment",
            duration_minutes=3.5,
            speaker_notes=(
                "Our crack detection engine monitors 7 multi-indicator patterns across all 20 economies. "
                "This week's assessment: Turkey remains the most stressed economy with 5 active crack "
                "patterns at 87% confidence — consumer credit, corporate spreads, labor market, EM "
                "vulnerability, and infrastructure constraints all firing. Russia is showing stress at 55% "
                "as the war economy burns through reserves. China at 45% — property sector is the big one. "
                "Argentina at 72% — no surprises there. The interesting one is Saudi Arabia creeping up to "
                "30% — the Iran deal headline is what's driving that, because if oil stays below Saudi's "
                "fiscal breakeven of $78, Vision 2030 starts getting cut.\n\n"
                "Key indicator moves: US credit card delinquencies at 3.8%, above our 3.5% threshold but "
                "no pattern activation because the other consumer indicators are holding. Watch this one — "
                "auto loan delinquencies at 4.2% are also above threshold. If savings rate drops below 3%, "
                "the Consumer Credit Stress pattern activates for the US."
            ),
            visual_cues=[
                "Full crack dashboard chart (20 countries, color-coded)",
                "Zoom on Turkey — 5 pattern breakdown",
                "Zoom on US — near-miss indicators highlighted",
                "Saudi Arabia trend with oil price overlay",
            ],
            data_points=[
                "Turkey: 87% confidence, 5 active patterns, 19/34 indicators breached",
                "Russia: 55% confidence, 3 active patterns (energy, labor, EM vulnerability)",
                "China: 45% confidence, 3 active patterns (property-driven)",
                "US: 4 indicators breached but 0 patterns activated (near-miss on consumer credit)",
                "Saudi Arabia: rising to 30% on oil price pressure",
            ],
            charts=["crack_dashboard.png"],
            transitions="'Now let's look at what happened this week that's moving these numbers.'",
            social_clip_hook="Turkey has 5 economic crack patterns firing simultaneously. Saudi Arabia just entered our watch list. Here's why.",
        )

    def _build_the_wire(self) -> ScriptSegment:
        return ScriptSegment(
            segment_id="seg2_the_wire",
            title="THE WIRE",
            subtitle="This Week's Headlines Decoded",
            duration_minutes=5.0,
            speaker_notes=(
                "Every headline is a signal. Our system maps each one to specific actors, affected stats, "
                "scenario probability shifts, and generational plan impacts. Let me walk you through the "
                "7 most consequential signals from this week.\n\n"
                "Number 1: Carney appoints a Chief Trade Negotiator. This isn't just a personnel announcement. "
                "This is Canada signaling to the US that USMCA is a top-of-government priority. It shifts "
                "our USMCA Renegotiation scenario from 50% to 55%, and the Withdrawal Threat drops from "
                "15% to 12%. For Carney, this is objective CAN_OBJ_1 — USMCA survival is existential.\n\n"
                "Number 2: Rubio in Slovakia and Hungary. This is Energy Dominance in action — USA_OBJ_1. "
                "The US is actively trying to replace Russian gas in Central Eastern Europe with American LNG. "
                "Every pipeline that switches from Gazprom to Cheniere is a direct revenue hit to Russia's "
                "war economy. This threatens RUS_OBJ_1 and advances DEU_OBJ_1.\n\n"
                "Number 3: Iran nuclear compromise signal. This is the big one. If Iran returns to full "
                "production, that's 1.5 million barrels per day back on the market. Oil Below $55 "
                "jumped 5 percentage points on this alone. Saudi Arabia's fiscal breakeven is $78 — "
                "that's the gap between Vision 2030 continuing and Vision 2030 getting cut.\n\n"
                "Number 4: Venezuela tanker boarding. The US military is now physically enforcing energy "
                "policy. This is a precedent — the same interdiction framework applies to Russian shadow "
                "fleet tankers. Russia noticed.\n\n"
                "I'll cover the remaining signals in the script notes, but the pattern is clear: "
                "energy is the dominant axis this week. Five of seven headlines touch energy markets."
            ),
            visual_cues=[
                "Headline impact chart — all 7 headlines with country impacts",
                "Scenario shift arrows for each headline",
                "Map: Rubio's CEE energy tour route",
                "Oil supply infographic: Iran + Venezuela + Saudi dynamics",
            ],
            data_points=[
                "USMCA Renegotiation: 50% → 55%",
                "Oil Below $55: 40% → 50.5% (biggest single-week shift)",
                "European Energy Shock: 8% → 6% (Rubio's CEE play helping)",
                "USMCA Withdrawal: 15% → 12%",
                "5 of 7 headlines touch energy markets directly",
            ],
            charts=["headline_impact.png", "scenario_shifts.png"],
            transitions="'Now let's go deeper on one country. This week: the United States.'",
            social_clip_hook="5 of 7 global headlines this week are about one thing: energy. Here's the chess game nobody's explaining.",
        )

    def _build_deep_dive(self, country: str) -> ScriptSegment:
        return ScriptSegment(
            segment_id="seg3_deep_dive",
            title="DEEP DIVE",
            subtitle=f"{country} — Generational Strategic Plan",
            duration_minutes=6.0,
            speaker_notes=(
                "Every major power is executing a multi-decade plan. This week we focus on the United States — "
                "'American Primacy Renewal.' Five objectives on the board.\n\n"
                "Objective 1: Energy Dominance. Status: ON TRACK. This week validates it — Rubio selling LNG "
                "in Europe, Venezuela blockade controlling supply, Iran deal potentially adding supply that "
                "advantages US as price-maker not price-taker. The US is playing this brilliantly.\n\n"
                "Objective 2: Technology Supremacy. ON TRACK. CHIPS Act fabs coming online 2026-2027. But "
                "there's a hidden dependency: AI needs energy. Objective 2 depends on Objective 1. This is "
                "why the grid reserve margin stat matters — if the US can't power the data centers, the AI "
                "lead evaporates.\n\n"
                "Objective 3: Dollar Hegemony Defense. BEHIND. This is the weak flank. The Kallas-Rubio "
                "clash this week is symptomatic — every time the US alienates allies, it weakens the "
                "coalition that upholds dollar primacy. Meanwhile, China's RMB internationalization is "
                "quiet but persistent. The debt trajectory is the structural threat — $1.9T annual deficit.\n\n"
                "Objective 4: Supply Chain Reshoring. BEHIND. USMCA is the vehicle here. Carney's trade "
                "negotiator appointment is actually good news for this objective — a stable USMCA is "
                "the foundation for nearshoring.\n\n"
                "Objective 5: Fiscal Sustainability. BEHIND. This is the meta-risk. If the debt spiral "
                "becomes self-reinforcing, it undermines all other objectives simultaneously. "
                "OBBBA adds $3.7T — the bet is that growth outruns debt. That's a bet, not a plan.\n\n"
                "The biggest risk: forced austerity from debt crisis cuts defense, AI investment, and "
                "energy infrastructure simultaneously. Everything is connected."
            ),
            visual_cues=[
                "Generational plan visual — 5 objectives on timeline",
                "Status indicators: 2 green, 3 yellow",
                "Dependency map: Energy → Tech → Fiscal chain",
                "Debt trajectory chart with OBBBA impact",
            ],
            data_points=[
                "Energy Dominance: ON TRACK — US now world's largest oil producer",
                "Tech Supremacy: ON TRACK — CHIPS Act fabs 2026-2027 timeline",
                "Dollar Hegemony: BEHIND — $1.9T annual deficit trajectory",
                "Supply Chain: BEHIND — USMCA renewal critical",
                "Fiscal: BEHIND — OBBBA adds $3.7T, betting on growth",
            ],
            charts=[f"deep_dive_{country.lower().replace(' ', '_')}.png"],
            transitions="'Let's pull back to the global view.'",
            social_clip_hook="The US has 5 strategic objectives. 2 are on track, 3 are behind. The one nobody talks about could unravel them all.",
        )

    def _build_the_board(self) -> ScriptSegment:
        return ScriptSegment(
            segment_id="seg4_the_board",
            title="THE BOARD",
            subtitle="Resource Control & Competition Matrix",
            duration_minutes=3.0,
            speaker_notes=(
                "Quick hits on the resource game. Semiconductors: unchanged, US/NLD/JPN/KOR alliance "
                "holding. Taiwan remains the single point of failure. Rare earths: Canada advancing on "
                "processing — watch Saskatchewan. Copper: China's Africa hospital strategy is fundamentally "
                "a copper strategy. Every BRI infrastructure project in Zambia, Congo, Peru is about "
                "locking up the copper supply chain. Oil: the big mover this week. Three simultaneous "
                "developments — Iran compromise, Venezuela enforcement, Rubio CEE tour — all shift the "
                "supply picture. LNG: Rubio's Slovakia/Hungary visit is about creating new demand "
                "for US LNG exports. This is Energy Dominance in real time.\n\n"
                "Competition update: USA vs China intensifying on tech. USA vs Russia escalating on "
                "energy and shadow fleet interdiction. Canada vs Australia heating up on critical minerals. "
                "India quietly advancing on manufacturing — Apple/Samsung shift accelerating."
            ),
            visual_cues=[
                "Resource control matrix chart",
                "Highlight changes from last week in yellow",
                "Competition pair arrows with intensity indicators",
            ],
            data_points=[
                "Oil: 3 supply-side developments this week",
                "China controls 87% of rare earth processing",
                "Canada positioning as democratic-aligned mineral alternative",
                "US LNG exports to Europe growing 15% YoY",
            ],
            charts=["resource_control.png"],
            transitions="'Finally, what are we watching for next week.'",
            social_clip_hook="China builds hospitals in Africa. Sounds nice. Follow the copper.",
        )

    def _build_whats_next(self) -> ScriptSegment:
        return ScriptSegment(
            segment_id="seg5_whats_next",
            title="WHAT'S NEXT",
            subtitle="Predictions & Monitoring Priorities",
            duration_minutes=3.0,
            speaker_notes=(
                "Five things to watch next week:\n\n"
                "1. Iran back-channel status. If the BBC interview translates into actual diplomatic "
                "contact, the Oil Below $55 scenario accelerates. Watch for State Department leaks.\n\n"
                "2. US credit card delinquency data. We're at 3.8% against a 3.5% threshold. If next "
                "month's reading exceeds 4%, and the savings rate data comes in below 3%, the Consumer "
                "Credit Stress pattern activates for the US. That would be the first crack pattern "
                "activation for the world's largest economy since 2019.\n\n"
                "3. Canada trade negotiator identity. Who Carney picks tells us the negotiating posture — "
                "an industry insider signals pragmatic deal-making, a political appointee signals hardball.\n\n"
                "4. Saudi Aramco earnings. If revenue is down more than 12% YoY, MBS has to make hard "
                "choices about Vision 2030 spending.\n\n"
                "5. European Parliament response to Kallas-Rubio exchange. If the rhetoric hardens into "
                "trade policy proposals, the transatlantic crack widens."
            ),
            visual_cues=[
                "5-item watchlist with probability/impact scores",
                "Calendar overlay showing data release dates",
                "US credit delinquency trend approaching threshold line",
            ],
            data_points=[
                "US CC delinquency: 3.8% (threshold 3.5%, pattern trigger at 4%)",
                "Saudi Aramco earnings: expected -12% YoY revenue",
                "Iran: first diplomatic signal in 18 months",
            ],
            charts=[],
            transitions="'That's our view for the week.'",
            social_clip_hook="5 things to watch next week. Number 2 could trigger the first US economic crack pattern since 2019.",
        )

    def _build_closing(self) -> ScriptSegment:
        return ScriptSegment(
            segment_id="closing",
            title="CLOSING",
            subtitle="",
            duration_minutes=1.0,
            speaker_notes=(
                "That's the Greenshoe Intelligence Report for this week. If you found this useful, "
                "subscribe and hit the bell — we publish every Sunday evening so you start the trading "
                "week informed. If you're a professional analyst, trader, or policy researcher and you "
                "have data or insights that would strengthen our analysis, reach out — the link's in "
                "the description. The best intelligence comes from networks, not algorithms. "
                "See you next week."
            ),
            visual_cues=[
                "Subscribe call-to-action overlay",
                "Contact info / social links",
                "Next week preview teaser card",
            ],
            data_points=[],
            charts=[],
            transitions="",
            social_clip_hook="",
        )

    def _generate_episode_title(self) -> str:
        return "Iran Moves the Needle — Oil, Energy Wars, and Cracks in the System"

    def _generate_description(self, segments: list[ScriptSegment]) -> str:
        lines = [
            f"Greenshoe Intelligence Report #{self.episode_number}",
            "",
            "This week: Iran signals nuclear compromise and the oil probability model shifts 10.5%. "
            "We break down 7 global headlines through our sovereign risk knowledge graph, tracking "
            "110 economic indicators across 20 countries.",
            "",
            "TIMESTAMPS:",
        ]

        time_offset = 2.0  # after cold open + intro
        for seg in segments:
            minutes = int(time_offset)
            seconds = int((time_offset - minutes) * 60)
            lines.append(f"{minutes:02d}:{seconds:02d} — {seg.title}: {seg.subtitle}")
            time_offset += seg.duration_minutes

        lines.extend([
            "",
            "SOURCES & DATA:",
            "• 110 economic statistics from FRED, World Bank, IMF, BIS, OECD",
            "• 50 risk scenarios with probability modeling",
            "• 7 generational strategic plans for major powers",
            "• Crack detection: 7 multi-indicator warning patterns",
            "",
            "CONNECT:",
            "If you have data, insights, or corrections, reach out:",
            "• Email: intel@greenshoeinvestments.com",
            "• This is independent research, not financial advice.",
            "",
            "#GeopoliticalAnalysis #SovereignRisk #EconomicIntelligence #GlobalMacro "
            "#TradingIntelligence #OilMarket #Iran #USMCA #BeltAndRoad #GreenshoeIntelligence",
        ])
        return "\n".join(lines)

    def _generate_tags(self) -> list[str]:
        return [
            "geopolitical analysis", "sovereign risk", "economic intelligence",
            "global macro", "oil market", "iran nuclear deal", "USMCA",
            "belt and road", "crack detection", "trading intelligence",
            "greenshoe investments", "canada trade", "energy dominance",
            "US economy", "China strategy", "Russia sanctions",
            "Saudi Arabia Vision 2030", "credit delinquency",
            "weekly intelligence briefing", "market analysis",
        ]

    def _generate_thumbnail_concept(self) -> str:
        return (
            "THUMBNAIL CONCEPT:\n"
            "  Background: Dark navy (#0a0f1a) with subtle world map overlay\n"
            "  Main visual: Oil barrel with cracking effect + Iran flag colors\n"
            "  Text (large, bold): 'OIL SHIFTED 10.5%'\n"
            "  Text (smaller): 'Iran • Venezuela • Saudi' with flag icons\n"
            "  Bottom bar: 'GREENSHOE INTELLIGENCE REPORT #XX'\n"
            "  Color accents: Red crack lines radiating from barrel\n"
            "  Style: Dark, serious, data-forward (NOT clickbait bright)"
        )

    def _write_script_markdown(
        self,
        cold_open: ScriptSegment,
        intro: ScriptSegment,
        segments: list[ScriptSegment],
        closing: ScriptSegment,
        week_date: str,
        title: str,
    ) -> Path:
        """Write the complete episode script as a markdown file."""
        filepath = self.output_dir / f"episode_{self.episode_number:03d}_script.md"

        all_segments = [cold_open, intro] + segments + [closing]
        time_offset = 0.0

        lines = [
            f"# Greenshoe Intelligence Report — Episode #{self.episode_number}",
            f"## {title}",
            f"### Week of {week_date}",
            "",
            f"**Estimated Duration:** {sum(s.duration_minutes for s in all_segments):.0f} minutes",
            "",
            "---",
            "",
        ]

        for seg in all_segments:
            minutes = int(time_offset)
            seconds = int((time_offset - minutes) * 60)
            timestamp = f"[{minutes:02d}:{seconds:02d}]"

            lines.extend([
                f"## {timestamp} {seg.title}" + (f" — {seg.subtitle}" if seg.subtitle else ""),
                "",
                f"**Duration:** {seg.duration_minutes:.1f} minutes",
                "",
                "### Speaker Notes",
                "",
                seg.speaker_notes,
                "",
            ])

            if seg.data_points:
                lines.append("### Key Data Points")
                lines.append("")
                for dp in seg.data_points:
                    lines.append(f"- {dp}")
                lines.append("")

            if seg.visual_cues:
                lines.append("### Visual Cues")
                lines.append("")
                for vc in seg.visual_cues:
                    lines.append(f"- 🎬 {vc}")
                lines.append("")

            if seg.charts:
                lines.append("### Charts")
                lines.append("")
                for chart in seg.charts:
                    lines.append(f"- 📊 `{chart}`")
                lines.append("")

            if seg.transitions:
                lines.append("### Transition")
                lines.append("")
                lines.append(f"*{seg.transitions}*")
                lines.append("")

            if seg.social_clip_hook:
                lines.append("### Social Clip")
                lines.append("")
                lines.append(f"> {seg.social_clip_hook}")
                lines.append("")

            lines.append("---")
            lines.append("")
            time_offset += seg.duration_minutes

        # Append thumbnail concept
        lines.extend([
            "## Production Notes",
            "",
            self._generate_thumbnail_concept(),
            "",
            "## YouTube Description",
            "",
            self._generate_description(segments),
            "",
            "## Tags",
            "",
            ", ".join(self._generate_tags()),
        ])

        filepath.write_text("\n".join(lines))
        return filepath
