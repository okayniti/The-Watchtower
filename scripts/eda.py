"""Exploratory figures for The Watchtower dataset (Deliverable 1 validation).

Run with::

    python scripts/eda.py

Loads ``data/access_logs.csv`` and ``data/labels.csv``, regenerating them via
:mod:`src.data_generator` if either is missing, joins them on ``event_id``, and writes
four presentation-ready figures to ``reports/figures/`` at 150 dpi:

======================================  ==============================================
``events_over_time.png``                Daily volume: benign against the six attacks.
``class_distribution.png``              All eight classes, log scale, counts labelled.
``entity_timeline_normal_vs_injected``  One victim's month, normal vs injected.
``naive_rule_precision.png``            Why the dataset is not circular.
======================================  ==============================================

The fourth figure is the important one. It computes — live, never hardcoded — the
precision of one naive single-feature rule per attack class. If those rules scored well,
the dataset would be trivially separable and every downstream metric would be
meaningless. They do not, which is the evidence that the generator's documented
attack/benign overlaps are real.

Everything here is deterministic: no sampling, and the entity featured in figure 3 is
chosen by a fixed rule rather than at random, so the figures are stable across runs.

Design notes
------------
Colour is assigned by *class identity* and is consistent across all four figures — the
green that marks ``low_and_slow_exfiltration`` in the volume chart is the same green on
its bar in the naive-rule chart. Benign traffic is deliberately a neutral grey rather
than a hue: it is the background against which findings are read, not a finding.

The palette is the validated 7-slot categorical set (adjacent-pair CVD ΔE 9.1, normal
vision 19.6, both above floor). Three of its hues sit below 3:1 against the light
surface, so every figure carries visible legends and direct labels — identity is never
communicated by colour alone.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.generator.config import (  # noqa: E402  (path setup must precede import)
    AMBIGUOUS_LABEL,
    BEHAVIOURS,
    BENIGN_LABEL,
    HARD_ANOMALIES,
    haversine_km,
    parse_geo,
)

DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
DPI = 150

# --- chrome & ink (light surface) ----------------------------------------------------
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

#: Validated categorical slots 1-7, in fixed order. Never cycled, never regenerated.
SLOTS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7")

#: One colour per behaviour, held constant across every figure in the deck.
CLASS_COLOR: dict[str, str] = {
    BENIGN_LABEL: INK_MUTED,  # neutral: the background, not a finding
    "brute_force": SLOTS[0],
    "impossible_travel": SLOTS[1],
    "credential_stuffing": SLOTS[2],
    "lateral_movement": SLOTS[3],
    "device_spoofing": SLOTS[4],
    "low_and_slow_exfiltration": SLOTS[5],
    AMBIGUOUS_LABEL: SLOTS[6],
}

#: Shorter labels for axes, where the full snake_case names would crowd.
PRETTY: dict[str, str] = {
    BENIGN_LABEL: "normal_baseline",
    "brute_force": "brute_force",
    "impossible_travel": "impossible_travel",
    "credential_stuffing": "credential_stuffing",
    "lateral_movement": "lateral_movement",
    "device_spoofing": "device_spoofing",
    "low_and_slow_exfiltration": "low_and_slow_exfil",
    AMBIGUOUS_LABEL: "insider_drift",
}


def setup_style() -> None:
    """Apply the shared figure style: recessive chrome, system sans, hairline grid."""
    plt.rcParams.update(
        {
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "figure.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
            "font.size": 9.5,
            "axes.titlesize": 12,
            "axes.titleweight": "600",
            "axes.titlecolor": INK_PRIMARY,
            "axes.labelsize": 9.5,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": BASELINE,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "grid.color": GRIDLINE,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",  # solid hairlines; dashed grids read as thresholds
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "legend.labelcolor": INK_SECONDARY,
        }
    )


# --------------------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------------------


def load_data(regenerate_if_missing: bool = True) -> pd.DataFrame:
    """Load the access log joined to its labels.

    The two files are stored separately so the feature file cannot leak ground truth
    (see ``CLAUDE.md`` §2). Joining them is legitimate here: this is analysis, not
    inference.

    Args:
        regenerate_if_missing: If ``True`` and either file is absent, invoke the
            generator to rebuild them. It is deterministic, so this reproduces the
            same dataset rather than a new one.

    Returns:
        One row per event with the feature columns plus ``label`` and ``episode_id``,
        sorted by timestamp.

    Raises:
        FileNotFoundError: If a file is missing and ``regenerate_if_missing`` is
            ``False``, or if regeneration did not produce it.
    """
    logs_path = DATA_DIR / "access_logs.csv"
    labels_path = DATA_DIR / "labels.csv"

    if not (logs_path.exists() and labels_path.exists()):
        if not regenerate_if_missing:
            raise FileNotFoundError(f"missing {logs_path} or {labels_path}")
        print("  dataset not found — regenerating via src.data_generator ...")
        subprocess.run(
            [sys.executable, "-m", "src.data_generator", "--quiet"],
            cwd=PROJECT_ROOT,
            check=True,
        )
        if not (logs_path.exists() and labels_path.exists()):
            raise FileNotFoundError("generator ran but produced no dataset")

    events = pd.read_csv(logs_path)
    labels = pd.read_csv(labels_path)
    df = events.merge(labels, on="event_id", validate="one_to_one")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.normalize()
    df["hour"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
    df["episode_id"] = df["episode_id"].fillna("")
    return df.sort_values(["timestamp", "entity_id"]).reset_index(drop=True)


def _style_time_axis(ax: plt.Axes) -> None:
    """Format a date x-axis with weekly major ticks and daily minor ticks.

    Args:
        ax: The axes whose x-axis should be formatted.
    """
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.xaxis.set_minor_locator(mdates.DayLocator())


def _shade_weekends(ax: plt.Axes, dates: pd.Series) -> None:
    """Shade weekend days, which explains the recurring dips in human activity.

    Args:
        ax: The axes to shade.
        dates: The date range covered by the plot.
    """
    for day in pd.date_range(dates.min(), dates.max(), freq="D"):
        if day.weekday() >= 5:
            ax.axvspan(
                day, day + pd.Timedelta(days=1), color=GRIDLINE, alpha=0.55, zorder=0, lw=0
            )


# --------------------------------------------------------------------------------------
# Figure 1 — events over time
# --------------------------------------------------------------------------------------


def fig_events_over_time(df: pd.DataFrame) -> Path:
    """Plot daily event volume: benign traffic above, the six attack classes below.

    Two stacked panels sharing one x-axis rather than a dual-axis chart. Benign volume
    runs ~3,000 events/day and the attacks ~40/day; forcing both onto one scale would
    flatten the attacks into the axis, and a second y-scale would invent a relationship
    between them that does not exist. Separate panels keep both readable and keep every
    comparison honest.

    The lower panel is what shows ``low_and_slow_exfiltration`` spanning many days,
    where the burst classes appear as single-day spikes.

    Args:
        df: The joined event frame.

    Returns:
        The path written.
    """
    benign = df[df.label == BENIGN_LABEL].groupby("date").size()
    attacks = df[df.label.isin(HARD_ANOMALIES)]
    daily = (
        attacks.groupby(["date", "label"]).size().unstack(fill_value=0).reindex(
            columns=list(HARD_ANOMALIES), fill_value=0
        )
    )
    daily = daily.reindex(benign.index, fill_value=0)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, height_ratios=[1.15, 1]
    )

    # --- benign ---------------------------------------------------------------------
    _shade_weekends(ax_top, benign.index.to_series())
    ax_top.fill_between(
        benign.index, benign.to_numpy(), color=INK_MUTED, alpha=0.16, zorder=2, lw=0
    )
    ax_top.plot(benign.index, benign.to_numpy(), color=INK_MUTED, lw=2, zorder=3)
    ax_top.set_ylabel("benign events / day")
    ax_top.set_ylim(0, benign.max() * 1.18)
    ax_top.grid(axis="y", zorder=1)
    ax_top.set_axisbelow(True)
    ax_top.annotate(
        f"normal_baseline — {benign.sum():,} events ({benign.sum()/len(df):.1%} of all traffic)",
        xy=(0.012, 0.88),
        xycoords="axes fraction",
        color=INK_SECONDARY,
        fontsize=9,
    )
    # Name the weekend shading once rather than spending a legend row on it.
    weekend = benign.index.to_series()[benign.index.to_series().dt.weekday >= 5]
    if len(weekend):
        ax_top.annotate(
            "weekends",
            xy=(weekend.iloc[1] + pd.Timedelta(hours=12), benign.max() * 0.06),
            color=INK_MUTED,
            fontsize=7.5,
            ha="center",
        )

    # --- attacks --------------------------------------------------------------------
    _shade_weekends(ax_bot, benign.index.to_series())
    bottom = np.zeros(len(daily))
    for label in HARD_ANOMALIES:
        values = daily[label].to_numpy()
        ax_bot.bar(
            daily.index,
            values,
            bottom=bottom,
            width=0.82,
            color=CLASS_COLOR[label],
            label=PRETTY[label],
            edgecolor=SURFACE,  # 2px surface gap between stacked segments
            linewidth=0.6,
            zorder=3,
        )
        bottom += values
    ax_bot.set_ylabel("attack events / day")
    ax_bot.set_xlabel("")
    ax_bot.grid(axis="y", zorder=1)
    ax_bot.set_axisbelow(True)
    ax_bot.legend(
        ncol=3, loc="upper left", bbox_to_anchor=(0.0, 1.30), columnspacing=1.6,
        handlelength=1.1, handleheight=1.1,
    )
    _style_time_axis(ax_bot)

    fig.suptitle(
        "Event volume over the 30-day window",
        x=0.055, y=0.975, ha="left", fontsize=14, color=INK_PRIMARY, weight="600",
    )
    fig.text(
        0.055, 0.935,
        f"Note the scales: {benign.sum():,} benign events against "
        f"{len(attacks):,} attack events. Panels are separate because one shared axis "
        "would flatten the lower series into the baseline.",
        ha="left", fontsize=9, color=INK_SECONDARY,
    )
    fig.text(
        0.055, 0.015,
        "low_and_slow_exfiltration is the only class spread across many days — every other\n"
        "attack is a burst. insider_drift is excluded here (it is benign, and spans weeks "
        "by design).",
        ha="left", va="bottom", fontsize=8.5, color=INK_MUTED, linespacing=1.5,
    )

    fig.tight_layout(rect=(0.0, 0.045, 1.0, 0.915))
    path = FIGURES_DIR / "events_over_time.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------------------
# Figure 2 — class distribution
# --------------------------------------------------------------------------------------


def fig_class_distribution(df: pd.DataFrame) -> Path:
    """Plot the eight-class distribution on a log axis with counts labelled.

    A linear axis is useless here: ``normal_baseline`` is ~98% of the data, so the
    seven other bars collapse to invisible slivers. The log axis is what makes the
    imbalance legible *as* imbalance — the whole point of the figure.

    Args:
        df: The joined event frame.

    Returns:
        The path written.
    """
    counts = df.label.value_counts().reindex(list(BEHAVIOURS), fill_value=0)
    total = len(df)
    order = list(BEHAVIOURS)[::-1]  # benign at top when drawn bottom-up

    fig, ax = plt.subplots(figsize=(10, 5.6))
    y_positions = np.arange(len(order))
    values = [counts[label] for label in order]

    ax.barh(
        y_positions,
        values,
        height=0.66,
        color=[CLASS_COLOR[label] for label in order],
        zorder=3,
    )
    ax.set_yticks(y_positions)
    ax.set_yticklabels([PRETTY[label] for label in order], fontsize=9.5)
    ax.set_xscale("log")
    ax.set_xlim(1, counts.max() * 6)
    ax.set_xlabel("events (log scale)")
    ax.grid(axis="x", zorder=1)
    ax.set_axisbelow(True)

    for y, label in zip(y_positions, order):
        count = counts[label]
        ax.annotate(
            f"{count:,}   ({count/total:.3%})",
            xy=(count, y),
            xytext=(7, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=INK_SECONDARY,
        )

    # Mark the two structural groupings without spending colour on them.
    hard_total = int(counts[list(HARD_ANOMALIES)].sum())
    ax.legend(
        handles=[
            Patch(facecolor=INK_MUTED, label="benign"),
            Patch(facecolor=SLOTS[0], label="attack (6 classes)"),
            Patch(facecolor=SLOTS[6], label="ambiguous — benign, attack-shaped"),
        ],
        loc="lower right",
        handlelength=1.1,
        handleheight=1.1,
    )

    fig.suptitle(
        "Class distribution — extreme imbalance by design",
        x=0.055, y=0.985, ha="left", fontsize=14, color=INK_PRIMARY, weight="600",
    )
    fig.text(
        0.055, 0.905,
        f"{hard_total:,} attack events in {total:,} total = "
        f"{hard_total/total:.2%}, inside the 0.5-3% band required by CLAUDE.md §4.2.",
        ha="left", va="bottom", fontsize=9, color=INK_SECONDARY,
    )
    fig.text(
        0.055, 0.015,
        f"insider_drift is excluded from that {hard_total/total:.2%}: it is a legitimate "
        "role change, and counting it\nas an anomaly would misstate the imbalance the "
        "detector actually faces.",
        ha="left", va="bottom", fontsize=8.5, color=INK_MUTED, linespacing=1.5,
    )

    fig.tight_layout(rect=(0.0, 0.075, 1.0, 0.88))
    path = FIGURES_DIR / "class_distribution.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------------------
# Figure 3 — one entity's timeline
# --------------------------------------------------------------------------------------


def pick_featured_entity(df: pd.DataFrame, preferred: str = "low_and_slow_exfiltration") -> tuple[str, str]:
    """Choose the entity to feature, deterministically.

    Prefers a ``low_and_slow_exfiltration`` victim because that class is the clearest
    "needle in a haystack" story — its events are individually unremarkable and only
    the cadence gives it away.

    Among victims it takes those with an above-median episode, then picks the one with
    the most *benign* events. Selecting purely on episode size would surface a
    low-volume entity whose episode is half its traffic, which makes the attack look
    obvious and undersells the problem. A busy entity with a substantial episode is the
    honest picture: a real haystack with a real needle in it.

    Args:
        df: The joined event frame.
        preferred: Behaviour class to prefer when choosing a victim.

    Returns:
        An ``(entity_id, label)`` tuple naming the featured entity and the class of the
        episode it was hit by.

    Raises:
        ValueError: If no entity in the dataset carries an injected episode.
    """
    benign_counts = df[df.label == BENIGN_LABEL].groupby("entity_id").size()
    for label in (preferred, "lateral_movement", *HARD_ANOMALIES):
        episode_sizes = df[df.label == label].groupby("entity_id").size()
        if episode_sizes.empty:
            continue
        candidates = episode_sizes[episode_sizes >= episode_sizes.median()].index
        volumes = benign_counts.reindex(candidates).dropna()
        if volumes.empty:
            continue
        best = sorted(volumes.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        return str(best), label
    raise ValueError("no injected episodes found in the dataset")


def fig_entity_timeline(df: pd.DataFrame) -> Path:
    """Plot one victim's month, colour-coded normal versus injected.

    Two panels, both against the same time axis. The upper panel puts hour-of-day on y,
    which reveals the entity's working rhythm and — for an exfiltration victim — the
    fixed daily slot the injected sessions occupy. The lower panel shows session
    duration on a log scale, demonstrating that the injected sessions sit *inside* the
    entity's own normal range rather than beyond it.

    Args:
        df: The joined event frame.

    Returns:
        The path written.
    """
    entity_id, episode_label = pick_featured_entity(df)
    entity = df[df.entity_id == entity_id].sort_values("timestamp")
    injected = entity[entity.label != BENIGN_LABEL]
    normal = entity[entity.label == BENIGN_LABEL]
    accent = CLASS_COLOR[episode_label]

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(11, 7.2), sharex=True, height_ratios=[1.25, 1]
    )

    for ax in (ax_top, ax_bot):
        _shade_weekends(ax, entity.timestamp.dt.normalize())
        ax.grid(axis="y", zorder=1)
        ax.set_axisbelow(True)

    # --- hour of day ------------------------------------------------------------------
    ax_top.scatter(
        normal.timestamp, normal.hour,
        s=26, color=INK_MUTED, alpha=0.55, linewidths=0, zorder=3,
    )
    ax_top.scatter(
        injected.timestamp, injected.hour,
        s=62, color=accent, edgecolor=SURFACE, linewidths=1.4, zorder=4,  # surface ring
    )
    ax_top.set_ylabel("hour of day")
    ax_top.set_ylim(-1, 25)
    ax_top.set_yticks([0, 6, 12, 18, 24])

    if not injected.empty:
        ax_top.axvspan(
            injected.timestamp.min(), injected.timestamp.max(),
            color=accent, alpha=0.07, zorder=1, lw=0,
        )
        ax_top.annotate(
            f"{PRETTY[episode_label]} episode — {len(injected)} events "
            f"over {(injected.timestamp.max() - injected.timestamp.min()).days} days",
            xy=(injected.timestamp.min(), 24.2),
            xytext=(4, 0), textcoords="offset points",
            fontsize=8.5, color=accent, va="top", weight="600",
        )

    ax_top.legend(
        handles=[
            Line2D([], [], marker="o", ls="", markersize=6,
                   markerfacecolor=INK_MUTED, markeredgecolor="none",
                   label=f"normal_baseline  ({len(normal)})"),
            Line2D([], [], marker="o", ls="", markersize=8,
                   markerfacecolor=accent, markeredgecolor=SURFACE,
                   label=f"{PRETTY[episode_label]}  ({len(injected)})"),
        ],
        loc="upper left", bbox_to_anchor=(0.0, 1.20), ncol=2,
    )

    # --- session duration -------------------------------------------------------------
    ax_bot.scatter(
        normal.timestamp, normal.session_duration,
        s=26, color=INK_MUTED, alpha=0.55, linewidths=0, zorder=3,
    )
    ax_bot.scatter(
        injected.timestamp, injected.session_duration,
        s=62, color=accent, edgecolor=SURFACE, linewidths=1.4, zorder=4,
    )
    ax_bot.set_yscale("log")
    ax_bot.set_ylabel("session duration (s, log)")
    _style_time_axis(ax_bot)

    share_below = 0.0
    if not normal.empty:
        p95 = float(normal.session_duration.quantile(0.95))
        ax_bot.axhline(p95, color=BASELINE, lw=1.2, zorder=2)
        ax_bot.annotate(
            f"this entity's own p95 = {p95:,.0f}s",
            xy=(entity.timestamp.min(), p95), xytext=(2, 4),
            textcoords="offset points", fontsize=8, color=INK_MUTED,
        )
        share_below = (injected.session_duration <= p95).mean() if len(injected) else 0.0

    fig.suptitle(
        f"One entity's month — {entity_id}",
        x=0.055, y=0.988, ha="left", fontsize=14, color=INK_PRIMARY, weight="600",
    )
    fig.text(
        0.055, 0.895,
        f"{len(entity):,} events, of which {len(injected)} are injected "
        f"({len(injected)/len(entity):.1%} of this entity, {len(df[df.label != BENIGN_LABEL])/len(df):.1%} "
        "of the estate). No single field puts the\nattack outside this entity's normal "
        "range — what gives it away is rhythm.",
        ha="left", va="bottom", fontsize=9, color=INK_SECONDARY, linespacing=1.5,
    )
    if not normal.empty and len(injected):
        fig.text(
            0.055, 0.015,
            f"{share_below:.0%} of the injected sessions fall below this entity's own "
            "95th-percentile duration: a per-event\nthreshold cannot separate them, which "
            "is what forces a sequence-aware model.",
            ha="left", va="bottom", fontsize=8.5, color=INK_MUTED, linespacing=1.5,
        )

    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.875))
    path = FIGURES_DIR / "entity_timeline_normal_vs_injected.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------------------
# Figure 4 — naive rule precision
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class NaiveRule:
    """One single-feature detector and how it scored.

    Attributes:
        name: Human-readable description of the rule.
        target: The behaviour the rule is trying to catch.
        fired: Number of events the rule alerted on.
        true_positives: How many of those carried the target label.
        base_rate: Prevalence of the target class, i.e. the precision a random
            selector of the same size would achieve.
    """

    name: str
    target: str
    fired: int
    true_positives: int
    base_rate: float

    @property
    def precision(self) -> float:
        """Fraction of the rule's alerts that were the target class."""
        return self.true_positives / self.fired if self.fired else 0.0

    @property
    def lift(self) -> float:
        """How many times better than chance the rule is."""
        return self.precision / self.base_rate if self.base_rate else 0.0


def compute_naive_rules(df: pd.DataFrame, novelty_warmup: int = 5) -> list[NaiveRule]:
    """Score one naive single-feature rule per attack class, computed from the data.

    These are the rules a reasonable engineer writes first, and each targets the class
    whose documented signature it matches. Novelty-based rules skip an entity's first
    few events, where "never seen before" is vacuously true — that handicap makes the
    naive baselines *stronger*, so the conclusion that they remain unusable is
    conservative rather than rigged.

    Args:
        df: The joined event frame.
        novelty_warmup: Events per entity to skip before novelty rules may fire.

    Returns:
        One :class:`NaiveRule` per attack class, in :data:`HARD_ANOMALIES` order.
    """
    work = df.sort_values(["entity_id", "timestamp"]).copy()
    grouped = work.groupby("entity_id", sort=False)
    work["seq"] = grouped.cumcount()
    warm = work["seq"] >= novelty_warmup

    # 1. brute_force — a run of failed authentications in the command sequence.
    fail = work.command_sequence.str.contains("AUTH_FAIL", regex=False)

    # 2. impossible_travel — implied geo-velocity between consecutive events.
    coords = work.geo_location.map(parse_geo)
    work["lat"] = [c[2] for c in coords]
    work["lon"] = [c[3] for c in coords]
    by_entity = work.groupby("entity_id", sort=False)
    prev_lat = by_entity["lat"].shift()
    prev_lon = by_entity["lon"].shift()
    gap_h = by_entity["timestamp"].diff().dt.total_seconds() / 3600.0
    movable = (prev_lat.notna() & (gap_h > 0)).to_numpy()

    velocity = np.zeros(len(work))
    idx = np.flatnonzero(movable)
    distances = np.array(
        [
            haversine_km(a, b, c, d)
            for a, b, c, d in zip(
                prev_lat.to_numpy()[idx], prev_lon.to_numpy()[idx],
                work.lat.to_numpy()[idx], work.lon.to_numpy()[idx],
            )
        ]
    )
    velocity[idx] = distances / gap_h.to_numpy()[idx]
    fast = pd.Series(velocity > 900.0, index=work.index)

    # 3. credential_stuffing — an address shared by many distinct identities.
    ip_entities = work.groupby("source_ip").entity_id.transform("nunique")
    shared_ip = ip_entities >= 10

    # 4. lateral_movement — a resource this entity has never touched.
    first_resource = ~work.duplicated(["entity_id", "resource_accessed"])
    novel_resource = first_resource & warm

    # 5. device_spoofing — a device fingerprint this entity has never presented.
    first_device = ~work.duplicated(["entity_id", "device_fingerprint"])
    novel_device = first_device & warm

    # 6. low_and_slow_exfiltration — an unusually long session, globally.
    p95 = work.session_duration.quantile(0.95)
    long_session = work.session_duration > p95

    specs: list[tuple[str, str, pd.Series]] = [
        ("command_sequence contains AUTH_FAIL", "brute_force", fail),
        ("geo-velocity > 900 km/h", "impossible_travel", fast),
        ("source_ip shared by ≥ 10 entities", "credential_stuffing", shared_ip),
        ("resource never seen for this entity", "lateral_movement", novel_resource),
        ("device never seen for this entity", "device_spoofing", novel_device),
        (f"session_duration > global p95 ({p95:,.0f}s)", "low_and_slow_exfiltration", long_session),
    ]

    order = {label: i for i, label in enumerate(HARD_ANOMALIES)}
    rules = [
        NaiveRule(
            name=name,
            target=target,
            fired=int(mask.sum()),
            true_positives=int((mask & (work.label == target)).sum()),
            base_rate=float((work.label == target).mean()),
        )
        for name, target, mask in specs
    ]
    rules.sort(key=lambda r: order[r.target])
    return rules


def fig_naive_rule_precision(df: pd.DataFrame) -> tuple[Path, list[NaiveRule]]:
    """Plot naive single-feature precision — the evidence the dataset is not circular.

    Each bar is one hand-written rule aimed at one attack class, with a marker showing
    the precision a random selector of the same size would reach. The gap between the
    two is real lift; the absolute height is what matters operationally, and it is low
    enough that every one of these rules would bury an analyst.

    Args:
        df: The joined event frame.

    Returns:
        The path written and the computed rules, so the caller can print them.
    """
    rules = compute_naive_rules(df)
    order = list(reversed(rules))  # highest class first when drawn bottom-up
    y = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.barh(
        y,
        [r.precision * 100 for r in order],
        height=0.6,
        color=[CLASS_COLOR[r.target] for r in order],
        zorder=3,
    )

    for i, rule in enumerate(order):
        ax.annotate(
            f"{rule.precision:.1%}",
            xy=(rule.precision * 100, i), xytext=(8, 0), textcoords="offset points",
            va="center", fontsize=10, color=INK_PRIMARY, weight="600",
        )
        ax.annotate(
            f"{rule.fired:,} alerts → {rule.true_positives} correct "
            f"· {rule.lift:.0f}× chance",
            xy=(rule.precision * 100, i), xytext=(52, 0), textcoords="offset points",
            va="center", fontsize=8.5, color=INK_MUTED,
        )
        ax.plot(
            [rule.base_rate * 100], [i],
            marker="|", markersize=15, markeredgewidth=2.5,
            color=INK_PRIMARY, zorder=4,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{r.name}\n→ {PRETTY[r.target]}" for r in order], fontsize=8.8
    )
    ax.set_xlabel("precision of the rule's alerts (%)")
    ax.set_xlim(0, max(r.precision for r in rules) * 100 * 2.05)
    ax.grid(axis="x", zorder=1)
    ax.set_axisbelow(True)
    ax.legend(
        handles=[
            Line2D([], [], marker="|", ls="", markersize=13, markeredgewidth=2.5,
                   color=INK_PRIMARY, label="precision of a random selector (class base rate)")
        ],
        loc="lower right",
    )

    worst, best = min(r.precision for r in rules), max(r.precision for r in rules)
    fig.suptitle(
        "Naive single-feature rules — why this dataset is not circular",
        x=0.055, y=0.985, ha="left", fontsize=14, color=INK_PRIMARY, weight="600",
    )
    fig.text(
        0.055, 0.885,
        "One hand-written rule per attack class, each matched to that class's documented\n"
        f"signature. Every one has real lift over chance — and every one is still "
        f"{1-best:.0%} to {1-worst:.0%} wrong.",
        ha="left", va="bottom", fontsize=9, color=INK_SECONDARY, linespacing=1.5,
    )
    # Captions are hand-wrapped: bbox_inches="tight" expands the canvas to fit any
    # overflowing text, so one long line silently stretches the whole figure.
    fig.text(
        0.055, 0.015,
        "If these bars were tall, the anomalies would be trivially separable and every\n"
        "downstream metric would be meaningless. They are not, because each attack was\n"
        "generated to overlap a specific legitimate behaviour: VPN egress, hardware refresh,\n"
        "shared gateways, mistyped passwords, one-off access.",
        ha="left", va="bottom", fontsize=8.5, color=INK_MUTED, linespacing=1.5,
    )

    fig.tight_layout(rect=(0.0, 0.055, 1.0, 0.905))
    path = FIGURES_DIR / "naive_rule_precision.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path, rules


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


def main() -> int:
    """Generate every EDA figure.

    Returns:
        Process exit code: ``0`` on success.
    """
    setup_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("The Watchtower — EDA figures")
    df = load_data()
    print(f"  loaded {len(df):,} events, {df.entity_id.nunique()} entities, "
          f"{df.label.nunique()} classes")

    written: list[Path] = []
    print("  rendering events_over_time ...")
    written.append(fig_events_over_time(df))
    print("  rendering class_distribution ...")
    written.append(fig_class_distribution(df))
    print("  rendering entity_timeline_normal_vs_injected ...")
    written.append(fig_entity_timeline(df))
    print("  rendering naive_rule_precision ...")
    path, rules = fig_naive_rule_precision(df)
    written.append(path)

    print("\nNaive single-feature rule scores:")
    print(f"  {'rule':<46} {'target':<26} {'fired':>7} {'prec':>7} {'lift':>6}")
    for rule in rules:
        print(f"  {rule.name:<46} {rule.target:<26} {rule.fired:>7,} "
              f"{rule.precision:>6.2%} {rule.lift:>5.0f}x")

    print("\nFigures written:")
    for figure_path in written:
        size_kb = figure_path.stat().st_size / 1024
        print(f"  {figure_path.relative_to(PROJECT_ROOT)}  ({size_kb:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
