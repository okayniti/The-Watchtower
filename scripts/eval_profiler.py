"""Evaluation harness for the baseline profiler (Deliverable 2).

**This is the only module in the project that reads ``labels.csv``.** Everything
upstream — :mod:`src.features`, :mod:`src.profiler` — runs on ``access_logs.csv``
alone. The join happens here, after scoring, purely to measure what the unsupervised
model already decided.

Run with::

    python scripts/eval_profiler.py

Reports three things:

1. **Score separation** — where each attack class lands in the benign score
   distribution, plus recall at the top-1% alert budget that ``CLAUDE.md`` §5 makes the
   headline metric. This is a baseline, not the final number: the sequence model and
   classifier come next.
2. **Cold-start behaviour** — that entities with almost no history score comparably to
   established ones rather than saturating the queue, which is the failure mode the
   peer-group prior exists to prevent.
3. **Concept drift / time-to-unflag** — the important one. After an ``insider_drift``
   ramp settles, does the entity's score fall back to baseline, and how fast? This is
   swept across several decay half-lives to show the knob actually controls it.

Figures are written to ``reports/figures/``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features import build_feature_matrix, load_events  # noqa: E402
from src.generator.config import (  # noqa: E402
    AMBIGUOUS_LABEL,
    BENIGN_LABEL,
    HARD_ANOMALIES,
)
from src.profiler import EntityProfiler, ProfilerConfig  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from eda import (  # noqa: E402  (reuse the deck's palette so figures match)
    CLASS_COLOR,
    GRIDLINE,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    PRETTY,
    SURFACE,
    setup_style,
)

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"

#: Alert budget from CLAUDE.md §5 — the top 1% of events by score.
ALERT_BUDGET = 0.01

#: Decay half-lives swept in the drift experiment, in days. The last is effectively
#: "never forget", the control case.
HALF_LIVES = (3.0, 7.0, 21.0, 3650.0)


def load_labels() -> pd.DataFrame:
    """Load ground truth. Called only from this evaluation module.

    Returns:
        A frame of ``event_id``, ``label``, ``episode_id``.

    Raises:
        FileNotFoundError: If the labels file is missing.
    """
    path = PROJECT_ROOT / "data" / "labels.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `python -m src.data_generator` first"
        )
    labels = pd.read_csv(path)
    labels["episode_id"] = labels["episode_id"].fillna("")
    return labels


# --------------------------------------------------------------------------------------
# 1. Score separation
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class ClassSeparation:
    """How one behaviour class scored relative to benign traffic.

    Attributes:
        label: The behaviour class.
        n_events: Events carrying that label.
        median_score: Median risk score for the class.
        benign_percentile: Where that median sits in the benign score distribution.
            50 means indistinguishable from ordinary traffic; 99 means the typical
            event of this class outscores 99% of benign events.
        recall_at_budget: Share of the class captured inside the top-1% alert budget.
        lift: ``recall_at_budget`` divided by the budget — how many times better than
            selecting 1% of events at random.
    """

    label: str
    n_events: int
    median_score: float
    benign_percentile: float
    recall_at_budget: float
    lift: float


def score_separation(
    scored: pd.DataFrame, budget: float = ALERT_BUDGET
) -> tuple[list[ClassSeparation], float, float]:
    """Measure how far each class separates from benign traffic.

    Args:
        scored: Events with ``risk_score`` and ``label``.
        budget: Alert-budget fraction.

    Returns:
        A ``(per_class, threshold, precision_at_budget)`` tuple. ``precision`` counts
        any of the six hard-anomaly classes as a true positive.
    """
    benign_scores = scored.loc[scored.label == BENIGN_LABEL, "risk_score"].to_numpy()
    benign_sorted = np.sort(benign_scores)

    threshold = float(scored.risk_score.quantile(1.0 - budget))
    alerted = scored[scored.risk_score >= threshold]
    hard = alerted.label.isin(HARD_ANOMALIES)
    precision = float(hard.mean()) if len(alerted) else 0.0

    results: list[ClassSeparation] = []
    for label in (*HARD_ANOMALIES, AMBIGUOUS_LABEL):
        subset = scored[scored.label == label]
        if subset.empty:
            continue
        median = float(subset.risk_score.median())
        percentile = float(np.searchsorted(benign_sorted, median) / len(benign_sorted) * 100)
        recall = float((subset.risk_score >= threshold).mean())
        results.append(
            ClassSeparation(
                label=label,
                n_events=len(subset),
                median_score=median,
                benign_percentile=percentile,
                recall_at_budget=recall,
                lift=recall / budget if budget else 0.0,
            )
        )
    return results, threshold, precision


def fig_score_by_class(scored: pd.DataFrame, threshold: float) -> Path:
    """Plot the risk-score distribution per behaviour class.

    Args:
        scored: Events with ``risk_score`` and ``label``.
        threshold: The top-1% alert threshold, drawn as a reference line.

    Returns:
        The path written.
    """
    order = [BENIGN_LABEL, *HARD_ANOMALIES, AMBIGUOUS_LABEL]
    fig, ax = plt.subplots(figsize=(11, 6.0))

    positions = np.arange(len(order))
    for i, label in enumerate(order):
        values = scored.loc[scored.label == label, "risk_score"].to_numpy()
        if values.size == 0:
            continue
        parts = ax.violinplot(
            values, positions=[i], widths=0.75, showextrema=False, showmedians=False
        )
        for body in parts["bodies"]:
            body.set_facecolor(CLASS_COLOR[label])
            body.set_alpha(0.55)
            body.set_edgecolor(SURFACE)
            body.set_linewidth(1.0)
        ax.plot(
            [i], [np.median(values)],
            marker="_", markersize=22, markeredgewidth=2.4,
            color=INK_PRIMARY, zorder=5,
        )

    ax.axhline(threshold, color=INK_SECONDARY, lw=1.4, zorder=4)
    ax.annotate(
        f"top-1% alert threshold  ({threshold:.2f})",
        xy=(len(order) - 0.45, threshold), xytext=(0, 6),
        textcoords="offset points", ha="right", fontsize=8.5, color=INK_SECONDARY,
    )

    ax.set_xticks(positions)
    ax.set_xticklabels([PRETTY[label] for label in order], rotation=28, ha="right")
    ax.set_ylabel("profiler risk score")
    ax.grid(axis="y", zorder=1)
    ax.set_axisbelow(True)
    ax.legend(
        handles=[Line2D([], [], marker="_", ls="", markersize=14, markeredgewidth=2.4,
                        color=INK_PRIMARY, label="class median")],
        loc="upper left",
    )

    fig.suptitle(
        "Baseline profiler — score separation by class",
        x=0.055, y=0.985, ha="left", fontsize=14, color=INK_PRIMARY, weight="600",
    )
    fig.text(
        0.055, 0.885,
        "Unsupervised, no labels used in scoring. Wide benign mass below the threshold is\n"
        "the point: the alert budget is spent on the classes that rise above it.",
        ha="left", va="bottom", fontsize=9, color=INK_SECONDARY, linespacing=1.5,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.87))
    path = FIGURES_DIR / "profiler_score_by_class.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------------------
# 2. Cold start
# --------------------------------------------------------------------------------------


def cold_start_report(scored: pd.DataFrame, threshold: float) -> dict[str, float]:
    """Check that history-poor entities are not swamping the alert queue.

    The failure mode this guards against is a profiler that scores every unfamiliar
    entity as maximally anomalous — technically defensible, operationally useless,
    because a new hire's first week would bury every real detection.

    Args:
        scored: Events with ``risk_score``, ``cold_start`` and ``label``.
        threshold: The top-1% alert threshold.

    Returns:
        Summary statistics comparing cold-start events with established ones.
    """
    cold = scored[scored.cold_start]
    warm = scored[~scored.cold_start]
    cold_benign = cold[cold.label == BENIGN_LABEL]
    warm_benign = warm[warm.label == BENIGN_LABEL]
    return {
        "cold_events": len(cold),
        "cold_share_of_log": len(cold) / len(scored),
        "cold_benign_median": float(cold_benign.risk_score.median()),
        "warm_benign_median": float(warm_benign.risk_score.median()),
        "cold_benign_alert_rate": float((cold_benign.risk_score >= threshold).mean()),
        "warm_benign_alert_rate": float((warm_benign.risk_score >= threshold).mean()),
        "cold_share_of_alerts": float(
            (scored[scored.risk_score >= threshold].cold_start).mean()
        ),
    }


# --------------------------------------------------------------------------------------
# 3. Concept drift / time-to-unflag
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class DriftResult:
    """Concept-drift measurement for one decay half-life.

    Attributes:
        half_life_days: The configured decay half-life.
        curve: Mean risk score by day relative to the end of the drift ramp.
        threshold: The top-1% alert threshold under this configuration.
        alert_rate_during: Share of ramp-phase events that would alert.
        alert_rate_settled: Share of post-ramp events that would alert, measured after
            the settling period.
        benign_alert_rate: Share of *all* benign events that would alert. The reference
            the drift rates must be read against — a drift alert rate at or below this
            means the class was never singled out in the first place.
        precision_at_budget: Precision on hard anomalies at the top-1% budget.
        hard_recall_at_budget: Share of hard-anomaly events captured at that budget.
        days_to_unflag: Days after the ramp ends before drifted entities' mean score
            falls back below threshold. ``None`` when the question does not apply
            because the class was never flagged — see :attr:`was_ever_flagged`.
        was_ever_flagged: Whether ramp-phase events alerted at a materially higher rate
            than ordinary benign traffic. When ``False``, ``days_to_unflag`` is not a
            measurement, it is a category error, and is reported as such.
        post_ramp_events: How many post-ramp events the settled figures rest on.
    """

    half_life_days: float
    curve: pd.Series
    threshold: float
    alert_rate_during: float
    alert_rate_settled: float
    benign_alert_rate: float
    precision_at_budget: float
    hard_recall_at_budget: float
    days_to_unflag: float | None
    was_ever_flagged: bool
    post_ramp_events: int


def measure_drift(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    half_life_days: float,
    settle_days: float = 3.0,
) -> DriftResult:
    """Score the log at one decay half-life and measure drift recovery.

    For every ``insider_drift`` episode, events are aligned on the end of the ramp: day
    0 is the last event the generator labelled ``insider_drift``, negative days are the
    ramp itself, and positive days are the settled period where the entity is using its
    expanded resource set legitimately. A profiler that handles drift should show scores
    falling back toward baseline as those positive days accumulate.

    The decay half-life is also swept against the headline metrics, because the
    parameter is a genuine trade-off rather than a free win: a short half-life holds a
    tight recent baseline and reacts sharply to any change (more sensitive, more false
    positives), a long one keeps a broad baseline that tolerates change (fewer false
    positives, slower to notice). Reporting only the drift side of that trade would
    misrepresent it.

    Args:
        features: The causal feature matrix.
        labels: Ground truth, joined here only.
        half_life_days: Decay half-life to test.
        settle_days: Days after the ramp to skip before measuring the settled rate,
            allowing the baseline time to absorb the change.

    Returns:
        The measurement for this configuration.
    """
    profiler = EntityProfiler(ProfilerConfig(decay_half_life_days=half_life_days))
    scores = profiler.run(features).scores
    scored = scores.merge(labels, on="event_id", validate="one_to_one")
    threshold = float(scored.risk_score.quantile(1.0 - ALERT_BUDGET))

    alerted = scored[scored.risk_score >= threshold]
    precision = float(alerted.label.isin(HARD_ANOMALIES).mean()) if len(alerted) else 0.0
    hard = scored[scored.label.isin(HARD_ANOMALIES)]
    hard_recall = float((hard.risk_score >= threshold).mean()) if len(hard) else 0.0
    benign = scored[scored.label == BENIGN_LABEL]
    benign_alert_rate = float((benign.risk_score >= threshold).mean())

    drift_events = scored[scored.label == AMBIGUOUS_LABEL]
    if drift_events.empty:
        return DriftResult(half_life_days, pd.Series(dtype=float), threshold,
                           0.0, 0.0, benign_alert_rate, precision, hard_recall,
                           None, False, 0)

    # Ramp end per entity: the last event the generator still called insider_drift.
    ramp_end = drift_events.groupby("entity_id").timestamp.max()
    drifted = scored[scored.entity_id.isin(ramp_end.index)].copy()
    drifted["ramp_end"] = drifted.entity_id.map(ramp_end)
    drifted["days_rel"] = (
        drifted.timestamp - drifted.ramp_end
    ).dt.total_seconds() / 86_400.0

    # Post-ramp events are benign by construction — the role change is complete and the
    # entity is simply doing its new job.
    post = drifted[(drifted.days_rel > 0) & (drifted.label == BENIGN_LABEL)]
    settled = post[post.days_rel >= settle_days]

    curve = (
        drifted[drifted.days_rel.between(-14, 14)]
        .assign(day=lambda d: np.floor(d.days_rel).astype(int))
        .groupby("day")
        .risk_score.mean()
    )

    alert_rate_during = float((drift_events.risk_score >= threshold).mean())

    # "Time to unflag" only means something if the class was flagged to begin with.
    # Require the ramp to alert at materially more than twice the ordinary benign rate
    # before treating the question as answerable.
    was_ever_flagged = alert_rate_during > 2.0 * benign_alert_rate

    days_to_unflag: float | None = None
    if was_ever_flagged:
        daily_post = post.assign(day=lambda d: np.ceil(d.days_rel).astype(int)).groupby(
            "day"
        ).risk_score.mean()
        for day, value in daily_post.sort_index().items():
            if value < threshold:
                days_to_unflag = float(day)
                break

    return DriftResult(
        half_life_days=half_life_days,
        curve=curve,
        threshold=threshold,
        alert_rate_during=alert_rate_during,
        alert_rate_settled=float((settled.risk_score >= threshold).mean())
        if len(settled)
        else 0.0,
        benign_alert_rate=benign_alert_rate,
        precision_at_budget=precision,
        hard_recall_at_budget=hard_recall,
        days_to_unflag=days_to_unflag,
        was_ever_flagged=was_ever_flagged,
        post_ramp_events=len(post),
    )


def fig_drift_decay(results: list[DriftResult]) -> Path:
    """Plot risk score around the end of a legitimate role change, per half-life.

    Args:
        results: One measurement per decay half-life.

    Returns:
        The path written.
    """
    fig, ax = plt.subplots(figsize=(11, 6.0))
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]

    ax.axvspan(-14, 0, color=GRIDLINE, alpha=0.6, zorder=0, lw=0)
    ax.annotate(
        "role change in progress\n(labelled insider_drift)",
        xy=(-13.4, 0.02), xycoords=("data", "axes fraction"),
        fontsize=8.5, color=INK_MUTED, va="bottom",
    )
    ax.annotate(
        "settled — legitimately doing the new job\n(labelled normal_baseline)",
        xy=(0.6, 0.02), xycoords=("data", "axes fraction"),
        fontsize=8.5, color=INK_MUTED, va="bottom",
    )

    for result, color in zip(results, palette):
        if result.curve.empty:
            continue
        never = result.half_life_days >= 365
        name = "no decay (control)" if never else f"half-life {result.half_life_days:.0f}d"
        ax.plot(
            result.curve.index, result.curve.to_numpy(),
            lw=2.2 if not never else 1.8,
            color=color,
            linestyle="-" if not never else (0, (4, 2)),
            label=name, zorder=3,
        )

    default = next((r for r in results if r.half_life_days == 7.0), results[0])
    ax.axhline(default.threshold, color=INK_SECONDARY, lw=1.3, zorder=2)
    ax.annotate(
        f"top-1% alert threshold ({default.threshold:.2f}) — no curve comes close",
        xy=(13.6, default.threshold), xytext=(0, 6), textcoords="offset points",
        ha="right", fontsize=8.5, color=INK_SECONDARY,
    )
    ax.axvline(0, color=INK_MUTED, lw=1.0, zorder=2)

    ax.set_xlabel("days relative to the end of the role change")
    ax.set_ylabel("mean risk score")
    ax.set_xlim(-14, 14)
    ax.grid(axis="y", zorder=1)
    ax.set_axisbelow(True)
    # Seated below the threshold line so it cannot collide with that annotation.
    ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.80), ncol=2)

    ax.set_ylim(0, max(default.threshold * 1.12, 2.5))

    fig.suptitle(
        "Concept drift — a legitimate role change never disturbs the baseline",
        x=0.055, y=0.995, ha="left", fontsize=14, color=INK_PRIMARY, weight="600",
    )
    fig.text(
        0.055, 0.815,
        "insider_drift entities, aligned on the end of their role change. Scores stay far\n"
        "below the alert threshold throughout — good for false positives, but it also means\n"
        "there is no alert to decay away, so 'time to unflag' is not measurable here. That\n"
        "test belongs to the sequence detector, which will actually flag this pattern.",
        ha="left", va="bottom", fontsize=9, color=INK_SECONDARY, linespacing=1.5,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.79))
    path = FIGURES_DIR / "profiler_drift_decay.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


def main() -> int:
    """Run the full profiler evaluation.

    Returns:
        Process exit code.
    """
    setup_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("The Watchtower — profiler evaluation")
    print("  (labels are joined in this module only, after scoring)\n")

    events = load_events()
    print(f"  building features for {len(events):,} events ...")
    features = build_feature_matrix(events)
    labels = load_labels()

    print("  scoring with default config (half-life 7d) ...")
    result = EntityProfiler(ProfilerConfig()).run(features)
    scored = result.scores.merge(labels, on="event_id", validate="one_to_one")

    separations, threshold, precision = score_separation(scored)

    print("\n" + "=" * 78)
    print("SCORE SEPARATION  (alert budget = top 1%, threshold = "
          f"{threshold:.3f})")
    print("=" * 78)
    print(f"  {'class':<28} {'n':>6} {'median':>8} {'benign %ile':>12} "
          f"{'recall@1%':>10} {'lift':>7}")
    for row in separations:
        print(f"  {row.label:<28} {row.n_events:>6,} {row.median_score:>8.2f} "
              f"{row.benign_percentile:>11.1f}% {row.recall_at_budget:>9.1%} "
              f"{row.lift:>6.0f}x")
    benign_median = float(scored.loc[scored.label == BENIGN_LABEL, "risk_score"].median())
    print(f"  {'normal_baseline':<28} "
          f"{int((scored.label == BENIGN_LABEL).sum()):>6,} {benign_median:>8.2f} "
          f"{'—':>12} {'—':>10} {'—':>7}")
    print(f"\n  precision @ top-1% budget: {precision:.1%} "
          f"(hard anomalies only; base rate "
          f"{scored.label.isin(HARD_ANOMALIES).mean():.2%})")

    cold = cold_start_report(scored, threshold)
    print("\n" + "=" * 78)
    print("COLD START")
    print("=" * 78)
    print(f"  cold-start events                 {cold['cold_events']:,} "
          f"({cold['cold_share_of_log']:.1%} of the log)")
    print(f"  median score, benign cold-start   {cold['cold_benign_median']:.3f}")
    print(f"  median score, benign established  {cold['warm_benign_median']:.3f}")
    print(f"  alert rate, benign cold-start     {cold['cold_benign_alert_rate']:.2%}")
    print(f"  alert rate, benign established    {cold['warm_benign_alert_rate']:.2%}")
    print(f"  cold-start share of the queue     {cold['cold_share_of_alerts']:.1%}")

    print("\n" + "=" * 78)
    print("CONCEPT DRIFT — time to unflag after a legitimate role change")
    print("=" * 78)
    drift_results: list[DriftResult] = []
    for half_life in HALF_LIVES:
        print(f"  scoring at half-life {half_life:g}d ...", flush=True)
        drift_results.append(measure_drift(features, labels, half_life))

    print(f"\n  {'half-life':>10} {'benign':>9} {'drift ramp':>11} {'settled':>9} "
          f"{'unflag':>9} {'prec@1%':>9} {'recall@1%':>10}")
    for item in drift_results:
        name = "no decay" if item.half_life_days >= 365 else f"{item.half_life_days:g}d"
        if not item.was_ever_flagged:
            unflag = "n/a"
        elif item.days_to_unflag is None:
            unflag = "never"
        else:
            unflag = f"{item.days_to_unflag:.0f}d"
        print(f"  {name:>10} {item.benign_alert_rate:>8.2%} "
              f"{item.alert_rate_during:>10.2%} {item.alert_rate_settled:>8.2%} "
              f"{unflag:>9} {item.precision_at_budget:>8.1%} "
              f"{item.hard_recall_at_budget:>9.1%}")
    print(f"\n  measured over {drift_results[0].post_ramp_events:,} post-ramp events "
          "from drifted entities (all labelled normal_baseline).")

    if not any(item.was_ever_flagged for item in drift_results):
        print(
            "\n  READ THIS: 'days to unflag' is n/a at every setting because the class\n"
            "  is never flagged. insider_drift alerts at or below the ordinary benign\n"
            "  rate under all four half-lives, so there is no false positive to decay\n"
            "  away and the metric has nothing to measure. That is a genuine result for\n"
            "  false-positive control — a legitimate role change does not disturb this\n"
            "  profiler — but it is NOT evidence that the decay mechanism works. The\n"
            "  time-to-unflag test needs a detector that flags drift in the first place,\n"
            "  which is the sequence model (Deliverable 3), not this per-event baseline.\n"
            "\n  What the half-life demonstrably does control is the sensitivity/tolerance\n"
            "  trade-off in the two right-hand columns: a shorter half-life holds a\n"
            "  tighter recent baseline, reacting more sharply to any change."
        )

    figures = [
        fig_score_by_class(scored, threshold),
        fig_drift_decay(drift_results),
    ]
    print("\nFigures written:")
    for path in figures:
        print(f"  {path.relative_to(PROJECT_ROOT)}")

    summary = pd.DataFrame(
        [
            {
                "class": row.label,
                "n_events": row.n_events,
                "median_score": round(row.median_score, 4),
                "benign_percentile": round(row.benign_percentile, 2),
                "recall_at_1pct": round(row.recall_at_budget, 4),
                "lift": round(row.lift, 1),
            }
            for row in separations
        ]
    )
    summary.to_csv(REPORTS_DIR / "profiler_separation.csv", index=False)
    print(f"  {(REPORTS_DIR / 'profiler_separation.csv').relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
