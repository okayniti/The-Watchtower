"""Deliverable 7 (metrics) — end-to-end evaluation of the detection engine.

Runs the full pipeline once and reports the numbers ``CLAUDE.md`` §5 asks for::

    python -m src.evaluate

**This module and ``scripts/eval_profiler.py`` are the only readers of
``labels.csv``.** Features, profiler, detector and classifier all receive targets as
arguments; none of them opens ground truth.

What is reported, and on what
-----------------------------
All headline metrics are computed on the **held-out validation slice** — the last 20% of
the time window, which the detector and classifier were never trained on. Reporting them
over the full dataset would be reporting training accuracy.

Precision at a top-1% alert budget is the headline (``CLAUDE.md`` §5). AUC is not
computed anywhere: at a 1.4% base rate it is dominated by the benign majority and would
flatter every model here.

The one exception to the validation-only rule is the ``insider_drift`` trust metric,
which needs the whole drift lifecycle — ramps largely complete before the validation
window opens, so a validation-only view would contain almost no pre-ramp events. It is
computed over the full window and labelled as partly in-sample where that applies.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from src.classifier import (  # noqa: E402
    CLASSES,
    FUNNEL_FRACTION,
    AnomalyClassifier,
    to_class_index,
)
from src.features import build_feature_matrix, load_events  # noqa: E402
from src.generator.config import AMBIGUOUS_LABEL, BENIGN_LABEL, HARD_ANOMALIES  # noqa: E402
from src.profiler import EntityProfiler, ProfilerConfig  # noqa: E402
from src.sequence_detector import SequenceConfig, SequenceDetector  # noqa: E402
from eda import (  # noqa: E402  (reuse the deck's exact palette and style)
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

#: Alert budgets to report. 1% is the headline (CLAUDE.md §5).
BUDGETS: Final[tuple[float, ...]] = (0.005, 0.01, 0.02, 0.05)
HEADLINE_BUDGET: Final[float] = 0.01

#: Best naive single-feature rule from scripts/eda.py, measured over the full dataset.
#: Hardcoded on purpose — it is a fixed published reference point, not something to
#: recompute here.
NAIVE_BEST_RULE: Final[str] = "source_ip shared by >= 10 entities"
NAIVE_BEST_PRECISION: Final[float] = 0.0702

#: Documented sequential blue ramp, light to dark, for the confusion matrix.
_SEQUENTIAL_BLUE: Final[tuple[str, ...]] = (
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#0d366b",
)


def load_labels() -> pd.DataFrame:
    """Load ground truth. One of only two places this happens.

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


@dataclass(slots=True)
class Pipeline:
    """Everything one end-to-end run produced.

    Attributes:
        frame: Per-event context, scores, predictions and ground truth, validation
            membership included.
        detector: The trained sequence detector.
        classifier: The trained type classifier.
        cutoff: Timestamp separating training from validation.
    """

    frame: pd.DataFrame
    detector: SequenceDetector
    classifier: AnomalyClassifier
    cutoff: pd.Timestamp


def run_pipeline(verbose: bool = True) -> Pipeline:
    """Execute features → profiler → detector → classifier, then attach labels.

    Args:
        verbose: Print stage progress.

    Returns:
        The assembled :class:`Pipeline`.
    """
    def log(message: str) -> None:
        if verbose:
            print(message, flush=True)

    log("  loading events and building causal features ...")
    events = load_events()
    features = build_feature_matrix(events)

    log("  running baseline profiler ...")
    profiler_scores = EntityProfiler(ProfilerConfig()).run(features).scores

    log("  preparing sequences ...")
    detector = SequenceDetector(SequenceConfig())
    data = detector.prepare(features, profiler_scores)

    # Labels enter here and are passed to the models as arrays — the models themselves
    # never read the file.
    labels = load_labels()
    aligned = (
        pd.DataFrame({"event_id": data.event_ids})
        .merge(labels, on="event_id", validate="one_to_one")
    )
    binary_targets = aligned.label.isin(HARD_ANOMALIES).to_numpy().astype(np.float32)
    class_targets = to_class_index(aligned.label)

    log(f"  training GRU detector (train < {data.cutoff:%Y-%m-%d %H:%M}) ...")
    detector.fit(data, binary_targets, verbose=verbose)
    sequence_scores = detector.predict(data)

    log("  training type classifier on the top-5% funnel ...")
    classifier = AnomalyClassifier()
    matrix, frame = classifier.build_inputs(features, profiler_scores, sequence_scores)
    funnel_mask = classifier.select_funnel(sequence_scores)
    train_mask = funnel_mask & ~data.is_validation
    classifier.fit(matrix, class_targets, train_mask, verbose=verbose)
    predictions = classifier.predict(matrix, funnel_mask)

    frame = frame[["event_id", "entity_id", "entity_type", "timestamp", "risk_score",
                   "sequence_score"]].copy()
    frame["cold_start"] = profiler_scores.set_index("event_id").loc[
        frame.event_id, "cold_start"
    ].to_numpy()
    frame["label"] = aligned.label.to_numpy()
    frame["episode_id"] = aligned.episode_id.to_numpy()
    frame["true_class"] = class_targets
    frame["pred_class"] = predictions
    frame["in_funnel"] = funnel_mask
    frame["is_validation"] = data.is_validation
    frame["is_attack"] = frame.label.isin(HARD_ANOMALIES)

    return Pipeline(frame=frame, detector=detector, classifier=classifier,
                    cutoff=data.cutoff)


# --------------------------------------------------------------------------------------
# 1. Precision at k
# --------------------------------------------------------------------------------------


def precision_at_k(
    scores: np.ndarray, is_attack: np.ndarray, budgets: tuple[float, ...] = BUDGETS
) -> pd.DataFrame:
    """Compute precision, recall and false-positive rate at each alert budget.

    Args:
        scores: Ranking score per event.
        is_attack: Whether each event is one of the six hard-anomaly classes.
        budgets: Alert-budget fractions.

    Returns:
        One row per budget.
    """
    total = len(scores)
    order = np.argsort(-scores, kind="stable")
    ranked = is_attack[order]
    positives = int(is_attack.sum())

    rows = []
    for budget in budgets:
        k = max(1, int(round(total * budget)))
        caught = int(ranked[:k].sum())
        rows.append(
            {
                "budget": budget,
                "alerts": k,
                "true_positives": caught,
                "precision": caught / k,
                "recall": caught / positives if positives else 0.0,
                "false_positive_rate": (k - caught) / k,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# 2. Per-class metrics
# --------------------------------------------------------------------------------------


def per_class_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute precision, recall and F1 per class from the classifier's predictions.

    Args:
        frame: Validation-slice rows with ``true_class`` and ``pred_class``.

    Returns:
        One row per class in :data:`~src.classifier.CLASSES` order.
    """
    rows = []
    for index, name in enumerate(CLASSES):
        true_positive = int(((frame.pred_class == index) & (frame.true_class == index)).sum())
        predicted = int((frame.pred_class == index).sum())
        actual = int((frame.true_class == index).sum())
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / actual if actual else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        rows.append(
            {
                "class": name,
                "support": actual,
                "predicted": predicted,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return pd.DataFrame(rows)


def confusion_matrix(frame: pd.DataFrame) -> np.ndarray:
    """Build the class confusion matrix.

    Args:
        frame: Validation-slice rows with ``true_class`` and ``pred_class``.

    Returns:
        A ``(n_classes, n_classes)`` count matrix, rows true and columns predicted.
    """
    size = len(CLASSES)
    matrix = np.zeros((size, size), dtype=int)
    for true, predicted in zip(frame.true_class, frame.pred_class):
        matrix[int(true), int(predicted)] += 1
    return matrix


# --------------------------------------------------------------------------------------
# 3-5. Trust metrics
# --------------------------------------------------------------------------------------


def insider_drift_false_positives(
    frame: pd.DataFrame, threshold: float
) -> dict[str, float]:
    """Measure the false-positive rate on legitimate role change, before and after.

    The trust metric. ``insider_drift`` is benign, so *any* alert on it is a false
    positive. Splitting pre-ramp from post-ramp answers the question that matters
    operationally: once an employee has settled into a new role, does the system stop
    accusing them?

    Args:
        frame: All scored events with labels.
        threshold: Detection score at or above which an event alerts.

    Returns:
        Alert rates and counts for the ramp and settled phases.
    """
    drift = frame[frame.label == AMBIGUOUS_LABEL]
    if drift.empty:
        return {"ramp_events": 0, "ramp_fp_rate": 0.0, "settled_events": 0,
                "settled_fp_rate": 0.0, "entities": 0}

    ramp_end = drift.groupby("entity_id").timestamp.max()
    drifted = frame[frame.entity_id.isin(ramp_end.index)].copy()
    drifted["ramp_end"] = drifted.entity_id.map(ramp_end)

    ramp = drifted[
        (drifted.timestamp <= drifted.ramp_end) & (drifted.label == AMBIGUOUS_LABEL)
    ]
    settled = drifted[
        (drifted.timestamp > drifted.ramp_end) & (drifted.label == BENIGN_LABEL)
    ]
    return {
        "ramp_events": len(ramp),
        "ramp_fp_rate": float((ramp.sequence_score >= threshold).mean()) if len(ramp) else 0.0,
        "settled_events": len(settled),
        "settled_fp_rate": float((settled.sequence_score >= threshold).mean())
        if len(settled)
        else 0.0,
        "entities": int(drifted.entity_id.nunique()),
    }


def cold_start_summary(frame: pd.DataFrame, threshold: float) -> dict[str, float]:
    """Summarise how history-poor entities were scored.

    Args:
        frame: Validation-slice rows.
        threshold: Detection score at or above which an event alerts.

    Returns:
        Mean, max and alert rate for cold-start versus established events.
    """
    cold = frame[frame.cold_start.astype(bool)]
    warm = frame[~frame.cold_start.astype(bool)]
    cold_benign = cold[~cold.is_attack]
    return {
        "cold_events": len(cold),
        "cold_entities": int(cold.entity_id.nunique()),
        "cold_mean_score": float(cold.sequence_score.mean()) if len(cold) else 0.0,
        "cold_max_score": float(cold.sequence_score.max()) if len(cold) else 0.0,
        "warm_mean_score": float(warm.sequence_score.mean()) if len(warm) else 0.0,
        "cold_benign_alert_rate": float((cold_benign.sequence_score >= threshold).mean())
        if len(cold_benign)
        else 0.0,
        "cold_share_of_alerts": float(
            frame[frame.sequence_score >= threshold].cold_start.astype(bool).mean()
        ),
    }


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------


def fig_precision_at_k(
    table: pd.DataFrame, profiler_table: pd.DataFrame
) -> Path:
    """Plot precision against alert budget, with both baselines marked.

    Args:
        table: Detector precision-at-k table.
        profiler_table: Same table computed from the profiler score alone.

    Returns:
        The path written.
    """
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    x = np.arange(len(table))
    width = 0.38

    ax.bar(x - width / 2, table.precision * 100, width, color=CLASS_COLOR["brute_force"],
           label="GRU sequence detector", zorder=3)
    ax.bar(x + width / 2, profiler_table.precision * 100, width,
           color=INK_MUTED, label="profiler baseline", zorder=3)

    for i, (detector_value, profiler_value) in enumerate(
        zip(table.precision, profiler_table.precision)
    ):
        ax.annotate(f"{detector_value:.1%}", xy=(i - width / 2, detector_value * 100),
                    xytext=(0, 4), textcoords="offset points", ha="center",
                    fontsize=9, color=INK_PRIMARY, weight="600")
        ax.annotate(f"{profiler_value:.1%}", xy=(i + width / 2, profiler_value * 100),
                    xytext=(0, 4), textcoords="offset points", ha="center",
                    fontsize=8.5, color=INK_SECONDARY)

    ax.axhline(NAIVE_BEST_PRECISION * 100, color=INK_SECONDARY, lw=1.3,
               linestyle=(0, (5, 3)), zorder=2)
    ax.annotate(
        f"best naive rule ({NAIVE_BEST_PRECISION:.1%})",
        xy=(len(table) - 0.45, NAIVE_BEST_PRECISION * 100), xytext=(0, 5),
        textcoords="offset points", ha="right", fontsize=8.5, color=INK_SECONDARY,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"top {b:.1%}".replace(".0%", "%") for b in table.budget])
    ax.set_xlabel("alert budget")
    ax.set_ylabel("precision (%)")
    ax.grid(axis="y", zorder=1)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right")

    headline = table[np.isclose(table.budget, HEADLINE_BUDGET)].precision.iloc[0]
    fig.suptitle("Precision at alert budget — headline metric",
                 x=0.055, y=0.99, ha="left", fontsize=14, color=INK_PRIMARY, weight="600")
    fig.text(
        0.055, 0.875,
        f"Validation slice, held out by time. At the top-1% budget the detector reaches\n"
        f"{headline:.1%} precision against a {NAIVE_BEST_PRECISION:.1%} best naive rule.",
        ha="left", va="bottom", fontsize=9, color=INK_SECONDARY, linespacing=1.5,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.86))
    path = FIGURES_DIR / "precision_at_k.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_confusion_matrix(matrix: np.ndarray) -> Path:
    """Plot the class confusion matrix, row-normalised.

    Args:
        matrix: Raw count matrix.

    Returns:
        The path written.
    """
    cmap = LinearSegmentedColormap.from_list("watchtower_blue", _SEQUENTIAL_BLUE)
    totals = matrix.sum(axis=1, keepdims=True)
    normalised = np.divide(matrix, np.maximum(totals, 1))

    fig, ax = plt.subplots(figsize=(9.5, 7.6))
    ax.imshow(normalised, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")

    labels = [PRETTY[name] for name in CLASSES]
    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(labels, rotation=32, ha="right", fontsize=8.5)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")

    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if matrix[i, j] == 0:
                continue
            # Ink flips to light once the cell is dark enough to need it.
            colour = "#ffffff" if normalised[i, j] > 0.55 else INK_PRIMARY
            ax.annotate(
                f"{matrix[i, j]:,}\n{normalised[i, j]:.0%}",
                xy=(j, i), ha="center", va="center", fontsize=8, color=colour,
            )

    ax.set_xticks(np.arange(len(CLASSES) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(CLASSES) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)

    fig.suptitle("Confusion matrix — anomaly type", x=0.055, y=0.985, ha="left",
                 fontsize=14, color=INK_PRIMARY, weight="600")
    fig.text(
        0.055, 0.905,
        "Validation slice. Cells are row-normalised: each row sums to 100% of that\n"
        "class's true events. insider_drift is folded into normal_baseline — it is benign.",
        ha="left", va="bottom", fontsize=9, color=INK_SECONDARY, linespacing=1.5,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.885))
    path = FIGURES_DIR / "confusion_matrix.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_per_class_f1(table: pd.DataFrame) -> Path:
    """Plot per-class precision, recall and F1.

    Args:
        table: Output of :func:`per_class_metrics`.

    Returns:
        The path written.
    """
    attacks = table[table["class"] != BENIGN_LABEL].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    y = np.arange(len(attacks))

    ax.barh(y, attacks.f1, height=0.6,
            color=[CLASS_COLOR[name] for name in attacks["class"]], zorder=3)
    for i, row in attacks.iterrows():
        ax.annotate(f"F1 {row.f1:.2f}", xy=(row.f1, i), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=9.5,
                    color=INK_PRIMARY, weight="600")
        ax.annotate(
            f"P {row.precision:.2f} · R {row.recall:.2f} · n={row.support}",
            xy=(row.f1, i), xytext=(64, 0), textcoords="offset points",
            va="center", fontsize=8.5, color=INK_MUTED,
        )

    ax.set_yticks(y)
    ax.set_yticklabels([PRETTY[name] for name in attacks["class"]], fontsize=9)
    ax.set_xlabel("F1")
    ax.set_xlim(0, 1.0)
    ax.grid(axis="x", zorder=1)
    ax.set_axisbelow(True)

    fig.suptitle("Per-class detection quality", x=0.055, y=0.99, ha="left",
                 fontsize=14, color=INK_PRIMARY, weight="600")
    fig.text(
        0.055, 0.885,
        "Validation slice, six attack classes. normal_baseline is omitted — at 98% of\n"
        "events its F1 is near 1.00 by construction and tells you nothing.",
        ha="left", va="bottom", fontsize=9, color=INK_SECONDARY, linespacing=1.5,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.87))
    path = FIGURES_DIR / "per_class_f1.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


def main() -> int:
    """Run the pipeline and print every required metric.

    Returns:
        Process exit code.
    """
    setup_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("The Watchtower — detection engine evaluation")
    pipeline = run_pipeline(verbose=True)
    frame = pipeline.frame
    validation = frame[frame.is_validation].reset_index(drop=True)

    detector_table = precision_at_k(
        validation.sequence_score.to_numpy(), validation.is_attack.to_numpy()
    )
    profiler_table = precision_at_k(
        validation.risk_score.to_numpy(), validation.is_attack.to_numpy()
    )
    headline = detector_table[np.isclose(detector_table.budget, HEADLINE_BUDGET)].iloc[0]
    profiler_headline = profiler_table[
        np.isclose(profiler_table.budget, HEADLINE_BUDGET)
    ].iloc[0]
    threshold = float(
        np.quantile(validation.sequence_score.to_numpy(), 1.0 - HEADLINE_BUDGET)
    )

    bar = "=" * 78
    print(f"\n{bar}")
    print(f"PRECISION @ K   (validation slice: {len(validation):,} events, "
          f"held out after {pipeline.cutoff:%Y-%m-%d %H:%M})")
    print(bar)
    print(f"  {'budget':>8} {'alerts':>8} {'TP':>6} {'precision':>11} {'recall':>9} "
          f"{'FP rate':>9}")
    for _, row in detector_table.iterrows():
        marker = "  <-- HEADLINE" if np.isclose(row.budget, HEADLINE_BUDGET) else ""
        print(f"  {row.budget:>7.1%} {int(row.alerts):>8,} {int(row.true_positives):>6} "
              f"{row.precision:>10.1%} {row.recall:>8.1%} {row.false_positive_rate:>8.1%}"
              f"{marker}")
    print(f"\n  base rate in validation slice: {validation.is_attack.mean():.2%}")

    class_table = per_class_metrics(validation)
    print(f"\n{bar}")
    print("PER-CLASS PRECISION / RECALL / F1   (validation slice)")
    print(bar)
    print(f"  {'class':<28} {'support':>8} {'precision':>10} {'recall':>8} {'F1':>7}")
    for _, row in class_table.iterrows():
        print(f"  {row['class']:<28} {int(row.support):>8,} {row.precision:>9.2f} "
              f"{row.recall:>7.2f} {row.f1:>6.2f}")
    attack_rows = class_table[class_table["class"] != BENIGN_LABEL]
    print(f"\n  macro-F1 over the six attack classes: {attack_rows.f1.mean():.3f}")

    matrix = confusion_matrix(validation)
    print(f"\n{bar}")
    print("CONFUSION MATRIX   (rows true, columns predicted)")
    print(bar)
    header = "".join(f"{PRETTY[name][:11]:>13}" for name in CLASSES)
    print(f"  {'':<24}{header}")
    for i, name in enumerate(CLASSES):
        counts = "".join(f"{matrix[i, j]:>13,}" for j in range(len(CLASSES)))
        print(f"  {PRETTY[name]:<24}{counts}")

    drift = insider_drift_false_positives(frame, threshold)
    print(f"\n{bar}")
    print("INSIDER_DRIFT FALSE-POSITIVE RATE   (the trust metric)")
    print(bar)
    print(f"  entities with a role change        {drift['entities']}")
    print(f"  during ramp   ({drift['ramp_events']:>4} events)   "
          f"FP rate {drift['ramp_fp_rate']:.2%}")
    print(f"  after settling ({drift['settled_events']:>4} events)  "
          f"FP rate {drift['settled_fp_rate']:.2%}")
    print("\n  Any alert here is a false positive — the employee did nothing wrong.")
    print("  Computed over the full window, not the validation slice: drift ramps mostly")
    print("  complete before validation opens, so the ramp figure is partly in-sample.")

    cold = cold_start_summary(validation, threshold)
    print(f"\n{bar}")
    print("COLD-START ENTITIES   (validation slice)")
    print(bar)
    print(f"  cold-start events / entities       {cold['cold_events']:,} / "
          f"{cold['cold_entities']}")
    print(f"  mean detection score, cold-start   {cold['cold_mean_score']:.4f}")
    print(f"  mean detection score, established  {cold['warm_mean_score']:.4f}")
    print(f"  max detection score, cold-start    {cold['cold_max_score']:.4f}")
    print(f"  benign cold-start alert rate       {cold['cold_benign_alert_rate']:.2%}")
    print(f"  cold-start share of the queue      {cold['cold_share_of_alerts']:.1%}")

    print(f"\n{bar}")
    print("LIFT VS BASELINES   (precision @ top-1%)")
    print(bar)
    naive_lift = headline.precision / NAIVE_BEST_PRECISION
    profiler_lift = (
        headline.precision / profiler_headline.precision
        if profiler_headline.precision > 0
        else float("inf")
    )
    print(f"  {'GRU sequence detector':<34} {headline.precision:>8.1%}")
    print(f"  {'profiler score alone':<34} {profiler_headline.precision:>8.1%}   "
          f"-> {profiler_lift:.2f}x")
    print(f"  {'best naive rule':<34} {NAIVE_BEST_PRECISION:>8.1%}   "
          f"-> {naive_lift:.2f}x")
    print(f"  ({NAIVE_BEST_RULE})")
    print(f"\n  random selection at this budget would score "
          f"{validation.is_attack.mean():.2%}")

    figures = [
        fig_precision_at_k(detector_table, profiler_table),
        fig_confusion_matrix(matrix),
        fig_per_class_f1(class_table),
    ]
    detector_table.to_csv(REPORTS_DIR / "precision_at_k.csv", index=False)
    class_table.to_csv(REPORTS_DIR / "per_class_metrics.csv", index=False)

    print("\nFigures written:")
    for path in figures:
        print(f"  {path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
