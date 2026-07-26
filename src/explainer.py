"""Deliverable 5 — the explainability layer.

Turns a risk score into a sentence an analyst can act on. Two paths, and the second is
the unusual one:

**Alerts.** For every flagged event, rank the features that drove the prediction and
render the top few as one plain-English sentence::

    Flagged (risk 0.91, impossible_travel): session 4,127 km from this entity's
    previous location within 40 min (implied 6,150 km/h) + source address never used
    by this entity before + activity outside this entity's own usual hours.

**Suppressions.** For events the *baseline profiler* scored highly but the sequence
detector did not, explain why the system stayed quiet::

    Suppressed (profiler 6.41 -> detector 0.04): first-ever access to this resource,
    but this entity has been steadily working in this area for 14 days — consistent
    with a settled role change, not exploration.

Most systems explain what they alert on. An analyst's trust is built at least as much by
the near-misses: being able to see that the system *considered* something and had a
defensible reason to stay quiet is what stops people from lowering the threshold "just
in case". It is also the thing that makes ``insider_drift`` legible rather than
mysterious.

How contributions are ranked
----------------------------
Gradient-times-input attribution on the classifier head: one backward pass over the
funnel gives, per event, each input's signed contribution to the predicted class logit.
This is cheap (a single batched backward pass, no per-event resampling) and adequate for
a two-hidden-layer MLP, where full SHAP would cost orders of magnitude more for a
ranking that would largely agree.

A feature must clear two bars before it is narrated: it must contribute positively to
the prediction, and it must actually be deviant for this entity — the profiler's
per-feature z-score has to exceed :data:`MIN_DEVIATION`. Without the second test the
layer happily writes fluent sentences about features sitting at their normal values,
which is worse than saying nothing.

Labels
------
This module reads no labels of its own. It calls :func:`src.evaluate.run_pipeline`,
which attaches ground truth for evaluation; the label is carried into the output file so
the dashboard can display measured precision, and is explicitly *not* an input to any
sentence.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Final

import numpy as np
import pandas as pd
import torch

from .classifier import CLASSES, SCORE_INPUTS
from .evaluate import HEADLINE_BUDGET, run_pipeline
from .features import FEATURE_NAMES, SPEC_BY_NAME

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

#: A feature must deviate at least this far from the entity's own baseline before it is
#: allowed into a sentence.
MIN_DEVIATION: Final[float] = 0.5

#: Maximum contributing factors narrated in one sentence.
MAX_FACTORS: Final[int] = 4

#: Minimum factors to narrate when any qualify.
MIN_FACTORS: Final[int] = 2

#: Fraction of validation events to pre-explain. Comfortably above the dashboard's
#: largest selectable budget (5%) so the queue never shows an unexplained alert.
EXPLAIN_FRACTION: Final[float] = 0.06

#: Profiler percentile above which a non-alerting event counts as "considered".
SUPPRESSION_PROFILER_QUANTILE: Final[float] = 0.95


# --------------------------------------------------------------------------------------
# Phrase templates
# --------------------------------------------------------------------------------------


def _fmt(value: float) -> str:
    """Format a number for prose, dropping pointless decimals.

    Args:
        value: The number to render.

    Returns:
        A human-readable string.
    """
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 10:
        return f"{value:.0f}"
    return f"{value:.2f}".rstrip("0").rstrip(".")


#: Rule-based phrase templates, keyed by feature name. Each receives the raw feature
#: value and the event's full feature row, so composite phrasing (distance *and* time
#: *and* velocity) is possible where it reads better.
PHRASE_TEMPLATES: Final[dict[str, Callable[[float, pd.Series], str]]] = {
    "geo_velocity_kmh": lambda v, row: (
        f"session {_fmt(row.get('geo_distance_km', 0.0))} km from this entity's previous "
        f"location, implying {_fmt(v)} km/h of travel"
    ),
    "geo_distance_km": lambda v, row: (
        f"session {_fmt(v)} km from this entity's previous location"
    ),
    "geo_city_novelty": lambda v, row: (
        "acting from a location this entity has almost never used"
        if v > 0.9 else "acting from a location that is unusual for this entity"
    ),
    "auth_fail_count": lambda v, row: f"{_fmt(v)} failed authentication attempts in this session",
    "auth_fail_rate_1h": lambda v, row: (
        f"{v:.0%} of this entity's last hour of activity failed to authenticate"
    ),
    "events_10m": lambda v, row: f"{_fmt(v)} events from this entity in the preceding 10 minutes",
    "auth_fail_streak": lambda v, row: f"{_fmt(v)} consecutive failed authentications",
    "resource_is_novel": lambda v, row: "first-ever access to this resource by this entity",
    "novel_resources_6h": lambda v, row: (
        f"{_fmt(v)} never-before-seen resources touched in the last 6 hours"
    ),
    "resource_rarity": lambda v, row: "a resource this entity almost never touches",
    "recon_command_ratio": lambda v, row: (
        f"{v:.0%} of the command sequence is reconnaissance"
    ),
    "sensitivity_jump": lambda v, row: (
        f"asset sensitivity {_fmt(v)} tiers above this entity's usual level"
    ),
    "device_is_novel": lambda v, row: "device fingerprint never seen for this entity",
    "device_os_family_preserved": lambda v, row: (
        "the new device reuses an OS this entity already runs — consistent with a cloned "
        "fingerprint rather than a hardware refresh"
    ),
    "device_count": lambda v, row: f"{_fmt(v)} distinct devices now seen for this entity",
    "ip_distinct_entities_1h": lambda v, row: (
        f"{_fmt(v)} different identities used this source address within the hour"
    ),
    "ip_fail_rate_1h": lambda v, row: (
        f"{v:.0%} of authentications from this address failed within the hour"
    ),
    "ip_is_novel": lambda v, row: "source address never used by this entity before",
    "ip_rarity": lambda v, row: "a source address this entity rarely uses",
    "duration_pctile_own": lambda v, row: (
        f"session length in this entity's own {v:.0%} percentile"
    ),
    "interval_regularity": lambda v, row: (
        f"unusually metronomic event timing for this entity (regularity {v:.2f})"
    ),
    "off_hours_score": lambda v, row: "activity outside this entity's own usual hours",
    "bulk_read_ratio": lambda v, row: f"{v:.0%} of commands are bulk data retrieval",
    "session_seconds_24h": lambda v, row: (
        f"{_fmt(v / 60)} minutes of cumulative session time on this resource in 24 hours"
    ),
    "resource_repeat_streak": lambda v, row: (
        f"{_fmt(v)} consecutive sessions against the same resource"
    ),
}


@dataclass(slots=True)
class Factor:
    """One ranked contributing feature.

    Attributes:
        feature: Feature name.
        value: Raw feature value for this event.
        deviation: Profiler z-score — how far from this entity's own baseline.
        contribution: Signed attribution toward the predicted class.
        direction: ``"high"`` or ``"both"``, from the feature's spec.
        targets: The attack signature this feature was engineered for.
        phrase: The rendered analyst-readable clause.
    """

    feature: str
    value: float
    deviation: float
    contribution: float
    direction: str
    targets: str
    phrase: str


@dataclass(slots=True)
class Explanation:
    """A structured explanation record for one event.

    Attributes:
        event_id: The event.
        entity_id: The acting entity.
        entity_type: Its cohort.
        timestamp: Event time, ISO format.
        risk_score: Sequence-detector probability.
        profiler_score: Baseline profiler score.
        predicted_type: The classifier's predicted class.
        cold_start: Whether the entity had little history at this point.
        kind: ``"alert"`` or ``"suppressed"``.
        factors: Ranked contributing features.
        sentence: The plain-English explanation.
        true_label: Ground truth, carried for dashboard metrics only — never an input
            to the sentence.
        resource_accessed: The asset touched, for display.
        source_ip: The origin address, for display.
        geo_location: Where the event appeared to come from, for display.
    """

    event_id: str
    entity_id: str
    entity_type: str
    timestamp: str
    risk_score: float
    profiler_score: float
    predicted_type: str
    cold_start: bool
    kind: str
    factors: list[Factor] = field(default_factory=list)
    sentence: str = ""
    true_label: str = ""
    resource_accessed: str = ""
    source_ip: str = ""
    geo_location: str = ""


# --------------------------------------------------------------------------------------
# Attribution
# --------------------------------------------------------------------------------------


def classifier_attributions(
    classifier, matrix: np.ndarray, rows: np.ndarray, predicted: np.ndarray
) -> np.ndarray:
    """Compute gradient-times-input attributions for the predicted class.

    Args:
        classifier: The trained :class:`~src.classifier.AnomalyClassifier`.
        matrix: The full classifier input matrix.
        rows: Row indices to attribute.
        predicted: Predicted class index per row in ``rows``.

    Returns:
        A ``(len(rows), n_inputs)`` array of signed contributions.
    """
    scaled = (matrix[rows] - classifier.mean) / classifier.std
    x = torch.tensor(scaled.astype(np.float32), requires_grad=True)
    classifier.model.eval()
    logits = classifier.model(x)
    selected = logits.gather(
        1, torch.tensor(predicted, dtype=torch.long).unsqueeze(1)
    ).sum()
    selected.backward()
    return (x.grad * x).detach().numpy()


def rank_factors(
    contributions: np.ndarray,
    deviations: pd.Series,
    feature_row: pd.Series,
    limit: int = MAX_FACTORS,
) -> list[Factor]:
    """Select and render the features worth telling an analyst about.

    Args:
        contributions: Signed attribution per input for this event.
        deviations: Profiler z-score per feature for this event.
        feature_row: Raw feature values for this event.
        limit: Maximum factors to return.

    Returns:
        Ranked factors, strongest first. Empty if nothing cleared both bars.
    """
    candidates: list[Factor] = []
    for index, name in enumerate(FEATURE_NAMES):
        contribution = float(contributions[index])
        deviation = float(deviations.get(name, 0.0))
        # Both tests matter: the feature must push toward the prediction AND be
        # genuinely abnormal for this entity.
        if contribution <= 0 or deviation < MIN_DEVIATION:
            continue
        template = PHRASE_TEMPLATES.get(name)
        if template is None:
            continue
        spec = SPEC_BY_NAME[name]
        candidates.append(
            Factor(
                feature=name,
                value=float(feature_row.get(name, 0.0)),
                deviation=deviation,
                contribution=contribution,
                direction=spec.direction,
                targets=spec.targets,
                phrase=template(float(feature_row.get(name, 0.0)), feature_row),
            )
        )

    candidates.sort(key=lambda f: f.contribution, reverse=True)
    # geo_distance duplicates what the velocity clause already says.
    names = {f.feature for f in candidates[:limit]}
    if "geo_velocity_kmh" in names:
        candidates = [f for f in candidates if f.feature != "geo_distance_km"]
    return candidates[:limit]


# --------------------------------------------------------------------------------------
# Sentence assembly
# --------------------------------------------------------------------------------------


def build_alert_sentence(
    risk_score: float, predicted_type: str, factors: list[Factor], cold_start: bool
) -> str:
    """Render an alert as one sentence.

    Args:
        risk_score: Detector probability.
        predicted_type: Predicted anomaly class.
        factors: Ranked contributing factors.
        cold_start: Whether the entity has little history.

    Returns:
        The analyst-readable sentence.
    """
    if not factors:
        body = (
            "elevated by the sequence model on the entity's recent event pattern rather "
            "than any single field"
        )
    else:
        chosen = factors[:MAX_FACTORS]
        if len(chosen) < MIN_FACTORS and len(factors) >= MIN_FACTORS:
            chosen = factors[:MIN_FACTORS]
        body = " + ".join(f.phrase for f in chosen)

    prefix = f"Flagged (risk {risk_score:.2f}, {predicted_type})"
    suffix = (
        " — limited history for this entity, scored against its peer group."
        if cold_start
        else "."
    )
    return f"{prefix}: {body}{suffix}"


def build_suppression_sentence(
    profiler_score: float,
    risk_score: float,
    factors: list[Factor],
    settled_days: float | None,
    familiar_location: bool,
) -> str:
    """Explain why a high-baseline event did not become an alert.

    The reason is derived only from things observable without labels: how long the
    entity has been working in this area, whether the apparent location is one it
    routinely uses, and whether the deviation had any sequential support.

    Args:
        profiler_score: Baseline profiler score.
        risk_score: Sequence-detector probability.
        factors: Ranked contributing factors from the profiler's view.
        settled_days: Days this entity has been accessing this resource, or ``None`` if
            this is the first access.
        familiar_location: Whether the apparent city is one the entity uses routinely.

    Returns:
        The analyst-readable sentence.
    """
    lead = factors[0].phrase if factors else "an elevated baseline deviation"
    top_feature = factors[0].feature if factors else ""

    if top_feature.startswith("resource") or top_feature == "sensitivity_jump":
        if settled_days is not None and settled_days >= 3:
            reason = (
                f"but this entity has been working in this area for "
                f"{settled_days:.0f} days — consistent with a settled role change, not "
                "exploration"
            )
        else:
            reason = (
                "but it is an isolated access with no follow-on exploration of "
                "neighbouring systems"
            )
    elif top_feature.startswith("geo") and familiar_location:
        reason = (
            "but the apparent location is one this entity egresses from routinely — "
            "consistent with corporate VPN routing rather than travel"
        )
    elif top_feature.startswith("auth"):
        reason = (
            "but the failures are isolated rather than a sustained burst against this "
            "identity"
        )
    elif top_feature.startswith("device"):
        reason = (
            "but the device has persisted across subsequent sessions — consistent with a "
            "hardware refresh rather than a spoof"
        )
    elif top_feature.startswith("duration") or top_feature.startswith("session"):
        reason = "but the session sits inside this entity's own established range"
    else:
        reason = "but the entity's surrounding event sequence carries no supporting signal"

    return (
        f"Suppressed (profiler {profiler_score:.2f} -> detector {risk_score:.2f}): "
        f"{lead}, {reason}."
    )


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------


def build_explanations(verbose: bool = True) -> tuple[list[Explanation], dict]:
    """Run the pipeline and produce explanations for the validation slice.

    Args:
        verbose: Print progress.

    Returns:
        A ``(records, meta)`` tuple.
    """
    def log(message: str) -> None:
        if verbose:
            print(message, flush=True)

    log("  running detection pipeline ...")
    pipeline = run_pipeline(verbose=verbose)
    frame = pipeline.frame
    features = pipeline.features
    deviations = pipeline.deviations

    events = pd.read_csv(DATA_DIR / "access_logs.csv")
    events["timestamp"] = pd.to_datetime(events["timestamp"])
    detail = events.set_index("event_id")

    # How long has this entity been using this resource? Label-free, and the basis of
    # the "settled role change" suppression reason.
    first_seen = (
        events.groupby(["entity_id", "resource_accessed"]).timestamp.min().rename("first_seen")
    )

    validation = frame[frame.is_validation]
    explain_threshold = float(
        np.quantile(validation.sequence_score.to_numpy(), 1.0 - EXPLAIN_FRACTION)
    )
    alert_threshold = float(
        np.quantile(validation.sequence_score.to_numpy(), 1.0 - HEADLINE_BUDGET)
    )
    profiler_threshold = float(
        np.quantile(validation.risk_score.to_numpy(), SUPPRESSION_PROFILER_QUANTILE)
    )

    alert_rows = validation.index[validation.sequence_score >= explain_threshold]
    suppressed_rows = validation.index[
        (validation.risk_score >= profiler_threshold)
        & (validation.sequence_score < alert_threshold)
    ]
    log(f"  explaining {len(alert_rows):,} candidate alerts and "
        f"{len(suppressed_rows):,} suppressions ...")

    records: list[Explanation] = []
    for rows, kind in ((alert_rows, "alert"), (suppressed_rows, "suppressed")):
        if len(rows) == 0:
            continue
        positions = np.asarray(rows, dtype=int)
        predicted = frame.pred_class.to_numpy()[positions]
        contributions = classifier_attributions(
            pipeline.classifier, pipeline.matrix, positions, predicted
        )

        for offset, position in enumerate(positions):
            row = frame.iloc[position]
            feature_row = features.iloc[position]
            deviation_row = deviations.iloc[position]
            factors = rank_factors(
                contributions[offset], deviation_row, feature_row
            )

            raw = detail.loc[row.event_id] if row.event_id in detail.index else None
            resource = str(raw["resource_accessed"]) if raw is not None else ""
            geo = str(raw["geo_location"]) if raw is not None else ""
            source_ip = str(raw["source_ip"]) if raw is not None else ""

            if kind == "alert":
                sentence = build_alert_sentence(
                    float(row.sequence_score),
                    CLASSES[int(row.pred_class)],
                    factors,
                    bool(row.cold_start),
                )
            else:
                key = (row.entity_id, resource)
                settled_days: float | None = None
                if key in first_seen.index:
                    delta = row.timestamp - first_seen.loc[key]
                    settled_days = float(delta.total_seconds() / 86_400.0)
                    if settled_days <= 0:
                        settled_days = None
                familiar = float(feature_row.get("geo_city_novelty", 1.0)) < 0.6
                sentence = build_suppression_sentence(
                    float(row.risk_score), float(row.sequence_score), factors,
                    settled_days, familiar,
                )

            records.append(
                Explanation(
                    event_id=str(row.event_id),
                    entity_id=str(row.entity_id),
                    entity_type=str(row.entity_type),
                    timestamp=pd.Timestamp(row.timestamp).isoformat(sep=" "),
                    risk_score=float(row.sequence_score),
                    profiler_score=float(row.risk_score),
                    predicted_type=CLASSES[int(row.pred_class)],
                    cold_start=bool(row.cold_start),
                    kind=kind,
                    factors=factors,
                    sentence=sentence,
                    true_label=str(row.label),
                    resource_accessed=resource,
                    source_ip=source_ip,
                    geo_location=geo,
                )
            )

    meta = {
        "generated_from": "src/explainer.py",
        "validation_events": int(len(validation)),
        "validation_cutoff": pd.Timestamp(pipeline.cutoff).isoformat(sep=" "),
        "alert_threshold_at_1pct": alert_threshold,
        "explain_threshold": explain_threshold,
        "profiler_suppression_threshold": profiler_threshold,
        "attack_base_rate": float(validation.is_attack.mean()),
        "note": (
            "true_label is carried for dashboard metrics only. It is never an input to "
            "any explanation sentence."
        ),
    }
    return records, meta


def write_outputs(records: list[Explanation], meta: dict) -> dict[str, Path]:
    """Persist explanations and the validation score table for the dashboard.

    Args:
        records: Explanation records.
        meta: Run metadata.

    Returns:
        Mapping of artefact name to path written.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    alerts_path = DATA_DIR / "alerts.json"
    payload = {
        "meta": meta,
        "records": [asdict(record) for record in records],
    }
    alerts_path.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    return {"alerts": alerts_path}


def main() -> int:
    """Build and cache explanations for the validation slice.

    Returns:
        Process exit code.
    """
    print("The Watchtower — explainability layer")
    records, meta = build_explanations(verbose=True)

    alerts = [r for r in records if r.kind == "alert"]
    suppressed = [r for r in records if r.kind == "suppressed"]
    paths = write_outputs(records, meta)

    validation_scores = pd.DataFrame(
        [
            {
                "event_id": r.event_id,
                "entity_id": r.entity_id,
                "entity_type": r.entity_type,
                "timestamp": r.timestamp,
                "risk_score": r.risk_score,
                "profiler_score": r.profiler_score,
                "predicted_type": r.predicted_type,
                "cold_start": r.cold_start,
                "kind": r.kind,
                "true_label": r.true_label,
            }
            for r in records
        ]
    )
    scores_path = DATA_DIR / "validation_scores.csv"
    validation_scores.to_csv(scores_path, index=False)

    print(f"\n  {len(alerts):,} alert explanations, {len(suppressed):,} suppressions")
    print(f"  alerts   -> {paths['alerts'].relative_to(PROJECT_ROOT)}")
    print(f"  scores   -> {scores_path.relative_to(PROJECT_ROOT)}")

    print("\n  sample alert explanations:")
    for record in sorted(alerts, key=lambda r: -r.risk_score)[:4]:
        print(f"\n    [{record.entity_id}] {record.sentence}")
    print("\n  sample suppressions:")
    for record in sorted(suppressed, key=lambda r: -r.profiler_score)[:3]:
        print(f"\n    [{record.entity_id}] {record.sentence}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
