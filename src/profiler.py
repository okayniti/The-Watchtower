"""Deliverable 2 — the per-entity behavioural baseline.

Learns what "normal" looks like for each entity and scores every event against it,
without labels. The output is an unsupervised risk score per event plus the per-feature
deviations that produced it.

Why this model
--------------
The estate has ~500 entities and ~180 events each. That number drives the whole design:

* **Not a per-entity one-class model** (isolation forest, one-class SVM, autoencoder).
  Fitting a few hundred parameters per entity on 180 samples overfits, and 500 fitted
  models is 500 things to refit as behaviour drifts.
* **Not a single global model.** "Unusual" is entity-relative here by construction —
  the generator gives every entity its own hours, resources, locations and session
  distribution. A global model learns the population's average and flags the tails of
  the population rather than the tails of each entity.

What this uses instead is an **exponentially-weighted per-entity, per-feature moment
estimator with shrinkage toward a peer-group prior**. Each entity holds a decayed mean
and variance for every feature; each event is scored by how far it sits from that
baseline, in units of that entity's own spread.

The reason to prefer it here is that one mechanism satisfies three of the five hard
requirements at once, rather than bolting each on afterwards:

===================  ==============================================================
``CLAUDE.md`` §4.3   **Concept drift.** Exponential decay means the baseline forgets.
   drift             A gradual legitimate change is absorbed into "normal" within a
                     few half-lives, so a drifted entity stops alerting on its own —
                     no retraining, no manual allow-listing. :attr:`ProfilerConfig`
                     exposes the half-life directly.
``CLAUDE.md`` §4.5   **Cold start.** Shrinkage toward the peer group is a weighted
   cold start       average whose weight is the entity's own effective sample size.
                     With no history the peer prior supplies the baseline entirely;
                     as evidence accumulates the entity's own profile takes over. A
                     brand-new identity is scored against its cohort, which is the
                     right answer and requires no special-casing.
``CLAUDE.md`` §4.4   **Explainability.** The score is an aggregate of per-feature
   explainability   deviations, so "why" is already computed: the contributing
                     features and their magnitudes fall out of scoring. Deliverable 5
                     renders them as sentences; nothing extra needs to be inferred.
===================  ==============================================================

Scoring is **causal and streaming**, matching :mod:`src.features`: an event is scored
against the baseline as it stands *before* that event, and only then folded in. An
attack therefore cannot mask itself by contaminating the baseline it is measured
against — though a slow enough attack can still be absorbed, which is exactly the
tension the decay half-life trades off.

One-sided by default
--------------------
Most features are suspicious only when large: a session that is unusually *short*, or
an address used unusually *rarely*, is not an intrusion signal.
:attr:`~src.features.FeatureSpec.direction` records this per feature and the scorer
clamps one-sided deviations at zero, so an entity is never flagged for being
unremarkable in an unusual way.

Labels are never read here.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .features import FEATURE_NAMES, FEATURE_SPECS, build_feature_matrix, load_events

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Numerical floor so a feature that has never varied cannot divide by zero.
_VAR_FLOOR: float = 1e-4

#: Variance assigned to a freshly-created profile, in transformed units. Deliberately
#: wide: during warm-up the model should under-alert rather than flood the queue.
_VAR_INIT: float = 1.0


@dataclass(slots=True)
class ProfilerConfig:
    """Parameters of the baseline profiler.

    Attributes:
        decay_half_life_days: Time for a past observation's weight to halve. **The
            concept-drift knob.** Short values adapt fast and forgive genuine change
            quickly, but also let a patient attacker normalise their own behaviour;
            long values hold a firmer baseline and keep flagging drifted-but-benign
            entities for longer. 7 days is the default because it lets the generator's
            10-21 day ``insider_drift`` ramps settle within the window while still
            remembering a fortnight of habit.
        min_history: Effective sample size below which an entity is reported as
            cold-start. Scoring degrades smoothly either side of it — this threshold
            only sets the reporting flag, it is not a branch in the maths.
        prior_strength: Weight of the peer-group prior, in pseudo-events. An entity
            needs roughly this much of its own history before its profile outweighs its
            cohort's.
        top_k: How many of the largest per-feature deviations contribute to the score.
            Aggregating over a few rather than all of them keeps the score from being
            diluted by the ~20 features that are quiet during any given attack.
        z_clip: Ceiling on a single feature's deviation, so one saturated binary
            feature cannot dominate the ranking on its own.
    """

    decay_half_life_days: float = 7.0
    min_history: int = 10
    prior_strength: float = 12.0
    top_k: int = 3
    z_clip: float = 8.0

    def __post_init__(self) -> None:
        """Validate the configuration.

        Raises:
            ValueError: If any parameter is out of range.
        """
        if self.decay_half_life_days <= 0:
            raise ValueError("decay_half_life_days must be positive")
        if self.prior_strength < 0:
            raise ValueError("prior_strength must be non-negative")
        if not 1 <= self.top_k <= len(FEATURE_NAMES):
            raise ValueError(f"top_k must be in 1..{len(FEATURE_NAMES)}")
        if self.z_clip <= 0:
            raise ValueError("z_clip must be positive")


@dataclass(slots=True)
class _Moments:
    """Exponentially-weighted mean and variance for one profile.

    Uses West's incremental weighted-variance recurrence, with the accumulated weight
    decayed by elapsed time before each update — so "how much evidence do we have" and
    "how old is it" are the same quantity.

    Attributes:
        weight: Effective sample size, after decay.
        mean: Per-feature decayed mean.
        sq: Per-feature weighted sum of squared deviations.
        last_timestamp: When this profile was last updated.
    """

    weight: float
    mean: np.ndarray
    sq: np.ndarray
    last_timestamp: dt.datetime | None = None

    @classmethod
    def empty(cls, n_features: int) -> "_Moments":
        """Create an unpopulated profile.

        Args:
            n_features: Width of the feature vector.

        Returns:
            A zeroed profile carrying no evidence.
        """
        return cls(
            weight=0.0,
            mean=np.zeros(n_features, dtype=float),
            sq=np.zeros(n_features, dtype=float),
        )

    def decay_factor(self, now: dt.datetime, half_life_days: float) -> float:
        """Return the weight multiplier for the time elapsed since the last update.

        Args:
            now: Current event time.
            half_life_days: Configured half-life.

        Returns:
            A multiplier in ``(0, 1]``.
        """
        if self.last_timestamp is None:
            return 1.0
        gap_days = (now - self.last_timestamp).total_seconds() / 86_400.0
        if gap_days <= 0:
            return 1.0
        return float(0.5 ** (gap_days / half_life_days))

    def effective_weight(self, now: dt.datetime, half_life_days: float) -> float:
        """Return the evidence this profile carries as of ``now``.

        Args:
            now: Current event time.
            half_life_days: Configured half-life.

        Returns:
            The decayed effective sample size.
        """
        return self.weight * self.decay_factor(now, half_life_days)

    def variance(self) -> np.ndarray:
        """Return the per-feature variance estimate, floored away from zero."""
        if self.weight <= 0:
            return np.full_like(self.mean, _VAR_INIT)
        return np.maximum(self.sq / self.weight, _VAR_FLOOR)

    def update(
        self, x: np.ndarray, now: dt.datetime, half_life_days: float
    ) -> None:
        """Fold one observation into the profile.

        Args:
            x: Transformed feature vector.
            now: Event time.
            half_life_days: Configured half-life.
        """
        if self.weight <= 0:
            self.mean = x.copy()
            self.sq = np.full_like(x, _VAR_INIT)
            self.weight = 1.0
            self.last_timestamp = now
            return

        decayed = self.weight * self.decay_factor(now, half_life_days)
        new_weight = decayed + 1.0
        previous_mean = self.mean.copy()
        self.mean = previous_mean + (x - previous_mean) / new_weight
        self.sq = self.sq * (decayed / self.weight) + (x - previous_mean) * (x - self.mean)
        self.weight = new_weight
        self.last_timestamp = now


@dataclass(slots=True)
class ProfilerResult:
    """Output of a scoring pass.

    Attributes:
        scores: One row per event: ``event_id``, ``entity_id``, ``timestamp``,
            ``risk_score``, ``cold_start``, ``history_weight`` and the top contributing
            feature names and magnitudes.
        deviations: The full per-feature deviation matrix, one row per event, aligned
            with :data:`~src.features.FEATURE_NAMES`. Feeds the explainability layer.
    """

    scores: pd.DataFrame
    deviations: pd.DataFrame


class EntityProfiler:
    """Per-entity behavioural baseline with peer-group shrinkage and time decay.

    Attributes:
        config: The profiler configuration.
        entities: Per-entity decayed profiles.
        peers: Per-``entity_type`` cohort profiles, used as the shrinkage prior.
    """

    def __init__(self, config: ProfilerConfig | None = None) -> None:
        """Initialise an empty profiler.

        Args:
            config: Parameters. Defaults to :class:`ProfilerConfig`.
        """
        self.config = config or ProfilerConfig()
        self.entities: dict[str, _Moments] = {}
        self.peers: dict[str, _Moments] = {}

        self._n_features = len(FEATURE_NAMES)
        self._log_mask = np.array(
            [spec.log_transform for spec in FEATURE_SPECS], dtype=bool
        )
        self._one_sided = np.array(
            [spec.direction == "high" for spec in FEATURE_SPECS], dtype=bool
        )

    def transform(self, raw: np.ndarray) -> np.ndarray:
        """Apply the per-feature variance-stabilising transform.

        Heavy-tailed counts and rates are ``log1p``-compressed so that a single extreme
        value — a 13-million km/h geo-velocity, say — does not swamp the mean and
        variance estimates for every subsequent event.

        Args:
            raw: Raw feature vector.

        Returns:
            The transformed vector.
        """
        out = raw.astype(float, copy=True)
        out[self._log_mask] = np.log1p(np.clip(out[self._log_mask], 0.0, None))
        return out

    def score_vector(
        self, entity_id: str, entity_type: str, raw: np.ndarray, now: dt.datetime
    ) -> tuple[float, np.ndarray, float, bool]:
        """Score one event against the current baseline, without updating it.

        Args:
            entity_id: The acting entity.
            entity_type: Its cohort, used for the shrinkage prior.
            raw: Raw feature vector.
            now: Event time.

        Returns:
            A ``(risk_score, deviations, history_weight, is_cold_start)`` tuple, where
            ``deviations`` is the per-feature clipped deviation vector.
        """
        cfg = self.config
        x = self.transform(raw)

        entity = self.entities.get(entity_id)
        peer = self.peers.get(entity_type)

        entity_weight = (
            entity.effective_weight(now, cfg.decay_half_life_days) if entity else 0.0
        )
        peer_weight = cfg.prior_strength if peer is not None else 0.0

        if entity is None and peer is None:
            # The very first event of the very first cohort member: no evidence at all
            # exists yet, so there is nothing to be anomalous relative to.
            return 0.0, np.zeros(self._n_features), 0.0, True

        total = entity_weight + peer_weight
        if total <= 0:
            mean = peer.mean if peer is not None else entity.mean  # type: ignore[union-attr]
            variance = peer.variance() if peer is not None else entity.variance()  # type: ignore[union-attr]
        else:
            entity_mean = entity.mean if entity is not None else 0.0
            entity_var = entity.variance() if entity is not None else 0.0
            peer_mean = peer.mean if peer is not None else 0.0
            peer_var = peer.variance() if peer is not None else 0.0
            mean = (entity_weight * entity_mean + peer_weight * peer_mean) / total
            variance = (entity_weight * entity_var + peer_weight * peer_var) / total

        variance = np.maximum(variance, _VAR_FLOOR)
        z = (x - mean) / np.sqrt(variance)
        # One-sided features contribute only when the value is unusually LARGE.
        z = np.where(self._one_sided, np.maximum(z, 0.0), np.abs(z))
        z = np.clip(z, 0.0, cfg.z_clip)

        top = np.sort(z)[-cfg.top_k :]
        score = float(np.sqrt(np.mean(top**2)))
        return score, z, entity_weight, entity_weight < cfg.min_history

    def update(
        self, entity_id: str, entity_type: str, raw: np.ndarray, now: dt.datetime
    ) -> None:
        """Fold an event into both the entity profile and its cohort prior.

        Args:
            entity_id: The acting entity.
            entity_type: Its cohort.
            raw: Raw feature vector.
            now: Event time.
        """
        x = self.transform(raw)
        entity = self.entities.get(entity_id)
        if entity is None:
            entity = _Moments.empty(self._n_features)
            self.entities[entity_id] = entity
        entity.update(x, now, self.config.decay_half_life_days)

        peer = self.peers.get(entity_type)
        if peer is None:
            peer = _Moments.empty(self._n_features)
            self.peers[entity_type] = peer
        peer.update(x, now, self.config.decay_half_life_days)

    def run(self, features: pd.DataFrame, verbose: bool = False) -> ProfilerResult:
        """Score an entire feature matrix in one causal pass.

        Args:
            features: Output of :func:`~src.features.build_feature_matrix`, sorted by
                timestamp.
            verbose: Print progress every 25,000 events.

        Returns:
            The scores and the per-feature deviation matrix.

        Raises:
            ValueError: If the matrix is not sorted by timestamp.
        """
        frame = features.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        if not frame["timestamp"].is_monotonic_increasing:
            raise ValueError("features must be sorted by timestamp")

        matrix = frame[list(FEATURE_NAMES)].to_numpy(dtype=float)
        entity_ids = frame["entity_id"].to_numpy()
        entity_types = frame["entity_type"].to_numpy()
        timestamps = frame["timestamp"].to_list()
        event_ids = frame["event_id"].to_numpy()

        n = len(frame)
        scores = np.zeros(n)
        weights = np.zeros(n)
        cold = np.zeros(n, dtype=bool)
        deviations = np.zeros((n, self._n_features))

        for i in range(n):
            entity_id = entity_ids[i]
            entity_type = entity_types[i]
            now = timestamps[i]
            raw = matrix[i]

            score, z, weight, is_cold = self.score_vector(entity_id, entity_type, raw, now)
            scores[i] = score
            deviations[i] = z
            weights[i] = weight
            cold[i] = is_cold

            self.update(entity_id, entity_type, raw, now)
            if verbose and (i + 1) % 25_000 == 0:
                print(f"    {i + 1:,} events ...", flush=True)

        order = np.argsort(-deviations, axis=1)[:, : self.config.top_k]
        top_names = [
            "|".join(
                FEATURE_NAMES[j] for j in order[i] if deviations[i, j] > 0.0
            )
            for i in range(n)
        ]

        score_frame = pd.DataFrame(
            {
                "event_id": event_ids,
                "entity_id": entity_ids,
                "entity_type": entity_types,
                "timestamp": frame["timestamp"].to_numpy(),
                "risk_score": scores,
                "history_weight": weights,
                "cold_start": cold,
                "top_features": top_names,
            }
        )
        deviation_frame = pd.DataFrame(deviations, columns=list(FEATURE_NAMES))
        deviation_frame.insert(0, "event_id", event_ids)
        return ProfilerResult(scores=score_frame, deviations=deviation_frame)


def profile_events(
    events: pd.DataFrame | None = None,
    config: ProfilerConfig | None = None,
    verbose: bool = False,
) -> ProfilerResult:
    """Convenience path: load the log, build features, score them.

    Args:
        events: Access log. Loaded from ``data/access_logs.csv`` if omitted.
        config: Profiler parameters.
        verbose: Print progress.

    Returns:
        The scoring result.
    """
    if events is None:
        events = load_events()
    features = build_feature_matrix(events, verbose=verbose)
    return EntityProfiler(config).run(features, verbose=verbose)


def main() -> int:
    """Score the default dataset and print an unlabelled summary.

    No labels are read here — the evaluation harness in ``scripts/eval_profiler.py``
    owns that.

    Returns:
        Process exit code.
    """
    print("The Watchtower — baseline profiler")
    result = profile_events(verbose=True)
    scores = result.scores

    print(f"\n  {len(scores):,} events scored")
    print(f"  cold-start events: {int(scores.cold_start.sum()):,} "
          f"({scores.cold_start.mean():.2%})")
    print("\n  risk_score distribution:")
    for q in (0.5, 0.9, 0.99, 0.999, 1.0):
        print(f"    p{q * 100:<6g} {scores.risk_score.quantile(q):8.3f}")

    out_dir = PROJECT_ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out_dir / "profiler_scores.csv", index=False)
    print(f"\n  scores -> data/processed/profiler_scores.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
