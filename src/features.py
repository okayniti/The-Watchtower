"""Causal feature engineering for The Watchtower.

Turns ``data/access_logs.csv`` into a feature matrix keyed by ``event_id``. Every
feature targets a **documented attack signature** from
:mod:`src.generator.attacks` rather than being a generic summary statistic — the
mapping is recorded in :data:`FEATURE_SPECS` and surfaces later in the explainability
layer, so an analyst can be told *which* behaviour a score came from.

Causality
---------
Every rolling feature is computed from an entity's **past events only**. That is
enforced structurally: this module makes a single chronological pass over the log,
computes each event's features from accumulated state, and only *then* folds the event
into that state. There is no window that can accidentally see forward, because the
future has not been read yet when a row is scored.

That design has a second payoff. The same pass is what a streaming deployment would run
— `update_state` after `extract` is exactly the online loop — so the "near real-time"
requirement in ``CLAUDE.md`` §1 is satisfied by construction rather than by a later
rewrite. The batch entry point :func:`build_feature_matrix` is a thin wrapper over the
streaming core.

Labels
------
This module never reads ``labels.csv``. It takes the feature file alone, which is the
whole point of keeping the two apart (``CLAUDE.md`` §2).

One deliberate outside input
----------------------------
:data:`~src.generator.config.RESOURCES_BY_PATH` supplies a sensitivity tier per
resource. This is *asset inventory*, not ground truth — every real SOC has a CMDB that
grades assets by criticality, and withholding it would model an unrealistically blind
defender. It is derived from the resource path, which is in the log; it is not derived
from any label.
"""

from __future__ import annotations

import datetime as dt
import math
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Iterable, Iterator, Literal

import numpy as np
import pandas as pd

from .generator.config import (
    AUTH_FAIL,
    BULK_READ_COMMANDS,
    RECON_COMMANDS,
    RESOURCES_BY_PATH,
    haversine_km,
    parse_geo,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

Direction = Literal["high", "both"]


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """Declares one feature and how downstream scoring should treat it.

    Attributes:
        name: Column name in the feature matrix.
        targets: The attack class whose documented signature this feature encodes.
        direction: ``"high"`` if only large values are suspicious (most features here),
            ``"both"`` if deviation in either direction matters. The profiler uses this
            to avoid flagging an entity for being *unusually ordinary* — a one-sided
            feature scored two-sided would fire on, say, a suspiciously short session.
        log_transform: Apply ``log1p`` before standardising. Set for heavy-tailed
            counts and rates, where a raw z-score is dominated by the tail.
        description: Analyst-readable phrasing, reused by the explainability layer.
    """

    name: str
    targets: str
    direction: Direction
    log_transform: bool
    description: str


#: Every engineered feature, with the attack signature it targets.
FEATURE_SPECS: Final[tuple[FeatureSpec, ...]] = (
    # --- impossible_travel -----------------------------------------------------------
    FeatureSpec("geo_velocity_kmh", "impossible_travel", "high", True,
                "implied travel speed since this entity's previous event"),
    FeatureSpec("geo_distance_km", "impossible_travel", "high", True,
                "distance from this entity's previous location"),
    FeatureSpec("geo_city_novelty", "impossible_travel", "high", False,
                "how rarely this entity has acted from this city before"),
    # --- brute_force -----------------------------------------------------------------
    FeatureSpec("auth_fail_count", "brute_force", "high", True,
                "failed authentications in this session"),
    FeatureSpec("auth_fail_rate_1h", "brute_force", "high", False,
                "share of this entity's last hour of events that failed to authenticate"),
    FeatureSpec("events_10m", "brute_force", "high", True,
                "events by this entity in the preceding 10 minutes"),
    FeatureSpec("auth_fail_streak", "brute_force", "high", True,
                "consecutive preceding events that failed to authenticate"),
    # --- lateral_movement ------------------------------------------------------------
    FeatureSpec("resource_is_novel", "lateral_movement", "high", False,
                "this entity has never accessed this resource before"),
    FeatureSpec("novel_resources_6h", "lateral_movement", "high", True,
                "distinct never-before-seen resources touched in the last 6 hours"),
    FeatureSpec("resource_rarity", "lateral_movement", "high", False,
                "how rarely this entity has used this resource before"),
    FeatureSpec("recon_command_ratio", "lateral_movement", "high", False,
                "share of the command sequence that is reconnaissance"),
    FeatureSpec("sensitivity_jump", "lateral_movement", "high", False,
                "how far this asset's sensitivity exceeds this entity's usual level"),
    # --- device_spoofing -------------------------------------------------------------
    FeatureSpec("device_is_novel", "device_spoofing", "high", False,
                "this entity has never presented this device fingerprint"),
    FeatureSpec("device_os_family_preserved", "device_spoofing", "high", False,
                "novel device reusing an OS this entity already runs (a credible clone)"),
    FeatureSpec("device_count", "device_spoofing", "high", True,
                "distinct devices this entity has presented so far"),
    # --- credential_stuffing ---------------------------------------------------------
    FeatureSpec("ip_distinct_entities_1h", "credential_stuffing", "high", True,
                "distinct identities seen from this address in the last hour"),
    FeatureSpec("ip_fail_rate_1h", "credential_stuffing", "high", False,
                "authentication failure rate from this address in the last hour"),
    FeatureSpec("ip_is_novel", "credential_stuffing", "high", False,
                "this entity has never egressed from this address before"),
    FeatureSpec("ip_rarity", "credential_stuffing", "high", False,
                "how rarely this entity has used this address before"),
    # --- low_and_slow_exfiltration ---------------------------------------------------
    FeatureSpec("duration_pctile_own", "low_and_slow_exfiltration", "high", False,
                "session length as a percentile of this entity's own history"),
    FeatureSpec("interval_regularity", "low_and_slow_exfiltration", "high", False,
                "how metronomic this entity's recent event timing has become"),
    FeatureSpec("off_hours_score", "low_and_slow_exfiltration", "high", False,
                "how unusual this hour of day is for this entity"),
    FeatureSpec("bulk_read_ratio", "low_and_slow_exfiltration", "high", False,
                "share of the command sequence that is bulk data retrieval"),
    FeatureSpec("session_seconds_24h", "low_and_slow_exfiltration", "high", True,
                "cumulative session time on this resource in the last 24 hours"),
    FeatureSpec("resource_repeat_streak", "low_and_slow_exfiltration", "high", True,
                "consecutive recent sessions against the same resource"),
)

#: Feature column names, in matrix order.
FEATURE_NAMES: Final[tuple[str, ...]] = tuple(spec.name for spec in FEATURE_SPECS)

#: Lookup from feature name to its spec.
SPEC_BY_NAME: Final[dict[str, FeatureSpec]] = {s.name: s for s in FEATURE_SPECS}

#: Context columns carried alongside the features for joining and diagnostics.
CONTEXT_COLUMNS: Final[tuple[str, ...]] = (
    "event_id",
    "entity_id",
    "entity_type",
    "timestamp",
    "entity_event_index",
)

_RECON_SET: Final[frozenset[str]] = frozenset(RECON_COMMANDS)
_BULK_SET: Final[frozenset[str]] = frozenset(BULK_READ_COMMANDS)

#: Rolling window lengths.
_WINDOW_10M: Final[dt.timedelta] = dt.timedelta(minutes=10)
_WINDOW_1H: Final[dt.timedelta] = dt.timedelta(hours=1)
_WINDOW_6H: Final[dt.timedelta] = dt.timedelta(hours=6)
_WINDOW_24H: Final[dt.timedelta] = dt.timedelta(hours=24)

#: Cap on retained per-entity duration history, for the own-percentile feature.
_DURATION_MEMORY: Final[int] = 200

#: Number of recent inter-event intervals used for the regularity statistic.
_INTERVAL_MEMORY: Final[int] = 12


# --------------------------------------------------------------------------------------
# Per-entity streaming state
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class EntityState:
    """Accumulated history for one entity, holding only what past events established.

    Every attribute is updated *after* an event has been scored, which is what makes
    the features causal.

    Attributes:
        n_events: Events seen so far.
        last_timestamp: Time of the previous event.
        last_lat: Latitude of the previous event.
        last_lon: Longitude of the previous event.
        resource_counts: How often each resource has been touched.
        city_counts: How often each city has been acted from.
        ip_counts: How often each address has been used.
        devices: Device fingerprints presented so far.
        os_families: OS strings presented so far, for the clone-detection flag.
        hour_counts: Events per hour-of-day, the entity's own activity profile.
        durations: Recent session durations, for the own-percentile feature.
        intervals: Recent inter-event gaps in seconds, for the regularity statistic.
        recent: ``(timestamp, had_fail, resource, duration)`` inside the longest window.
        novel_resource_times: When each first-time resource access happened.
        fail_streak: Consecutive preceding events that failed to authenticate.
        last_resource: Resource touched by the previous event.
        resource_streak: Consecutive recent events against ``last_resource``.
        sensitivity_sum: Running total of accessed asset sensitivity.
    """

    n_events: int = 0
    last_timestamp: dt.datetime | None = None
    last_lat: float | None = None
    last_lon: float | None = None
    resource_counts: Counter = field(default_factory=Counter)
    city_counts: Counter = field(default_factory=Counter)
    ip_counts: Counter = field(default_factory=Counter)
    devices: set[str] = field(default_factory=set)
    os_families: set[str] = field(default_factory=set)
    hour_counts: np.ndarray = field(default_factory=lambda: np.zeros(24))
    durations: deque = field(default_factory=lambda: deque(maxlen=_DURATION_MEMORY))
    intervals: deque = field(default_factory=lambda: deque(maxlen=_INTERVAL_MEMORY))
    recent: deque = field(default_factory=deque)
    novel_resource_times: deque = field(default_factory=deque)
    fail_streak: int = 0
    last_resource: str | None = None
    resource_streak: int = 0
    sensitivity_sum: float = 0.0

    def prune(self, now: dt.datetime) -> None:
        """Drop state that has fallen outside the longest rolling window.

        Args:
            now: The current event's timestamp.
        """
        cutoff = now - _WINDOW_24H
        while self.recent and self.recent[0][0] < cutoff:
            self.recent.popleft()
        novel_cutoff = now - _WINDOW_6H
        while self.novel_resource_times and self.novel_resource_times[0] < novel_cutoff:
            self.novel_resource_times.popleft()


@dataclass(slots=True)
class OriginState:
    """Cross-entity history for one source address.

    This is the only state that spans entities. It exists because
    ``credential_stuffing``'s signature is structural — many identities behind one
    origin — and is invisible from any single entity's timeline.

    Attributes:
        recent: ``(timestamp, entity_id, had_fail)`` for events inside the 1-hour window.
    """

    recent: deque = field(default_factory=deque)

    def prune(self, now: dt.datetime) -> None:
        """Drop events older than the one-hour window.

        Args:
            now: The current event's timestamp.
        """
        cutoff = now - _WINDOW_1H
        while self.recent and self.recent[0][0] < cutoff:
            self.recent.popleft()


# --------------------------------------------------------------------------------------
# The extractor
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class _ParsedEvent:
    """One log row with its string fields already decoded."""

    event_id: str
    entity_id: str
    entity_type: str
    timestamp: dt.datetime
    source_ip: str
    lat: float
    lon: float
    city: str
    resource: str
    sensitivity: int
    duration: float
    commands: list[str]
    device: str
    os_family: str
    fail_count: int


class FeatureExtractor:
    """Streaming, causal feature extractor.

    Usage is a single chronological pass::

        extractor = FeatureExtractor()
        for row in events_in_time_order:
            features = extractor.process(row)   # scored against history only
                                                # (state updated internally, after)

    Attributes:
        entities: Per-entity accumulated state.
        origins: Per-source-address accumulated state.
    """

    def __init__(self) -> None:
        """Initialise empty state."""
        self.entities: dict[str, EntityState] = {}
        self.origins: dict[str, OriginState] = {}

    # -- parsing ---------------------------------------------------------------------

    @staticmethod
    def _parse(row: dict) -> _ParsedEvent:
        """Decode one raw log row into typed fields.

        Args:
            row: A mapping with the access-log columns.

        Returns:
            The decoded event.
        """
        _city, _country, lat, lon = parse_geo(row["geo_location"])
        commands = str(row["command_sequence"]).split("|") if row["command_sequence"] else []
        device = str(row["device_fingerprint"])
        resource = str(row["resource_accessed"])
        catalog_entry = RESOURCES_BY_PATH.get(resource)
        timestamp = row["timestamp"]
        if not isinstance(timestamp, dt.datetime):
            timestamp = pd.Timestamp(timestamp).to_pydatetime()
        return _ParsedEvent(
            event_id=str(row["event_id"]),
            entity_id=str(row["entity_id"]),
            entity_type=str(row["entity_type"]),
            timestamp=timestamp,
            source_ip=str(row["source_ip"]),
            lat=lat,
            lon=lon,
            city=_city,
            resource=resource,
            # Unknown assets default to mid sensitivity rather than zero, so a resource
            # missing from inventory is not implicitly treated as harmless.
            sensitivity=catalog_entry.sensitivity if catalog_entry else 2,
            duration=float(row["session_duration"]),
            commands=commands,
            device=device,
            os_family=device.split("/")[0].strip(),
            fail_count=sum(1 for c in commands if c == AUTH_FAIL),
        )

    # -- feature computation -----------------------------------------------------------

    def process(self, row: dict) -> dict[str, float]:
        """Compute features for one event, then fold it into state.

        Args:
            row: A mapping with the access-log columns.

        Returns:
            A dict of context fields plus every feature in :data:`FEATURE_NAMES`.
        """
        event = self._parse(row)
        state = self.entities.get(event.entity_id)
        if state is None:
            state = EntityState()
            self.entities[event.entity_id] = state
        origin = self.origins.get(event.source_ip)
        if origin is None:
            origin = OriginState()
            self.origins[event.source_ip] = origin

        state.prune(event.timestamp)
        origin.prune(event.timestamp)

        features = {
            **self._geo_features(event, state),
            **self._auth_features(event, state),
            **self._resource_features(event, state),
            **self._device_features(event, state),
            **self._origin_features(event, state, origin),
            **self._cadence_features(event, state),
        }

        record: dict[str, float] = {
            "event_id": event.event_id,
            "entity_id": event.entity_id,
            "entity_type": event.entity_type,
            "timestamp": event.timestamp,
            "entity_event_index": state.n_events,  # history depth BEFORE this event
            **features,
        }
        self._update(event, state, origin)
        return record

    def _geo_features(self, event: _ParsedEvent, state: EntityState) -> dict[str, float]:
        """Geo-velocity family — targets ``impossible_travel``.

        Args:
            event: The current event.
            state: The entity's accumulated history.

        Returns:
            The geo feature values.
        """
        distance = 0.0
        velocity = 0.0
        if state.last_timestamp is not None and state.last_lat is not None:
            distance = haversine_km(state.last_lat, state.last_lon, event.lat, event.lon)
            gap_hours = (event.timestamp - state.last_timestamp).total_seconds() / 3600.0
            if gap_hours > 0 and distance > 1.0:
                velocity = distance / gap_hours

        seen = state.city_counts.get(event.city, 0)
        # 1.0 for a city never used before, falling toward 0 as it becomes routine.
        novelty = 1.0 - (seen / state.n_events if state.n_events else 0.0)
        return {
            "geo_velocity_kmh": velocity,
            "geo_distance_km": distance,
            "geo_city_novelty": novelty,
        }

    def _auth_features(self, event: _ParsedEvent, state: EntityState) -> dict[str, float]:
        """Failed-authentication family — targets ``brute_force``.

        Args:
            event: The current event.
            state: The entity's accumulated history.

        Returns:
            The auth feature values.
        """
        cutoff_1h = event.timestamp - _WINDOW_1H
        cutoff_10m = event.timestamp - _WINDOW_10M
        window_1h = [r for r in state.recent if r[0] >= cutoff_1h]
        fails_1h = sum(1 for r in window_1h if r[1])
        events_10m = sum(1 for r in state.recent if r[0] >= cutoff_10m)
        return {
            "auth_fail_count": float(event.fail_count),
            "auth_fail_rate_1h": fails_1h / len(window_1h) if window_1h else 0.0,
            "events_10m": float(events_10m),
            "auth_fail_streak": float(state.fail_streak),
        }

    def _resource_features(
        self, event: _ParsedEvent, state: EntityState
    ) -> dict[str, float]:
        """Resource-novelty family — targets ``lateral_movement``.

        Args:
            event: The current event.
            state: The entity's accumulated history.

        Returns:
            The resource feature values.
        """
        seen = state.resource_counts.get(event.resource, 0)
        is_novel = 1.0 if seen == 0 else 0.0
        rarity = 1.0 - (seen / state.n_events if state.n_events else 0.0)
        recon = sum(1 for c in event.commands if c in _RECON_SET)

        mean_sensitivity = (
            state.sensitivity_sum / state.n_events if state.n_events else float(event.sensitivity)
        )
        return {
            "resource_is_novel": is_novel,
            "novel_resources_6h": float(len(state.novel_resource_times)),
            "resource_rarity": rarity,
            "recon_command_ratio": recon / len(event.commands) if event.commands else 0.0,
            "sensitivity_jump": max(0.0, float(event.sensitivity) - mean_sensitivity),
        }

    def _device_features(
        self, event: _ParsedEvent, state: EntityState
    ) -> dict[str, float]:
        """Device-mismatch family — targets ``device_spoofing``.

        The ``device_os_family_preserved`` flag is the discriminating one: a credible
        spoof clones the OS and changes only firmware and MAC, whereas a genuine
        hardware refresh often moves platform too.

        Args:
            event: The current event.
            state: The entity's accumulated history.

        Returns:
            The device feature values.
        """
        is_novel = 0.0 if event.device in state.devices else 1.0
        preserved = (
            1.0 if is_novel and event.os_family in state.os_families else 0.0
        )
        return {
            "device_is_novel": is_novel,
            "device_os_family_preserved": preserved,
            "device_count": float(len(state.devices)),
        }

    def _origin_features(
        self, event: _ParsedEvent, state: EntityState, origin: OriginState
    ) -> dict[str, float]:
        """Shared-origin family — targets ``credential_stuffing``.

        Args:
            event: The current event.
            state: The entity's accumulated history.
            origin: The address's cross-entity history.

        Returns:
            The origin feature values.
        """
        distinct = len({entity_id for _ts, entity_id, _fail in origin.recent})
        fails = sum(1 for _ts, _eid, fail in origin.recent if fail)
        seen = state.ip_counts.get(event.source_ip, 0)
        return {
            "ip_distinct_entities_1h": float(distinct),
            "ip_fail_rate_1h": fails / len(origin.recent) if origin.recent else 0.0,
            "ip_is_novel": 1.0 if seen == 0 else 0.0,
            "ip_rarity": 1.0 - (seen / state.n_events if state.n_events else 0.0),
        }

    def _cadence_features(
        self, event: _ParsedEvent, state: EntityState
    ) -> dict[str, float]:
        """Cadence and volume family — targets ``low_and_slow_exfiltration``.

        ``duration_pctile_own`` is measured against the entity's *own* history rather
        than a global threshold. That is the whole point: the generator draws
        exfiltration sessions from the 70th-95th percentile of the victim's own
        distribution, so a global cut-off cannot separate them (measured at ~1%
        precision), while a per-entity percentile can at least see them.

        ``interval_regularity`` captures the metronomic cadence of a scheduled drain.
        It is inverted coefficient-of-variation, so a perfectly regular series scores 1
        and bursty human activity scores near 0.

        Args:
            event: The current event.
            state: The entity's accumulated history.

        Returns:
            The cadence feature values.
        """
        if state.durations:
            history = np.fromiter(state.durations, dtype=float)
            pctile = float((history < event.duration).mean())
        else:
            pctile = 0.5  # no opinion yet; the peer prior will carry this entity

        regularity = 0.0
        if len(state.intervals) >= 4:
            intervals = np.fromiter(state.intervals, dtype=float)
            mean_gap = float(intervals.mean())
            if mean_gap > 0:
                cv = float(intervals.std()) / mean_gap
                regularity = 1.0 / (1.0 + cv)

        total_hours = state.hour_counts.sum()
        if total_hours > 0:
            share = state.hour_counts[event.timestamp.hour] / total_hours
            # Compare against a flat 1/24 baseline so "unusual hour" is entity-relative.
            off_hours = float(max(0.0, 1.0 - share * 24.0))
        else:
            off_hours = 0.0

        bulk = sum(1 for c in event.commands if c in _BULK_SET)
        cutoff = event.timestamp - _WINDOW_24H
        seconds_24h = sum(
            r[3] for r in state.recent if r[0] >= cutoff and r[2] == event.resource
        )
        return {
            "duration_pctile_own": pctile,
            "interval_regularity": regularity,
            "off_hours_score": off_hours,
            "bulk_read_ratio": bulk / len(event.commands) if event.commands else 0.0,
            "session_seconds_24h": float(seconds_24h),
            "resource_repeat_streak": float(
                state.resource_streak if state.last_resource == event.resource else 0
            ),
        }

    # -- state update -------------------------------------------------------------------

    def _update(
        self, event: _ParsedEvent, state: EntityState, origin: OriginState
    ) -> None:
        """Fold the scored event into state. Called only after features are computed.

        Args:
            event: The current event.
            state: The entity's accumulated history.
            origin: The address's cross-entity history.
        """
        had_fail = event.fail_count > 0
        if state.resource_counts.get(event.resource, 0) == 0:
            state.novel_resource_times.append(event.timestamp)

        if state.last_timestamp is not None:
            gap = (event.timestamp - state.last_timestamp).total_seconds()
            if gap > 0:
                state.intervals.append(gap)

        state.resource_streak = (
            state.resource_streak + 1 if state.last_resource == event.resource else 1
        )
        state.last_resource = event.resource
        state.fail_streak = state.fail_streak + 1 if had_fail else 0

        state.resource_counts[event.resource] += 1
        state.city_counts[event.city] += 1
        state.ip_counts[event.source_ip] += 1
        state.devices.add(event.device)
        state.os_families.add(event.os_family)
        state.hour_counts[event.timestamp.hour] += 1
        state.durations.append(event.duration)
        state.recent.append(
            (event.timestamp, had_fail, event.resource, event.duration)
        )
        state.sensitivity_sum += event.sensitivity
        state.last_timestamp = event.timestamp
        state.last_lat = event.lat
        state.last_lon = event.lon
        state.n_events += 1

        origin.recent.append((event.timestamp, event.entity_id, had_fail))


# --------------------------------------------------------------------------------------
# Batch entry points
# --------------------------------------------------------------------------------------


def stream_features(events: Iterable[dict]) -> Iterator[dict[str, float]]:
    """Yield feature records for a chronologically ordered event stream.

    Args:
        events: Log rows in ascending timestamp order.

    Yields:
        One feature record per event.
    """
    extractor = FeatureExtractor()
    for row in events:
        yield extractor.process(row)


def build_feature_matrix(events: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Build the full feature matrix from an access log.

    Args:
        events: The access log. Must contain the feature columns; ``label`` is neither
            required nor read.
        verbose: Print progress every 25,000 events.

    Returns:
        One row per event, indexed positionally, with :data:`CONTEXT_COLUMNS` followed
        by :data:`FEATURE_NAMES`.

    Raises:
        ValueError: If the frame is not sorted by timestamp, which would silently break
            the causality guarantee.
    """
    frame = events.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    if not frame["timestamp"].is_monotonic_increasing:
        raise ValueError(
            "events must be sorted by timestamp — causal features depend on it"
        )

    extractor = FeatureExtractor()
    records: list[dict] = []
    for position, row in enumerate(frame.to_dict("records"), start=1):
        records.append(extractor.process(row))
        if verbose and position % 25_000 == 0:
            print(f"    {position:,} events ...", flush=True)

    matrix = pd.DataFrame.from_records(records)
    return matrix[list(CONTEXT_COLUMNS) + list(FEATURE_NAMES)]


def load_events(path: Path | str | None = None) -> pd.DataFrame:
    """Load the access log, features only.

    Args:
        path: Path to ``access_logs.csv``. Defaults to ``data/access_logs.csv``.

    Returns:
        The log sorted by timestamp.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file carries a ``label`` column, which would mean ground
            truth had leaked into the feature file.
    """
    resolved = Path(path) if path else PROJECT_ROOT / "data" / "access_logs.csv"
    if not resolved.exists():
        raise FileNotFoundError(
            f"{resolved} not found — run `python -m src.data_generator` first"
        )
    frame = pd.read_csv(resolved)
    if "label" in frame.columns:
        raise ValueError(
            f"{resolved} contains a 'label' column. The feature file must not carry "
            "ground truth (CLAUDE.md §2); regenerate without --include-label-in-logs."
        )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame.sort_values(["timestamp", "entity_id"]).reset_index(drop=True)


def main() -> int:
    """Build and cache the feature matrix from the default paths.

    Returns:
        Process exit code.
    """
    print("The Watchtower — feature extraction")
    events = load_events()
    print(f"  {len(events):,} events, {events.entity_id.nunique()} entities")
    matrix = build_feature_matrix(events, verbose=True)

    out_dir = PROJECT_ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "features.parquet"
    try:
        matrix.to_parquet(out_path, index=False)
    except (ImportError, ValueError):  # no parquet engine installed
        out_path = out_dir / "features.csv"
        matrix.to_csv(out_path, index=False)

    print(f"  {len(FEATURE_NAMES)} features -> {out_path.relative_to(PROJECT_ROOT)}")
    described = matrix[list(FEATURE_NAMES)].describe().T[["mean", "50%", "max"]]
    print(described.to_string(float_format=lambda v: f"{v:,.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
