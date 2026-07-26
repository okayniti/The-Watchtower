"""Deliverable 1 — orchestration and CLI for The Watchtower synthetic data generator.

Run end to end with::

    python -m src.data_generator

which regenerates, deterministically for a fixed seed:

======================================  ==================================================
``data/access_logs.csv``                10 feature columns + ``event_id``. **No label.**
``data/labels.csv``                     ``event_id``, ``label``, ``episode_id``.
``data/entity_profiles.json``           Generator-side ground truth. Never a model input.
``reports/data_summary.md`` / ``.json`` The run summary, also printed to stdout.
======================================  ==================================================

Why the label lives in its own file
-----------------------------------
``CLAUDE.md`` §2 requires that no component read ``label`` outside training and
evaluation, and that inference paths run on a label-free frame. Enforcing that with a
convention ("remember to drop the column") fails the first time somebody writes
``df.drop(columns=['label'])`` in the wrong place. Splitting the files makes the leak
structurally impossible: the feature file simply does not contain ground truth, and the
two are rejoined on ``event_id`` only inside training and evaluation code.

The logical schema is still the full 11 fields — it is materialised as a join, not as one
table. Set :attr:`~src.generator.config.GeneratorConfig.include_label_in_logs` to ``True``
if a single denormalised file is genuinely wanted for inspection.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

from .generator.attacks import INJECTORS, Episode, InjectionContext
from .generator.config import (
    AMBIGUOUS_LABEL,
    BEHAVIOURS,
    BENIGN_LABEL,
    ENTITY_TYPES,
    FEATURE_COLUMNS,
    HARD_ANOMALIES,
    LABEL_COLUMNS,
    AccessEvent,
    GeneratorConfig,
)
from .generator.entities import EntityRoster, build_entities
from .generator.profiles import (
    BehaviouralProfile,
    build_profiles,
    generate_normal_events,
)

#: Repository root, resolved from this file's location so the CLI works from any cwd.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(slots=True)
class GenerationOutput:
    """Everything one generator run produced.

    Attributes:
        events: Every event, sorted by timestamp, with ``event_id`` assigned.
        roster: The entity roster used.
        profiles: Behavioural ground truth, keyed by ``entity_id``.
        episodes: Injected campaign records.
        config: The configuration the run used.
        summary: The computed run summary (see :func:`summarise`).
    """

    events: list[AccessEvent]
    roster: EntityRoster
    profiles: dict[str, BehaviouralProfile]
    episodes: list[Episode]
    config: GeneratorConfig
    summary: dict[str, object]


# --------------------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------------------


def generate(cfg: GeneratorConfig, verbose: bool = True) -> GenerationOutput:
    """Run the full simulation: roster, profiles, benign traffic, then injection.

    All randomness derives from :attr:`GeneratorConfig.seed`, and every stage consumes
    the generator in a fixed order, so the output is byte-reproducible.

    Args:
        cfg: The run configuration.
        verbose: If ``True``, print stage progress to stdout.

    Returns:
        The populated :class:`GenerationOutput`.
    """
    rng = np.random.default_rng(cfg.seed)
    Faker.seed(cfg.seed)
    faker = Faker("en_US")

    def log(message: str) -> None:
        if verbose:
            print(f"  {message}", flush=True)

    log(f"building roster of {cfg.n_entities} entities ...")
    roster = build_entities(cfg, rng, faker)

    log("building behavioural profiles ...")
    profiles = build_profiles(roster, cfg, rng)

    log(f"generating benign traffic over {cfg.days} days ...")
    normal_events = generate_normal_events(roster, profiles, cfg, rng, faker)
    log(f"    {len(normal_events):,} benign events")

    events_by_entity: dict[str, list[AccessEvent]] = {}
    for event in normal_events:
        events_by_entity.setdefault(event.entity_id, []).append(event)
    for history in events_by_entity.values():
        history.sort(key=lambda e: e.timestamp)

    window_start = dt.datetime.combine(
        dt.date.fromisoformat(cfg.start_date), dt.time.min
    )
    window_end = window_start + dt.timedelta(days=cfg.days) - dt.timedelta(seconds=1)

    ctx = InjectionContext(
        roster=roster,
        profiles=profiles,
        events_by_entity=events_by_entity,
        cfg=cfg,
        rng=rng,
        faker=faker,
        window_start=window_start,
        window_end=window_end,
    )

    # Budgets are shares of the *final* total, so solve for that total up front:
    #   total = normal / (1 - share_of_injected_new_events)
    # insider_drift rewrites rather than adds, so it does not grow the denominator.
    additive_share = sum(cfg.attack_share.get(name, 0.0) for name in HARD_ANOMALIES)
    estimated_total = len(normal_events) / max(1.0 - additive_share, 1e-6)

    all_events = list(normal_events)
    episodes: list[Episode] = []
    for name, injector in INJECTORS.items():
        share = cfg.attack_share.get(name, 0.0)
        budget = int(round(share * estimated_total))
        if budget <= 0:
            continue
        log(f"injecting {name} (budget {budget:,} events) ...")
        result = injector(ctx, budget)
        all_events.extend(result.events)
        episodes.extend(result.episodes)
        produced = sum(ep.n_events for ep in result.episodes)
        log(f"    {produced:,} events across {len(result.episodes)} episode(s)")

    # Truncate to whole seconds BEFORE sorting. Injectors compute offsets in fractional
    # minutes and hours, so their events would otherwise carry microseconds while benign
    # events land on exact seconds — a perfect label leak that would let any model score
    # ~100% off a timestamp artefact. Real SIEM logs are second-resolution anyway.
    for event in all_events:
        event.timestamp = event.timestamp.replace(microsecond=0)

    # Strict chronological order — this is time-series data (CLAUDE.md §4.1).
    log("sorting and assigning event ids ...")
    all_events.sort(key=lambda e: (e.timestamp, e.entity_id))
    width = max(6, len(str(len(all_events))))
    for index, event in enumerate(all_events, start=1):
        event.event_id = f"evt-{index:0{width}d}"

    summary = summarise(all_events, roster, episodes, cfg)
    return GenerationOutput(
        events=all_events,
        roster=roster,
        profiles=profiles,
        episodes=episodes,
        config=cfg,
        summary=summary,
    )


# --------------------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------------------


def summarise(
    events: list[AccessEvent],
    roster: EntityRoster,
    episodes: list[Episode],
    cfg: GeneratorConfig,
) -> dict[str, object]:
    """Compute the run summary: class balance, per-entity volume, cold-start cases.

    Args:
        events: Every generated event, already sorted and id-assigned.
        roster: The entity roster.
        episodes: Injected campaign records.
        cfg: The run configuration.

    Returns:
        A JSON-serialisable summary dict. ``checks`` holds the pass/fail assertions that
        matter for the spec — most importantly that the hard-anomaly rate landed inside
        the 0.5-3% band from ``CLAUDE.md`` §4.2.
    """
    total = len(events)
    counts = {label: 0 for label in BEHAVIOURS}
    per_entity: dict[str, int] = {}
    benign_per_entity: dict[str, int] = {}
    for event in events:
        counts[event.label] = counts.get(event.label, 0) + 1
        per_entity[event.entity_id] = per_entity.get(event.entity_id, 0) + 1
        if event.label == BENIGN_LABEL:
            benign_per_entity[event.entity_id] = (
                benign_per_entity.get(event.entity_id, 0) + 1
            )

    hard_count = sum(counts[label] for label in HARD_ANOMALIES)
    ambiguous_count = counts[AMBIGUOUS_LABEL]
    hard_rate = hard_count / total if total else 0.0

    episodes_by_label: dict[str, dict[str, float]] = {}
    for label in (*HARD_ANOMALIES, AMBIGUOUS_LABEL):
        matching = [ep for ep in episodes if ep.label == label]
        episodes_by_label[label] = {
            "episodes": len(matching),
            "events": sum(ep.n_events for ep in matching),
            "mean_events_per_episode": (
                round(statistics.fmean(ep.n_events for ep in matching), 1)
                if matching
                else 0.0
            ),
            "mean_span_days": (
                round(statistics.fmean(ep.span_days for ep in matching), 2)
                if matching
                else 0.0
            ),
            "entities_touched": len({eid for ep in matching for eid in ep.entity_ids}),
        }

    cold = roster.cold_start()
    # Report benign history separately from total volume: a cold-start entity can pick
    # up extra events by being *targeted* (credential-stuffing campaigns sweep the whole
    # estate, new hires included). Those are the hardest cold-start cases of all — an
    # attack on an identity with almost no baseline — so they are kept, and sparsity is
    # asserted against benign history, which is what "no history" actually means.
    cold_records = sorted(
        (
            {
                "entity_id": e.entity_id,
                "entity_type": e.entity_type,
                "events": per_entity.get(e.entity_id, 0),
                "benign_events": benign_per_entity.get(e.entity_id, 0),
                "targeted": per_entity.get(e.entity_id, 0)
                > benign_per_entity.get(e.entity_id, 0),
            }
            for e in cold
        ),
        key=lambda r: (r["benign_events"], r["entity_id"]),
    )

    volumes = sorted(per_entity.values())
    established_volumes = sorted(
        per_entity.get(e.entity_id, 0) for e in roster.established()
    )

    timestamps = [e.timestamp for e in events]
    return {
        "config": dataclasses.asdict(cfg),
        "total_events": total,
        "window": {
            "start": timestamps[0].isoformat() if timestamps else None,
            "end": timestamps[-1].isoformat() if timestamps else None,
            "days": cfg.days,
        },
        "entities": {
            "total": len(roster.entities),
            "by_type": {
                et: len(roster.of_type(et)) for et in ENTITY_TYPES
            },
            "active": len(per_entity),
            "cold_start": len(cold),
        },
        "class_balance": {
            label: {
                "count": counts[label],
                "pct_of_total": round(100.0 * counts[label] / total, 4) if total else 0.0,
            }
            for label in BEHAVIOURS
        },
        "anomaly_rates": {
            "hard_anomaly_events": hard_count,
            "hard_anomaly_pct": round(100.0 * hard_rate, 4),
            "ambiguous_events": ambiguous_count,
            "ambiguous_pct": round(100.0 * ambiguous_count / total, 4) if total else 0.0,
            "benign_pct": round(100.0 * counts[BENIGN_LABEL] / total, 4) if total else 0.0,
        },
        "episodes": episodes_by_label,
        "per_entity_events": {
            "min": volumes[0] if volumes else 0,
            "p25": int(np.percentile(volumes, 25)) if volumes else 0,
            "median": int(np.median(volumes)) if volumes else 0,
            "p75": int(np.percentile(volumes, 75)) if volumes else 0,
            "max": volumes[-1] if volumes else 0,
            "mean": round(statistics.fmean(volumes), 1) if volumes else 0.0,
            "established_min": established_volumes[0] if established_volumes else 0,
            "established_median": (
                int(np.median(established_volumes)) if established_volumes else 0
            ),
        },
        "cold_start_entities": cold_records,
        "checks": {
            "hard_anomaly_rate_in_0.5_3_pct": 0.005 <= hard_rate <= 0.03,
            "events_sorted_by_timestamp": all(
                events[i].timestamp <= events[i + 1].timestamp
                for i in range(len(events) - 1)
            ),
            "event_ids_unique": len({e.event_id for e in events}) == total,
            # Guards against a formatting artefact separating injected from benign
            # events. See the truncation step in generate().
            "timestamps_second_resolution": all(
                e.timestamp.microsecond == 0 for e in events
            ),
            "all_behaviours_present": all(counts[label] > 0 for label in BEHAVIOURS),
            "cold_start_entities_are_sparse": all(
                r["benign_events"] <= cfg.cold_start_max_events for r in cold_records
            ),
            "low_and_slow_spans_multiple_days": episodes_by_label[
                "low_and_slow_exfiltration"
            ]["mean_span_days"]
            > 1.0,
            "insider_drift_is_gradual": episodes_by_label[AMBIGUOUS_LABEL][
                "mean_span_days"
            ]
            > 5.0,
        },
    }


def format_summary(summary: dict) -> str:
    """Render the run summary as a Markdown document.

    The same text is printed to stdout and written to ``reports/data_summary.md``, so
    the console output and the report asset never diverge.

    Args:
        summary: The dict returned by :func:`summarise`.

    Returns:
        A Markdown string.
    """
    total = summary["total_events"]
    lines: list[str] = []
    add = lines.append

    add("# The Watchtower — synthetic dataset summary")
    add("")
    add(f"- **Total events:** {total:,}")
    add(f"- **Window:** {summary['window']['start']} to {summary['window']['end']} "
        f"({summary['window']['days']} days)")
    ent = summary["entities"]
    add(f"- **Entities:** {ent['total']} "
        f"({', '.join(f'{v} {k}' for k, v in ent['by_type'].items())})")
    add(f"- **Seed:** {summary['config']['seed']} (run is fully reproducible)")
    add("")

    add("## Class balance")
    add("")
    add("| Behaviour | Class | Events | % of total |")
    add("|---|---|---:|---:|")
    for label in BEHAVIOURS:
        row = summary["class_balance"][label]
        if label == BENIGN_LABEL:
            kind = "benign"
        elif label == AMBIGUOUS_LABEL:
            kind = "**ambiguous**"
        else:
            kind = "anomaly"
        add(f"| `{label}` | {kind} | {row['count']:,} | {row['pct_of_total']:.3f}% |")
    add("")

    rates = summary["anomaly_rates"]
    add(f"**Hard-anomaly rate: {rates['hard_anomaly_pct']:.3f}%** "
        f"({rates['hard_anomaly_events']:,} events across the six attack classes) — "
        "target band 0.5%-3% per CLAUDE.md §4.2.")
    add("")
    add(f"`insider_drift` contributes a further {rates['ambiguous_pct']:.3f}% "
        f"({rates['ambiguous_events']:,} events) and is excluded from the anomaly rate: "
        "it is benign-but-attack-shaped, and counting it would misstate the imbalance "
        "the detector actually faces.")
    add("")

    add("## Injected episodes")
    add("")
    add("| Behaviour | Episodes | Events | Mean events/episode | Mean span (days) | Entities |")
    add("|---|---:|---:|---:|---:|---:|")
    for label, row in summary["episodes"].items():
        add(f"| `{label}` | {row['episodes']} | {row['events']:,} | "
            f"{row['mean_events_per_episode']} | {row['mean_span_days']} | "
            f"{row['entities_touched']} |")
    add("")

    vol = summary["per_entity_events"]
    add("## Per-entity event volume")
    add("")
    add("| Statistic | All entities | Established only |")
    add("|---|---:|---:|")
    add(f"| min | {vol['min']} | {vol['established_min']} |")
    add(f"| median | {vol['median']} | {vol['established_median']} |")
    add(f"| p25 / p75 | {vol['p25']} / {vol['p75']} | — |")
    add(f"| max | {vol['max']} | — |")
    add(f"| mean | {vol['mean']} | — |")
    add("")

    cold = summary["cold_start_entities"]
    add(f"## Cold-start entities ({len(cold)})")
    add("")
    add("Held back deliberately: these identities first appear only in the final "
        f"{summary['config']['cold_start_window_days']} days of the window and emit a "
        "handful of events each. They are the test cases for CLAUDE.md §4.5 — the "
        "detector must score them without meaningful history.")
    add("")
    targeted = [r for r in cold if r["targeted"]]
    if targeted:
        add(f"{len(targeted)} of them are also *targeted* by an injected campaign — an "
            "attack against an identity with almost no baseline is the hardest cold-start "
            "case in the dataset, so these are kept rather than excluded.")
        add("")
    add("| entity_id | type | benign history | total events | targeted |")
    add("|---|---|---:|---:|---|")
    for record in cold[:15]:
        add(f"| `{record['entity_id']}` | {record['entity_type']} | "
            f"{record['benign_events']} | {record['events']} | "
            f"{'yes' if record['targeted'] else '—'} |")
    if len(cold) > 15:
        add(f"| … {len(cold) - 15} more | | | | |")
    add("")

    add("## Checks")
    add("")
    add("| Check | Result |")
    add("|---|---|")
    for name, passed in summary["checks"].items():
        add(f"| {name.replace('_', ' ')} | {'PASS' if passed else 'FAIL'} |")
    add("")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------


def to_frames(
    output: GenerationOutput,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the event list into the feature frame and the label frame.

    Args:
        output: A completed generation run.

    Returns:
        A ``(features, labels)`` tuple. ``features`` carries no ground truth unless
        :attr:`GeneratorConfig.include_label_in_logs` was enabled.
    """
    feature_rows = []
    label_rows = []
    for event in output.events:
        feature_rows.append(
            {
                "event_id": event.event_id,
                "entity_id": event.entity_id,
                "entity_type": event.entity_type,
                "timestamp": event.timestamp.isoformat(sep=" "),
                "source_ip": event.source_ip,
                "geo_location": event.geo_location,
                "resource_accessed": event.resource_accessed,
                "auth_method": event.auth_method,
                "session_duration": event.session_duration,
                "command_sequence": event.command_string(),
                "device_fingerprint": event.device_fingerprint,
            }
        )
        label_rows.append(
            {
                "event_id": event.event_id,
                "label": event.label,
                "episode_id": event.episode_id,
            }
        )

    features = pd.DataFrame(feature_rows, columns=list(FEATURE_COLUMNS))
    labels = pd.DataFrame(label_rows, columns=list(LABEL_COLUMNS))
    if output.config.include_label_in_logs:
        features["label"] = labels["label"].to_numpy()
    return features, labels


def build_sample(output: GenerationOutput) -> list[str]:
    """Select the ``event_id`` values for the committed showcase sample.

    The full dataset is git-ignored — it is 21 MB and regenerates byte-for-byte from the
    seed — so a small sample is tracked instead to make the schema and the shape of the
    data visible directly on the repository page.

    A chronological head would not do the job: the first few thousand events are all
    day-one traffic and contain almost none of the attack classes. Instead this takes
    **whole episodes** from every behaviour (so multi-event structure such as a
    brute-force burst or a multi-day exfiltration cadence survives intact) and fills the
    remainder with benign events strided evenly across the whole window.

    The sample is therefore **stratified, not proportional** — its class balance is
    deliberately unrepresentative and must never be used to compute the imbalance
    figures. ``reports/data_summary.md`` carries the real numbers.

    Args:
        output: A completed generation run.

    Returns:
        The selected ``event_id`` values, ordered by timestamp.
    """
    cfg = output.config
    by_episode: dict[str, list[AccessEvent]] = {}
    benign: list[AccessEvent] = []
    for event in output.events:
        if event.episode_id:
            by_episode.setdefault(event.episode_id, []).append(event)
        if event.label == BENIGN_LABEL:
            benign.append(event)

    selected: list[AccessEvent] = []
    for label in BEHAVIOURS:
        if label == BENIGN_LABEL:
            continue
        taken = 0
        # Episodes are keyed with a per-class prefix and ascending index, so iterating
        # sorted keys picks the earliest episodes first and stays deterministic.
        for episode_id in sorted(by_episode):
            if taken >= cfg.sample_min_per_class:
                break
            events = by_episode[episode_id]
            if not events or events[0].label != label:
                continue
            # Cap any single episode so one large campaign cannot dominate the sample.
            chunk = events[: max(cfg.sample_min_per_class, 25)]
            selected.extend(chunk)
            taken += len(chunk)

    remaining = max(cfg.sample_rows - len(selected), 0)
    if remaining and benign:
        stride = max(len(benign) // remaining, 1)
        selected.extend(benign[::stride][:remaining])

    selected.sort(key=lambda e: (e.timestamp, e.entity_id))
    return [e.event_id for e in selected]


def write_outputs(output: GenerationOutput, root: Path = PROJECT_ROOT) -> dict[str, Path]:
    """Write every artefact of a run to disk.

    Args:
        output: A completed generation run.
        root: Repository root. Output directories are resolved relative to it.

    Returns:
        Mapping of artefact name to the path written.
    """
    data_dir = root / output.config.output_dir
    reports_dir = root / output.config.reports_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    features, labels = to_frames(output)
    paths = {
        "access_logs": data_dir / "access_logs.csv",
        "labels": data_dir / "labels.csv",
        "sample_access_logs": data_dir / "sample_access_logs.csv",
        "sample_labels": data_dir / "sample_labels.csv",
        "entity_profiles": data_dir / "entity_profiles.json",
        "summary_md": reports_dir / "data_summary.md",
        "summary_json": reports_dir / "data_summary.json",
        "episodes": reports_dir / "injected_episodes.csv",
    }

    features.to_csv(paths["access_logs"], index=False)
    labels.to_csv(paths["labels"], index=False)

    # Tracked showcase sample — see build_sample() for why it is stratified.
    sample_ids = build_sample(output)
    keep = features.event_id.isin(sample_ids)
    features[keep].to_csv(paths["sample_access_logs"], index=False)
    labels[labels.event_id.isin(sample_ids)].to_csv(paths["sample_labels"], index=False)

    profile_payload = {
        "_warning": (
            "Generator-side ground truth for analysis and reporting only. This file is "
            "NOT a model input — the profiler must recover these patterns from "
            "access_logs.csv alone."
        ),
        "profiles": [p.to_record() for p in output.profiles.values()],
    }
    paths["entity_profiles"].write_text(
        json.dumps(profile_payload, indent=2), encoding="utf-8"
    )

    pd.DataFrame(
        [
            {
                "episode_id": ep.episode_id,
                "label": ep.label,
                "n_events": ep.n_events,
                "n_entities": len(ep.entity_ids),
                "entity_ids": ";".join(ep.entity_ids[:8]),
                "start": ep.start.isoformat(sep=" "),
                "end": ep.end.isoformat(sep=" "),
                "span_days": ep.span_days,
                "note": ep.note,
            }
            for ep in output.episodes
        ]
    ).to_csv(paths["episodes"], index=False)

    paths["summary_json"].write_text(
        json.dumps(output.summary, indent=2, default=str), encoding="utf-8"
    )
    paths["summary_md"].write_text(format_summary(output.summary), encoding="utf-8")
    return paths


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the command-line interface.

    Returns:
        The configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.data_generator",
        description="Generate The Watchtower synthetic access-log dataset.",
    )
    defaults = GeneratorConfig()
    parser.add_argument("--entities", type=int, default=defaults.n_entities,
                        help="number of entities in the roster")
    parser.add_argument("--days", type=int, default=defaults.days,
                        help="length of the simulation window in days")
    parser.add_argument("--start-date", type=str, default=defaults.start_date,
                        help="first simulated day, YYYY-MM-DD")
    parser.add_argument("--seed", type=int, default=defaults.seed,
                        help="master RNG seed; fixes the entire output")
    parser.add_argument("--output-dir", type=str, default=defaults.output_dir,
                        help="directory for access_logs.csv and labels.csv")
    parser.add_argument("--reports-dir", type=str, default=defaults.reports_dir,
                        help="directory for the run summary")
    parser.add_argument("--include-label-in-logs", action="store_true",
                        help="also write the label column into access_logs.csv "
                             "(off by default so the feature file cannot leak truth)")
    parser.add_argument("--quiet", action="store_true", help="suppress progress output")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m src.data_generator``.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        Process exit code: ``0`` on success, ``1`` if any summary check failed.
    """
    args = build_arg_parser().parse_args(argv)
    cfg = GeneratorConfig(
        n_entities=args.entities,
        days=args.days,
        start_date=args.start_date,
        seed=args.seed,
        output_dir=args.output_dir,
        reports_dir=args.reports_dir,
        include_label_in_logs=args.include_label_in_logs,
    )

    verbose = not args.quiet
    if verbose:
        print("The Watchtower — generating synthetic access logs")
    output = generate(cfg, verbose=verbose)
    paths = write_outputs(output)

    print()
    print(format_summary(output.summary))
    print("Artefacts written:")
    for name, path in paths.items():
        print(f"  {name:<16} {path.relative_to(PROJECT_ROOT)}")

    failed = [name for name, ok in output.summary["checks"].items() if not ok]
    if failed:
        print(f"\nFAILED CHECKS: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
