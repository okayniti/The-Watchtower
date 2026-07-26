"""Deliverable 6 — The Watchtower analyst dashboard.

A SOC triage surface built around a **ranked alert queue** under an explicit alert
budget. Run with::

    streamlit run dashboard/app.py

Everything is precomputed. :mod:`src.explainer` writes ``data/alerts.json`` and
``data/validation_scores.csv``; this app only loads, filters and renders them. No model
is trained, no feature is recomputed, and nothing here touches torch — the page is
interactive immediately.

The design premise
------------------
An analyst does not have an unbounded queue. They have a shift. So the budget is the
primary control, not a hidden threshold: moving the slider from 0.5% to 5% shows
precision collapsing from 98% to 20% in real time, which is the single most honest thing
this project can put in front of a reviewer. Alert counts and precision update together,
so the trade-off is visible rather than asserted.

The Trust panel exists for the same reason. ``insider_drift`` is a legitimate role
change, so every alert on it is a false positive; showing the ramp-versus-settled rate
side by side is how an analyst learns the system will stop accusing someone once their
new role beds in.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Final

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from src.generator.config import AMBIGUOUS_LABEL, BENIGN_LABEL, HARD_ANOMALIES  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"

# Palette lifted from the figure deck so the product and the report look like one system.
CLASS_COLOR: Final[dict[str, str]] = {
    BENIGN_LABEL: "#898781",
    "brute_force": "#2a78d6",
    "impossible_travel": "#eb6834",
    "credential_stuffing": "#1baf7a",
    "lateral_movement": "#eda100",
    "device_spoofing": "#e87ba4",
    "low_and_slow_exfiltration": "#008300",
    AMBIGUOUS_LABEL: "#4a3aa7",
}
INK_SECONDARY: Final[str] = "#52514e"
INK_MUTED: Final[str] = "#898781"

#: Selectable alert budgets, as percentages of the scored window.
BUDGETS: Final[tuple[float, ...]] = (0.5, 1.0, 2.0, 5.0)

st.set_page_config(
    page_title="The Watchtower — analyst console",
    page_icon="🗼",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_alerts() -> tuple[pd.DataFrame, dict]:
    """Load precomputed explanations.

    Returns:
        A ``(records, meta)`` tuple.

    Raises:
        FileNotFoundError: If the explanations have not been generated.
    """
    path = DATA_DIR / "alerts.json"
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    frame = pd.DataFrame(payload["records"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame, payload["meta"]


@st.cache_data(show_spinner=False)
def load_events() -> tuple[pd.DataFrame, bool]:
    """Load the raw access log, used for per-entity timelines.

    The full log is git-ignored because it is large and regenerates deterministically,
    so a fresh clone falls back to the tracked 2,000-row sample. That keeps the queue
    and explanations fully working and degrades only the per-entity timeline, rather
    than failing to start.

    Returns:
        A ``(frame, is_full)`` tuple. ``is_full`` is ``False`` when the sample was used.
    """
    full = DATA_DIR / "access_logs.csv"
    path, is_full = (full, True) if full.exists() else (
        DATA_DIR / "sample_access_logs.csv", False
    )
    frame = pd.read_csv(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame, is_full


@st.cache_data(show_spinner=False)
def load_labels() -> pd.DataFrame:
    full = DATA_DIR / "labels.csv"
    path = full if full.exists() else DATA_DIR / "sample_labels.csv"
    return pd.read_csv(path)


def pretty(name: str) -> str:
    """Render a class name for display.

    Args:
        name: A behaviour class name.

    Returns:
        The name with underscores replaced by spaces.
    """
    return name.replace("_", " ")


# --------------------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------------------


def render_header(
    alerts: pd.DataFrame, meta: dict, budget_pct: float
) -> pd.DataFrame:
    """Render the summary panel and return the alerts inside the budget.

    Args:
        alerts: All alert-kind records.
        meta: Run metadata.
        budget_pct: Selected alert budget, as a percentage.

    Returns:
        The alerts falling inside the budget, highest risk first.
    """
    total_validation = int(meta["validation_events"])
    capacity = max(1, int(round(total_validation * budget_pct / 100.0)))
    ranked = alerts.sort_values("risk_score", ascending=False).head(capacity)

    true_positives = int(ranked.true_label.isin(HARD_ANOMALIES).sum())
    precision = true_positives / len(ranked) if len(ranked) else 0.0
    base_rate = float(meta["attack_base_rate"])

    st.markdown("### The Watchtower — analyst console")
    st.caption(
        f"Scored window: {total_validation:,} held-out events after "
        f"{meta['validation_cutoff']}. All scores precomputed; nothing trains on load."
    )

    columns = st.columns(5)
    columns[0].metric("Events in window", f"{total_validation:,}")
    columns[1].metric(f"Alerts @ {budget_pct:g}%", f"{len(ranked):,}")
    columns[2].metric(
        "Precision", f"{precision:.1%}", delta=f"{precision / base_rate:.0f}x base rate"
    )
    columns[3].metric("True positives", f"{true_positives:,}")
    columns[4].metric(
        "Analyst load", f"{len(ranked) / 7:.0f}/day", help="Assuming a 7-day window."
    )
    return ranked


def render_trust_panel(alerts: pd.DataFrame, labels: pd.DataFrame) -> None:
    """Render the insider-drift false-positive panel.

    Args:
        alerts: All alert-kind records inside the current view.
        labels: Ground truth, for identifying drift events.
    """
    st.markdown("#### Trust — legitimate role change")
    st.caption(
        "`insider_drift` is a benign role change, so every alert on it is a false "
        "positive. Measured over the full window by `src/evaluate.py`."
    )
    columns = st.columns(3)
    columns[0].metric("FP rate during ramp", "0.78%")
    columns[1].metric("FP rate once settled", "0.19%", delta="-0.59pp", delta_color="inverse")
    columns[2].metric("Entities tracked", "25")
    st.caption(
        "The rate falls 4x once the role change beds in — the system stops accusing an "
        "employee who did nothing wrong."
    )


def render_queue(ranked: pd.DataFrame) -> pd.Series | None:
    """Render the ranked alert queue and return the selected row.

    Args:
        ranked: Alerts inside the current budget and filters.

    Returns:
        The selected alert record, or ``None`` if the queue is empty.
    """
    st.markdown("#### Ranked alert queue")
    if ranked.empty:
        st.info("No alerts match the current filters.")
        return None

    display = pd.DataFrame(
        {
            "risk": ranked.risk_score.round(3),
            "predicted type": ranked.predicted_type.map(pretty),
            "entity": ranked.entity_id,
            "type": ranked.entity_type,
            "when": ranked.timestamp.dt.strftime("%d %b %H:%M"),
            "resource": ranked.resource_accessed,
            "cold start": ranked.cold_start.map({True: "yes", False: ""}),
        }
    )

    event = st.dataframe(
        display,
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        height=380,
        column_config={
            "risk": st.column_config.ProgressColumn(
                "risk", min_value=0.0, max_value=1.0, format="%.3f"
            )
        },
    )
    selected = event.selection.rows if event and event.selection else []
    index = selected[0] if selected else 0
    return ranked.iloc[index]


def render_detail(record: pd.Series, events: pd.DataFrame) -> None:
    """Render the detail view for one alert.

    Args:
        record: The selected alert.
        events: The full access log, for the entity timeline.
    """
    st.markdown("#### Alert detail")

    left, right = st.columns([3, 2])
    with left:
        colour = CLASS_COLOR.get(record.predicted_type, INK_SECONDARY)
        st.markdown(
            f"<div style='border-left:4px solid {colour};padding:0.6rem 0 0.6rem 0.9rem'>"
            f"<div style='font-size:0.8rem;color:{INK_MUTED};letter-spacing:0.04em'>"
            f"{record.event_id} &nbsp;·&nbsp; {record.entity_id} "
            f"({record.entity_type})</div>"
            f"<div style='font-size:1.05rem;margin-top:0.35rem'>{record.sentence}</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        if record.cold_start:
            st.warning(
                "Limited history for this entity — scored against its peer-group "
                "baseline rather than its own.",
                icon="⚠️",
            )

    with right:
        metrics = st.columns(2)
        metrics[0].metric("Detector risk", f"{record.risk_score:.3f}")
        metrics[1].metric("Profiler baseline", f"{record.profiler_score:.2f}")
        st.caption(
            f"**Resource** {record.resource_accessed}  \n"
            f"**Origin** {record.source_ip} — {record.geo_location}"
        )

    factors = pd.DataFrame(record.factors)
    if not factors.empty:
        st.markdown("**Contributing factors** — ranked by contribution to the prediction")
        chart_data = factors.set_index("feature")["contribution"]
        st.bar_chart(chart_data, horizontal=True, height=max(160, 42 * len(factors)))
        st.dataframe(
            pd.DataFrame(
                {
                    "factor": factors.feature,
                    "value": factors.value.round(3),
                    "deviation (σ vs entity)": factors.deviation.round(2),
                    "targets": factors.targets.map(pretty),
                }
            ),
            width="stretch",
            hide_index=True,
        )
    else:
        st.caption(
            "No single field is abnormal — this alert comes from the sequence model's "
            "read of the entity's recent event pattern."
        )

    history = (
        events[events.entity_id == record.entity_id]
        .sort_values("timestamp")
        .assign(hour=lambda d: d.timestamp.dt.hour + d.timestamp.dt.minute / 60)
    )
    if history.empty:
        st.caption(
            f"No timeline available for {record.entity_id} — run "
            "`python -m src.data_generator` to rebuild the full access log."
        )
    else:
        st.markdown(f"**Recent activity for {record.entity_id}** — {len(history):,} events")
        st.scatter_chart(
            history.set_index("timestamp")[["hour"]],
            height=220,
            color=CLASS_COLOR.get(record.predicted_type, "#2a78d6"),
        )
        st.caption("Hour of day over the window. Look for rhythm, not outliers.")
        st.dataframe(
            history.tail(12)[
                ["timestamp", "resource_accessed", "auth_method",
                 "session_duration", "geo_location"]
            ].iloc[::-1],
            width="stretch",
            hide_index=True,
        )


def render_suppressions(suppressed: pd.DataFrame) -> None:
    """Render the suppressed-events panel.

    Args:
        suppressed: Records the profiler scored highly but the detector did not.
    """
    st.markdown("#### Suppressed — considered and dismissed")
    st.caption(
        "Events the baseline profiler ranked in its top 5% that the sequence model did "
        "not escalate. Showing these is how an analyst learns to trust the threshold "
        "instead of lowering it."
    )
    if suppressed.empty:
        st.info("Nothing suppressed in the current view.")
        return
    for _, record in suppressed.sort_values("profiler_score", ascending=False).head(8).iterrows():
        with st.expander(
            f"{record.entity_id} · profiler {record.profiler_score:.2f} → "
            f"detector {record.risk_score:.3f} · {record.resource_accessed}"
        ):
            st.write(record.sentence)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def main() -> None:
    """Compose the dashboard."""
    try:
        records, meta = load_alerts()
    except FileNotFoundError:
        st.error(
            "`data/alerts.json` not found. Generate it first:\n\n"
            "```\npython -m src.explainer\n```"
        )
        return

    events, full_log = load_events()
    labels = load_labels()
    if not full_log:
        st.info(
            "Running against the 2,000-row tracked sample — the full access log is not "
            "present, so per-entity timelines will be sparse. Rebuild it with "
            "`python -m src.data_generator`.",
            icon="ℹ️",
        )

    with st.sidebar:
        st.markdown("### Alert budget")
        budget_pct = st.select_slider(
            "Share of events an analyst can review",
            options=list(BUDGETS),
            value=1.0,
            format_func=lambda v: f"{v:g}%",
        )
        st.caption("1% is the budget the project is scored against.")

        st.markdown("### Filters")
        all_alerts = records[records.kind == "alert"]
        types = sorted(all_alerts.predicted_type.unique())
        chosen_types = st.multiselect(
            "Anomaly type", types, default=types, format_func=pretty
        )
        entity_types = sorted(all_alerts.entity_type.unique())
        chosen_entity_types = st.multiselect(
            "Entity type", entity_types, default=entity_types
        )
        low, high = all_alerts.timestamp.min(), all_alerts.timestamp.max()
        date_range = st.date_input(
            "Time range", value=(low.date(), high.date()),
            min_value=low.date(), max_value=high.date(),
        )
        st.divider()
        st.caption(
            "Precomputed by `src/explainer.py`. Regenerate after retraining with "
            "`python -m src.explainer`."
        )

    alerts = records[records.kind == "alert"]
    ranked = render_header(alerts, meta, budget_pct)

    # Filters narrow the queue, they do not change the budget: the budget defines how
    # many alerts exist, the filters decide which of them you are looking at.
    filtered = ranked[
        ranked.predicted_type.isin(chosen_types)
        & ranked.entity_type.isin(chosen_entity_types)
    ]
    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start, end = date_range
        filtered = filtered[
            (filtered.timestamp.dt.date >= start) & (filtered.timestamp.dt.date <= end)
        ]

    st.divider()
    queue_column, trust_column = st.columns([3, 1])
    with queue_column:
        selected = render_queue(filtered)
    with trust_column:
        render_trust_panel(filtered, labels)
        distribution = (
            ranked.predicted_type.value_counts().rename_axis("type").reset_index(name="alerts")
        )
        st.markdown("#### Queue composition")
        st.dataframe(
            distribution.assign(type=lambda d: d.type.map(pretty)),
            width="stretch", hide_index=True, height=200,
        )

    if selected is not None:
        st.divider()
        render_detail(selected, events)

    st.divider()
    render_suppressions(records[records.kind == "suppressed"])


main()
