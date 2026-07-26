# CLAUDE.md — Behavioral Anomaly Detection for Cybersecurity

Persistent project context. Read this first in any new session.

---

## 1. Goal

Model per-entity "normal" access behaviour, detect intrusions in **near real-time**, classify the
anomaly **TYPE**, and attach an **explainable risk score**.

This is a hackathon project. Judging is driven by the Hard Requirements (§4) and the Headline
Metric (§5) — not by model complexity.

---

## 2. Data Schema (11 fields)

| # | Field | Type | Notes |
|---|-------|------|-------|
| 1 | `entity_id` | str | Stable ID for the actor. |
| 2 | `entity_type` | enum | `user` \| `service_account` \| `edge_device` |
| 3 | `timestamp` | datetime | Event time. Ordering key for all sequence models. |
| 4 | `source_ip` | str | IPv4. |
| 5 | `geo_location` | str | Resolved from IP (city/country + lat/lon). |
| 6 | `resource_accessed` | str | Path / service / asset identifier. |
| 7 | `auth_method` | enum | `password` \| `token` \| `certificate` \| `biometric` |
| 8 | `session_duration` | float | Seconds. |
| 9 | `command_sequence` | list[str] | Ordered commands issued in the session. |
| 10 | `device_fingerprint` | str | Composite: OS / firmware / MAC. |
| 11 | `label` | enum | `normal` or one of the 7 anomaly types. **Training/eval only — hidden at inference.** |

**Rule:** no component of the pipeline may read `label` outside of training and evaluation.
Inference code paths must be able to run on a `label`-free frame.

---

## 3. The 8 Behaviours

| Behaviour | Class | Signature (what the generator must encode) |
|---|---|---|
| `normal_baseline` | benign | Per-entity habitual IPs, geos, hours, resources, auth method, command patterns. |
| `brute_force` | anomaly | Burst of failed auths, one entity, short window, repeated auth attempts. |
| `impossible_travel` | anomaly | Two geos too far apart for the elapsed time between events. |
| `credential_stuffing` | anomaly | Many *different* entity_ids from a shared IP/device, low success rate. |
| `lateral_movement` | anomaly | Entity walks across resources it has never touched, in an expanding chain. |
| `device_spoofing` | anomaly | Known entity_id arriving with a mismatched/novel device_fingerprint. |
| `low_and_slow_exfiltration` | anomaly | Small, regular, long-duration sessions on data resources over days — individually unremarkable. |
| `insider_drift` | **ambiguous** | Legitimate role change: behaviour genuinely shifts and *stays* shifted. Exists to punish detectors that flag forever. Used for false-positive tuning. |

`insider_drift` is deliberately the hardest call. Treat it as the FP-control test case, not as a
class to maximise recall on.

---

## 4. Hard Requirements (these drive scoring)

1. **Sequential / behavioural data** — events over time per entity, not static i.i.d. rows.
   Features and models must exploit order and inter-event timing.
2. **Extreme class imbalance** — anomalies are ~0.5–3% of events. No resampling that inflates the
   reported metric; imbalance must survive into evaluation.
3. **Concept drift** — legitimate behaviour evolves. A profile must adapt (e.g. decayed/rolling
   baselines) so a drifted-but-benign entity stops being flagged. "Flagged forever" is a failure.
4. **Explainability** — the analyst needs **WHY**, not just a number. Every alert carries the
   contributing features and a natural-language rationale.
5. **Cold-start** — a brand-new entity with zero history must still receive a sensible score
   (fall back to entity_type/peer-group priors, not a crash or a null).

---

## 5. Metrics

**Headline metric: Precision / false-positive rate at a top-1% alert budget.**
Rank all events by risk score, take the top 1%, report precision and FP rate on that slice.
This is the number that matters. **AUC is not the headline metric** — with 0.5–3% positives it is
misleading, and reporting it as the primary result is a scoring risk.

Also report:
- Per-class precision / recall / F1 for the 7 anomaly types.
- Confusion matrix across all 8 classes.
- Alert-budget curve (precision vs. budget at 0.1% / 0.5% / 1% / 5%).
- FP rate specifically on `insider_drift` events, and time-to-unflag after a drift event.

---

## 6. Deliverables

| # | Deliverable | Location |
|---|---|---|
| 1 | Synthetic data generator | `src/data_generator.py` |
| 2 | Baseline per-entity profiler | `src/profiler.py` |
| 3 | Sequence-aware detector | `src/sequence_detector.py` |
| 4 | Anomaly-type classifier | `src/classifier.py` |
| 5 | Explainability layer (analyst-readable sentences) | `src/explainer.py` |
| 6 | Streamlit analyst dashboard, ranked alert queue | `dashboard/app.py` |
| 7 | Report | `reports/` |

Supporting: `src/features.py` (sequence feature engineering), `src/evaluate.py` (§5 metrics),
`src/config.py` (schema constants, class names, thresholds).

---

## 7. Repo Layout

```
honeywell/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── .gitignore
├── src/                 # library code (no notebooks-only logic)
├── data/
│   ├── raw/             # generated event logs
│   └── processed/       # feature matrices, splits
├── notebooks/           # exploration only — nothing load-bearing
├── dashboard/           # Streamlit app
└── reports/
    └── figures/
```

Rule: notebooks import from `src/`. Logic never lives only in a notebook.

---

## 8. Current Status

**Deliverable 1 (generator) is complete and validated.** Everything downstream is unbuilt.

Built:
- `src/generator/{config,entities,profiles,attacks}.py` + `src/data_generator.py`.
- ~90k events, 500 entities, 30 days. Hard-anomaly rate **1.39%** (in the 0.5-3% band).
- Deterministic: same seed → byte-identical CSVs (verified).
- All eight behaviours present; eight self-checks pass and gate the CLI exit code.

Two decisions worth not re-litigating:
1. **Labels live in `data/labels.csv`, not in `access_logs.csv`.** Structural enforcement of
   §2, not convention. Rejoin on `event_id` inside training/eval only.
2. **`insider_drift` is excluded from the anomaly-rate figure.** It is benign-but-attack-shaped;
   counting it would misstate the imbalance the detector faces. Reported separately.

Realism is load-bearing, not decoration — naive single-feature rules score 1-16% precision
against this data (AUTH_FAIL 5.9%, unseen device 6.0%, shared IP 3.3%, novel resource 4.1%,
session duration ≤1.1%). Preserve that when touching the generator. In particular, ~45% of
entities tunnel via a VPN gateway and geolocate to the *gateway's* city, which is what stops
geo-velocity from being a clean separator.

EDA figures are built (`scripts/eda.py` → `reports/figures/`, 4 PNGs at 150 dpi, deterministic).
`naive_rule_precision.png` is the credibility figure: one naive rule per attack class, all
computed live, none above 7% precision. Keep it in the report and the deck.

Figures follow one shared palette (validated categorical slots, adjacent-pair CVD ΔE 9.1) with
colour keyed to class identity and held constant across every figure — if you add a chart,
reuse `CLASS_COLOR` from `scripts/eda.py` rather than picking new colours. Captions are
hand-wrapped because `bbox_inches="tight"` stretches the canvas to fit any overflowing line.

### Deliverable 2 + feature layer (complete)

`src/features.py` — 25 causal features, each mapped to an attack signature via
`FEATURE_SPECS`. Implemented as a **streaming single pass** (score-then-update), so
lookahead leakage is structurally impossible and the same code path serves near-real-time.
`src/profiler.py` — exponentially-weighted per-entity moments with peer-group shrinkage.
One mechanism covers drift (decay), cold start (shrinkage) and explainability (per-feature z).
`scripts/eval_profiler.py` — **the only module that reads `labels.csv`.**

Baseline results at the top-1% budget: **precision 22.2%**, hard-anomaly recall 16.0%.

| class | recall@1% | note |
|---|---:|---|
| credential_stuffing | 50.0% | point anomaly, cross-entity — profiler's best case |
| impossible_travel | 19.0% | |
| device_spoofing | 12.0% | |
| brute_force | 10.1% | |
| lateral_movement | **1.4%** | sequence-dependent — needs Deliverable 3 |
| low_and_slow_exfiltration | **0.6%** | sequence-dependent — needs Deliverable 3 |

Cold start works: benign cold-start events alert at 0.56% vs 0.81% for established
entities, and are 4.5% of the queue at 6.8% of the log. They are not swamping it.

### Two honest negatives — do not paper over these

1. **Time-to-unflag is not measurable at the profiler stage.** `insider_drift` alerts at
   0.39% against a benign rate of 0.79% — it is never flagged, so there is no false
   positive to decay away. Good for FP control, but it is *not* evidence the decay
   mechanism works. The test needs a detector that flags drift first — that is the
   sequence model's job. Re-run it there.
2. **Decay currently costs precision and buys nothing back.** Swept over half-lives:
   no-decay scores 25.9% precision / 18.6% recall, 7d scores 22.2% / 16.0%. Shorter
   half-lives hold a tighter baseline, which raises variance sensitivity faster than it
   improves mean adaptation. The 7d default is kept because §4.3 requires the mechanism
   and the sequence model may justify it — but **revisit the default once Deliverable 3
   exists**, and do not claim decay is helping until a measurement says so.

Build order: generator ✅ → EDA figures ✅ → features ✅ → profiler ✅ →
**sequence detector** → classifier → explainer → evaluation → dashboard → report.

---

## 9. Environment

- Python 3.13.7 (Windows, PowerShell).
- Deps in `requirements.txt`: numpy, pandas, scikit-learn, torch, faker, streamlit, matplotlib, shap.
- Note: `shap` + Python 3.13 can need a recent build; if install fails, pin `shap>=0.46`.
