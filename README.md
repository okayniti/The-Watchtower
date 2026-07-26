# Behavioral Anomaly Detection for Cybersecurity

Model per-entity "normal" access behaviour, detect intrusions in near real-time, classify the
anomaly type, and attach an explainable risk score.

> **Status: Deliverable 1 complete.** The synthetic data generator is built and
> validated. Detection, classification, explainability and the dashboard are next.
> Full spec lives in [CLAUDE.md](CLAUDE.md).
> Live link: https://the-watchtower.streamlit.app/

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/a6eb7cc0-6780-401e-bed8-d7f0bb3f1f59" />


## The problem

Security logs are sequential, overwhelmingly benign, and constantly shifting. A detector that
scores well on AUC can still drown an analyst in false positives. This project targets the number
an analyst actually feels: **precision at a top-1% alert budget**.

## What it does

1. Generates synthetic access-event logs covering 8 behaviours (1 benign, 6 attacks, 1 ambiguous
   insider-drift case for false-positive tuning).
2. Builds a rolling per-entity behavioural profile that adapts to concept drift.
3. Scores events with a sequence-aware detector.
4. Classifies the anomaly *type*, not just anomaly/normal.
5. Explains every alert in analyst-readable English.
6. Serves a ranked alert queue in a Streamlit dashboard.

## Constraints it is built against

| Constraint | Why it's hard |
|---|---|
| Sequential data | Order and inter-event timing carry the signal; static rows lose it. |
| ~0.5–3% anomalies | Extreme imbalance; accuracy and AUC both mislead. |
| Concept drift | Legit behaviour evolves — a drifted entity must stop being flagged. |
| Explainability | A score without a reason is not actionable. |
| Cold start | A brand-new entity with no history still needs a sensible score. |

## Data schema

`entity_id`, `entity_type`, `timestamp`, `source_ip`, `geo_location`, `resource_accessed`,
`auth_method`, `session_duration`, `command_sequence`, `device_fingerprint`, `label`

`label` is used for training and evaluation only — it is hidden at inference.

## Behaviours

`normal_baseline` · `brute_force` · `impossible_travel` · `credential_stuffing` ·
`lateral_movement` · `device_spoofing` · `low_and_slow_exfiltration` · `insider_drift`

## Metrics

**Headline: precision / false-positive rate at a top-1% alert budget.** Not AUC.
Also: per-class precision/recall/F1, an 8-class confusion matrix, and an alert-budget curve.

## Layout

```
src/          library code — generator, profiler, detector, classifier, explainer, evaluation
data/raw/     generated event logs
data/processed/  feature matrices and splits
notebooks/    exploration only; imports from src/
dashboard/    Streamlit analyst dashboard
reports/      write-up and figures
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Usage

```powershell
python -m src.data_generator          # regenerate the full dataset + summary
python -m src.data_generator --help   # entities, days, seed, output paths
python scripts/eda.py                 # rebuild the figures in reports/figures/
```

Roughly 90,000 events across 500 entities over 30 days, in about a minute.
`scripts/eda.py` regenerates the dataset automatically if it is missing.

### Figures

| Figure | What it shows |
|---|---|
| `events_over_time.png` | Daily volume, benign against the six attack classes. |
| `class_distribution.png` | All eight classes, log scale — the imbalance made legible. |
| `entity_timeline_normal_vs_injected.png` | One victim's month: the attack is a change in rhythm, not an outlier. |
| `naive_rule_precision.png` | **The credibility figure.** One naive rule per attack class, computed live. |

That last one is the one to read first. Six hand-written single-feature rules — the ones a
reasonable engineer writes on day one — score **1.1% to 7.0% precision** against this data.
They have genuine lift over chance (3× to 66×) and are still 93–99% wrong. If those bars
were tall, the anomalies would be trivially separable and every downstream metric would be
meaningless.

### Data files

| File | Tracked? | What it is |
|---|---|---|
| `data/access_logs.csv` | no (~21 MB) | Full log, 10 feature columns + `event_id`. **No label.** |
| `data/labels.csv` | no | `event_id` → `label`, `episode_id`. |
| `data/sample_access_logs.csv` | **yes** | 2,000-row showcase sample. |
| `data/sample_labels.csv` | **yes** | Its labels. |
| `data/entity_profiles.json` | no | Generator-side ground truth. Never a model input. |
| `reports/data_summary.md` | **yes** | Class balance, episodes, cold-start cases, checks. |
| `reports/injected_episodes.csv` | **yes** | One row per injected campaign. |

The full dataset is git-ignored because it is large and **deterministic**: the same seed
reproduces it byte-for-byte, so `python -m src.data_generator` is a complete substitute
for committing it.

The tracked sample is **stratified, not proportional** — it over-represents every attack
class so all eight behaviours are visible in a small file. Do not compute class balance
from it; `reports/data_summary.md` has the real figures.

### Why the label lives in a separate file

`CLAUDE.md` §2 requires that no component read `label` outside training and evaluation.
Enforcing that by convention fails the first time someone drops the column in the wrong
place. Splitting the files makes the leak structurally impossible — the feature file
simply does not contain ground truth, and the two are rejoined on `event_id` only inside
training and evaluation code.

## Baseline profiler (Deliverable 2)

```powershell
python -m src.features            # build the causal feature matrix
python -m src.profiler            # score events, no labels
python scripts/eval_profiler.py   # evaluate (the only place labels are read)
```

25 features, each mapped to a documented attack signature. Scoring is an
exponentially-weighted per-entity baseline with shrinkage toward the entity's peer
cohort — one mechanism that covers concept drift (decay), cold start (shrinkage) and
explainability (per-feature deviations) at once.

At the top-1% alert budget the baseline reaches **22.2% precision** and 16.0%
hard-anomaly recall, against a 1.39% base rate. It does well on point anomalies
(`credential_stuffing` 50% recall) and poorly on the sequence-dependent classes
(`lateral_movement` 1.4%, `low_and_slow_exfiltration` 0.6%) — precisely the gap the
sequence detector exists to close.

Cold-start entities alert at 0.56% versus 0.81% for established ones, so a brand-new
identity is scored sensibly rather than flooding the queue.

**Two negatives worth reading before trusting this:** `insider_drift` is never flagged at
all (0.39% against a 0.79% benign rate), so "time to unflag" has nothing to measure yet —
that test moves to the sequence detector. And the decay half-life currently *costs*
precision (25.9% with no decay vs 22.2% at 7 days) with no drift false-positive to offset
it. The mechanism stays because the spec requires it, but the default is unjustified on
present evidence. See [CLAUDE.md](CLAUDE.md) §8.

## Build order

generator ✅ → EDA figures ✅ → features ✅ → profiler ✅ → sequence detector →
classifier → explainer → evaluation → dashboard → report
