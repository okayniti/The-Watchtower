"""Deliverable 3 — sequence-aware detector.

A 2-layer GRU over each entity's ordered event sequence. Per-timestep input is the
causal feature vector from :mod:`src.features` plus the baseline risk score from
:mod:`src.profiler`; output is a per-event probability from a sigmoid head.

Why a GRU and not a Transformer
-------------------------------
Stated as a decision, not a comparison. The dataset is ~90k events across 500 entity
sequences averaging ~180 steps, trained on CPU inside a hackathon budget. A GRU is the
right size of tool for that: it is strictly causal by construction (no mask to get
wrong), it trains in minutes on CPU, and recurrence handles the long, irregularly-spaced
sequences here without the quadratic attention cost or the positional-encoding design
work a Transformer would add. Self-attention earns its keep when long-range
cross-position interactions matter and there is data to fit them; at this volume it
would mostly add parameters and training time. No alternatives were benchmarked — that
was a deliberate scope decision.

Why this catches what the profiler misses
-----------------------------------------
The baseline profiler scores each event against a per-entity distribution and does well
on point anomalies (``credential_stuffing`` 50% recall) but fails the two classes whose
signal is *ordering*: ``lateral_movement`` (1.4%) and ``low_and_slow_exfiltration``
(0.6%). Those are the classes a recurrent hidden state can accumulate evidence for — a
chain of individually-unremarkable novel-resource accesses, or a metronomic cadence
sustained over days. That gap is the reason this model exists.

Causality and leakage
---------------------
Three separate guarantees, none of them incidental:

* The GRU is unidirectional, so a timestep's output depends only on that timestep and
  its predecessors.
* The train/validation split is by **time**, not at random. The last 20% of the window
  is held out, and training sequences are truncated at the cutoff — the model never sees
  a gradient from a validation-period event.
* Feature standardisation statistics are fitted on the training slice alone.

Labels
------
This is a supervised model, so it needs targets — but it never opens ``labels.csv``.
Targets arrive as an array argument supplied by :mod:`src.evaluate`, which is the only
module in the project that reads ground truth from disk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from .features import FEATURE_NAMES

#: Extra per-timestep inputs appended to the engineered features.
PROFILER_INPUTS: Final[tuple[str, ...]] = ("risk_score", "history_weight", "cold_start")


@dataclass(slots=True)
class SequenceConfig:
    """Hyper-parameters for the sequence detector.

    Attributes:
        hidden_size: GRU hidden width.
        num_layers: Stacked GRU layers.
        dropout: Dropout between GRU layers and in the head.
        epochs: Training epochs. Held low deliberately — this is a hackathon demo, and
            further tuning would buy accuracy that the evaluation story does not need.
        batch_size: Entity sequences per batch.
        learning_rate: Adam learning rate.
        val_fraction: Share of the time window held out for validation.
        seed: Master seed for reproducible weights and shuffling.
        grad_clip: Gradient-norm clip, which recurrent nets want.
    """

    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    epochs: int = 12
    batch_size: int = 32
    learning_rate: float = 2e-3
    val_fraction: float = 0.2
    seed: int = 20260726
    grad_clip: float = 5.0


@dataclass(slots=True)
class SequenceData:
    """Model inputs assembled into per-entity sequences.

    Attributes:
        matrix: ``(n_events, n_inputs)`` standardised input matrix in global time order.
        event_ids: Event identifiers aligned with ``matrix`` rows.
        entity_ids: Entity identifier per row.
        timestamps: Event time per row.
        sequences: Row indices per entity, each already in ascending time order.
        cutoff: Timestamp separating training from validation.
        is_validation: Boolean mask marking validation-period rows.
        n_inputs: Width of the input vector.
    """

    matrix: np.ndarray
    event_ids: np.ndarray
    entity_ids: np.ndarray
    timestamps: pd.Series
    sequences: dict[str, np.ndarray]
    cutoff: pd.Timestamp
    is_validation: np.ndarray
    n_inputs: int


class _GRUScorer(nn.Module):
    """Per-timestep binary scorer over an entity's event sequence.

    Attributes:
        gru: The recurrent stack.
        head: Feedforward projection from hidden state to a single logit.
    """

    def __init__(self, n_inputs: int, config: SequenceConfig) -> None:
        """Build the network.

        Args:
            n_inputs: Width of the per-timestep input vector.
            config: Hyper-parameters.
        """
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_inputs,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(config.hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(32, 1),
        )

    def forward(self, padded: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Score every timestep of a padded batch.

        Args:
            padded: ``(batch, max_len, n_inputs)`` padded input.
            lengths: True sequence length per batch element.

        Returns:
            ``(batch, max_len)`` logits.
        """
        packed = pack_padded_sequence(
            padded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        output, _ = self.gru(packed)
        output, _ = pad_packed_sequence(output, batch_first=True, total_length=padded.size(1))
        return self.head(output).squeeze(-1)


class SequenceDetector:
    """GRU sequence detector with a time-ordered train/validation split.

    Attributes:
        config: Hyper-parameters.
        model: The trained network, or ``None`` before :meth:`fit`.
        mean: Per-feature training means used for standardisation.
        std: Per-feature training standard deviations.
        history: Per-epoch training and validation loss.
    """

    def __init__(self, config: SequenceConfig | None = None) -> None:
        """Initialise an untrained detector.

        Args:
            config: Hyper-parameters. Defaults to :class:`SequenceConfig`.
        """
        self.config = config or SequenceConfig()
        self.model: _GRUScorer | None = None
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.history: list[dict[str, float]] = []

    # -- data preparation ---------------------------------------------------------------

    def prepare(
        self, features: pd.DataFrame, profiler_scores: pd.DataFrame
    ) -> SequenceData:
        """Join features to profiler output and assemble per-entity sequences.

        Args:
            features: Output of :func:`~src.features.build_feature_matrix`.
            profiler_scores: ``scores`` frame from :class:`~src.profiler.EntityProfiler`.

        Returns:
            The assembled :class:`SequenceData`.

        Raises:
            ValueError: If the two frames do not align on ``event_id``.
        """
        merged = features.merge(
            profiler_scores[["event_id", *PROFILER_INPUTS]],
            on="event_id",
            validate="one_to_one",
        )
        if len(merged) != len(features):
            raise ValueError("features and profiler scores did not align on event_id")

        merged = merged.sort_values(["timestamp", "entity_id"]).reset_index(drop=True)
        merged["cold_start"] = merged["cold_start"].astype(float)

        columns = [*FEATURE_NAMES, *PROFILER_INPUTS]
        raw = merged[columns].to_numpy(dtype=np.float32)
        # Heavy tails everywhere (geo-velocity reaches 1.4e7); compress before scaling
        # so standardisation is not dictated by a handful of extreme rows.
        raw = np.sign(raw) * np.log1p(np.abs(raw))

        timestamps = pd.to_datetime(merged["timestamp"])
        cutoff = timestamps.quantile(1.0 - self.config.val_fraction)
        is_validation = (timestamps > cutoff).to_numpy()

        train_rows = raw[~is_validation]
        self.mean = train_rows.mean(axis=0)
        self.std = train_rows.std(axis=0)
        self.std[self.std < 1e-6] = 1.0
        matrix = (raw - self.mean) / self.std

        entity_ids = merged["entity_id"].to_numpy()
        sequences: dict[str, np.ndarray] = {
            entity: np.asarray(rows, dtype=np.int64)
            for entity, rows in merged.groupby("entity_id", sort=True).groups.items()
        }
        return SequenceData(
            matrix=matrix.astype(np.float32),
            event_ids=merged["event_id"].to_numpy(),
            entity_ids=entity_ids,
            timestamps=timestamps,
            sequences=sequences,
            cutoff=cutoff,
            is_validation=is_validation,
            n_inputs=matrix.shape[1],
        )

    # -- training -----------------------------------------------------------------------

    def fit(self, data: SequenceData, targets: np.ndarray, verbose: bool = True) -> None:
        """Train on the pre-cutoff portion of every entity sequence.

        Args:
            data: Prepared sequences.
            targets: Binary target per event, aligned with ``data.matrix`` rows. ``1``
                for the six hard-anomaly classes, ``0`` otherwise — ``insider_drift`` is
                a zero, because it is benign and the model must learn not to flag it.
            verbose: Print per-epoch losses.
        """
        cfg = self.config
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        train_sequences: list[tuple[torch.Tensor, torch.Tensor]] = []
        for rows in data.sequences.values():
            keep = rows[~data.is_validation[rows]]
            if keep.size == 0:
                continue
            train_sequences.append(
                (
                    torch.from_numpy(data.matrix[keep]),
                    torch.from_numpy(targets[keep].astype(np.float32)),
                )
            )

        positives = float(targets[~data.is_validation].sum())
        negatives = float((~data.is_validation).sum() - positives)
        # Inverse-frequency weighting: at a ~1.4% base rate the model would otherwise
        # minimise loss by predicting "benign" for everything.
        pos_weight = torch.tensor([negatives / max(positives, 1.0)], dtype=torch.float32)

        self.model = _GRUScorer(data.n_inputs, cfg)
        optimiser = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

        generator = torch.Generator().manual_seed(cfg.seed)
        order = np.arange(len(train_sequences))

        for epoch in range(1, cfg.epochs + 1):
            self.model.train()
            shuffled = torch.randperm(len(order), generator=generator).numpy()
            total_loss, total_steps = 0.0, 0

            for start in range(0, len(shuffled), cfg.batch_size):
                batch_idx = shuffled[start : start + cfg.batch_size]
                batch = [train_sequences[int(i)] for i in batch_idx]
                lengths = torch.tensor([len(x) for x, _ in batch], dtype=torch.long)
                padded = pad_sequence([x for x, _ in batch], batch_first=True)
                target = pad_sequence([y for _, y in batch], batch_first=True)
                mask = (
                    torch.arange(padded.size(1))[None, :] < lengths[:, None]
                ).float()

                optimiser.zero_grad()
                logits = self.model(padded, lengths)
                loss = (criterion(logits, target) * mask).sum() / mask.sum()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                optimiser.step()

                total_loss += float(loss) * float(mask.sum())
                total_steps += int(mask.sum())

            epoch_loss = total_loss / max(total_steps, 1)
            self.history.append({"epoch": epoch, "train_loss": epoch_loss})
            if verbose:
                print(f"    epoch {epoch:>2}/{cfg.epochs}  train_loss {epoch_loss:.4f}",
                      flush=True)

    # -- inference ----------------------------------------------------------------------

    def predict(self, data: SequenceData) -> np.ndarray:
        """Score every event, running each entity's full sequence.

        Validation-period events are scored with the hidden state accumulated over that
        entity's earlier events, which is exactly how the model would run online. The
        GRU's unidirectionality is what makes that safe.

        Args:
            data: Prepared sequences.

        Returns:
            Per-event probability, aligned with ``data.matrix`` rows.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if self.model is None:
            raise RuntimeError("call fit() before predict()")

        self.model.eval()
        scores = np.zeros(len(data.matrix), dtype=np.float32)
        entities = list(data.sequences.items())

        with torch.no_grad():
            for start in range(0, len(entities), self.config.batch_size):
                chunk = entities[start : start + self.config.batch_size]
                tensors = [torch.from_numpy(data.matrix[rows]) for _e, rows in chunk]
                lengths = torch.tensor([len(t) for t in tensors], dtype=torch.long)
                padded = pad_sequence(tensors, batch_first=True)
                logits = self.model(padded, lengths)
                probabilities = torch.sigmoid(logits).numpy()
                for position, (_entity, rows) in enumerate(chunk):
                    scores[rows] = probabilities[position, : len(rows)]
        return scores
