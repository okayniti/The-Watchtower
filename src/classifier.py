"""Deliverable 4 — anomaly-type classifier.

A small feedforward head that answers "*which* attack is this?" for events the detector
has already surfaced. Two hidden layers over the same per-event features, plus the
profiler and sequence scores, with a 7-way softmax over ``normal_baseline`` and the six
attack classes.

Why a head rather than a second sequence model
----------------------------------------------
The ordering evidence has already been extracted: the GRU's per-event probability is one
of this model's inputs, so the recurrent signal is available without paying to learn it
twice. Detection and typing are also different problems — detection must scan 90k events
and is dominated by the imbalance, whereas typing sees only the funnel and needs to
separate six classes from each other. Keeping the second stage cheap is what makes the
funnel design worthwhile.

The funnel
----------
The classifier runs only on the top :data:`FUNNEL_FRACTION` of events by detection
score. That is deliberately generous — five times the 1% budget that
:mod:`src.evaluate` actually reports on — so that thresholding decisions live in
evaluation rather than being baked into the model. Events below the funnel are recorded
as ``normal_baseline`` by default, which is the correct prediction for the overwhelming
majority of them.

``insider_drift`` is folded into ``normal_baseline``. It is a legitimate role change, so
"normal" *is* the right answer; giving it a class of its own would train the model to
report a benign employee as a finding.

Labels
------
Targets arrive as an array argument. This module never opens ``labels.csv`` —
:mod:`src.evaluate` is the only reader.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
import torch
from torch import nn

from .features import FEATURE_NAMES
from .generator.config import BENIGN_LABEL, HARD_ANOMALIES

#: Softmax output classes, in index order. Index 0 is the benign catch-all.
CLASSES: Final[tuple[str, ...]] = (BENIGN_LABEL, *HARD_ANOMALIES)

#: Class index lookup.
CLASS_INDEX: Final[dict[str, int]] = {name: i for i, name in enumerate(CLASSES)}

#: Share of events passed to the classifier, ranked by detection score.
FUNNEL_FRACTION: Final[float] = 0.05

#: Score columns appended to the engineered features.
SCORE_INPUTS: Final[tuple[str, ...]] = ("risk_score", "sequence_score")


def to_class_index(labels: pd.Series) -> np.ndarray:
    """Map ground-truth label strings to classifier target indices.

    ``insider_drift`` maps to ``normal_baseline`` — it is benign by construction, and the
    classifier's job is to not report it.

    Args:
        labels: Ground-truth label strings.

    Returns:
        Integer class indices aligned with :data:`CLASSES`.
    """
    return labels.map(lambda name: CLASS_INDEX.get(name, 0)).to_numpy(dtype=np.int64)


@dataclass(slots=True)
class ClassifierConfig:
    """Hyper-parameters for the type classifier.

    Attributes:
        hidden_sizes: Widths of the two hidden layers.
        dropout: Dropout between hidden layers.
        epochs: Training epochs, kept low for the same hackathon-budget reason as the
            detector.
        batch_size: Minibatch size.
        learning_rate: Adam learning rate.
        seed: Master seed for reproducible weights and shuffling.
    """

    hidden_sizes: tuple[int, int] = (96, 48)
    dropout: float = 0.2
    epochs: int = 15
    batch_size: int = 128
    learning_rate: float = 3e-3
    seed: int = 20260726


class _MLP(nn.Module):
    """Two-hidden-layer feedforward classifier."""

    def __init__(self, n_inputs: int, n_classes: int, config: ClassifierConfig) -> None:
        """Build the network.

        Args:
            n_inputs: Input width.
            n_classes: Number of output classes.
            config: Hyper-parameters.
        """
        super().__init__()
        first, second = config.hidden_sizes
        self.net = nn.Sequential(
            nn.Linear(n_inputs, first),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(first, second),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(second, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits.

        Args:
            x: ``(batch, n_inputs)`` input.

        Returns:
            ``(batch, n_classes)`` logits.
        """
        return self.net(x)


class AnomalyClassifier:
    """Feedforward anomaly-type classifier operating on the detector's funnel.

    Attributes:
        config: Hyper-parameters.
        model: The trained network, or ``None`` before :meth:`fit`.
        mean: Per-feature training means.
        std: Per-feature training standard deviations.
        funnel_threshold: Detection score above which events enter the funnel.
    """

    def __init__(self, config: ClassifierConfig | None = None) -> None:
        """Initialise an untrained classifier.

        Args:
            config: Hyper-parameters. Defaults to :class:`ClassifierConfig`.
        """
        self.config = config or ClassifierConfig()
        self.model: _MLP | None = None
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.funnel_threshold: float = 0.0

    @staticmethod
    def build_inputs(
        features: pd.DataFrame,
        profiler_scores: pd.DataFrame,
        sequence_scores: np.ndarray,
    ) -> tuple[np.ndarray, pd.DataFrame]:
        """Assemble the classifier's input matrix.

        Args:
            features: Causal feature matrix.
            profiler_scores: Baseline profiler output.
            sequence_scores: Per-event detector probabilities aligned with ``features``.

        Returns:
            A ``(matrix, frame)`` tuple, where ``frame`` carries the aligned context
            columns and the two score inputs.
        """
        frame = features.merge(
            profiler_scores[["event_id", "risk_score"]], on="event_id", validate="one_to_one"
        )
        frame = frame.sort_values(["timestamp", "entity_id"]).reset_index(drop=True)
        frame["sequence_score"] = sequence_scores

        raw = frame[[*FEATURE_NAMES, *SCORE_INPUTS]].to_numpy(dtype=np.float32)
        raw = np.sign(raw) * np.log1p(np.abs(raw))
        return raw, frame

    def select_funnel(self, sequence_scores: np.ndarray) -> np.ndarray:
        """Return the mask of events entering the classifier.

        Args:
            sequence_scores: Per-event detection probabilities.

        Returns:
            Boolean mask of events above the funnel threshold.
        """
        self.funnel_threshold = float(
            np.quantile(sequence_scores, 1.0 - FUNNEL_FRACTION)
        )
        return sequence_scores >= self.funnel_threshold

    def fit(
        self,
        matrix: np.ndarray,
        targets: np.ndarray,
        train_mask: np.ndarray,
        verbose: bool = True,
    ) -> None:
        """Train on funnel events from the training period only.

        Args:
            matrix: Full input matrix.
            targets: Class index per event.
            train_mask: Events eligible for training — funnel members inside the
                training window.
            verbose: Print per-epoch loss.
        """
        cfg = self.config
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        train_rows = matrix[train_mask]
        train_targets = targets[train_mask]

        self.mean = train_rows.mean(axis=0)
        self.std = train_rows.std(axis=0)
        self.std[self.std < 1e-6] = 1.0
        scaled = (train_rows - self.mean) / self.std

        counts = np.bincount(train_targets, minlength=len(CLASSES)).astype(np.float64)
        # Inverse frequency, with unseen classes neutralised rather than infinite.
        weights = np.where(counts > 0, len(train_targets) / (len(CLASSES) * np.maximum(counts, 1)), 0.0)
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32)
        )

        self.model = _MLP(matrix.shape[1], len(CLASSES), cfg)
        optimiser = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate)

        x = torch.from_numpy(scaled.astype(np.float32))
        y = torch.from_numpy(train_targets)
        generator = torch.Generator().manual_seed(cfg.seed)

        for epoch in range(1, cfg.epochs + 1):
            self.model.train()
            shuffled = torch.randperm(len(x), generator=generator)
            total, seen = 0.0, 0
            for start in range(0, len(shuffled), cfg.batch_size):
                idx = shuffled[start : start + cfg.batch_size]
                optimiser.zero_grad()
                loss = criterion(self.model(x[idx]), y[idx])
                loss.backward()
                optimiser.step()
                total += float(loss) * len(idx)
                seen += len(idx)
            if verbose:
                print(f"    epoch {epoch:>2}/{cfg.epochs}  train_loss {total / max(seen, 1):.4f}",
                      flush=True)

    def predict(self, matrix: np.ndarray, funnel_mask: np.ndarray) -> np.ndarray:
        """Predict a class index for every event.

        Events outside the funnel are assigned ``normal_baseline`` without being run
        through the network — the detector already decided they were unremarkable.

        Args:
            matrix: Full input matrix.
            funnel_mask: Which events to actually classify.

        Returns:
            Class index per event.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if self.model is None:
            raise RuntimeError("call fit() before predict()")

        predictions = np.zeros(len(matrix), dtype=np.int64)
        if not funnel_mask.any():
            return predictions

        scaled = (matrix[funnel_mask] - self.mean) / self.std
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.from_numpy(scaled.astype(np.float32)))
            predictions[funnel_mask] = logits.argmax(dim=1).numpy()
        return predictions
