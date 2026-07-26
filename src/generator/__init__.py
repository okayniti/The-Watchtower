"""The Watchtower synthetic access-log generator.

Public surface:

* :class:`~src.generator.config.GeneratorConfig` — the run configuration.
* :class:`~src.generator.config.AccessEvent` — the 11-field event record.
* :func:`~src.generator.entities.build_entities` — roster construction.
* :func:`~src.generator.profiles.build_profiles` /
  :func:`~src.generator.profiles.generate_normal_events` — benign traffic.
* :data:`~src.generator.attacks.INJECTORS` — the seven behaviour injectors.

Orchestration lives one level up in :mod:`src.data_generator`.
"""

from __future__ import annotations

from .attacks import INJECTORS, Episode, InjectionContext, InjectionResult
from .config import (
    AMBIGUOUS_LABEL,
    BEHAVIOURS,
    BENIGN_LABEL,
    ENTITY_TYPES,
    FEATURE_COLUMNS,
    HARD_ANOMALIES,
    LABEL_COLUMNS,
    AccessEvent,
    City,
    GeneratorConfig,
    Resource,
    haversine_km,
    parse_geo,
)
from .entities import Entity, EntityRoster, build_entities
from .profiles import BehaviouralProfile, build_profiles, generate_normal_events

__all__ = [
    "AMBIGUOUS_LABEL",
    "BEHAVIOURS",
    "BENIGN_LABEL",
    "ENTITY_TYPES",
    "FEATURE_COLUMNS",
    "HARD_ANOMALIES",
    "INJECTORS",
    "LABEL_COLUMNS",
    "AccessEvent",
    "BehaviouralProfile",
    "City",
    "Entity",
    "EntityRoster",
    "Episode",
    "GeneratorConfig",
    "InjectionContext",
    "InjectionResult",
    "Resource",
    "build_entities",
    "build_profiles",
    "generate_normal_events",
    "haversine_km",
    "parse_geo",
]
