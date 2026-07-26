"""Entity roster construction for The Watchtower generator.

An *entity* is the stable identity that acts in the log: a human user, a service
account, or an edge device. This module owns everything about an entity that is
**static** across the simulation window — its identifier, where it is based, which
addresses it egresses from, and what hardware it runs on. The *behavioural*
distributions (when it is active, what it touches, how long sessions run) live in
:mod:`src.generator.profiles`.

The split matters because the attack injectors in :mod:`src.generator.attacks` need to
violate exactly one of these two layers at a time. ``device_spoofing`` violates the
static layer (wrong hardware, right behaviour); ``lateral_movement`` violates the
behavioural layer (right hardware, wrong resources). Keeping them separate keeps each
attack's signature clean to reason about — and keeps the *benign* counterpart of each
(a genuine laptop refresh, a genuinely curious engineer) equally easy to inject.

Cold start
----------
A configurable slice of the roster (:attr:`GeneratorConfig.cold_start_fraction`) is held
back: those entities first appear only in the final few days of the window and emit a
handful of events. They exist so that the detector must produce a sensible score for an
identity it has essentially never seen — hard requirement §4.5 in ``CLAUDE.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from .config import (
    CITIES,
    DEVICE_POOLS,
    ENTITY_TYPES,
    OFFICE_CITIES,
    City,
    GeneratorConfig,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from faker import Faker

#: Prefix used to build entity identifiers, keyed by entity type.
_ID_PREFIX: dict[str, str] = {
    "user": "usr",
    "service_account": "svc",
    "edge_device": "dev",
}

_CITIES_BY_NAME: dict[str, City] = {c.name: c for c in CITIES}
_OFFICE_CITY_OBJECTS: tuple[City, ...] = tuple(
    _CITIES_BY_NAME[name] for name in OFFICE_CITIES
)


@dataclass(slots=True)
class Entity:
    """A single actor in the simulation, with its static identity attributes.

    Attributes:
        entity_id: Stable identifier, e.g. ``"usr-0142"``.
        entity_type: One of :data:`~src.generator.config.ENTITY_TYPES`.
        home_city: Where the entity is normally located.
        secondary_cities: Legitimate alternate locations — a second office, a home
            town, a customer site. Visited occasionally at *plausible* travel speeds.
        primary_device: The ``"<os> / fw <firmware> / <MAC>"`` fingerprint the entity
            normally presents.
        replacement_device: A second, equally legitimate fingerprint adopted partway
            through the window (hardware refresh), or ``None`` if the entity never
            changes device.
        device_change_day: Day index on which :attr:`replacement_device` takes over.
        first_day: First day index (0-based) on which the entity may emit events.
        last_day: Last day index on which the entity may emit events.
        is_cold_start: ``True`` if the entity is a deliberately history-poor case.
        max_events: Hard cap on the entity's normal-event count, or ``None`` for
            uncapped. Only cold-start entities are capped.
        rate_multiplier: Entity-specific multiplier on the type-level daily event rate,
            so activity volume varies realistically across the roster.
        _ip_by_city: Lazily-populated map from city name to the IPv4 address this entity
            egresses from when located there. Deterministic per entity.
    """

    entity_id: str
    entity_type: str
    home_city: City
    secondary_cities: tuple[City, ...]
    primary_device: str
    replacement_device: str | None
    device_change_day: int
    first_day: int
    last_day: int
    is_cold_start: bool
    max_events: int | None
    rate_multiplier: float
    _ip_by_city: dict[str, str] = field(default_factory=dict, repr=False)

    def device_on_day(self, day_index: int) -> str:
        """Return the fingerprint this entity legitimately presents on a given day.

        Args:
            day_index: 0-based day offset from the start of the window.

        Returns:
            The active device fingerprint string.
        """
        if self.replacement_device is not None and day_index >= self.device_change_day:
            return self.replacement_device
        return self.primary_device

    def ip_for_city(self, city: City, rng: np.random.Generator, faker: "Faker") -> str:
        """Return this entity's egress IP when acting from ``city``.

        The mapping is memoised, so an entity always presents the same address from the
        same location — which is what makes a *changed* address informative.

        Args:
            city: The location the entity is acting from.
            rng: Random generator, used only on first allocation for a city.
            faker: Faker instance, used only on first allocation for a city.

        Returns:
            An IPv4 address string.
        """
        cached = self._ip_by_city.get(city.name)
        if cached is not None:
            return cached

        if city.name in OFFICE_CITIES:
            # Corporate site: RFC1918 space, one /16 per office.
            office_index = OFFICE_CITIES.index(city.name)
            ip = (
                f"10.{office_index + 10}."
                f"{int(rng.integers(0, 256))}.{int(rng.integers(1, 255))}"
            )
        else:
            ip = faker.ipv4_public()
        self._ip_by_city[city.name] = ip
        return ip

    def all_legitimate_cities(self) -> tuple[City, ...]:
        """Return every location this entity may legitimately appear from."""
        return (self.home_city, *self.secondary_cities)


@dataclass(frozen=True, slots=True)
class SharedGateway:
    """A corporate VPN / NAT egress point that many entities route through.

    Gateways carry a location because that is how IP geolocation actually behaves: a
    session tunnelled through the London POP resolves to London regardless of where the
    operator is physically sitting. This is the single largest source of false
    impossible-travel alerts in real security operations, and modelling it is what stops
    geo-velocity from being a clean separator in this dataset.

    Attributes:
        ip: The shared egress address.
        city: Where the gateway geolocates to.
    """

    ip: str
    city: City


@dataclass(slots=True)
class EntityRoster:
    """The full population plus organisation-level shared infrastructure.

    Attributes:
        entities: Every entity in the simulation, cold-start ones included.
        shared_gateways: Egress points (corporate VPN / NAT) that *many* entities
            legitimately share. They apply false-positive pressure twice over: on
            ``credential_stuffing``, whose signature is "many identities, one address",
            and on ``impossible_travel``, because tunnelled sessions appear to teleport.
    """

    entities: list[Entity]
    shared_gateways: list[SharedGateway]

    @property
    def shared_gateway_ips(self) -> list[str]:
        """Return just the gateway addresses."""
        return [g.ip for g in self.shared_gateways]

    def by_id(self) -> dict[str, Entity]:
        """Return an ``entity_id -> Entity`` lookup."""
        return {e.entity_id: e for e in self.entities}

    def of_type(self, entity_type: str) -> list[Entity]:
        """Return every entity of a given type.

        Args:
            entity_type: One of :data:`~src.generator.config.ENTITY_TYPES`.

        Returns:
            The matching entities, in roster order.
        """
        return [e for e in self.entities if e.entity_type == entity_type]

    def established(self) -> list[Entity]:
        """Return only entities with a full history (i.e. excluding cold-start cases)."""
        return [e for e in self.entities if not e.is_cold_start]

    def cold_start(self) -> list[Entity]:
        """Return only the deliberately history-poor entities."""
        return [e for e in self.entities if e.is_cold_start]


def _make_device_fingerprint(
    entity_type: str, rng: np.random.Generator, faker: "Faker"
) -> str:
    """Compose one ``"<os> / fw <firmware> / <MAC>"`` fingerprint.

    Args:
        entity_type: Determines which hardware pool to draw from.
        rng: Random generator.
        faker: Faker instance, used for the MAC address.

    Returns:
        A rendered device fingerprint string.
    """
    pool = DEVICE_POOLS[entity_type]
    os_name, firmware = pool[int(rng.integers(0, len(pool)))]
    return f"{os_name} / fw {firmware} / {faker.mac_address().upper()}"


def _allocate_type_counts(cfg: GeneratorConfig) -> dict[str, int]:
    """Split the roster across entity types according to the configured mix.

    Uses largest-remainder allocation so the counts sum exactly to
    :attr:`GeneratorConfig.n_entities` regardless of rounding.

    Args:
        cfg: The run configuration.

    Returns:
        Mapping of entity type to entity count.
    """
    exact = {et: cfg.entity_type_mix[et] * cfg.n_entities for et in ENTITY_TYPES}
    counts = {et: int(v) for et, v in exact.items()}
    remainder = cfg.n_entities - sum(counts.values())
    # Hand the leftover slots to the types with the largest fractional part.
    order = sorted(ENTITY_TYPES, key=lambda et: exact[et] - counts[et], reverse=True)
    for i in range(remainder):
        counts[order[i % len(order)]] += 1
    return counts


def _pick_secondary_cities(
    home: City, cfg: GeneratorConfig, rng: np.random.Generator
) -> tuple[City, ...]:
    """Choose the legitimate alternate locations for an entity.

    Same-region destinations are strongly favoured, because that is how real travel
    works — and because it means a benign trip often produces a *moderate* geo jump
    rather than an obviously-fine local one. That overlap is deliberate.

    Args:
        home: The entity's home city.
        cfg: The run configuration.
        rng: Random generator.

    Returns:
        Zero to three alternate cities. Empty for entities that never travel.
    """
    if rng.random() > cfg.benign_travel_prob:
        return ()

    candidates = [c for c in CITIES if c.name != home.name]
    weights = np.array(
        [3.0 if c.region == home.region else 1.0 for c in candidates], dtype=float
    )
    weights /= weights.sum()
    n_secondary = int(rng.integers(1, 4))
    chosen_idx = rng.choice(len(candidates), size=n_secondary, replace=False, p=weights)
    return tuple(candidates[int(i)] for i in chosen_idx)


def build_entities(
    cfg: GeneratorConfig, rng: np.random.Generator, faker: "Faker"
) -> EntityRoster:
    """Construct the full entity roster for a run.

    Args:
        cfg: The run configuration.
        rng: Seeded random generator. Consumed in a fixed order, so the roster is
            reproducible for a given seed.
        faker: Seeded Faker instance, used for MAC and public-IP allocation.

    Returns:
        The populated :class:`EntityRoster`.
    """
    # Gateways sit at corporate sites, spread across the office estate so that some
    # entities inevitably end up tunnelling through a different continent.
    shared_gateways = [
        SharedGateway(
            ip=faker.ipv4_public(),
            city=_OFFICE_CITY_OBJECTS[i % len(_OFFICE_CITY_OBJECTS)],
        )
        for i in range(cfg.shared_gateway_count)
    ]

    counts = _allocate_type_counts(cfg)
    n_cold = int(round(cfg.n_entities * cfg.cold_start_fraction))

    entities: list[Entity] = []
    serial = 0
    for entity_type in ENTITY_TYPES:
        for _ in range(counts[entity_type]):
            serial += 1
            entity_id = f"{_ID_PREFIX[entity_type]}-{serial:04d}"

            # Edge devices and service accounts are pinned to corporate sites; humans
            # may live anywhere.
            if entity_type == "user":
                home = CITIES[int(rng.integers(0, len(CITIES)))]
            else:
                home = _OFFICE_CITY_OBJECTS[
                    int(rng.integers(0, len(_OFFICE_CITY_OBJECTS)))
                ]

            # Only humans travel. A service account "travelling" would be a real signal.
            secondary = (
                _pick_secondary_cities(home, cfg, rng) if entity_type == "user" else ()
            )

            primary_device = _make_device_fingerprint(entity_type, rng, faker)
            if rng.random() < cfg.benign_device_change_prob:
                replacement_device = _make_device_fingerprint(entity_type, rng, faker)
                # Refresh lands in the middle stretch so there is history either side.
                device_change_day = int(rng.integers(cfg.days // 4, (3 * cfg.days) // 4))
            else:
                replacement_device = None
                device_change_day = cfg.days + 1

            # Log-normal spread: most entities near the type mean, a few much busier.
            rate_multiplier = float(np.clip(rng.lognormal(mean=-0.12, sigma=0.5), 0.25, 4.0))

            entities.append(
                Entity(
                    entity_id=entity_id,
                    entity_type=entity_type,
                    home_city=home,
                    secondary_cities=secondary,
                    primary_device=primary_device,
                    replacement_device=replacement_device,
                    device_change_day=device_change_day,
                    first_day=0,
                    last_day=cfg.days - 1,
                    is_cold_start=False,
                    max_events=None,
                    rate_multiplier=rate_multiplier,
                )
            )

    # Promote a random slice to cold-start: late arrival, tiny history.
    if n_cold > 0:
        cold_idx = rng.choice(len(entities), size=n_cold, replace=False)
        for i in cold_idx:
            entity = entities[int(i)]
            entity.is_cold_start = True
            entity.first_day = max(0, cfg.days - cfg.cold_start_window_days)
            entity.last_day = cfg.days - 1
            entity.max_events = int(rng.integers(1, cfg.cold_start_max_events + 1))
            # A brand-new identity has no travel history and no hardware refresh.
            entity.secondary_cities = ()
            entity.replacement_device = None
            entity.device_change_day = cfg.days + 1

    return EntityRoster(entities=entities, shared_gateways=shared_gateways)
