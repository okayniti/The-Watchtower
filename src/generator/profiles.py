"""Per-entity behavioural profiles and normal-traffic generation.

Where :mod:`src.generator.entities` fixes *who* an entity is, this module fixes *how it
behaves*: the hours it is active, the resources it habitually touches, how it
authenticates, and how long its sessions run. Every entity draws its own parameters, so
no two entities have the same footprint — which is the whole premise of per-entity
anomaly detection.

Realism budget
--------------
A generator that emits only clean, habitual traffic produces a trivially separable
dataset: any deviation is an attack, and a naive threshold scores 100%. To avoid that,
normal traffic here deliberately includes rare-but-legitimate behaviour, each item
chosen to collide with a specific attack signature:

============================  =========================================================
Benign behaviour              Attack it creates false-positive pressure on
============================  =========================================================
Business travel               ``impossible_travel`` — real geo jumps, plausible speed
Hardware refresh              ``device_spoofing`` — a genuinely new fingerprint
Shared VPN/NAT egress         ``credential_stuffing`` — many identities, one address
   ...and its apparent geo     ``impossible_travel`` — benign teleportation
Curiosity / one-off access    ``lateral_movement`` — genuinely novel resources
Mistyped password             ``brute_force`` — real ``AUTH_FAIL`` tokens in the stream
Late-night incident work      any hour-of-day rule
============================  =========================================================

Travel is modelled with an explicit transit constraint: after a location change, the
entity emits nothing until enough time has passed to cover the distance at commercial
flight speed. That is what makes ``impossible_travel`` detectable *and* keeps benign
trips from firing — the two differ only in velocity, not in the fact of moving.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from .config import (
    AUTH_FAIL,
    AUTH_METHODS,
    AUTH_OK,
    BENIGN_LABEL,
    COMMAND_VOCAB,
    DEPARTMENT_ADJACENCY,
    DEPARTMENTS_BY_ENTITY_TYPE,
    RECON_COMMANDS,
    RESOURCES,
    AccessEvent,
    City,
    GeneratorConfig,
    Resource,
)
from .entities import Entity, EntityRoster, SharedGateway

if TYPE_CHECKING:  # pragma: no cover - typing only
    from faker import Faker

#: Cruise speed used for the transit constraint on benign travel, km/h. Chosen just
#: below real commercial cruise (~900 km/h) so that legitimate trips land *near* the
#: physical limit — the ambiguity is the point.
BENIGN_TRAVEL_SPEED_KMH: float = 780.0

#: Fixed overhead added to every trip (check-in, transfers, ground transport), hours.
TRAVEL_OVERHEAD_HOURS: float = 3.0


@dataclass(slots=True)
class TravelWindow:
    """A legitimate trip: the entity acts from ``city`` for a contiguous day range.

    Attributes:
        city: Destination the entity works from during the window.
        start_day: First day index of the trip, inclusive.
        end_day: Last day index of the trip, inclusive.
    """

    city: City
    start_day: int
    end_day: int

    def covers(self, day_index: int) -> bool:
        """Return whether ``day_index`` falls inside this trip."""
        return self.start_day <= day_index <= self.end_day


@dataclass(slots=True)
class BehaviouralProfile:
    """The stable behavioural signature of one entity.

    This is generator-side ground truth. It is written to ``data/entity_profiles.json``
    for analysis and for the report only — **it is never a model input**. The whole
    point of Deliverable 2 is to *recover* an approximation of this from the log alone.

    Attributes:
        entity_id: The entity this profile belongs to.
        hour_weights: Length-24 probability vector over hour-of-day. Entity-specific,
            never uniform for humans.
        habitual_resources: Resources the entity routinely touches.
        resource_weights: Access probabilities aligned with :attr:`habitual_resources`.
        home_department: The department most of the entity's habitual resources sit in.
        auth_primary: The entity's preferred authentication method.
        auth_weights: Probability vector over :data:`~src.generator.config.AUTH_METHODS`.
        duration_mu: ``mu`` of the log-normal session-duration distribution.
        duration_sigma: ``sigma`` of the log-normal session-duration distribution.
        command_rate: Poisson rate for the number of commands issued after auth.
        recon_affinity: Per-event probability of issuing a reconnaissance-flavoured
            command. Non-zero for some legitimate engineers by design.
        gateway: The corporate VPN/NAT egress this entity habitually tunnels through,
            or ``None`` if it always egresses locally. Entities keep one gateway rather
            than picking randomly, because real VPN clients pin to a POP. Around one in
            six gateway users is assigned an out-of-region POP — a contractor routed
            through headquarters — and those are the hardest benign geo-velocity cases
            in the dataset.
        travel: Legitimate trips taken during the window.
        drifted_resources: Resources added to the habitual set by an ``insider_drift``
            episode. Populated by :mod:`src.generator.attacks`, empty otherwise.
    """

    entity_id: str
    hour_weights: np.ndarray
    habitual_resources: list[Resource]
    resource_weights: np.ndarray
    home_department: str
    auth_primary: str
    auth_weights: np.ndarray
    duration_mu: float
    duration_sigma: float
    command_rate: float
    recon_affinity: float
    gateway: SharedGateway | None = None
    travel: list[TravelWindow] = field(default_factory=list)
    drifted_resources: list[Resource] = field(default_factory=list)

    def city_on_day(self, entity: Entity, day_index: int) -> City:
        """Return where the entity is legitimately located on a given day.

        Args:
            entity: The entity being placed.
            day_index: 0-based day offset from the start of the window.

        Returns:
            The entity's home city, or a trip destination if a trip covers that day.
        """
        for window in self.travel:
            if window.covers(day_index):
                return window.city
        return entity.home_city

    def sample_resource(self, rng: np.random.Generator) -> Resource:
        """Draw a resource from the entity's habitual set.

        Args:
            rng: Random generator.

        Returns:
            One habitual resource, weighted by how routine it is for this entity.
        """
        idx = rng.choice(len(self.habitual_resources), p=self.resource_weights)
        return self.habitual_resources[int(idx)]

    def to_record(self) -> dict[str, object]:
        """Serialise the profile to a JSON-friendly dict for the ground-truth dump."""
        return {
            "entity_id": self.entity_id,
            "home_department": self.home_department,
            "auth_primary": self.auth_primary,
            "peak_hour": int(np.argmax(self.hour_weights)),
            "hour_weights": [round(float(w), 5) for w in self.hour_weights],
            "habitual_resources": [r.path for r in self.habitual_resources],
            "median_session_seconds": round(float(np.exp(self.duration_mu)), 1),
            "command_rate": round(self.command_rate, 2),
            "gateway": None if self.gateway is None else {
                "ip": self.gateway.ip,
                "city": self.gateway.city.name,
            },
            "travel": [
                {"city": w.city.name, "start_day": w.start_day, "end_day": w.end_day}
                for w in self.travel
            ],
            "drifted_resources": [r.path for r in self.drifted_resources],
        }


# --------------------------------------------------------------------------------------
# Profile construction
# --------------------------------------------------------------------------------------


def _circular_bump(centre: float, width: float) -> np.ndarray:
    """Return an unnormalised 24-vector with a Gaussian bump wrapped around the clock.

    Args:
        centre: Peak hour, may be fractional and may exceed 24 (it wraps).
        width: Standard deviation of the bump, in hours.

    Returns:
        A length-24 array of non-negative weights.
    """
    hours = np.arange(24, dtype=float)
    delta = np.abs(hours - centre)
    circular_delta = np.minimum(delta, 24.0 - delta)
    return np.exp(-0.5 * (circular_delta / width) ** 2)


def _build_hour_weights(entity_type: str, rng: np.random.Generator) -> np.ndarray:
    """Build an entity-specific hour-of-day distribution.

    Humans get a shifted working-hours bump, sometimes with an evening second shift.
    Service accounts are either cron-spiky or close to flat. Edge devices run all day
    with a mild diurnal ripple. None of the three is uniform, so "unusual hour" is a
    genuinely per-entity notion rather than a global one.

    Args:
        entity_type: One of :data:`~src.generator.config.ENTITY_TYPES`.
        rng: Random generator.

    Returns:
        A length-24 probability vector summing to 1.
    """
    if entity_type == "user":
        centre = float(rng.normal(10.5, 1.8))
        width = float(rng.uniform(1.8, 3.2))
        weights = _circular_bump(centre, width)
        if rng.random() < 0.30:  # evening second shift
            weights += rng.uniform(0.3, 0.7) * _circular_bump(
                centre + float(rng.uniform(7.0, 11.0)), float(rng.uniform(1.2, 2.2))
            )
        floor = 0.02
    elif entity_type == "service_account":
        if rng.random() < 0.5:  # cron-driven: a few sharp spikes
            weights = np.full(24, 0.05)
            for _ in range(int(rng.integers(2, 5))):
                weights += _circular_bump(float(rng.integers(0, 24)), 0.7)
        else:  # steady polling
            weights = np.full(24, 1.0) + 0.25 * _circular_bump(
                float(rng.integers(0, 24)), 4.0
            )
        floor = 0.05
    else:  # edge_device
        weights = np.full(24, 1.0) + 0.4 * _circular_bump(float(rng.normal(13, 4)), 5.0)
        floor = 0.15

    weights = weights + floor * weights.max()
    return weights / weights.sum()


def _pick_habitual_resources(
    entity: Entity, rng: np.random.Generator
) -> tuple[list[Resource], np.ndarray, str]:
    """Choose the resource set an entity routinely touches.

    The set is anchored in one department, with a minority of accesses reaching into an
    adjacent one — real people do have legitimate cross-team access, and that overlap is
    what makes ``lateral_movement`` non-trivial.

    Args:
        entity: The entity being profiled.
        rng: Random generator.

    Returns:
        A ``(resources, weights, home_department)`` tuple, where ``weights`` is a
        probability vector aligned with ``resources``.
    """
    departments = DEPARTMENTS_BY_ENTITY_TYPE[entity.entity_type]
    home_department = departments[int(rng.integers(0, len(departments)))]

    in_dept = [r for r in RESOURCES if r.department == home_department]
    adjacent = [
        r
        for r in RESOURCES
        if r.department in DEPARTMENT_ADJACENCY.get(home_department, ())
    ]

    n_core = int(min(len(in_dept), rng.integers(2, min(6, len(in_dept) + 1))))
    core_idx = rng.choice(len(in_dept), size=n_core, replace=False)
    chosen = [in_dept[int(i)] for i in core_idx]

    if adjacent and rng.random() < 0.6:
        n_adj = int(min(len(adjacent), rng.integers(1, 3)))
        adj_idx = rng.choice(len(adjacent), size=n_adj, replace=False)
        chosen.extend(adjacent[int(i)] for i in adj_idx)

    # Heavy-tailed access weights: one or two resources dominate, the rest are rare.
    alpha = np.linspace(2.5, 0.4, num=len(chosen))
    weights = rng.dirichlet(alpha)
    return chosen, weights, home_department


def _build_travel_windows(
    entity: Entity, cfg: GeneratorConfig, rng: np.random.Generator
) -> list[TravelWindow]:
    """Lay out non-overlapping legitimate trips across the window.

    Args:
        entity: The entity being profiled. Only entities with secondary cities travel.
        cfg: The run configuration.
        rng: Random generator.

    Returns:
        Chronologically ordered, non-overlapping trips. May be empty.
    """
    if not entity.secondary_cities:
        return []

    windows: list[TravelWindow] = []
    cursor = int(rng.integers(2, max(3, cfg.days // 3)))
    for city in entity.secondary_cities:
        if rng.random() > 0.7:  # not every listed city is actually visited
            continue
        length = int(rng.integers(2, 7))
        if cursor + length >= cfg.days - 1:
            break
        windows.append(TravelWindow(city=city, start_day=cursor, end_day=cursor + length))
        # Leave a gap of at least a few days before the next trip.
        cursor += length + int(rng.integers(3, 9))
    return windows


def build_profiles(
    roster: EntityRoster, cfg: GeneratorConfig, rng: np.random.Generator
) -> dict[str, BehaviouralProfile]:
    """Build a behavioural profile for every entity in the roster.

    Args:
        roster: The entity roster.
        cfg: The run configuration.
        rng: Seeded random generator, consumed in roster order for reproducibility.

    Returns:
        Mapping of ``entity_id`` to its :class:`BehaviouralProfile`.
    """
    profiles: dict[str, BehaviouralProfile] = {}
    for entity in roster.entities:
        resources, weights, home_department = _pick_habitual_resources(entity, rng)

        auth_weights = rng.dirichlet(_auth_alpha(entity.entity_type, rng))
        auth_primary = AUTH_METHODS[int(np.argmax(auth_weights))]

        if entity.entity_type == "user":
            duration_mu, duration_sigma = float(rng.normal(6.1, 0.45)), float(
                rng.uniform(0.55, 0.95)
            )
            command_rate = float(rng.uniform(3.0, 9.0))
            recon_affinity = float(rng.uniform(0.0, 0.03))
        elif entity.entity_type == "service_account":
            duration_mu, duration_sigma = float(rng.normal(4.4, 0.4)), float(
                rng.uniform(0.3, 0.6)
            )
            command_rate = float(rng.uniform(2.0, 5.0))
            recon_affinity = float(rng.uniform(0.0, 0.01))
        else:  # edge_device
            duration_mu, duration_sigma = float(rng.normal(3.5, 0.35)), float(
                rng.uniform(0.25, 0.5)
            )
            command_rate = float(rng.uniform(1.5, 3.5))
            recon_affinity = 0.0

        profiles[entity.entity_id] = BehaviouralProfile(
            entity_id=entity.entity_id,
            gateway=_assign_gateway(entity, roster, rng),
            hour_weights=_build_hour_weights(entity.entity_type, rng),
            habitual_resources=resources,
            resource_weights=weights,
            home_department=home_department,
            auth_primary=auth_primary,
            auth_weights=auth_weights,
            duration_mu=duration_mu,
            duration_sigma=duration_sigma,
            command_rate=command_rate,
            recon_affinity=recon_affinity,
            travel=_build_travel_windows(entity, cfg, rng),
        )
    return profiles


def _assign_gateway(
    entity: Entity, roster: EntityRoster, rng: np.random.Generator
) -> SharedGateway | None:
    """Pin an entity to the corporate VPN/NAT egress it habitually tunnels through.

    Roughly 45% of non-device entities use a gateway at all. Of those, most are routed
    to a POP in their own region, but about one in six is deliberately assigned an
    out-of-region gateway. Those entities appear to teleport between continents at
    implausible speed while doing nothing whatsoever wrong — which is exactly the false
    positive an ``impossible_travel`` detector has to survive.

    Args:
        entity: The entity being profiled.
        roster: Supplies the gateway pool.
        rng: Random generator.

    Returns:
        The entity's habitual gateway, or ``None`` if it always egresses locally.
    """
    if entity.entity_type == "edge_device" or not roster.shared_gateways:
        return None  # edge devices sit on the site network and do not tunnel
    if rng.random() > 0.45:
        return None

    same_region = [
        g for g in roster.shared_gateways if g.city.region == entity.home_city.region
    ]
    if same_region and rng.random() < 0.84:
        return same_region[int(rng.integers(0, len(same_region)))]
    return roster.shared_gateways[int(rng.integers(0, len(roster.shared_gateways)))]


def _auth_alpha(entity_type: str, rng: np.random.Generator) -> np.ndarray:
    """Return Dirichlet concentration parameters over auth methods for an entity type.

    Humans mostly use passwords or biometrics; service accounts use tokens and
    certificates; edge devices are certificate-first. One method dominates per entity,
    with a realistic fallback tail.

    Args:
        entity_type: One of :data:`~src.generator.config.ENTITY_TYPES`.
        rng: Random generator, used to pick which method dominates.

    Returns:
        A length-4 concentration vector aligned with
        :data:`~src.generator.config.AUTH_METHODS`.
    """
    base = {
        "user": np.array([6.0, 1.2, 0.4, 2.0]),
        "service_account": np.array([0.3, 7.0, 4.0, 0.05]),
        "edge_device": np.array([0.2, 2.0, 7.0, 0.05]),
    }[entity_type]
    # Jitter so entities of the same type still differ from one another.
    return base * rng.uniform(0.7, 1.4, size=len(AUTH_METHODS))


# --------------------------------------------------------------------------------------
# Normal-traffic generation
# --------------------------------------------------------------------------------------


def _sample_commands(
    resource: Resource,
    profile: BehaviouralProfile,
    cfg: GeneratorConfig,
    rng: np.random.Generator,
) -> list[str]:
    """Compose a plausible command sequence for one normal session.

    The sequence opens with auth-outcome tokens (see the note in
    :mod:`src.generator.config` on why outcome lives here), then draws commands from the
    resource's family vocabulary.

    Args:
        resource: The resource being accessed, which selects the command vocabulary.
        profile: The acting entity's profile.
        cfg: The run configuration.
        rng: Random generator.

    Returns:
        The ordered command list.
    """
    commands: list[str] = []
    if rng.random() < cfg.benign_auth_fail_prob:
        # A mistyped credential. One or two failures, then success — this is what stops
        # "contains AUTH_FAIL" from separating brute force for free.
        commands.extend([AUTH_FAIL] * int(rng.integers(1, 3)))
    commands.append(AUTH_OK)

    vocab = COMMAND_VOCAB[resource.family]
    n_commands = 1 + int(rng.poisson(profile.command_rate))
    idx = rng.integers(0, len(vocab), size=n_commands)
    commands.extend(vocab[int(i)] for i in idx)

    if rng.random() < profile.recon_affinity:
        # A legitimately curious engineer. Rare, but it happens.
        commands.append(RECON_COMMANDS[int(rng.integers(0, len(RECON_COMMANDS)))])
    return commands


def _sample_hour(
    profile: BehaviouralProfile, cfg: GeneratorConfig, rng: np.random.Generator
) -> int:
    """Draw an hour-of-day for one event.

    Args:
        profile: The acting entity's profile.
        cfg: The run configuration.
        rng: Random generator.

    Returns:
        An hour in ``[0, 23]``. Usually from the entity's habitual distribution, but
        occasionally from its *least* likely hours (late-night incident work).
    """
    if rng.random() < cfg.off_hours_prob:
        inverted = profile.hour_weights.max() - profile.hour_weights + 1e-6
        inverted /= inverted.sum()
        return int(rng.choice(24, p=inverted))
    return int(rng.choice(24, p=profile.hour_weights))


def _daily_event_count(
    entity: Entity,
    profile: BehaviouralProfile,
    day: dt.date,
    cfg: GeneratorConfig,
    rng: np.random.Generator,
) -> int:
    """Draw how many events an entity emits on one day.

    Args:
        entity: The acting entity.
        profile: Its behavioural profile (unused today, kept for signature symmetry
            with the other samplers).
        day: The calendar date, used for the weekday effect.
        cfg: The run configuration.
        rng: Random generator.

    Returns:
        A non-negative event count.
    """
    rate = cfg.events_per_day[entity.entity_type] * entity.rate_multiplier
    if entity.entity_type == "user" and day.weekday() >= 5:
        rate *= cfg.weekend_activity_factor
    return int(rng.poisson(rate))


def _pick_egress(
    entity: Entity,
    profile: BehaviouralProfile,
    city: City,
    cfg: GeneratorConfig,
    rng: np.random.Generator,
    faker: "Faker",
) -> tuple[str, City]:
    """Choose the egress address for one normal session, and where it geolocates to.

    When the session tunnels through the entity's gateway, the *apparent* location
    becomes the gateway's city while the entity stays physically put. That decoupling of
    apparent from physical location is deliberate: it is the mechanism behind both the
    "many identities, one IP" pattern that ``credential_stuffing`` relies on, and the
    benign teleportation that makes ``impossible_travel`` hard.

    Args:
        entity: The acting entity.
        profile: Its behavioural profile, which holds the pinned gateway.
        city: Where the entity is physically located for this event.
        cfg: The run configuration.
        rng: Random generator.
        faker: Faker instance, for first-time IP allocation.

    Returns:
        An ``(ip, apparent_city)`` tuple.
    """
    if profile.gateway is not None and rng.random() < cfg.shared_gateway_prob:
        return profile.gateway.ip, profile.gateway.city
    return entity.ip_for_city(city, rng, faker), city


def generate_normal_events(
    roster: EntityRoster,
    profiles: dict[str, BehaviouralProfile],
    cfg: GeneratorConfig,
    rng: np.random.Generator,
    faker: "Faker",
) -> list[AccessEvent]:
    """Generate the benign background traffic for the whole window.

    Events are produced per entity, day by day, so that the travel transit constraint
    can be enforced against the entity's own event history. The caller is responsible
    for the final global sort by timestamp.

    Args:
        roster: The entity roster.
        profiles: Behavioural profiles keyed by ``entity_id``.
        cfg: The run configuration.
        rng: Seeded random generator.
        faker: Seeded Faker instance.

    Returns:
        All benign events, labelled :data:`~src.generator.config.BENIGN_LABEL`.
        Unsorted across entities; sorted within each entity.
    """
    from .config import haversine_km  # local import keeps the module header tidy

    start = dt.date.fromisoformat(cfg.start_date)
    events: list[AccessEvent] = []

    for entity in roster.entities:
        profile = profiles[entity.entity_id]
        emitted = 0
        last_timestamp: dt.datetime | None = None
        last_city: City = entity.home_city

        for day_index in range(entity.first_day, entity.last_day + 1):
            if entity.max_events is not None and emitted >= entity.max_events:
                break

            day = start + dt.timedelta(days=day_index)
            city = profile.city_on_day(entity, day_index)

            # Enforce physically plausible transit after a location change.
            not_before: dt.datetime | None = None
            if city.name != last_city.name and last_timestamp is not None:
                distance = haversine_km(
                    last_city.lat, last_city.lon, city.lat, city.lon
                )
                transit_hours = distance / BENIGN_TRAVEL_SPEED_KMH + TRAVEL_OVERHEAD_HOURS
                not_before = last_timestamp + dt.timedelta(hours=transit_hours)

            n_events = _daily_event_count(entity, profile, day, cfg, rng)
            if entity.max_events is not None:
                n_events = min(n_events, entity.max_events - emitted)
            if n_events <= 0:
                continue

            timestamps: list[dt.datetime] = []
            for _ in range(n_events):
                hour = _sample_hour(profile, cfg, rng)
                timestamps.append(
                    dt.datetime.combine(day, dt.time(hour=hour))
                    + dt.timedelta(
                        minutes=int(rng.integers(0, 60)),
                        seconds=int(rng.integers(0, 60)),
                    )
                )
            timestamps.sort()

            for timestamp in timestamps:
                if not_before is not None and timestamp < not_before:
                    continue  # still in transit — the entity cannot act yet
                events.append(
                    _make_normal_event(entity, profile, timestamp, city, cfg, rng, faker)
                )
                last_timestamp = timestamp
                last_city = city
                emitted += 1
                if entity.max_events is not None and emitted >= entity.max_events:
                    break

    return events


def _make_normal_event(
    entity: Entity,
    profile: BehaviouralProfile,
    timestamp: dt.datetime,
    city: City,
    cfg: GeneratorConfig,
    rng: np.random.Generator,
    faker: "Faker",
) -> AccessEvent:
    """Materialise one benign event from an entity's profile.

    Args:
        entity: The acting entity.
        profile: Its behavioural profile.
        timestamp: When the entity acts.
        city: Where the entity is physically located.
        cfg: The run configuration.
        rng: Random generator.
        faker: Faker instance.

    Returns:
        A fully populated benign :class:`~src.generator.config.AccessEvent`.
    """
    day_index = (timestamp.date() - dt.date.fromisoformat(cfg.start_date)).days

    if rng.random() < cfg.benign_novel_resource_prob:
        # Legitimate one-off access outside the habitual set: a stand-in for a
        # colleague, an on-call escalation, an audit. Indistinguishable in isolation
        # from the first step of lateral movement.
        resource = RESOURCES[int(rng.integers(0, len(RESOURCES)))]
    else:
        resource = profile.sample_resource(rng)

    auth_method = AUTH_METHODS[int(rng.choice(len(AUTH_METHODS), p=profile.auth_weights))]
    duration = float(rng.lognormal(profile.duration_mu, profile.duration_sigma))
    source_ip, apparent_city = _pick_egress(entity, profile, city, cfg, rng, faker)

    return AccessEvent(
        entity_id=entity.entity_id,
        entity_type=entity.entity_type,
        timestamp=timestamp,
        source_ip=source_ip,
        geo_location=apparent_city.render(),
        resource_accessed=resource.path,
        auth_method=auth_method,
        session_duration=round(min(duration, 86_400.0), 2),
        command_sequence=_sample_commands(resource, profile, cfg, rng),
        device_fingerprint=entity.device_on_day(day_index),
        label=BENIGN_LABEL,
        episode_id="",
    )
