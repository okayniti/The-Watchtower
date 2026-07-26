"""Attack injectors and their behavioural assumptions.

This docstring is the **injected attack taxonomy** required by Deliverable 1 and is
reproduced verbatim in the report. Each entry states the behavioural assumption the
injector encodes, the fields it perturbs, and — critically — the *overlap*: the
legitimate behaviour it is designed to be confusable with. If an injector had no
documented overlap, it would be trivially separable, the headline precision number would
be meaningless, and the dataset would not be worth modelling.

A shared principle runs through all seven: **an attack perturbs as few fields as it
plausibly can.** Real intrusions reuse valid credentials, valid devices and valid
network paths wherever possible. An injector that made every field simultaneously
strange would produce a dataset any threshold could solve.

--------------------------------------------------------------------------------------
1. ``brute_force``
--------------------------------------------------------------------------------------
*Assumption.* An adversary with no valid credential repeatedly guesses against one
identity on an auth-bearing surface (VPN, mail, IAM). The signature is **rate**: many
authentication attempts against a single ``entity_id`` inside a short window, dominated
by failures, with very short sessions.

*Perturbs.* ``command_sequence`` (runs of ``AUTH_FAIL``), ``session_duration`` (short),
inter-event gap (seconds to a couple of minutes), sometimes ``source_ip`` and
``geo_location``.

*Overlap.* Legitimate users mistype passwords — normal traffic carries ``AUTH_FAIL``
tokens at :attr:`~src.generator.config.GeneratorConfig.benign_auth_fail_prob`, so
"contains a failure" separates nothing. Bursts are sized from 6 attempts upward, and the
low end overlaps a frustrated user retrying after a password rotation. Roughly 40% of
campaigns originate from an address or city the victim genuinely uses (compromised
host / insider), so geography is not a reliable tell either. Only the *density* of
attempts per entity per minute is genuinely discriminative.

--------------------------------------------------------------------------------------
2. ``impossible_travel``
--------------------------------------------------------------------------------------
*Assumption.* A stolen credential is used from a second location while the legitimate
owner is still active elsewhere. The signature is **geo-velocity**: the implied speed
between consecutive events for one entity exceeds what physical travel allows.

*Perturbs.* ``geo_location`` and ``source_ip`` only. Resource, auth method, device and
session shape are all drawn from the victim's own profile — the attacker is using a
valid session.

*Overlap.* Two sources, and the second is the important one.

First, normal traffic contains real business travel (see :mod:`src.generator.profiles`),
which produces genuine multi-thousand-kilometre jumps at door-to-door speeds up to
~780 km/h. Injected velocities start just above that band, so the slowest injected cases
sit within measurement error of a real flight whose egress IP geolocated imprecisely.

Second — and this is what makes the class genuinely hard — around 45% of entities
habitually tunnel through a corporate VPN/NAT gateway, and a tunnelled session
geolocates to the *gateway's* city, not the operator's. An engineer in Bengaluru whose
traffic egresses via the London POP appears to cross Eurasia and back within minutes,
repeatedly, while doing nothing wrong. Roughly one in six gateway users is pinned to an
out-of-region POP specifically to manufacture these cases. This is the dominant source
of false impossible-travel alerts in real security operations, and it means raw
geo-velocity is *not* a sufficient statistic here: the detector has to learn which
apparent locations are routine for a given entity.

--------------------------------------------------------------------------------------
3. ``credential_stuffing``
--------------------------------------------------------------------------------------
*Assumption.* A breach dump is replayed against the estate: **many different
``entity_id`` values from one address and one device fingerprint**, in a tight window,
with a low success rate.

*Perturbs.* The cross-entity structure of the log rather than any single row. Individual
rows look like ordinary failed logins.

*Overlap.* The organisation runs shared VPN/NAT gateways
(:attr:`~src.generator.config.GeneratorConfig.shared_gateway_count`) through which many
legitimate entities egress every day — "many identities, one IP" is *normal* here. One
campaign in four is deliberately routed through one of those very gateways, so address
reputation cannot carry the decision. What remains discriminative is the joint pattern:
identical device fingerprint across identities, compressed timing, and failure ratio.

--------------------------------------------------------------------------------------
4. ``lateral_movement``
--------------------------------------------------------------------------------------
*Assumption.* An adversary already inside a valid session explores outward, touching
resources the entity has never touched, in an **expanding chain** that walks the
department adjacency graph toward higher sensitivity, interleaved with reconnaissance
commands.

*Perturbs.* ``resource_accessed`` (novel, in sequence) and ``command_sequence`` (recon
verbs). Deliberately **not** ``source_ip``, ``device_fingerprint`` or ``auth_method`` —
the session itself is legitimate.

*Overlap.* Normal traffic includes one-off access outside the habitual set at
:attr:`~src.generator.config.GeneratorConfig.benign_novel_resource_prob` (covering for a
colleague, on-call escalation, audit work), and some engineers have a non-zero
``recon_affinity`` because running ``iam.whoami`` is a normal thing to do. Any single
event in a lateral-movement chain is therefore indistinguishable from benign curiosity.
Only the ordered chain — novelty compounding across consecutive events — carries signal,
which is precisely why this class needs a sequence-aware model.

--------------------------------------------------------------------------------------
5. ``device_spoofing``
--------------------------------------------------------------------------------------
*Assumption.* An attacker replays a valid identity from hardware that is not the
entity's. Sophisticated spoofing clones the plausible parts, so the **OS family is
preserved** while firmware build and MAC differ.

*Perturbs.* ``device_fingerprint``, and often ``source_ip`` within a plausible region.

*Overlap.* This is the closest benign/malicious pair in the dataset. Legitimate hardware
refreshes occur at
:attr:`~src.generator.config.GeneratorConfig.benign_device_change_prob` and present
exactly the same surface signal: an ``entity_id`` appearing with a fingerprint never
seen before. The only real difference is *persistence* — a refresh becomes the entity's
new normal and continues indefinitely, whereas a spoof appears for a handful of events
and vanishes. A detector that flags "unseen device" alone will fire on every laptop
upgrade in the estate.

--------------------------------------------------------------------------------------
6. ``low_and_slow_exfiltration``
--------------------------------------------------------------------------------------
*Assumption.* A patient adversary drains a sensitive store beneath per-event alerting
thresholds: a small number of sessions per day, on the same one or two data resources,
at a **regular cadence**, sustained across 5-14 days.

*Perturbs.* Nothing, per event. Session durations are drawn from the *upper-normal*
range of the entity's own distribution — around the 70th-95th percentile, never beyond
it. Command counts stay inside the entity's normal envelope. Resources are ones the
entity can legitimately reach.

*Overlap.* Total, by construction. Every individual row is inside the entity's normal
operating range, so a per-event detector *cannot* catch this class no matter how it is
tuned — the point of including it. The only signal is aggregate: cadence regularity
(low variance in inter-event interval), cumulative session-time on high-sensitivity
resources, and a bulk-read command mix sustained over days. Legitimate scheduled
reporting jobs produce a similar cadence, which is the residual false-positive risk.

--------------------------------------------------------------------------------------
7. ``insider_drift`` — the ambiguous class
--------------------------------------------------------------------------------------
*Assumption.* A legitimate role change. Over 10-21 days an entity **gradually** adopts
resources from an adjacent department, ramping from occasional to routine, and then
**keeps using them permanently**. Nothing about this is an attack.

*Mechanic.* Unlike the six above, this injector does not add events — it *rewrites*
existing benign ones. During the ramp, an increasing fraction of the entity's sessions
are redirected to the newly-adopted resources and labelled ``insider_drift``. After the
ramp completes, the entity's remaining events continue to use the expanded set but are
labelled ``normal_baseline``. Event volume is unchanged, because a person changing role
does the same amount of work on different things.

*Why it is here.* It shares its entire surface signature with ``lateral_movement`` — the
same department-adjacency walk into novel, higher-sensitivity resources by an otherwise
valid session. The differences are that drift is *gradual* rather than bursty, and
*permanent* rather than transient.

*What it tests.* Two of the five hard requirements at once. It is the false-positive
control for the top-1% alert budget (``CLAUDE.md`` §5), and the post-ramp tail is the
concept-drift test (``CLAUDE.md`` §4.3): a profiler that does not adapt will keep
flagging this entity forever, which the metric ``time-to-unflag`` measures directly.
It should **not** be treated as a class to maximise recall on.

--------------------------------------------------------------------------------------
Budgeting
--------------------------------------------------------------------------------------
Each injector is handed a target event count derived from
:attr:`~src.generator.config.GeneratorConfig.attack_share` and generates whole episodes
until that budget is met, so per-class volume is controllable while episode sizes stay
realistically variable. The six hard classes sum to ~1.35% of events, inside the 0.5-3%
band of ``CLAUDE.md`` §4.2. ``insider_drift`` is budgeted separately and excluded from
that figure, because counting a benign role change as an anomaly would misstate the
imbalance the detector actually faces.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import numpy as np

from .config import (
    ADVERSARY_CITIES,
    AUTH_FAIL,
    AUTH_OK,
    BENIGN_LABEL,
    BULK_READ_COMMANDS,
    CITIES,
    COMMAND_VOCAB,
    DEPARTMENT_ADJACENCY,
    RECON_COMMANDS,
    RESOURCES,
    RESOURCES_BY_PATH,
    AccessEvent,
    City,
    GeneratorConfig,
    Resource,
    haversine_km,
    parse_geo,
)
from .entities import Entity, EntityRoster
from .profiles import BehaviouralProfile

if TYPE_CHECKING:  # pragma: no cover - typing only
    from faker import Faker

#: Auth-bearing surfaces an attacker can hammer without valid credentials.
_AUTH_SURFACES: tuple[str, ...] = (
    "vpn/gateway",
    "corp/mail",
    "admin/iam/directory",
    "corp/wiki",
)

#: Minimum implied speed (km/h) an injected ``impossible_travel`` pair must exceed.
#: Sits just above the benign travel ceiling so the slowest cases stay ambiguous.
_IMPOSSIBLE_SPEED_FLOOR_KMH: float = 950.0


@dataclass(slots=True)
class Episode:
    """Bookkeeping for one injected campaign.

    Attributes:
        episode_id: Identifier shared by every event in the campaign.
        label: The behaviour class this episode represents.
        entity_ids: Entities involved. Usually one; ``credential_stuffing`` spans many.
        start: Timestamp of the campaign's first event.
        end: Timestamp of the campaign's last event.
        n_events: Number of events carrying this episode's label.
        span_days: Calendar days the campaign covers, for the summary table.
        note: Short human-readable description, surfaced in the run summary.
    """

    episode_id: str
    label: str
    entity_ids: list[str]
    start: dt.datetime
    end: dt.datetime
    n_events: int
    span_days: float
    note: str


@dataclass(slots=True)
class InjectionResult:
    """What one injector produced.

    Attributes:
        events: Newly created events. Empty for ``insider_drift``, which rewrites
            existing events in place rather than adding new ones.
        episodes: Campaign records for the summary.
    """

    events: list[AccessEvent] = field(default_factory=list)
    episodes: list[Episode] = field(default_factory=list)


@dataclass(slots=True)
class InjectionContext:
    """Everything an injector needs to place a realistic campaign.

    Attributes:
        roster: The entity roster.
        profiles: Behavioural profiles keyed by ``entity_id``.
        events_by_entity: Existing benign events grouped by ``entity_id`` and sorted by
            time. Injectors anchor onto real activity rather than inventing timelines,
            which is what keeps injected events consistent with their victim's history.
        cfg: The run configuration.
        rng: Seeded random generator.
        faker: Seeded Faker instance.
        window_start: First datetime of the simulation window.
        window_end: Last datetime of the simulation window.
    """

    roster: EntityRoster
    profiles: dict[str, BehaviouralProfile]
    events_by_entity: dict[str, list[AccessEvent]]
    cfg: GeneratorConfig
    rng: np.random.Generator
    faker: "Faker"
    window_start: dt.datetime
    window_end: dt.datetime


# --------------------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------------------


def _victim_pool(ctx: InjectionContext, min_events: int = 12) -> list[Entity]:
    """Return entities with enough history to be a credible victim.

    Cold-start entities are excluded: an attack against an identity with three events
    would conflate the cold-start test with the detection test.

    Args:
        ctx: The injection context.
        min_events: Minimum benign events the entity must already have.

    Returns:
        Eligible entities, in roster order.
    """
    return [
        e
        for e in ctx.roster.established()
        if len(ctx.events_by_entity.get(e.entity_id, [])) >= min_events
    ]


def _pick(items: list, rng: np.random.Generator):
    """Return one uniformly random element of ``items``.

    Args:
        items: A non-empty sequence.
        rng: Random generator.

    Returns:
        The chosen element.
    """
    return items[int(rng.integers(0, len(items)))]


def _anchor_event(entity_id: str, ctx: InjectionContext) -> AccessEvent:
    """Pick one of an entity's existing benign events to anchor a campaign onto.

    Args:
        entity_id: The victim's identifier.
        ctx: The injection context.

    Returns:
        A randomly chosen benign event belonging to that entity.
    """
    history = ctx.events_by_entity[entity_id]
    return history[int(ctx.rng.integers(0, len(history)))]


def _adversary_ip(ctx: InjectionContext) -> str:
    """Return a fresh externally-routable address for an adversary."""
    return ctx.faker.ipv4_public()


def _adversary_city(ctx: InjectionContext) -> City:
    """Pick an origin city for an adversary.

    Two thirds of the time this is drawn from :data:`ADVERSARY_CITIES`; the rest come
    from the ordinary city pool, so "unusual country" never becomes a free separator.

    Args:
        ctx: The injection context.

    Returns:
        The chosen city.
    """
    if ctx.rng.random() < 0.66:
        return ADVERSARY_CITIES[int(ctx.rng.integers(0, len(ADVERSARY_CITIES)))]
    return CITIES[int(ctx.rng.integers(0, len(CITIES)))]


def _clone_fingerprint(fingerprint: str, ctx: InjectionContext) -> str:
    """Produce a spoofed fingerprint that preserves the OS family.

    A credible spoof gets the easy parts right. Only the firmware build and MAC differ,
    so the mismatch is subtle rather than a wholesale platform change.

    Args:
        fingerprint: The victim's genuine ``"<os> / fw <firmware> / <MAC>"`` string.
        ctx: The injection context.

    Returns:
        A modified fingerprint string.
    """
    os_part, fw_part, _mac_part = (p.strip() for p in fingerprint.split("/", 2))
    fw_version = fw_part.removeprefix("fw ").strip()
    pieces = fw_version.split(".")
    if pieces and pieces[-1].isdigit():
        pieces[-1] = str(int(pieces[-1]) + int(ctx.rng.integers(1, 40)))
    spoofed_fw = ".".join(pieces)
    return f"{os_part} / fw {spoofed_fw} / {ctx.faker.mac_address().upper()}"


def _upper_normal_duration(
    profile: BehaviouralProfile, ctx: InjectionContext, low: float = 0.70, high: float = 0.95
) -> float:
    """Draw a session duration from the upper tail of an entity's *normal* range.

    Used by ``low_and_slow_exfiltration``: long enough to move data, never long enough
    to leave the entity's own envelope.

    Args:
        profile: The entity's behavioural profile.
        ctx: The injection context.
        low: Lower quantile bound of the entity's own duration distribution.
        high: Upper quantile bound.

    Returns:
        A duration in seconds.
    """
    from math import erf, sqrt

    quantile = float(ctx.rng.uniform(low, high))
    # Invert the standard normal CDF by bisection — avoids a scipy dependency.
    lo, hi = -6.0, 6.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if 0.5 * (1 + erf(mid / sqrt(2))) < quantile:
            lo = mid
        else:
            hi = mid
    z = (lo + hi) / 2
    return round(float(np.exp(profile.duration_mu + profile.duration_sigma * z)), 2)


def _novel_resources_for(
    profile: BehaviouralProfile, count: int, ctx: InjectionContext
) -> list[Resource]:
    """Build an expansion chain of resources the entity has never touched.

    The chain walks the department adjacency graph outward from the entity's home
    department and is ordered by ascending sensitivity, mimicking an adversary (or a
    newly-promoted employee) working toward higher-value assets.

    Args:
        profile: The entity's behavioural profile.
        count: How many novel resources to return.
        ctx: The injection context.

    Returns:
        Up to ``count`` novel resources, ordered by ascending sensitivity.
    """
    habitual = {r.path for r in profile.habitual_resources}
    habitual.update(r.path for r in profile.drifted_resources)
    reachable = set(DEPARTMENT_ADJACENCY.get(profile.home_department, ()))
    reachable.add(profile.home_department)

    near = [r for r in RESOURCES if r.path not in habitual and r.department in reachable]
    far = [r for r in RESOURCES if r.path not in habitual and r.department not in reachable]
    ctx.rng.shuffle(near)
    ctx.rng.shuffle(far)

    chain = (near + far)[:count]
    chain.sort(key=lambda r: r.sensitivity)
    return chain


def _new_event(
    entity: Entity,
    timestamp: dt.datetime,
    source_ip: str,
    geo: str,
    resource: str,
    auth_method: str,
    duration: float,
    commands: list[str],
    fingerprint: str,
    label: str,
    episode_id: str,
) -> AccessEvent:
    """Construct an injected event. Thin wrapper for readability at call sites.

    Args:
        entity: The entity the event is attributed to.
        timestamp: Event time.
        source_ip: Egress address.
        geo: Rendered ``geo_location`` string.
        resource: Resource path touched.
        auth_method: Authentication method used.
        duration: Session duration in seconds.
        commands: Ordered command sequence.
        fingerprint: Device fingerprint string.
        label: Ground-truth behaviour class.
        episode_id: Campaign identifier.

    Returns:
        The assembled :class:`~src.generator.config.AccessEvent`.
    """
    return AccessEvent(
        entity_id=entity.entity_id,
        entity_type=entity.entity_type,
        timestamp=timestamp,
        source_ip=source_ip,
        geo_location=geo,
        resource_accessed=resource,
        auth_method=auth_method,
        session_duration=round(float(duration), 2),
        command_sequence=commands,
        device_fingerprint=fingerprint,
        label=label,
        episode_id=episode_id,
    )


def _episode_from(
    episode_id: str, label: str, events: list[AccessEvent], note: str
) -> Episode:
    """Summarise a set of injected events as an :class:`Episode` record.

    Args:
        episode_id: The campaign identifier.
        label: The behaviour class.
        events: The campaign's events. Must be non-empty.
        note: Human-readable description for the summary.

    Returns:
        The episode record.
    """
    times = sorted(e.timestamp for e in events)
    span = (times[-1] - times[0]).total_seconds() / 86_400.0
    return Episode(
        episode_id=episode_id,
        label=label,
        entity_ids=sorted({e.entity_id for e in events}),
        start=times[0],
        end=times[-1],
        n_events=len(events),
        span_days=round(span, 3),
        note=note,
    )


# --------------------------------------------------------------------------------------
# 1. brute_force
# --------------------------------------------------------------------------------------


def inject_brute_force(ctx: InjectionContext, budget: int) -> InjectionResult:
    """Inject password-guessing bursts against individual identities.

    See the module docstring, section 1, for the behavioural assumption and the
    documented overlap with legitimate mistyped credentials.

    Args:
        ctx: The injection context.
        budget: Target number of events to label ``brute_force``.

    Returns:
        The injected events and their episode records.
    """
    result = InjectionResult()
    pool = _victim_pool(ctx)
    if not pool:
        return result

    episode_index = 0
    while sum(e.n_events for e in result.episodes) < budget:
        victim = _pick(pool, ctx.rng)
        profile = ctx.profiles[victim.entity_id]
        episode_index += 1
        episode_id = f"bf-{episode_index:03d}"

        anchor = _anchor_event(victim.entity_id, ctx)
        # Attackers do not respect the victim's calendar, but they do not avoid it
        # either — offset from a real event by anything from minutes to a day.
        start = anchor.timestamp + dt.timedelta(
            minutes=float(ctx.rng.uniform(-720, 1440))
        )
        start = min(max(start, ctx.window_start), ctx.window_end - dt.timedelta(hours=1))

        # 40% of campaigns come from somewhere the victim genuinely uses, so that
        # geography alone cannot carry the decision.
        if ctx.rng.random() < 0.40:
            city = _pick(list(victim.all_legitimate_cities()), ctx.rng)
            source_ip = victim.ip_for_city(city, ctx.rng, ctx.faker)
        else:
            city = _adversary_city(ctx)
            source_ip = _adversary_ip(ctx)

        # Half the time the attacker is on the victim's own compromised host.
        fingerprint = (
            anchor.device_fingerprint
            if ctx.rng.random() < 0.5
            else _clone_fingerprint(anchor.device_fingerprint, ctx)
        )
        surface = _pick(list(_AUTH_SURFACES), ctx.rng)

        n_attempts = int(ctx.rng.integers(6, 31))
        burst_minutes = float(ctx.rng.uniform(3.0, 25.0))
        gaps = np.sort(ctx.rng.uniform(0.0, burst_minutes, size=n_attempts))

        events: list[AccessEvent] = []
        succeeded_at = (
            int(ctx.rng.integers(n_attempts // 2, n_attempts))
            if ctx.rng.random() < 0.35
            else None
        )
        for i, gap in enumerate(gaps):
            is_success = succeeded_at is not None and i >= succeeded_at
            commands = (
                [AUTH_OK, *ctx.rng.choice(COMMAND_VOCAB["webapp"], size=2)]
                if is_success
                else [AUTH_FAIL] * int(ctx.rng.integers(1, 4))
            )
            events.append(
                _new_event(
                    entity=victim,
                    timestamp=start + dt.timedelta(minutes=float(gap)),
                    source_ip=source_ip,
                    geo=city.render(),
                    resource=surface,
                    auth_method="password",
                    # Short, but inside the low tail of a normal session.
                    duration=float(ctx.rng.uniform(4.0, 48.0)),
                    commands=[str(c) for c in commands],
                    fingerprint=fingerprint,
                    label="brute_force",
                    episode_id=episode_id,
                )
            )
            if is_success:
                break  # a successful guess ends the guessing

        result.events.extend(events)
        result.episodes.append(
            _episode_from(
                episode_id,
                "brute_force",
                events,
                f"{len(events)} attempts against {victim.entity_id} on {surface} "
                f"over {burst_minutes:.0f}m from {city.name}"
                + (" (credential guessed)" if succeeded_at is not None else ""),
            )
        )
        _ = profile  # profile is intentionally unused: the attacker does not know it
    return result


# --------------------------------------------------------------------------------------
# 2. impossible_travel
# --------------------------------------------------------------------------------------


def inject_impossible_travel(ctx: InjectionContext, budget: int) -> InjectionResult:
    """Inject sessions from a location unreachable in the elapsed time.

    See the module docstring, section 2. Only ``geo_location`` and ``source_ip`` are
    perturbed — everything else is drawn from the victim's own profile, because the
    attacker holds a valid session.

    Args:
        ctx: The injection context.
        budget: Target number of events to label ``impossible_travel``.

    Returns:
        The injected events and their episode records.
    """
    result = InjectionResult()
    # Humans travel, so a human is the credible victim for a geo-velocity anomaly.
    pool = [e for e in _victim_pool(ctx) if e.entity_type == "user"]
    if not pool:
        pool = _victim_pool(ctx)
    if not pool:
        return result

    episode_index = 0
    while sum(e.n_events for e in result.episodes) < budget:
        victim = _pick(pool, ctx.rng)
        profile = ctx.profiles[victim.entity_id]
        anchor = _anchor_event(victim.entity_id, ctx)
        _, _, anchor_lat, anchor_lon = parse_geo(anchor.geo_location)

        # Choose a gap first, then demand a city far enough away to break the speed
        # limit. Short gaps make almost any distant city qualify; long gaps force a
        # genuinely extreme jump.
        gap_hours = float(ctx.rng.uniform(0.4, 4.0))
        required_km = _IMPOSSIBLE_SPEED_FLOOR_KMH * gap_hours
        candidates = [
            c
            for c in (*CITIES, *ADVERSARY_CITIES)
            if haversine_km(anchor_lat, anchor_lon, c.lat, c.lon) >= required_km
        ]
        if not candidates:
            continue  # anchor sits somewhere with no qualifying destination; retry

        destination = _pick(candidates, ctx.rng)
        episode_index += 1
        episode_id = f"it-{episode_index:03d}"

        distance = haversine_km(anchor_lat, anchor_lon, destination.lat, destination.lon)
        implied_kmh = distance / gap_hours
        source_ip = _adversary_ip(ctx)

        n_events = int(ctx.rng.integers(1, 4))
        events: list[AccessEvent] = []
        cursor = anchor.timestamp + dt.timedelta(hours=gap_hours)
        for _ in range(n_events):
            if cursor > ctx.window_end:
                break
            resource = profile.sample_resource(ctx.rng)
            commands = [AUTH_OK] + [
                str(c)
                for c in ctx.rng.choice(
                    COMMAND_VOCAB[resource.family],
                    size=1 + int(ctx.rng.poisson(profile.command_rate)),
                )
            ]
            events.append(
                _new_event(
                    entity=victim,
                    timestamp=cursor,
                    source_ip=source_ip,
                    geo=destination.render(),
                    resource=resource.path,
                    # A valid session: the attacker inherits the victim's auth method.
                    auth_method=anchor.auth_method,
                    duration=float(
                        ctx.rng.lognormal(profile.duration_mu, profile.duration_sigma)
                    ),
                    commands=commands,
                    fingerprint=anchor.device_fingerprint,
                    label="impossible_travel",
                    episode_id=episode_id,
                )
            )
            cursor += dt.timedelta(minutes=float(ctx.rng.uniform(4.0, 55.0)))

        if not events:
            continue
        result.events.extend(events)
        result.episodes.append(
            _episode_from(
                episode_id,
                "impossible_travel",
                events,
                f"{victim.entity_id} appears in {destination.name} "
                f"{gap_hours:.1f}h after {anchor.geo_location.split(',')[0]} "
                f"({distance:,.0f} km, {implied_kmh:,.0f} km/h implied)",
            )
        )
    return result


# --------------------------------------------------------------------------------------
# 3. credential_stuffing
# --------------------------------------------------------------------------------------


def inject_credential_stuffing(ctx: InjectionContext, budget: int) -> InjectionResult:
    """Inject breach-dump replay campaigns across many identities from one origin.

    See the module docstring, section 3. One campaign in four egresses from a genuine
    shared corporate gateway, so address reputation cannot separate the class.

    Args:
        ctx: The injection context.
        budget: Target number of events to label ``credential_stuffing``.

    Returns:
        The injected events and their episode records.
    """
    result = InjectionResult()
    pool = [e for e in ctx.roster.entities if e.entity_type != "edge_device"]
    if len(pool) < 10:
        return result

    episode_index = 0
    while sum(e.n_events for e in result.episodes) < budget:
        episode_index += 1
        episode_id = f"cs-{episode_index:03d}"

        # 1 in 4 campaigns hides behind infrastructure that legitimately carries many
        # identities — the deliberate overlap documented in the module docstring. Such a
        # campaign also inherits the gateway's apparent location, so it geolocates to a
        # corporate office like any ordinary tunnelled session.
        if ctx.rng.random() < 0.25 and ctx.roster.shared_gateways:
            gateway = _pick(ctx.roster.shared_gateways, ctx.rng)
            source_ip, city = gateway.ip, gateway.city
        else:
            source_ip = _adversary_ip(ctx)
            city = _adversary_city(ctx)
        os_name, firmware = ("Windows 10 22H2", "10.0.19045")
        fingerprint = f"{os_name} / fw {firmware} / {ctx.faker.mac_address().upper()}"

        n_targets = int(ctx.rng.integers(18, 61))
        target_idx = ctx.rng.choice(
            len(pool), size=min(n_targets, len(pool)), replace=False
        )
        targets = [pool[int(i)] for i in target_idx]

        start = ctx.window_start + dt.timedelta(
            seconds=float(
                ctx.rng.uniform(0, (ctx.window_end - ctx.window_start).total_seconds())
            )
        )
        campaign_minutes = float(ctx.rng.uniform(10.0, 90.0))
        surface = _pick(["vpn/gateway", "corp/mail"], ctx.rng)
        # Realistic hit rate for a stuffing run: most credentials in a dump are stale.
        success_rate = float(ctx.rng.uniform(0.01, 0.06))

        events: list[AccessEvent] = []
        n_success = 0
        for target in targets:
            for _ in range(int(ctx.rng.integers(1, 3))):
                offset = float(ctx.rng.uniform(0, campaign_minutes))
                timestamp = start + dt.timedelta(minutes=offset)
                if timestamp > ctx.window_end:
                    continue
                succeeded = ctx.rng.random() < success_rate
                if succeeded:
                    n_success += 1
                    commands = [AUTH_OK] + [
                        str(c)
                        for c in ctx.rng.choice(COMMAND_VOCAB["webapp"], size=2)
                    ]
                else:
                    commands = [AUTH_FAIL]
                events.append(
                    _new_event(
                        entity=target,
                        timestamp=timestamp,
                        source_ip=source_ip,
                        geo=city.render(),
                        resource=surface,
                        auth_method="password",
                        duration=float(ctx.rng.uniform(2.0, 30.0)),
                        commands=commands,
                        fingerprint=fingerprint,
                        label="credential_stuffing",
                        episode_id=episode_id,
                    )
                )

        if not events:
            continue
        result.events.extend(events)
        result.episodes.append(
            _episode_from(
                episode_id,
                "credential_stuffing",
                events,
                f"{len(targets)} identities probed from {source_ip} ({city.name}) "
                f"over {campaign_minutes:.0f}m, {n_success} succeeded",
            )
        )
    return result


# --------------------------------------------------------------------------------------
# 4. lateral_movement
# --------------------------------------------------------------------------------------


def inject_lateral_movement(ctx: InjectionContext, budget: int) -> InjectionResult:
    """Inject expanding resource-exploration chains inside otherwise valid sessions.

    See the module docstring, section 4. Network origin, device and auth method are all
    the victim's own — only the resource sequence and command mix are anomalous.

    Args:
        ctx: The injection context.
        budget: Target number of events to label ``lateral_movement``.

    Returns:
        The injected events and their episode records.
    """
    result = InjectionResult()
    pool = _victim_pool(ctx)
    if not pool:
        return result

    episode_index = 0
    while sum(e.n_events for e in result.episodes) < budget:
        victim = _pick(pool, ctx.rng)
        profile = ctx.profiles[victim.entity_id]
        anchor = _anchor_event(victim.entity_id, ctx)

        chain = _novel_resources_for(profile, int(ctx.rng.integers(4, 13)), ctx)
        if len(chain) < 3:
            continue

        episode_index += 1
        episode_id = f"lm-{episode_index:03d}"
        cursor = anchor.timestamp + dt.timedelta(minutes=float(ctx.rng.uniform(1, 45)))

        events: list[AccessEvent] = []
        for step, resource in enumerate(chain):
            if cursor > ctx.window_end:
                break
            vocab = COMMAND_VOCAB[resource.family]
            commands = [AUTH_OK]
            # Recon density rises as the chain progresses and the adversary orients.
            n_recon = int(ctx.rng.binomial(3, 0.25 + 0.05 * step))
            commands.extend(
                str(c) for c in ctx.rng.choice(RECON_COMMANDS, size=max(n_recon, 0))
            )
            commands.extend(
                str(c)
                for c in ctx.rng.choice(
                    vocab, size=1 + int(ctx.rng.poisson(max(profile.command_rate, 1.0)))
                )
            )
            events.append(
                _new_event(
                    entity=victim,
                    timestamp=cursor,
                    source_ip=anchor.source_ip,
                    geo=anchor.geo_location,
                    resource=resource.path,
                    auth_method=anchor.auth_method,
                    duration=float(
                        ctx.rng.lognormal(profile.duration_mu, profile.duration_sigma)
                    ),
                    commands=commands,
                    fingerprint=anchor.device_fingerprint,
                    label="lateral_movement",
                    episode_id=episode_id,
                )
            )
            cursor += dt.timedelta(minutes=float(ctx.rng.uniform(2.0, 40.0)))

        if len(events) < 3:
            continue
        result.events.extend(events)
        max_sensitivity = max(RESOURCES_BY_PATH[e.resource_accessed].sensitivity for e in events)
        result.episodes.append(
            _episode_from(
                episode_id,
                "lateral_movement",
                events,
                f"{victim.entity_id} walks {len(events)} novel resources from "
                f"{profile.home_department}, reaching sensitivity {max_sensitivity}",
            )
        )
    return result


# --------------------------------------------------------------------------------------
# 5. device_spoofing
# --------------------------------------------------------------------------------------


def inject_device_spoofing(ctx: InjectionContext, budget: int) -> InjectionResult:
    """Inject valid identities presenting from cloned hardware.

    See the module docstring, section 5. The spoof preserves the victim's OS family and
    is transient — persistence is the only thing separating it from a genuine hardware
    refresh.

    Args:
        ctx: The injection context.
        budget: Target number of events to label ``device_spoofing``.

    Returns:
        The injected events and their episode records.
    """
    result = InjectionResult()
    pool = _victim_pool(ctx)
    if not pool:
        return result

    episode_index = 0
    while sum(e.n_events for e in result.episodes) < budget:
        victim = _pick(pool, ctx.rng)
        profile = ctx.profiles[victim.entity_id]
        anchor = _anchor_event(victim.entity_id, ctx)
        episode_index += 1
        episode_id = f"ds-{episode_index:03d}"

        fingerprint = _clone_fingerprint(anchor.device_fingerprint, ctx)
        # Half the campaigns keep the victim's own network origin too, leaving the
        # fingerprint as the single perturbed field.
        if ctx.rng.random() < 0.5:
            city = _pick(list(victim.all_legitimate_cities()), ctx.rng)
            source_ip = victim.ip_for_city(city, ctx.rng, ctx.faker)
        else:
            city = _adversary_city(ctx)
            source_ip = _adversary_ip(ctx)

        n_events = int(ctx.rng.integers(2, 9))
        cursor = anchor.timestamp + dt.timedelta(hours=float(ctx.rng.uniform(0.5, 20.0)))

        events: list[AccessEvent] = []
        for _ in range(n_events):
            if cursor > ctx.window_end:
                break
            resource = profile.sample_resource(ctx.rng)
            commands = [AUTH_OK] + [
                str(c)
                for c in ctx.rng.choice(
                    COMMAND_VOCAB[resource.family],
                    size=1 + int(ctx.rng.poisson(profile.command_rate)),
                )
            ]
            events.append(
                _new_event(
                    entity=victim,
                    timestamp=cursor,
                    source_ip=source_ip,
                    geo=city.render(),
                    resource=resource.path,
                    auth_method=profile.auth_primary,
                    duration=float(
                        ctx.rng.lognormal(profile.duration_mu, profile.duration_sigma)
                    ),
                    commands=commands,
                    fingerprint=fingerprint,
                    label="device_spoofing",
                    episode_id=episode_id,
                )
            )
            cursor += dt.timedelta(minutes=float(ctx.rng.uniform(5.0, 120.0)))

        if not events:
            continue
        result.events.extend(events)
        result.episodes.append(
            _episode_from(
                episode_id,
                "device_spoofing",
                events,
                f"{victim.entity_id} presents cloned hardware "
                f"({fingerprint.split('/')[0].strip()}) from {city.name} "
                f"for {len(events)} sessions",
            )
        )
    return result


# --------------------------------------------------------------------------------------
# 6. low_and_slow_exfiltration
# --------------------------------------------------------------------------------------


def inject_low_and_slow_exfiltration(ctx: InjectionContext, budget: int) -> InjectionResult:
    """Inject multi-day, sub-threshold data drains.

    See the module docstring, section 6. Every event sits inside the victim's own
    normal envelope; only cadence regularity and cumulative volume carry signal.

    Args:
        ctx: The injection context.
        budget: Target number of events to label ``low_and_slow_exfiltration``.

    Returns:
        The injected events and their episode records.
    """
    result = InjectionResult()
    pool = _victim_pool(ctx)
    if not pool:
        return result

    data_families = {"database", "fileshare"}
    episode_index = 0
    while sum(e.n_events for e in result.episodes) < budget:
        victim = _pick(pool, ctx.rng)
        profile = ctx.profiles[victim.entity_id]
        anchor = _anchor_event(victim.entity_id, ctx)

        # Prefer a store the entity can already reach — insiders exfiltrate what they
        # already have access to, which is what makes the resource unremarkable.
        reachable = [r for r in profile.habitual_resources if r.family in data_families]
        if not reachable:
            reachable = [
                r
                for r in RESOURCES
                if r.family in data_families and r.sensitivity >= 3
            ]
        targets = [
            _pick(reachable, ctx.rng)
            for _ in range(int(ctx.rng.integers(1, min(3, len(reachable)) + 1)))
        ]

        span_days = int(ctx.rng.integers(5, 15))
        start_day_offset = float(
            ctx.rng.uniform(0, max(1.0, ctx.cfg.days - span_days - 1))
        )
        base = ctx.window_start + dt.timedelta(days=start_day_offset)
        # A fixed daily slot with only a few minutes of jitter — the cadence regularity
        # that distinguishes this class at the sequence level.
        base_hour = int(ctx.rng.integers(0, 24))
        jitter_minutes = float(ctx.rng.uniform(4.0, 25.0))
        per_day = int(ctx.rng.integers(1, 4))

        episode_index += 1
        episode_id = f"ls-{episode_index:03d}"
        events: list[AccessEvent] = []
        for day in range(span_days):
            for slot in range(per_day):
                timestamp = (
                    dt.datetime.combine(
                        (base + dt.timedelta(days=day)).date(), dt.time(hour=base_hour)
                    )
                    + dt.timedelta(hours=float(slot) * 6.0)
                    + dt.timedelta(minutes=float(ctx.rng.normal(0, jitter_minutes)))
                )
                if not (ctx.window_start <= timestamp <= ctx.window_end):
                    continue
                resource = targets[slot % len(targets)]
                commands = [AUTH_OK] + [
                    str(c)
                    for c in ctx.rng.choice(
                        BULK_READ_COMMANDS,
                        # Command count stays inside the entity's own normal range.
                        size=max(1, int(ctx.rng.poisson(profile.command_rate))),
                    )
                ]
                events.append(
                    _new_event(
                        entity=victim,
                        timestamp=timestamp,
                        source_ip=anchor.source_ip,
                        geo=anchor.geo_location,
                        resource=resource.path,
                        auth_method=profile.auth_primary,
                        duration=_upper_normal_duration(profile, ctx),
                        commands=commands,
                        fingerprint=anchor.device_fingerprint,
                        label="low_and_slow_exfiltration",
                        episode_id=episode_id,
                    )
                )

        if len(events) < 5:
            continue
        result.events.extend(events)
        result.episodes.append(
            _episode_from(
                episode_id,
                "low_and_slow_exfiltration",
                events,
                f"{victim.entity_id} drains {'/'.join(sorted({t.path for t in targets}))} "
                f"in {per_day}x daily sessions across {span_days} days",
            )
        )
    return result


# --------------------------------------------------------------------------------------
# 7. insider_drift (ambiguous)
# --------------------------------------------------------------------------------------


def inject_insider_drift(ctx: InjectionContext, budget: int) -> InjectionResult:
    """Rewrite benign traffic to reflect a gradual, permanent role change.

    See the module docstring, section 7. This injector adds no events — it redirects
    existing benign sessions onto newly-adopted resources. Ramp-phase events are
    labelled ``insider_drift``; post-ramp events keep using the expanded set but stay
    labelled ``normal_baseline``, which is what makes ``time-to-unflag`` measurable.

    Args:
        ctx: The injection context.
        budget: Target number of events to label ``insider_drift``.

    Returns:
        Episode records only; :attr:`InjectionResult.events` is always empty because the
        rewritten events already live in the caller's list.
    """
    result = InjectionResult()
    # A role change is a human thing, and needs enough runway to be gradual.
    pool = [
        e
        for e in _victim_pool(ctx, min_events=30)
        if e.entity_type == "user"
    ]
    if not pool:
        return result

    used: set[str] = set()
    episode_index = 0
    labelled = 0
    attempts = 0
    while labelled < budget and attempts < len(pool) * 3:
        attempts += 1
        victim = _pick(pool, ctx.rng)
        if victim.entity_id in used:
            continue
        used.add(victim.entity_id)

        profile = ctx.profiles[victim.entity_id]
        new_resources = _novel_resources_for(profile, int(ctx.rng.integers(2, 6)), ctx)
        if len(new_resources) < 2:
            continue

        ramp_days = int(ctx.rng.integers(10, 22))
        # Start early enough that a post-ramp tail exists to test un-flagging.
        latest_start = max(1, ctx.cfg.days - ramp_days - 4)
        ramp_start_day = int(ctx.rng.integers(1, latest_start + 1))
        ramp_start = ctx.window_start + dt.timedelta(days=ramp_start_day)
        ramp_end = ramp_start + dt.timedelta(days=ramp_days)

        episode_index += 1
        episode_id = f"id-{episode_index:03d}"

        touched: list[AccessEvent] = []
        history = ctx.events_by_entity.get(victim.entity_id, [])
        for event in history:
            if event.label != BENIGN_LABEL:
                continue  # never overwrite another injector's work
            timestamp = event.timestamp
            if timestamp < ramp_start:
                continue

            if timestamp <= ramp_end:
                # Adoption probability ramps linearly from 0 to ~0.75 across the window.
                progress = (timestamp - ramp_start).total_seconds() / max(
                    (ramp_end - ramp_start).total_seconds(), 1.0
                )
                adopt_prob = 0.75 * progress
                new_label = "insider_drift"
            else:
                # Settled into the new role: the expanded set is simply this entity's
                # normal now, and must stop generating alerts.
                adopt_prob = 0.55
                new_label = BENIGN_LABEL

            if ctx.rng.random() >= adopt_prob:
                continue

            resource = new_resources[int(ctx.rng.integers(0, len(new_resources)))]
            vocab = COMMAND_VOCAB[resource.family]
            event.resource_accessed = resource.path
            event.command_sequence = [event.command_sequence[0]] + [
                str(c)
                for c in ctx.rng.choice(
                    vocab, size=1 + int(ctx.rng.poisson(profile.command_rate))
                )
            ]
            event.label = new_label
            event.episode_id = episode_id
            if new_label == "insider_drift":
                touched.append(event)

        if len(touched) < 4:
            continue

        profile.drifted_resources.extend(new_resources)
        labelled += len(touched)
        result.episodes.append(
            _episode_from(
                episode_id,
                "insider_drift",
                touched,
                f"{victim.entity_id} gradually adopts "
                f"{', '.join(r.path for r in new_resources)} over {ramp_days} days, "
                f"then keeps them permanently",
            )
        )
    return result


# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------

Injector = Callable[[InjectionContext, int], InjectionResult]

#: Injectors in execution order. ``insider_drift`` runs last because it rewrites benign
#: events in place and must not clobber events another injector has already claimed.
INJECTORS: dict[str, Injector] = {
    "brute_force": inject_brute_force,
    "impossible_travel": inject_impossible_travel,
    "credential_stuffing": inject_credential_stuffing,
    "lateral_movement": inject_lateral_movement,
    "device_spoofing": inject_device_spoofing,
    "low_and_slow_exfiltration": inject_low_and_slow_exfiltration,
    "insider_drift": inject_insider_drift,
}
