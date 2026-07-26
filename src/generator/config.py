"""Configuration, schema constants and reference data for The Watchtower generator.

This module is the single source of truth for:

* the :class:`GeneratorConfig` dataclass that parameterises a full simulation run,
* the 11-field event schema (:class:`AccessEvent`) from ``CLAUDE.md`` §2,
* the class vocabulary (:data:`BEHAVIOURS`) from ``CLAUDE.md`` §3,
* reference data used to make events look like a real access log: real-world city
  coordinates, a resource catalogue with sensitivity tiers, per-resource command
  vocabularies, and OS/firmware pools for device fingerprints.

Design notes
------------
**Schema tension — auth outcome.** ``CLAUDE.md`` §3 requires ``brute_force`` to be a
"burst of failed auths" and ``credential_stuffing`` to have a "low success rate", but the
11-field schema has no success/failure column. Rather than adding a 12th field (which
would break the spec and hand the detector a giveaway feature), auth outcome is encoded
inside ``command_sequence`` as :data:`AUTH_OK` / :data:`AUTH_FAIL` tokens. This is
faithful to how real SIEM logs work — the outcome lives in the event stream — and it
keeps the failure signal *sequential*, which is what ``CLAUDE.md`` §4.1 asks for.
Normal events also emit occasional ``AUTH_FAIL`` tokens (fat-fingered passwords), so the
token alone is not a separator.

**Geo format.** ``geo_location`` is emitted as ``"City, CC (lat, lon)"`` — human-readable
in the dashboard, and machine-parseable via :func:`parse_geo` so that geo-velocity
(km/h between consecutive events) can be computed for ``impossible_travel``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Final, Literal, Sequence

# --------------------------------------------------------------------------------------
# Class vocabulary (CLAUDE.md §3)
# --------------------------------------------------------------------------------------

BENIGN_LABEL: Final[str] = "normal_baseline"

#: The six unambiguous attack classes. These are what the 0.5-3% imbalance budget covers.
HARD_ANOMALIES: Final[tuple[str, ...]] = (
    "brute_force",
    "impossible_travel",
    "credential_stuffing",
    "lateral_movement",
    "device_spoofing",
    "low_and_slow_exfiltration",
)

#: Deliberately ambiguous. Benign in truth, attack-shaped in the features. Reported
#: separately from the anomaly budget — see CLAUDE.md §3 and §5.
AMBIGUOUS_LABEL: Final[str] = "insider_drift"

#: All eight behaviours, in report order.
BEHAVIOURS: Final[tuple[str, ...]] = (BENIGN_LABEL, *HARD_ANOMALIES, AMBIGUOUS_LABEL)

EntityType = Literal["user", "service_account", "edge_device"]
ENTITY_TYPES: Final[tuple[EntityType, ...]] = ("user", "service_account", "edge_device")

AuthMethod = Literal["password", "token", "certificate", "biometric"]
AUTH_METHODS: Final[tuple[AuthMethod, ...]] = (
    "password",
    "token",
    "certificate",
    "biometric",
)

#: Auth-outcome tokens embedded in ``command_sequence`` (see module docstring).
AUTH_OK: Final[str] = "AUTH_OK"
AUTH_FAIL: Final[str] = "AUTH_FAIL"

#: Column order of ``data/access_logs.csv`` (the label is *not* here — see labels.csv).
FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "event_id",
    "entity_id",
    "entity_type",
    "timestamp",
    "source_ip",
    "geo_location",
    "resource_accessed",
    "auth_method",
    "session_duration",
    "command_sequence",
    "device_fingerprint",
)

#: Column order of ``data/labels.csv``.
LABEL_COLUMNS: Final[tuple[str, ...]] = ("event_id", "label", "episode_id")


# --------------------------------------------------------------------------------------
# Geography — real coordinates so geo-velocity is physically meaningful
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class City:
    """A real-world location with true coordinates.

    Attributes:
        name: City name as rendered in ``geo_location``.
        country: ISO-3166 alpha-2 country code.
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.
        region: Coarse continent bucket, used to pick *plausible* business-travel
            destinations (same-region trips are far more common than cross-globe ones).
    """

    name: str
    country: str
    lat: float
    lon: float
    region: str

    def render(self) -> str:
        """Return the ``geo_location`` string form: ``"City, CC (lat, lon)"``."""
        return f"{self.name}, {self.country} ({self.lat:.4f}, {self.lon:.4f})"


CITIES: Final[tuple[City, ...]] = (
    City("Bengaluru", "IN", 12.9716, 77.5946, "APAC"),
    City("Mumbai", "IN", 19.0760, 72.8777, "APAC"),
    City("Hyderabad", "IN", 17.3850, 78.4867, "APAC"),
    City("Singapore", "SG", 1.3521, 103.8198, "APAC"),
    City("Tokyo", "JP", 35.6762, 139.6503, "APAC"),
    City("Sydney", "AU", -33.8688, 151.2093, "APAC"),
    City("Seoul", "KR", 37.5665, 126.9780, "APAC"),
    City("London", "GB", 51.5074, -0.1278, "EMEA"),
    City("Berlin", "DE", 52.5200, 13.4050, "EMEA"),
    City("Amsterdam", "NL", 52.3676, 4.9041, "EMEA"),
    City("Paris", "FR", 48.8566, 2.3522, "EMEA"),
    City("Dublin", "IE", 53.3498, -6.2603, "EMEA"),
    City("Warsaw", "PL", 52.2297, 21.0122, "EMEA"),
    City("Dubai", "AE", 25.2048, 55.2708, "EMEA"),
    City("Tel Aviv", "IL", 32.0853, 34.7818, "EMEA"),
    City("New York", "US", 40.7128, -74.0060, "AMER"),
    City("Austin", "US", 30.2672, -97.7431, "AMER"),
    City("San Jose", "US", 37.3382, -121.8863, "AMER"),
    City("Seattle", "US", 47.6062, -122.3321, "AMER"),
    City("Chicago", "US", 41.8781, -87.6298, "AMER"),
    City("Toronto", "CA", 43.6532, -79.3832, "AMER"),
    City("Sao Paulo", "BR", -23.5505, -46.6333, "AMER"),
    City("Mexico City", "MX", 19.4326, -99.1332, "AMER"),
    City("Bogota", "CO", 4.7110, -74.0721, "AMER"),
)

#: Cities that host a corporate office (entities based here get a 10.x office subnet).
OFFICE_CITIES: Final[tuple[str, ...]] = (
    "Bengaluru",
    "Hyderabad",
    "London",
    "Berlin",
    "Amsterdam",
    "New York",
    "Austin",
    "Singapore",
)

#: Locations favoured by attackers. Not exclusive — attacks also originate from
#: ordinary cities, so "unusual country" alone must not be a sufficient detector.
ADVERSARY_CITIES: Final[tuple[City, ...]] = (
    City("Sofia", "BG", 42.6977, 23.3219, "EMEA"),
    City("Lagos", "NG", 6.5244, 3.3792, "EMEA"),
    City("Kyiv", "UA", 50.4501, 30.5234, "EMEA"),
    City("Ho Chi Minh City", "VN", 10.8231, 106.6297, "APAC"),
    City("Caracas", "VE", 10.4806, -66.9036, "AMER"),
    City("Jakarta", "ID", -6.2088, 106.8456, "APAC"),
)

_GEO_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?P<city>.+?),\s*(?P<country>[A-Z]{2})\s*\(\s*(?P<lat>-?\d+\.\d+),\s*(?P<lon>-?\d+\.\d+)\s*\)$"
)


def parse_geo(geo_location: str) -> tuple[str, str, float, float]:
    """Parse a ``geo_location`` string back into its components.

    Args:
        geo_location: A string in the form ``"City, CC (lat, lon)"``.

    Returns:
        A ``(city, country, lat, lon)`` tuple.

    Raises:
        ValueError: If the string does not match the expected format.
    """
    match = _GEO_RE.match(geo_location.strip())
    if match is None:
        raise ValueError(f"Unparseable geo_location: {geo_location!r}")
    return (
        match["city"],
        match["country"],
        float(match["lat"]),
        float(match["lon"]),
    )


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two coordinates, in kilometres.

    Args:
        lat1: Latitude of the first point, decimal degrees.
        lon1: Longitude of the first point, decimal degrees.
        lat2: Latitude of the second point, decimal degrees.
        lon2: Longitude of the second point, decimal degrees.

    Returns:
        Distance in kilometres along the Earth's surface.
    """
    earth_radius_km = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = phi2 - phi1
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return 2 * earth_radius_km * math.asin(math.sqrt(a))


# --------------------------------------------------------------------------------------
# Resource catalogue
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Resource:
    """An accessible asset.

    Attributes:
        path: Identifier written to ``resource_accessed``.
        family: Command-vocabulary family (see :data:`COMMAND_VOCAB`).
        sensitivity: 1 (routine) to 4 (crown jewels). Drives exfiltration targeting and
            is *not* exposed to the model — it is generator-side metadata only.
        department: Used by ``insider_drift`` and ``lateral_movement`` to build
            plausible expansion chains into adjacent departments.
    """

    path: str
    family: str
    sensitivity: int
    department: str


RESOURCES: Final[tuple[Resource, ...]] = (
    # --- engineering -------------------------------------------------------------
    Resource("eng/repo/watchtower", "repo", 2, "engineering"),
    Resource("eng/repo/platform-core", "repo", 2, "engineering"),
    Resource("eng/repo/edge-firmware", "repo", 3, "engineering"),
    Resource("eng/ci/build-runner", "compute", 2, "engineering"),
    Resource("eng/artifacts/releases", "fileshare", 2, "engineering"),
    # --- data ---------------------------------------------------------------------
    Resource("db/analytics/events", "database", 2, "data"),
    Resource("db/analytics/sessions", "database", 2, "data"),
    Resource("db/prod/customers", "database", 4, "data"),
    Resource("db/prod/transactions", "database", 4, "data"),
    Resource("db/prod/telemetry", "database", 3, "data"),
    Resource("warehouse/exports", "fileshare", 3, "data"),
    # --- corporate ----------------------------------------------------------------
    Resource("hr/records", "fileshare", 4, "corporate"),
    Resource("hr/payroll", "fileshare", 4, "corporate"),
    Resource("finance/ledger", "fileshare", 4, "corporate"),
    Resource("finance/invoices", "fileshare", 3, "corporate"),
    Resource("legal/contracts", "fileshare", 3, "corporate"),
    Resource("corp/wiki", "webapp", 1, "corporate"),
    Resource("corp/ticketing", "webapp", 1, "corporate"),
    Resource("corp/mail", "webapp", 2, "corporate"),
    # --- infrastructure -----------------------------------------------------------
    Resource("infra/k8s/prod", "compute", 4, "infrastructure"),
    Resource("infra/k8s/staging", "compute", 2, "infrastructure"),
    Resource("infra/vault/secrets", "secrets", 4, "infrastructure"),
    Resource("infra/monitoring", "webapp", 1, "infrastructure"),
    Resource("admin/iam/directory", "identity", 4, "infrastructure"),
    Resource("vpn/gateway", "network", 2, "infrastructure"),
    # --- edge / OT ------------------------------------------------------------------
    Resource("iot/telemetry/ingest", "telemetry", 2, "edge"),
    Resource("iot/fleet/registry", "telemetry", 3, "edge"),
    Resource("iot/firmware/ota", "telemetry", 3, "edge"),
    Resource("ot/scada/hmi", "telemetry", 4, "edge"),
    Resource("ot/historian", "database", 3, "edge"),
)

RESOURCES_BY_PATH: Final[dict[str, Resource]] = {r.path: r for r in RESOURCES}

#: Departments an entity type plausibly lives in, used when seeding habitual resources.
DEPARTMENTS_BY_ENTITY_TYPE: Final[dict[str, tuple[str, ...]]] = {
    "user": ("engineering", "data", "corporate", "infrastructure"),
    "service_account": ("data", "infrastructure", "engineering"),
    "edge_device": ("edge", "infrastructure"),
}

#: Adjacency graph used to build *plausible* expansion chains. Both ``lateral_movement``
#: and ``insider_drift`` walk this graph — that shared mechanic is exactly why the two
#: are hard to tell apart, which is the point (CLAUDE.md §3).
DEPARTMENT_ADJACENCY: Final[dict[str, tuple[str, ...]]] = {
    "engineering": ("data", "infrastructure"),
    "data": ("engineering", "corporate", "infrastructure"),
    "corporate": ("data",),
    "infrastructure": ("engineering", "data", "edge"),
    "edge": ("infrastructure", "data"),
}


# --------------------------------------------------------------------------------------
# Command vocabularies
# --------------------------------------------------------------------------------------

#: Per-family command pools. Sequences are drawn from these, so a session's commands are
#: coherent with the resource it touched.
COMMAND_VOCAB: Final[dict[str, tuple[str, ...]]] = {
    "repo": ("git.clone", "git.fetch", "git.push", "git.log", "code.search", "pr.review"),
    "compute": ("pod.list", "pod.logs", "pod.exec", "deploy.apply", "node.describe", "job.run"),
    "database": ("db.connect", "db.select", "db.explain", "db.export", "db.count", "db.describe"),
    "fileshare": ("fs.list", "fs.stat", "fs.read", "fs.download", "fs.search", "fs.upload"),
    "webapp": ("http.get", "page.view", "search.query", "doc.edit", "comment.add"),
    "secrets": ("secret.list", "secret.read", "secret.lease", "policy.read"),
    "identity": ("iam.list_users", "iam.describe", "iam.list_roles", "iam.whoami"),
    "network": ("vpn.connect", "vpn.status", "vpn.disconnect", "route.show"),
    "telemetry": ("tel.publish", "tel.heartbeat", "tel.batch", "fw.version", "cfg.pull"),
}

#: Reconnaissance-flavoured commands. Used by ``lateral_movement`` — but also, at low
#: rate, by ordinary curious engineers, so their presence is suggestive, not conclusive.
RECON_COMMANDS: Final[tuple[str, ...]] = (
    "iam.whoami",
    "iam.list_roles",
    "net.scan",
    "share.enumerate",
    "host.discover",
)

#: Bulk-read commands. Used by ``low_and_slow_exfiltration`` — and by legitimate
#: analysts running reports, which is what makes the exfil case genuinely subtle.
BULK_READ_COMMANDS: Final[tuple[str, ...]] = (
    "db.export",
    "fs.download",
    "db.select",
    "fs.read",
)


# --------------------------------------------------------------------------------------
# Device fingerprint pools
# --------------------------------------------------------------------------------------

#: ``(os_string, firmware_prefix)`` pools keyed by entity type. A fingerprint renders as
#: ``"<os> / fw <firmware> / <MAC>"`` per CLAUDE.md §2 (OS/firmware/MAC).
DEVICE_POOLS: Final[dict[str, tuple[tuple[str, str], ...]]] = {
    "user": (
        ("Windows 11 23H2", "10.0.22631"),
        ("Windows 10 22H2", "10.0.19045"),
        ("macOS 14 Sonoma", "23.5.0"),
        ("macOS 15 Sequoia", "24.1.0"),
        ("Ubuntu 22.04 LTS", "5.15.0"),
        ("Fedora 40", "6.8.9"),
    ),
    "service_account": (
        ("Debian 12 (container)", "6.1.0"),
        ("Alpine 3.20 (container)", "6.6.32"),
        ("Ubuntu 24.04 (container)", "6.8.0"),
        ("RHEL 9 (vm)", "5.14.0"),
    ),
    "edge_device": (
        ("Yocto Kirkstone", "4.1.12-rt"),
        ("BuildRoot 2024.02", "5.10.198"),
        ("Zephyr RTOS 3.6", "3.6.0"),
        ("FreeRTOS 10.6", "10.6.1"),
    ),
}


# --------------------------------------------------------------------------------------
# The run configuration
# --------------------------------------------------------------------------------------


def _default_attack_share() -> dict[str, float]:
    """Return the default per-attack share of total events.

    Each value is the fraction of *all* generated events that the injector should
    attempt to label as that class. The six hard anomalies sum to ~1.35%, comfortably
    inside the 0.5-3% band mandated by ``CLAUDE.md`` §4.2. ``insider_drift`` sits outside
    that budget because it is ambiguous-benign, not an attack.

    Returns:
        Mapping of behaviour name to target share of total events.
    """
    return {
        "brute_force": 0.0028,
        "impossible_travel": 0.0011,
        "credential_stuffing": 0.0031,
        "lateral_movement": 0.0023,
        "device_spoofing": 0.0009,
        "low_and_slow_exfiltration": 0.0033,
        # Ambiguous — reported separately from the anomaly budget.
        "insider_drift": 0.0055,
    }


@dataclass(slots=True)
class GeneratorConfig:
    """Everything that parameterises one reproducible simulation run.

    Every stochastic decision in the generator is driven by :attr:`seed`, so two runs
    with an identical config produce byte-identical CSVs.

    Attributes:
        n_entities: Total entities in the roster, including cold-start ones.
        days: Length of the simulation window in days.
        start_date: ISO date (``YYYY-MM-DD``) of the first simulated day.
        seed: Master RNG seed. Fixing this fixes the entire output.
        entity_type_mix: Proportion of the roster per entity type. Must sum to 1.
        events_per_day: Mean events per day per entity, keyed by entity type. Actual
            counts are Poisson-drawn around an entity-specific multiplier of this.
        cold_start_fraction: Fraction of entities held back as cold-start cases — they
            appear only near the end of the window with a handful of events, so the
            detector must score them with almost no history (``CLAUDE.md`` §4.5).
        cold_start_max_events: Upper bound on how many events a cold-start entity gets.
        cold_start_window_days: How many days before the window end a cold-start entity
            first appears.
        attack_share: Target share of total events per behaviour. See
            :func:`_default_attack_share`.
        weekend_activity_factor: Multiplier applied to a human user's event rate at
            weekends. Service accounts and edge devices ignore this.
        benign_travel_prob: Per-entity probability of taking at least one legitimate
            business trip during the window. Trips move the entity's geo at *plausible*
            velocity and are the main false-positive pressure on ``impossible_travel``.
        benign_device_change_prob: Per-entity probability of a legitimate hardware
            refresh mid-window — false-positive pressure on ``device_spoofing``.
        benign_novel_resource_prob: Per-event probability that a normal session touches
            a resource outside the entity's habitual set — false-positive pressure on
            ``lateral_movement``.
        benign_auth_fail_prob: Per-event probability that a normal session contains one
            or two ``AUTH_FAIL`` tokens (mistyped credentials) — this is what stops
            ``AUTH_FAIL`` from being a giveaway for ``brute_force``.
        off_hours_prob: Per-event probability that a human user's event is drawn from
            outside their habitual active hours (late-night incident work).
        shared_gateway_count: Number of shared VPN/NAT egress IPs. Many *legitimate*
            entities share these addresses, which is the false-positive pressure on
            ``credential_stuffing``'s "many identities, one IP" signature.
        shared_gateway_prob: Per-event probability a normal session egresses via a
            shared gateway rather than the entity's own address.
        output_dir: Directory for ``access_logs.csv`` / ``labels.csv``.
        reports_dir: Directory for the run summary.
        include_label_in_logs: If ``True``, also write ``label`` into
            ``access_logs.csv``. Defaults to ``False`` so the feature file physically
            cannot leak ground truth (``CLAUDE.md`` §2).
        sample_rows: Size of the committed showcase sample. The full dataset is ~21 MB
            and git-ignored (it regenerates deterministically), so a small stratified
            sample is tracked instead to make the data shape visible in the repo.
        sample_min_per_class: Minimum events per behaviour in that sample. Guarantees
            all eight classes appear, which a chronological head does not — the first
            few thousand events are day-one traffic and contain almost no attacks.
    """

    n_entities: int = 500
    days: int = 30
    start_date: str = "2026-05-01"
    seed: int = 20260726

    entity_type_mix: dict[str, float] = field(
        default_factory=lambda: {
            "user": 0.65,
            "service_account": 0.20,
            "edge_device": 0.15,
        }
    )
    events_per_day: dict[str, float] = field(
        default_factory=lambda: {
            "user": 5.0,
            "service_account": 12.0,
            "edge_device": 8.0,
        }
    )

    cold_start_fraction: float = 0.05
    cold_start_max_events: int = 8
    cold_start_window_days: int = 3

    attack_share: dict[str, float] = field(default_factory=_default_attack_share)

    # --- benign-but-unusual behaviour: the false-positive pressure ------------------
    weekend_activity_factor: float = 0.18
    benign_travel_prob: float = 0.22
    benign_device_change_prob: float = 0.08
    benign_novel_resource_prob: float = 0.035
    benign_auth_fail_prob: float = 0.045
    off_hours_prob: float = 0.06
    shared_gateway_count: int = 6
    shared_gateway_prob: float = 0.12

    output_dir: str = "data"
    reports_dir: str = "reports"
    include_label_in_logs: bool = False
    sample_rows: int = 2000
    sample_min_per_class: int = 10

    def __post_init__(self) -> None:
        """Validate the configuration.

        Raises:
            ValueError: If any field is out of range or internally inconsistent.
        """
        if self.n_entities < 10:
            raise ValueError("n_entities must be at least 10")
        if self.days < 7:
            raise ValueError("days must be at least 7 (attacks span multi-day windows)")

        mix_total = sum(self.entity_type_mix.values())
        if not math.isclose(mix_total, 1.0, abs_tol=1e-6):
            raise ValueError(f"entity_type_mix must sum to 1.0, got {mix_total}")
        missing = set(ENTITY_TYPES) - set(self.entity_type_mix)
        if missing:
            raise ValueError(f"entity_type_mix is missing entity types: {sorted(missing)}")

        unknown = set(self.attack_share) - set(HARD_ANOMALIES) - {AMBIGUOUS_LABEL}
        if unknown:
            raise ValueError(f"attack_share has unknown behaviours: {sorted(unknown)}")
        if any(v < 0 for v in self.attack_share.values()):
            raise ValueError("attack_share values must be non-negative")

        hard_total = sum(self.attack_share.get(name, 0.0) for name in HARD_ANOMALIES)
        if not 0.0005 <= hard_total <= 0.03:
            raise ValueError(
                f"hard-anomaly share {hard_total:.4%} is outside the mandated "
                "0.5%-3% band (CLAUDE.md §4.2)"
            )

        if not 0.0 <= self.cold_start_fraction < 0.5:
            raise ValueError("cold_start_fraction must be in [0, 0.5)")

    @property
    def hard_anomaly_share(self) -> float:
        """Total configured share of events belonging to the six hard attack classes."""
        return sum(self.attack_share.get(name, 0.0) for name in HARD_ANOMALIES)

    @property
    def expected_events(self) -> int:
        """Rough expected normal-event count, used to size attack budgets up front.

        Returns:
            An estimate of the number of normal events the run will produce, before
            injection. This only needs to be accurate enough to set episode counts.
        """
        per_day = sum(
            self.entity_type_mix[et] * self.n_entities * self.events_per_day[et]
            for et in ENTITY_TYPES
        )
        # Cold-start entities contribute a negligible tail; ignore them in the estimate.
        return int(per_day * self.days * (1.0 - self.cold_start_fraction))


# --------------------------------------------------------------------------------------
# The event record
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class AccessEvent:
    """One row of the access log — the 11-field schema from ``CLAUDE.md`` §2.

    ``label`` and ``episode_id`` are carried on the in-memory record but are written to
    ``labels.csv``, never to ``access_logs.csv`` (unless
    :attr:`GeneratorConfig.include_label_in_logs` is explicitly enabled).

    Attributes:
        entity_id: Stable identifier of the acting entity.
        entity_type: One of :data:`ENTITY_TYPES`.
        timestamp: Event time (timezone-naive UTC).
        source_ip: IPv4 address the session originated from.
        geo_location: ``"City, CC (lat, lon)"`` — parse with :func:`parse_geo`.
        resource_accessed: Path of the asset touched.
        auth_method: One of :data:`AUTH_METHODS`.
        session_duration: Session length in seconds.
        command_sequence: Ordered commands issued, including auth-outcome tokens.
        device_fingerprint: ``"<os> / fw <firmware> / <MAC>"``.
        label: Ground truth, one of :data:`BEHAVIOURS`.
        episode_id: Identifier grouping the events of a single injected episode, or
            ``""`` for ordinary background traffic. Useful for episode-level evaluation.
        event_id: Assigned after the global sort, so IDs ascend with time.
    """

    entity_id: str
    entity_type: str
    timestamp: "object"  # datetime.datetime; typed loosely to keep this module import-light
    source_ip: str
    geo_location: str
    resource_accessed: str
    auth_method: str
    session_duration: float
    command_sequence: Sequence[str]
    device_fingerprint: str
    label: str = BENIGN_LABEL
    episode_id: str = ""
    event_id: str = ""

    def command_string(self) -> str:
        """Render ``command_sequence`` for CSV as a ``|``-delimited string.

        Returns:
            The commands joined by ``"|"``. The delimiter is chosen because it never
            appears in :data:`COMMAND_VOCAB`, so the round-trip is lossless.
        """
        return "|".join(self.command_sequence)
