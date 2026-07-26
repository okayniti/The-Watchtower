# The Watchtower — synthetic dataset summary

- **Total events:** 90,495
- **Window:** 2026-05-01T00:01:12 to 2026-05-30T23:58:51 (30 days)
- **Entities:** 500 (325 user, 100 service_account, 75 edge_device)
- **Seed:** 20260726 (run is fully reproducible)

## Class balance

| Behaviour | Class | Events | % of total |
|---|---|---:|---:|
| `normal_baseline` | benign | 88,721 | 98.040% |
| `brute_force` | anomaly | 267 | 0.295% |
| `impossible_travel` | anomaly | 100 | 0.111% |
| `credential_stuffing` | anomaly | 280 | 0.309% |
| `lateral_movement` | anomaly | 208 | 0.230% |
| `device_spoofing` | anomaly | 83 | 0.092% |
| `low_and_slow_exfiltration` | anomaly | 321 | 0.355% |
| `insider_drift` | **ambiguous** | 515 | 0.569% |

**Hard-anomaly rate: 1.391%** (1,259 events across the six attack classes) — target band 0.5%-3% per CLAUDE.md §4.2.

`insider_drift` contributes a further 0.569% (515 events) and is excluded from the anomaly rate: it is benign-but-attack-shaped, and counting it would misstate the imbalance the detector actually faces.

## Injected episodes

| Behaviour | Episodes | Events | Mean events/episode | Mean span (days) | Entities |
|---|---:|---:|---:|---:|---:|
| `brute_force` | 15 | 267 | 17.8 | 0.01 | 15 |
| `impossible_travel` | 49 | 100 | 2.0 | 0.02 | 45 |
| `credential_stuffing` | 5 | 280 | 56.0 | 0.04 | 159 |
| `lateral_movement` | 26 | 208 | 8.0 | 0.1 | 26 |
| `device_spoofing` | 14 | 83 | 5.9 | 0.21 | 14 |
| `low_and_slow_exfiltration` | 16 | 321 | 20.1 | 9.02 | 16 |
| `insider_drift` | 25 | 515 | 20.6 | 11.7 | 25 |

## Per-entity event volume

| Statistic | All entities | Established only |
|---|---:|---:|
| min | 1 | 30 |
| median | 139 | 143 |
| p25 / p75 | 82 / 223 | — |
| max | 915 | — |
| mean | 181.0 | — |

## Cold-start entities (25)

Held back deliberately: these identities first appear only in the final 3 days of the window and emit a handful of events each. They are the test cases for CLAUDE.md §4.5 — the detector must score them without meaningful history.

12 of them are also *targeted* by an injected campaign — an attack against an identity with almost no baseline is the hardest cold-start case in the dataset, so these are kept rather than excluded.

| entity_id | type | benign history | total events | targeted |
|---|---|---:|---:|---|
| `usr-0017` | user | 1 | 1 | — |
| `usr-0158` | user | 1 | 3 | yes |
| `usr-0193` | user | 1 | 1 | — |
| `usr-0268` | user | 1 | 1 | — |
| `usr-0303` | user | 1 | 3 | yes |
| `usr-0180` | user | 2 | 3 | yes |
| `usr-0255` | user | 2 | 3 | yes |
| `usr-0316` | user | 2 | 3 | yes |
| `svc-0362` | service_account | 4 | 6 | yes |
| `usr-0064` | user | 4 | 4 | — |
| `usr-0086` | user | 4 | 4 | — |
| `usr-0183` | user | 4 | 6 | yes |
| `dev-0472` | edge_device | 5 | 5 | — |
| `svc-0418` | service_account | 5 | 6 | yes |
| `usr-0057` | user | 5 | 5 | — |
| … 10 more | | | | |

## Checks

| Check | Result |
|---|---|
| hard anomaly rate in 0.5 3 pct | PASS |
| events sorted by timestamp | PASS |
| event ids unique | PASS |
| timestamps second resolution | PASS |
| all behaviours present | PASS |
| cold start entities are sparse | PASS |
| low and slow spans multiple days | PASS |
| insider drift is gradual | PASS |
