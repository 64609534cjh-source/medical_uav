map_seed = 0, dev seeds 100000..100019

| λ | policy | orders | on-time | served | expired | km/deliv | charge | busy | pool | hidden |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.04 | edf | 20.9 | 0.99 ± 0.04 | 1.00 | 0.00 | 5.07 | 0.15 | 0.39 | 0.6 | 0.0 |
| 0.04 | random | 20.9 | 0.42 ± 0.14 | 0.75 | 0.25 | 11.15 | 0.24 | 0.45 | 1.7 | 0.0 |
| 0.06 | edf | 35.0 | 0.75 ± 0.26 | 0.98 | 0.02 | 4.78 | 0.23 | 0.59 | 2.6 | 0.0 |
| 0.06 | random | 35.0 | 0.31 ± 0.12 | 0.61 | 0.39 | 8.85 | 0.27 | 0.52 | 3.9 | 0.0 |
| 0.10 | edf | 56.4 | 0.27 ± 0.12 | 0.77 | 0.23 | 5.03 | 0.26 | 0.68 | 8.4 | 12.4 |
| 0.10 | random | 56.4 | 0.20 ± 0.04 | 0.44 | 0.56 | 8.46 | 0.30 | 0.57 | 7.9 | 4.7 |

Gate G1 decision (2026-10-11):
- Map: `map_seed = 0` (first seed with a max A*/straight-line ratio >= 1.15; its ratio is 2.04).
  200/200 map seeds generate a valid map, 1.01 attempts on average.
- Loads: low λ = 0.04, medium λ = 0.06, high λ = 0.10 (orders/min base rate).
  λ = 0.07 (the spec's initial medium) gave EDF 0.57 ± 0.23 on-time, at the lower edge of the
  55–85 % band; λ = 0.06 sits in the middle (0.75) with a charging share of 0.23.
- Throughput: ~9,650 decisions/s per process (random policy, medium load).

Columns: on-time / served / expired are fractions of all released orders; km/deliv = km flown per
delivered order; charge / busy = share of the episode spent charging / flying+handling;
pool = mean pending orders per decision; hidden = decisions × on-time-feasible orders beyond the 12 slots.
