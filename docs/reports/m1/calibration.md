Command: python scripts/calibrate.py --tune-theta --map-check 200 --benchmark

map_seed = 0; dev seeds 100000..100039 (40 days per row); charging threshold θ edf 20 %, slack 20 %

| λ | policy | orders | on-time (mean ± day SD) | served | expired | rejected | km/deliv | charge | busy | decisions | pool mean/p95/max | hidden % | hidden mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.04 | slack | 21.6 | 1.00 ± 0.01 | 1.00 | 0.00 | 0.0 | 4.71 | 0.15 | 0.39 | 3230 | 0.5/2/5 | 0.0 | 0.00 |
| 0.04 | edf | 21.6 | 1.00 ± 0.01 | 1.00 | 0.00 | 0.0 | 4.85 | 0.15 | 0.39 | 3226 | 0.5/2/6 | 0.0 | 0.00 |
| 0.04 | random | 21.6 | 0.47 ± 0.16 | 0.78 | 0.22 | 0.0 | 10.80 | 0.26 | 0.48 | 3775 | 1.8/5/12 | 0.0 | 0.00 |
| 0.06 | slack | 35.5 | 0.94 ± 0.07 | 0.97 | 0.03 | 0.0 | 4.01 | 0.21 | 0.57 | 3659 | 1.6/6/11 | 0.0 | 0.00 |
| 0.06 | edf | 35.5 | 0.81 ± 0.22 | 0.99 | 0.01 | 0.0 | 4.34 | 0.22 | 0.58 | 3739 | 2.2/7/11 | 0.0 | 0.00 |
| 0.06 | random | 35.5 | 0.31 ± 0.10 | 0.60 | 0.40 | 0.0 | 9.29 | 0.28 | 0.53 | 4018 | 4.1/10/16 | 0.0 | 0.00 |
| 0.08 | slack | 47.1 | 0.84 ± 0.08 | 0.92 | 0.08 | 0.0 | 3.58 | 0.23 | 0.65 | 3943 | 3.3/9/16 | 0.4 | 0.00 |
| 0.08 | edf | 47.1 | 0.51 ± 0.22 | 0.93 | 0.07 | 0.0 | 4.10 | 0.24 | 0.66 | 4162 | 4.9/12/22 | 0.9 | 0.01 |
| 0.08 | random | 47.1 | 0.23 ± 0.07 | 0.48 | 0.52 | 0.0 | 9.22 | 0.29 | 0.56 | 4120 | 6.3/14/23 | 0.8 | 0.01 |
| 0.10 | slack | 58.1 | 0.72 ± 0.09 | 0.85 | 0.15 | 0.0 | 3.23 | 0.23 | 0.70 | 4152 | 5.4/14/23 | 3.5 | 0.10 |
| 0.10 | edf | 58.1 | 0.30 ± 0.12 | 0.81 | 0.19 | 0.0 | 4.06 | 0.25 | 0.68 | 4453 | 7.8/17/27 | 4.9 | 0.16 |
| 0.10 | random | 58.1 | 0.19 ± 0.04 | 0.42 | 0.58 | 0.0 | 8.56 | 0.30 | 0.57 | 4316 | 8.4/18/29 | 3.5 | 0.11 |
| 0.12 | slack | 67.9 | 0.65 ± 0.10 | 0.80 | 0.20 | 0.0 | 2.99 | 0.22 | 0.72 | 4372 | 7.2/17/28 | 5.9 | 0.18 |
| 0.12 | edf | 67.9 | 0.26 ± 0.09 | 0.71 | 0.29 | 0.1 | 4.13 | 0.25 | 0.70 | 4591 | 10.1/21/30 | 10.9 | 0.34 |
| 0.12 | random | 67.9 | 0.18 ± 0.04 | 0.38 | 0.62 | 0.3 | 8.43 | 0.30 | 0.58 | 4403 | 10.2/22/30 | 7.8 | 0.22 |
| 0.14 | slack | 79.9 | 0.56 ± 0.08 | 0.72 | 0.28 | 0.1 | 2.79 | 0.22 | 0.73 | 4504 | 9.5/21/30 | 13.3 | 0.44 |
| 0.14 | edf | 79.9 | 0.23 ± 0.03 | 0.61 | 0.39 | 0.7 | 4.14 | 0.25 | 0.71 | 4713 | 12.4/24/30 | 18.6 | 0.70 |
| 0.14 | random | 79.9 | 0.16 ± 0.04 | 0.33 | 0.66 | 0.8 | 8.26 | 0.30 | 0.59 | 4426 | 12.5/25/30 | 15.1 | 0.51 |

Gate G1 (amended 2026-10-11): load levels are calibrated against the strong Slack-aware heuristic,
not EDF (EDF keeps chasing orders that can no longer be on time and falls far behind at moderate load).
- Map: `map_seed = 0` (max A*/straight-line ratio 2.04); 200/200 map seeds generate (1.01 attempts).
- Loads: low λ = 0.04, medium λ = 0.08, high λ = 0.12 orders/min. Slack-aware on-time: 1.00 / 0.84 / 0.65;
  EDF: 1.00 / 0.51 / 0.26. λ = 0.10 is a diagnostic point; λ = 0.14 is a stress setting only
  (an on-time-feasible order is hidden beyond the 12 slots in 13 % of decisions).
- Charging threshold θ tuned per heuristic on dev seeds at λ 0.08 from {20, 30, 40} %:
  EDF 0.514 / 0.444 / 0.421, Slack-aware 0.842 / 0.829 / 0.820 -> θ = 20 % for both.
- Throughput: ~9,100 decisions/s per process (random policy, medium load).

Columns: on-time is the mean over days ± the day-to-day standard deviation; on-time / served /
expired are fractions of all released orders; decisions = total decisions in the row's 40 days; rejected = orders per day
refused because the pending pool was full; km/deliv = km flown per delivered order; charge / busy =
share of the episode spent charging / flying+handling; pool = pending orders at each decision
(mean/p95/max); hidden % = share of decisions in which at least one order that could still be on
time was outside the 12 visible slots; hidden mean = average number of such orders per decision.
