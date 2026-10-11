map_seed = 0; dev seeds 100000..100039 (40 days per row); 3230 decisions in the first row

| λ | policy | orders | on-time | served | expired | rejected | km/deliv | charge | busy | pool mean/p95/max | hidden % | hidden mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.04 | slack | 21.6 | 1.00 ± 0.01 | 1.00 | 0.00 | 0.0 | 4.71 | 0.15 | 0.39 | 0.5/2/5 | 0.0 | 0.00 |
| 0.04 | edf | 21.6 | 0.99 ± 0.03 | 1.00 | 0.00 | 0.0 | 5.01 | 0.16 | 0.40 | 0.6/3/7 | 0.0 | 0.00 |
| 0.04 | random | 21.6 | 0.42 ± 0.14 | 0.75 | 0.25 | 0.0 | 10.97 | 0.25 | 0.46 | 1.9/5/12 | 0.0 | 0.00 |
| 0.08 | slack | 47.1 | 0.83 ± 0.10 | 0.92 | 0.08 | 0.0 | 3.59 | 0.23 | 0.66 | 3.3/9/17 | 0.5 | 0.01 |
| 0.08 | edf | 47.1 | 0.42 ± 0.22 | 0.88 | 0.12 | 0.0 | 4.88 | 0.25 | 0.66 | 6.2/15/23 | 2.4 | 0.05 |
| 0.08 | random | 47.1 | 0.23 ± 0.07 | 0.51 | 0.49 | 0.0 | 8.59 | 0.29 | 0.56 | 6.2/14/20 | 0.7 | 0.01 |
| 0.10 | slack | 58.1 | 0.70 ± 0.10 | 0.84 | 0.16 | 0.0 | 3.28 | 0.23 | 0.69 | 5.5/15/23 | 3.5 | 0.10 |
| 0.10 | edf | 58.1 | 0.26 ± 0.10 | 0.73 | 0.27 | 0.0 | 5.02 | 0.26 | 0.68 | 9.0/19/30 | 6.4 | 0.22 |
| 0.10 | random | 58.1 | 0.19 ± 0.05 | 0.41 | 0.59 | 0.0 | 8.65 | 0.29 | 0.56 | 8.5/19/28 | 3.9 | 0.13 |
| 0.12 | slack | 67.9 | 0.65 ± 0.10 | 0.79 | 0.21 | 0.0 | 3.00 | 0.23 | 0.72 | 7.2/17/28 | 6.1 | 0.18 |
| 0.12 | edf | 67.9 | 0.25 ± 0.08 | 0.63 | 0.36 | 0.2 | 5.13 | 0.26 | 0.69 | 11.1/22/30 | 13.6 | 0.42 |
| 0.12 | random | 67.9 | 0.17 ± 0.05 | 0.36 | 0.63 | 0.2 | 8.83 | 0.30 | 0.58 | 10.5/22/30 | 8.3 | 0.26 |
| 0.14 | slack | 79.9 | 0.55 ± 0.08 | 0.70 | 0.29 | 0.2 | 2.88 | 0.23 | 0.72 | 9.6/21/30 | 14.0 | 0.45 |
| 0.14 | edf | 79.9 | 0.24 ± 0.04 | 0.54 | 0.45 | 1.1 | 5.15 | 0.26 | 0.70 | 13.3/25/30 | 20.8 | 0.81 |
| 0.14 | random | 79.9 | 0.15 ± 0.04 | 0.32 | 0.67 | 1.1 | 8.53 | 0.30 | 0.58 | 12.4/25/30 | 14.8 | 0.50 |

Gate G1 (amended 2026-10-11): load levels are calibrated against the strong Slack-aware heuristic,
not EDF (EDF chases hopeless late orders and collapses at moderate load).
- Map: `map_seed = 0` (max A*/straight-line ratio 2.04); 200/200 map seeds generate (1.01 attempts).
- Loads: low λ = 0.04, medium λ = 0.08, high λ = 0.12 orders/min. Slack-aware on-time: 1.00 / 0.83 / 0.65.
  λ = 0.10 is a diagnostic point; λ = 0.14 is a stress setting only (on-time-feasible orders hidden
  beyond the 12 slots in 14 % of decisions).
- Slack-aware charging threshold θ = 20 % (dev, λ 0.08: θ 20 → 0.831, 30 → 0.823, 40 → 0.813;
  the differences are within day-to-day noise).
- Throughput: ~9,100 decisions/s per process (random policy, medium load).

Columns: on-time / served / expired are fractions of all released orders; rejected = orders per day
refused because the pending pool was full; km/deliv = km flown per delivered order; charge / busy =
share of the episode spent charging / flying+handling; pool = pending orders at each decision
(mean/p95/max); hidden % = share of decisions in which at least one order that could still be on
time was outside the 12 visible slots; hidden mean = average number of such orders per decision.
