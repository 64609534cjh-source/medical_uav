Command: python scripts/calibrate.py --tune-theta --out outputs/calibration_newmap

map seed 0; demand seeds 100000-100039 (40 days per row); charging threshold EDF 20%, Slack-aware 20% (tuned in this run).

λ is the base arrival rate in orders per minute. Rates are means of the daily rates over the days of a row, with the population SD across days; a day's on-time rate is on-time deliveries / all orders released that day (rejected included). Pool statistics are pooled over all decisions; hidden % is the share of decisions at which an order that could still be on time (direct estimate) is not among the 12 visible slots.

| λ | policy | orders | on-time (mean ± day SD) | served | expired | rejected | km/deliv | charge | busy | decisions | pool mean/p95/max | hidden % | hidden mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.04 | slack | 21.6 | 1.00 ± 0.00 | 1.00 | 0.00 | 0.0 | 4.43 | 0.14 | 0.37 | 3278 | 0.5/2/7 | 0.0 | 0.00 |
| 0.04 | edf | 21.6 | 1.00 ± 0.02 | 1.00 | 0.00 | 0.0 | 4.55 | 0.14 | 0.38 | 3283 | 0.5/2/8 | 0.0 | 0.00 |
| 0.04 | random | 21.6 | 0.59 ± 0.14 | 0.86 | 0.14 | 0.0 | 7.76 | 0.21 | 0.42 | 4349 | 1.5/5/11 | 0.0 | 0.00 |
| 0.06 | slack | 35.5 | 0.96 ± 0.06 | 0.99 | 0.01 | 0.0 | 3.81 | 0.20 | 0.56 | 3749 | 1.5/5/11 | 0.0 | 0.00 |
| 0.06 | edf | 35.5 | 0.86 ± 0.20 | 0.99 | 0.01 | 0.0 | 4.14 | 0.21 | 0.58 | 3806 | 2.0/7/12 | 0.0 | 0.00 |
| 0.06 | random | 35.5 | 0.39 ± 0.13 | 0.72 | 0.28 | 0.0 | 6.45 | 0.24 | 0.51 | 4619 | 3.6/9/14 | 0.0 | 0.00 |
| 0.08 | slack | 47.1 | 0.87 ± 0.08 | 0.95 | 0.05 | 0.0 | 3.33 | 0.22 | 0.65 | 4014 | 3.2/9/14 | 0.3 | 0.00 |
| 0.08 | edf | 47.1 | 0.56 ± 0.23 | 0.94 | 0.06 | 0.0 | 3.93 | 0.24 | 0.66 | 4236 | 4.7/12/22 | 1.3 | 0.02 |
| 0.08 | random | 47.1 | 0.32 ± 0.08 | 0.60 | 0.40 | 0.0 | 6.18 | 0.26 | 0.55 | 4796 | 5.6/13/20 | 0.6 | 0.01 |
| 0.10 | slack | 58.1 | 0.75 ± 0.09 | 0.88 | 0.12 | 0.0 | 3.01 | 0.22 | 0.70 | 4248 | 5.2/14/21 | 3.0 | 0.08 |
| 0.10 | edf | 58.1 | 0.32 ± 0.13 | 0.83 | 0.17 | 0.0 | 3.89 | 0.24 | 0.69 | 4538 | 7.7/17/27 | 4.8 | 0.15 |
| 0.10 | random | 58.1 | 0.27 ± 0.05 | 0.51 | 0.49 | 0.0 | 6.06 | 0.26 | 0.57 | 4898 | 7.7/17/29 | 3.1 | 0.09 |
| 0.12 | slack | 67.9 | 0.67 ± 0.09 | 0.81 | 0.19 | 0.0 | 2.87 | 0.22 | 0.72 | 4426 | 7.1/16/27 | 6.1 | 0.19 |
| 0.12 | edf | 67.9 | 0.29 ± 0.13 | 0.73 | 0.27 | 0.1 | 3.87 | 0.24 | 0.70 | 4646 | 9.8/21/30 | 11.1 | 0.33 |
| 0.12 | random | 67.9 | 0.23 ± 0.06 | 0.46 | 0.54 | 0.2 | 5.95 | 0.27 | 0.58 | 4992 | 9.7/20/30 | 6.8 | 0.20 |
| 0.14 | slack | 79.9 | 0.59 ± 0.07 | 0.73 | 0.27 | 0.1 | 2.68 | 0.22 | 0.74 | 4588 | 9.3/20/30 | 13.2 | 0.43 |
| 0.14 | edf | 79.9 | 0.24 ± 0.03 | 0.63 | 0.36 | 0.6 | 3.91 | 0.24 | 0.71 | 4838 | 12.4/25/30 | 19.6 | 0.73 |
| 0.14 | random | 79.9 | 0.21 ± 0.04 | 0.41 | 0.58 | 0.7 | 5.91 | 0.28 | 0.59 | 5139 | 11.8/24/30 | 13.7 | 0.44 |
