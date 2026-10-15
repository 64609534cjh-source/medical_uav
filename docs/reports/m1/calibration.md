Command: python scripts/calibrate.py --out outputs/calibration

map seed 0; demand seeds 100000-100039 (40 days per row); charging threshold EDF 20%, Slack-aware 20% (defaults).

λ is the base arrival rate in orders per minute. Rates are means of the daily rates over the days of a row, with the population SD across days; a day's on-time rate is on-time deliveries / all orders released that day. Pool statistics are pooled over all decisions; hidden % is the share of decisions at which an order that could still be on time (direct estimate) is not among the 12 visible slots.

| λ | policy | orders | on-time (mean ± day SD) | served | expired | km/deliv | charge | busy | decisions | pool mean/p95/max | hidden % | hidden mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.04 | slack | 21.6 | 0.99 ± 0.02 | 1.00 | 0.00 | 4.39 | 0.14 | 0.37 | 3270 | 0.5/2/5 | 0.0 | 0.00 |
| 0.04 | edf | 21.6 | 1.00 ± 0.02 | 1.00 | 0.00 | 4.54 | 0.14 | 0.38 | 3272 | 0.5/2/6 | 0.0 | 0.00 |
| 0.04 | random | 21.6 | 0.56 ± 0.16 | 0.84 | 0.16 | 8.83 | 0.23 | 0.45 | 3997 | 1.6/5/10 | 0.0 | 0.00 |
| 0.06 | slack | 35.5 | 0.95 ± 0.06 | 0.98 | 0.02 | 3.85 | 0.20 | 0.56 | 3685 | 1.6/6/11 | 0.0 | 0.00 |
| 0.06 | edf | 35.5 | 0.83 ± 0.20 | 0.99 | 0.01 | 4.23 | 0.21 | 0.58 | 3756 | 2.2/7/12 | 0.0 | 0.00 |
| 0.06 | random | 35.5 | 0.36 ± 0.11 | 0.67 | 0.33 | 7.68 | 0.26 | 0.53 | 4368 | 3.8/10/16 | 0.0 | 0.00 |
| 0.09 | slack | 51.8 | 0.81 ± 0.09 | 0.91 | 0.09 | 3.29 | 0.22 | 0.68 | 4084 | 4.0/11/17 | 0.7 | 0.01 |
| 0.09 | edf | 51.8 | 0.41 ± 0.19 | 0.89 | 0.11 | 4.04 | 0.24 | 0.68 | 4328 | 6.1/14/21 | 1.7 | 0.03 |
| 0.09 | random | 51.8 | 0.26 ± 0.07 | 0.52 | 0.48 | 7.06 | 0.28 | 0.57 | 4547 | 6.8/15/23 | 1.0 | 0.02 |
| 0.12 | slack | 67.9 | 0.66 ± 0.10 | 0.81 | 0.19 | 2.90 | 0.22 | 0.72 | 4398 | 7.2/17/29 | 5.8 | 0.17 |
| 0.12 | edf | 67.9 | 0.27 ± 0.11 | 0.72 | 0.28 | 4.00 | 0.25 | 0.70 | 4662 | 10.1/21/33 | 11.4 | 0.34 |
| 0.12 | random | 67.9 | 0.23 ± 0.04 | 0.44 | 0.56 | 6.56 | 0.28 | 0.59 | 4741 | 10.0/21/33 | 8.2 | 0.23 |
| 0.14 | slack | 79.9 | 0.56 ± 0.07 | 0.72 | 0.28 | 2.77 | 0.22 | 0.74 | 4525 | 9.7/21/33 | 14.4 | 0.50 |
| 0.14 | edf | 79.9 | 0.24 ± 0.04 | 0.60 | 0.40 | 4.20 | 0.25 | 0.71 | 4747 | 12.8/25/37 | 19.5 | 0.78 |
| 0.14 | random | 79.9 | 0.18 ± 0.04 | 0.37 | 0.63 | 6.88 | 0.29 | 0.59 | 4856 | 12.3/25/41 | 15.1 | 0.51 |
