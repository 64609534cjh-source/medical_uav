# Failure cases: the three registered days (2026-10-27)

Days chosen by the registered rule (roadmap §9.20). Analysis by Codex (gpt-6-astra, high) from the replay records of the shipped PPO and Slack-aware on each day (`python scripts/demo.py --policy {ppo,slack} --load high --seed <day>`); totals checked against the records. Verbatim.

All times below are minutes after 08:00; decision indices and order IDs are zero-based. Peak release windows are [60,180) and [300,390). Counts and timings come from the six replay records.

Day 100004, the median-selected case: PPO delivered 61/71 on time versus Slack’s 48/71, a **+18.3-point** difference; +19.0 points is the screen median.

- PPO missed 6 medicine, 3 plasma and 1 sample order; supplies had no misses. All 10 expired waiting; none expired on board or arrived late.
- By release hour: 09:00–10:00 had 4 misses, 10:00–11:00 had 2, 13:00–14:00 had 2, 14:00–15:00 had 1 and 15:00–16:00 had 1. Six belonged to the first peak, two to the second and two were off-peak.
- At peak starts, PPO/Slack battery was 100.0%/87.1% and 81.3%/36.0%. PPO charged at M during 284.85–289.06; Slack charged there during 296.05–308.79, crossing the second peak start. Actual charging inside peaks totalled 28.01 versus 44.53 minutes.
- PPO alone served 19 orders on time; Slack alone served 6: medicine #11, #12 and #28, sample #22, and plasma #52 and #68. Thus PPO’s higher total still included orders that Slack successfully served.
- Show decision **38 at 164.79**: deliver two samples to M, probability 50.69%, versus another H5 sample pickup, 49.11%. Samples #16/#19 arrived at 169.46/171.46, both on time; Slack never picked either up.
- Show decision **85 at 349.69**: pick medicine #50 for H4, 92.64%, versus plasma for H1, 7.35%. This filled the third slot; PPO then charged during 351.69–358.35 and completed the medicine batch by 367.41. Plasma #52 remained unpicked and expired at 407.96; Slack delivered it at 374.11, before its 377.96 deadline.

Day 100006, the smallest-margin case: PPO delivered 56/57 on time versus Slack’s 53/57, **+5.3 points**.

- PPO’s sole miss was supplies #46, released at 367.10, within the second peak and the 14:00–15:00 hour. It arrived 2.45 minutes late. All plasma, medicine and sample orders were on time; no order expired waiting or on board.
- This day had 57 orders, versus 71 and 82 in the other cases; first-peak arrivals numbered 19 versus 31 and 40. Both policies were near the ceiling: 98.2% and 93.0%. Slack had only four misses available for PPO to improve upon.
- PPO served all four of those orders on time: #20 and #45, which Slack left to expire, and #36 and #54, which Slack delivered late. PPO’s separate miss on #46 reduced the net gain to three orders.
- Peak-start batteries were 71.9%/33.1% and 100.0%/68.7%. PPO charged at M during 290.05–294.57; Slack’s M charge ran 291.09–302.26. Charging inside peaks totalled 35.72 versus 49.67 minutes.
- Show the costly decision **107 at 435.47**: deliver medicine #49 to H4, 94.29%, versus deliver supplies #46 to H2, 5.70%. PPO completed H4 at 440.49, then H2 at 444.55, missing #46’s 442.10 deadline. Slack picked up that same supplies order at 424.26 and delivered it directly at 429.27.

Day 100009, the plasma-shortfall case: PPO delivered 60/82 on time versus Slack’s 43/82, **+20.7 points**. Plasma service was 13/24 versus 9/24.

- PPO missed 11 plasma, 5 medicine, 4 sample and 2 supplies orders. Twenty expired waiting; none expired on board. Medicine #11 and sample #37 arrived late by 7.86 and 0.21 minutes.
- By release hour: 09:00–10:00 had 7 misses, 10:00–11:00 had 10, 11:00–12:00 had 1, 12:00–13:00 had 1 and 14:00–15:00 had 3. Seventeen belonged to the first peak and two to the second; three were off-peak, including the plasma order released at 393.17.
- Peak-start batteries were 100.0%/80.0% and 48.9%/56.8%. PPO charged at M during 301.69–311.91; Slack had charged there during 276.70–287.96. PPO therefore entered the second peak with less battery. Charging inside peaks nevertheless totalled 41.90 versus 57.11 minutes.
- PPO alone served 26 orders on time; Slack alone served 9. For plasma specifically, PPO alone served nine, while Slack alone served #15, #24, #44, #66 and #67.
- The later plasma split is visible: PPO charged at C2 during 376.31–385.23, then travelled to M and topped up, finishing at 389.59. It delivered newer plasma #68/#70 at 397.29/400.90 but left #66/#67 to expire. Slack delivered #66/#67 at 386.30/397.06 and left #70 to expire.
- Show decision **40 at 168.94**: charge M to 60%, 50.15%; pick H4 medicine, 41.20%; charge M to 100%, 6.93%. With medicine #11 aboard, charging ended at 176.51, after its 174.36 deadline. PPO added plasma #40 and delivered #11 at 182.22.
- Show decision **52 at 216.12**: pick another sample at H5, 99.82%, while carrying H4 sample #37. The resulting batch reached M at 223.54/225.54: #37 was 0.21 minutes late, while added sample #39 was on time. Slack left both to expire waiting.

Across these cases, 30 of PPO’s 33 misses expired before pickup; the other three were late deliveries, with no onboard expiries.  
Twenty-eight misses came from peak releases, including 17 from day 100009’s first peak.  
The traces contain successful batches alongside urgent orders left waiting, and delivery or pickup sequences followed by small deadline overruns.  
PPO spent fewer minutes charging inside peaks in every case, but neither higher peak-start battery nor better service on every individual order was universal.
