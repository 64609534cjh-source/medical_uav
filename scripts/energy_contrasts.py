"""Registered endpoints of the energy study from a behaviour.csv of scripts/analyze_behaviour.py.

Pre-registered in docs/superpowers/specs/2026-10-27-final-phase-preregistration.md, item 1.
Primary: the on-time rate of orders released inside the demand peaks, Slack-100-prepeak minus
Slack-100, mean of medium and high per day, 95 % bootstrap over whole days (both loads resampled
together). Manipulation checks: battery at peak start, charging minutes inside peaks, top-ups
per day. Secondary: Slack-100 - Slack, prepeak - Slack, and each checkpoint minus itself with
full charges hidden (<label>/no100).

Usage:
    python scripts/energy_contrasts.py outputs/energy_select/behaviour.csv
"""
import argparse
import csv
from collections import defaultdict

import numpy as np

from medical_uav.evaluation import paired_ci

LOADS = ("medium", "high")
MARGIN = 0.01  # a practically meaningful gain: 1 point
BATTERY_RISE = 5.0  # battery points at peak start that show the manipulation worked
FEW_TOP_UPS = 0.5  # top-ups per day below which the manipulation counts as failed


def per_day(data, method, key) -> np.ndarray:
    """key averaged over medium and high, one value per day."""
    days = sorted(data[method]["high"])
    return np.array([np.mean([float(data[method][load][d][key]) for load in LOADS])
                     for d in days])


def contrast(data, a, b, key) -> tuple[float, float, float]:
    """a - b on key (mean of medium and high per day), bootstrap over days."""
    return paired_ci(per_day(data, a, key) - per_day(data, b, key))


def verdict(gain, battery_rise, peak_charge_change, top_ups) -> str:
    """The registered interpretation of the primary endpoint; the intended energy changes are a
    higher battery at the peak starts and fewer charging minutes inside the peaks."""
    mean, low, high = gain
    if battery_rise < BATTERY_RISE or top_ups < FEW_TOP_UPS:
        return "failed manipulation: few top-ups or little change in peak-start battery"
    if high < 0:
        return "the intervention harms peak service"
    if peak_charge_change < 0 and mean >= MARGIN and low > 0:
        return "anticipatory top-ups improve peak service, with the predicted energy changes"
    if peak_charge_change < 0 and high < MARGIN:
        return "no practically meaningful peak-service benefit reproduced"
    return "inconclusive"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv")
    args = parser.parse_args()

    data = defaultdict(lambda: defaultdict(dict))
    for r in csv.DictReader(open(args.csv)):
        data[r["method"]][r["load"]][int(r["seed"])] = r
    fmt = lambda c, k=100: f"{c[0] * k:+6.2f} [{c[1] * k:+6.2f}, {c[2] * k:+6.2f}]"
    n_days = len(next(iter(data.values()))["high"])
    print(f"{n_days} days; differences in points (rates) or battery points / minutes; "
          "95 % bootstrap over days\n")

    print("condition means (medium / high):")
    keys = ("on_time_rate", "on_time_peak", "on_time_offpeak", "battery_at_peak",
            "peak_charge_minutes", "charge_minutes", "charge_full_share", "km_flown", "overrides")
    print(f"  {'method':32s}" + "".join(f"{k[:17]:>20s}" for k in keys))
    for method in data:
        cells = []
        for k in keys:
            v = [np.nanmean([float(r[k]) for r in data[method][load].values()]) for load in LOADS]
            cells.append(f"{v[0]:9.3f} /{v[1]:8.3f}")
        print(f"  {method:32s}" + "".join(f"{c:>20s}" for c in cells))

    a, b = "prepeak", "slack100"
    gain = contrast(data, a, b, "on_time_peak")
    battery = contrast(data, a, b, "battery_at_peak")
    peak_charge = contrast(data, a, b, "peak_charge_minutes")
    print(f"\nPRIMARY  peak on-time, {a} - {b}: {fmt(gain)}")
    for load in LOADS:
        d = [float(data[a][load][s]["on_time_peak"]) - float(data[b][load][s]["on_time_peak"])
             for s in sorted(data[a][load])]
        print(f"  {load:7s} {fmt(paired_ci(d))}")
    top_ups = np.mean(per_day(data, a, "overrides"))  # per day and load
    print(f"  battery at peak start {fmt(battery, 1)}; charging minutes in peaks "
          f"{fmt(peak_charge, 1)}; top-ups per day {top_ups:.2f}")
    print(f"  VERDICT: {verdict(gain, battery[0], peak_charge[0], top_ups)}")

    print("\nsecondary (peak on-time | all-day on-time):")
    for x, y in (("slack100", "slack"), ("prepeak", "slack")):
        print(f"  {x} - {y}: {fmt(contrast(data, x, y, 'on_time_peak'))} | "
              f"{fmt(contrast(data, x, y, 'on_time_rate'))}")
    overall = contrast(data, "prepeak", "slack", "on_time_rate")
    print(f"  prepeak overall useful (all-day lower bound > -1 point vs Slack): "
          f"{'yes' if overall[1] > -MARGIN else 'no'}")
    masked = [m[:-len("/no100")] for m in data if m.endswith("/no100")]
    if masked:
        diffs = {key: [per_day(data, m, key) - per_day(data, m + "/no100", key) for m in masked]
                 for key in ("on_time_peak", "on_time_rate")}
        for m, peak, day in zip(masked, diffs["on_time_peak"], diffs["on_time_rate"]):
            print(f"  {m} - no100: {fmt(paired_ci(peak))} | {fmt(paired_ci(day))}")
        print(f"  seed-averaged PPO - no100: {fmt(paired_ci(np.mean(diffs['on_time_peak'], 0)))} | "
              f"{fmt(paired_ci(np.mean(diffs['on_time_rate'], 0)))}")


if __name__ == "__main__":
    main()
