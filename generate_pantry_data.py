from __future__ import annotations

import argparse
import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


SCENARIO_PRESETS = {
    "baseline": dict(
        hardship_beta=(2.2, 2.4),
        econ_shift=0.00,
        precip_scale=1.00,
        precip_add=0.00,
        extra_storm_rate=0.02,
        turnout_penalty=(0.85, 0.94),
        needs_boost=(1.05, 1.12),
        donations_scale=1.00,
    ),
    "extreme_weather": dict(
        hardship_beta=(2.2, 2.4),
        econ_shift=0.02,
        precip_scale=1.65,
        precip_add=0.22,
        extra_storm_rate=0.25,
        turnout_penalty=(0.68, 0.90),
        needs_boost=(1.12, 1.30),
        donations_scale=0.92,
    ),
    "extreme_hardship": dict(
        hardship_beta=(4.9, 1.6),
        econ_shift=0.22,
        precip_scale=1.00,
        precip_add=0.00,
        extra_storm_rate=0.03,
        turnout_penalty=(0.84, 0.94),
        needs_boost=(1.05, 1.15),
        donations_scale=0.92,
    ),
    "low_hardship": dict(
        hardship_beta=(1.6, 5.2),
        econ_shift=-0.06,
        precip_scale=1.00,
        precip_add=0.00,
        extra_storm_rate=0.01,
        turnout_penalty=(0.86, 0.95),
        needs_boost=(1.03, 1.10),
        donations_scale=1.06,
    ),
}


def _make_event_blocks(
    rng: np.random.Generator,
    n_cycles: int,
    blocks_range=(2, 5),
    dur_range=(2, 10),
) -> np.ndarray:
    ev = np.zeros(n_cycles, dtype=np.int8)
    n_blocks = int(rng.integers(blocks_range[0], blocks_range[1] + 1))
    for _ in range(n_blocks):
        start = int(rng.integers(0, max(1, n_cycles - dur_range[1] - 1)))
        dur = int(rng.integers(dur_range[0], dur_range[1] + 1))
        ev[start : start + dur] = 1
    return ev


def generate_df(
    n_rows: int,
    scenario: str,
    seed: int,
    noise_scale: float,
    randomness: float,
    crisis_level: str = "med",
    start_date: str = "2021-01-03",
    years: int = 5,
    cycle_days: int = 14,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cfg = SCENARIO_PRESETS[scenario]

    # biweekly cycle dates
    start = datetime.fromisoformat(start_date)
    end = start + timedelta(days=365 * years)
    dates = []
    d = start
    while d < end:
        dates.append(d)
        d += timedelta(days=cycle_days)
    dates = np.array(dates, dtype="datetime64[D]")
    n_cycles = len(dates)

    # replicate pantries to reach n_rows
    n_pantries = math.ceil(n_rows / n_cycles)
    total = n_pantries * n_cycles

    pantry_id = np.repeat(np.arange(n_pantries, dtype=np.int32), n_cycles)
    cycle_start = np.tile(dates, n_pantries)

    # pantry-level attributes
    a, b = cfg["hardship_beta"]
    hardship = np.clip(rng.beta(a, b, size=n_pantries), 0, 1)
    immigrant_share = np.clip(rng.beta(2.0, 3.2, size=n_pantries), 0, 1)
    capacity = rng.integers(120, 780, size=n_pantries)

    # pantry-specific sensitivities
    econ_sens = np.clip(rng.normal(0.25, 0.10 * randomness, size=n_pantries), 0.05, 0.65)
    weather_sens = np.clip(rng.normal(1.0, 0.18 * randomness, size=n_pantries), 0.6, 1.6)
    donation_sens = np.clip(rng.normal(1.0, 0.20 * randomness, size=n_pantries), 0.6, 1.5)
    lbs_base_shift = rng.normal(0.0, 2.0 * randomness, size=n_pantries)

    baseline_households = (
        55
        + 230 * hardship
        + 90 * immigrant_share
        + rng.normal(0, 22 * randomness, size=n_pantries)
    )
    baseline_households = np.clip(baseline_households, 20, 720)

    hh_size = np.clip(rng.normal(2.9 + 0.5 * immigrant_share, 0.32, size=n_pantries), 1.8, 5.7)

    # expand (length total)
    hardship_r = hardship[pantry_id]
    immigrant_r = immigrant_share[pantry_id]
    capacity_r = capacity[pantry_id]
    base_hh_r = baseline_households[pantry_id]
    hh_size_r = hh_size[pantry_id]
    econ_sens_r = econ_sens[pantry_id]
    weather_sens_r = weather_sens[pantry_id]
    donation_sens_r = donation_sens[pantry_id]
    lbs_shift_r = lbs_base_shift[pantry_id]

    # date features
    cycle_start_pd = pd.to_datetime(cycle_start.astype("datetime64[ns]"))
    year = cycle_start_pd.year.values.astype(np.int16)
    month = cycle_start_pd.month.values.astype(np.int8)
    weekofyear = cycle_start_pd.isocalendar().week.values.astype(np.int16)

    season = np.where(
        np.isin(month, [12, 1, 2]), "winter",
        np.where(
            np.isin(month, [3, 4, 5]),
            "spring",
            np.where(np.isin(month, [6, 7, 8]), "summer", "fall"),
        ),
    )

    holiday = (
        ((month == 11) & (rng.random(total) < 0.40))
        | ((month == 12) & (rng.random(total) < 0.55))
        | ((month == 1) & (rng.random(total) < 0.10))
    ).astype(np.int8)

    school_break = (
        ((month == 12) & (rng.random(total) < 0.40))
        | ((month == 2) & (rng.random(total) < 0.16))
        | ((month == 4) & (rng.random(total) < 0.14))
        | ((month >= 6) & (month <= 8) & (rng.random(total) < 0.40))
    ).astype(np.int8)

    # ---------- MACRO SHOCKS ----------
    unique_years = np.unique(year)
    y_mult = np.exp(rng.normal(0, 0.12 * randomness, size=len(unique_years)))
    year_to_mult = {int(y): float(m) for y, m in zip(unique_years, y_mult)}
    year_mult = np.array([year_to_mult[int(y)] for y in year], dtype=np.float32)

    m_mult = np.exp(rng.normal(0, 0.08 * randomness, size=12))
    month_mult = m_mult[month - 1].astype(np.float32)

    # ---------- timeline events (flags length = total) ----------
    if crisis_level == "low":
        crisis_blocks = (2, 5)
        crisis_durs = (2, 8)
    elif crisis_level == "high":
        crisis_blocks = (10, 18)
        crisis_durs = (6, 16)
    else:  # "med"
        crisis_blocks = (6, 12)
        crisis_durs = (4, 12)

    crisis_by_cycle = _make_event_blocks(rng, n_cycles, blocks_range=crisis_blocks, dur_range=crisis_durs)
    drive_by_cycle = _make_event_blocks(rng, n_cycles, blocks_range=(1, 4), dur_range=(1, 6))

    cycle_index = np.tile(np.arange(n_cycles, dtype=np.int32), n_pantries)
    crisis_flag = crisis_by_cycle[cycle_index].astype(np.int8)
    food_drive_flag = drive_by_cycle[cycle_index].astype(np.int8)

    # ---------- weather ----------
    month_temp_mean = np.array([31, 34, 43, 53, 62, 71, 76, 75, 67, 56, 46, 35])
    temp_f = month_temp_mean[month - 1] + rng.normal(0, 6.0, size=total)

    precip_in = np.clip(
        (rng.gamma(shape=2.0, scale=0.55, size=total) - 0.18) * cfg["precip_scale"] + cfg["precip_add"],
        0,
        None,
    )

    severe_weather = (
        ((month <= 3) & (precip_in > 1.6) & (rng.random(total) < 0.55))
        | ((month == 12) & (precip_in > 1.5) & (rng.random(total) < 0.50))
    )

    extra = cfg["extra_storm_rate"]
    if extra > 0:
        winterish = (month <= 3) | (month == 12)
        severe_weather = severe_weather | (winterish & (rng.random(total) < extra))
    severe_weather = severe_weather.astype(np.int8)

    # ---------- econ pressure ----------
    year_idx = (year - year.min()).astype(np.float32)
    econ_pressure = np.clip(
        (0.25 + 0.08 * year_idx + rng.normal(0, 0.06, size=total) + cfg["econ_shift"]) * year_mult,
        0,
        1.5,
    )

    # ---------- donations ----------
    donations = (
        120
        + 1.6 * capacity_r
        + 140 * (season == "fall")
        + 220 * (season == "winter")
        + rng.normal(0, 120 * randomness, size=total)
    )
    donations *= cfg["donations_scale"]
    donations *= (1.0 + 0.10 * food_drive_flag)
    # emergency aid / shipments in crisis (SUPPLY can rise in emergencies)
    donations *= (1.0 + rng.uniform(0.05, 0.20, size=total) * crisis_flag)
    donations *= donation_sens_r
    donations *= month_mult
    donations = np.clip(donations, 0, None)

    # ---------- Targets ----------
    seasonal_mult = np.select(
        [season == "winter", season == "spring", season == "summer", season == "fall"],
        [1.10, 1.02, 0.95, 1.05],
        default=1.0,
    ).astype(np.float32)

    holiday_mult = (1.0 + 0.12 * holiday).astype(np.float32)
    school_mult = (1.0 + 0.06 * school_break).astype(np.float32)

    lo, hi = cfg["turnout_penalty"]
    base_turnout_weather = np.where(
        severe_weather == 1, rng.uniform(lo, hi, size=total), 1.0
    ).astype(np.float32)
    turnout_weather_mult = np.clip(base_turnout_weather * weather_sens_r, 0.55, 1.10)

    # crisis pushes demand up (more people show up)
    crisis_demand_mult = 1.0 + rng.uniform(0.10, 0.35, size=total) * crisis_flag

    expected_households = (
        base_hh_r
        * (1.0 + econ_sens_r * econ_pressure)
        * seasonal_mult
        * holiday_mult
        * school_mult
        * turnout_weather_mult
        * crisis_demand_mult
        * (1.0 + rng.normal(0, noise_scale * randomness, size=total))
    )

    households = np.clip(expected_households, 8, capacity_r * rng.uniform(0.80, 1.10, size=total))
    households = np.round(households).astype(np.int16)

    individuals = np.round(households * (hh_size_r + rng.normal(0, 0.24, size=total))).astype(np.int16)
    individuals = np.clip(individuals, households, None)

    lo, hi = cfg["needs_boost"]
    needs_weather_mult = np.where(
        severe_weather == 1, rng.uniform(lo, hi, size=total), 1.0
    ).astype(np.float32)

    lbs_per_household = (
        28
        + 12 * hardship_r
        + 3.0 * (season == "winter").astype(np.float32)
        + lbs_shift_r
        + rng.normal(0, 5.0 * randomness, size=total)
    )
    lbs_per_household = np.clip(lbs_per_household, 16, 62) * needs_weather_mult

    # emergency need = more food per household (NEED ↑ in crisis)
    crisis_need_mult = 1.0 + rng.uniform(0.15, 0.45, size=total) * crisis_flag
    lbs_per_household *= crisis_need_mult

    total_food_lbs = households.astype(np.float32) * lbs_per_household
    total_food_lbs *= (1.0 + np.clip(donations / 3500, 0, 0.15))
    # extra emergency distribution effort
    total_food_lbs *= (1.0 + rng.uniform(0.05, 0.18, size=total) * crisis_flag)
    total_food_lbs = np.round(total_food_lbs).astype(np.int32)

    overhead = 1.0 + 0.12 * holiday + 0.06 * school_break + 0.12 * severe_weather + 0.08 * crisis_flag
    hh_per_vol_hr = np.clip(rng.normal(8.2 - 1.0 * hardship_r, 1.0 * randomness, size=total), 4.8, 11.5)
    volunteer_hours = (households / hh_per_vol_hr) * overhead
    volunteers = np.ceil(volunteer_hours / 3.0).astype(np.int16)
    volunteers = np.clip(volunteers, 3, 55)

    df = pd.DataFrame(
        {
            "scenario": scenario,
            "pantry_id": pantry_id[:n_rows],
            "cycle_start_date": cycle_start_pd[:n_rows].strftime("%Y-%m-%d"),
            "year": year[:n_rows],
            "month": month[:n_rows],
            "weekofyear": weekofyear[:n_rows],
            "season": season[:n_rows],
            "holiday_flag": holiday[:n_rows],
            "school_break_flag": school_break[:n_rows],
            "severe_weather_flag": severe_weather[:n_rows],
            "avg_temp_f": np.round(temp_f[:n_rows], 1),
            "precip_in": np.round(precip_in[:n_rows], 2),
            "econ_pressure_index": np.round(econ_pressure[:n_rows], 3),
            "donations_lbs": np.round(donations[:n_rows], 1),
            "hardship_index": np.round(hardship_r[:n_rows], 3),
            "immigrant_share": np.round(immigrant_r[:n_rows], 3),
            "capacity_households": capacity_r[:n_rows].astype(np.int16),
            "crisis_flag": crisis_flag[:n_rows],
            "food_drive_flag": food_drive_flag[:n_rows],
            "households_served": households[:n_rows],
            "individuals_served": individuals[:n_rows],
            "total_food_lbs_distributed": total_food_lbs[:n_rows],
            "volunteers_recommended": volunteers[:n_rows],
        }
    )
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=500_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--noise", type=float, default=0.025)
    p.add_argument("--randomness", type=float, default=2.0, help=">1 = more run-to-run randomness")
    p.add_argument("--scenario", choices=list(SCENARIO_PRESETS.keys()), default="baseline")
    p.add_argument("--crisis-level", choices=["low", "med", "high"], default="med")
    p.add_argument("--out", type=str, default="pantry_data.csv.gz")
    args = p.parse_args()

    df = generate_df(
        n_rows=args.rows,
        scenario=args.scenario,
        seed=args.seed,
        noise_scale=args.noise,
        randomness=args.randomness,
        crisis_level=args.crisis_level,
    )

    out_path = Path(args.out).expanduser().resolve()
    if str(out_path).endswith(".gz"):
        df.to_csv(out_path, index=False, compression="gzip")
    else:
        df.to_csv(out_path, index=False)

    print(f"Saved {len(df):,} rows → {out_path}")


if __name__ == "__main__":
    main()