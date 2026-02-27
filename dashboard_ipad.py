# dashboard.py
from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Pantry Forecast Dashboard", page_icon="🥫", layout="wide")

# -----------------------------
# CLI args (optional; Streamlit Cloud usually won't pass these)
# -----------------------------
def get_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--data", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--metrics", default=None)
    args, _ = p.parse_known_args()
    return args

ARGS = get_args()

ROOT = Path(__file__).parent.resolve()

# -----------------------------
# Helpers
# -----------------------------
def load_json(path: str | Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def season_from_month(m: int) -> str:
    if m in [12, 1, 2]:
        return "winter"
    if m in [3, 4, 5]:
        return "spring"
    if m in [6, 7, 8]:
        return "summer"
    return "fall"

def pred_range(pred, mape_frac):
    pred = float(pred)
    m = float(mape_frac)
    lo = max(0.0, pred * (1.0 - m))
    hi = max(0.0, pred * (1.0 + m))
    return lo, hi

def fmt_int(x): return f"{int(round(float(x))):,}"
def fmt_lbs(x): return f"{int(round(float(x))):,} lbs"
def fmt_float(x, d=1): return f"{float(x):.{d}f}"

FEATURE_FALLBACK = [
    "pantry_id","year","month","weekofyear","season",
    "holiday_flag","school_break_flag","severe_weather_flag",
    "avg_temp_f","precip_in","econ_pressure_index","donations_lbs",
    "hardship_index","immigrant_share","capacity_households",
    "crisis_flag","food_drive_flag"
]
TARGET_FALLBACK = ["households_served","total_food_lbs_distributed","volunteers_recommended"]

def pick_default_data() -> str | None:
    """
    Prefer a small demo file in the repo (best for Streamlit Cloud).
    If not present, fall back to newest file in scenario_runs/.
    """
    demo = ROOT / "demo_data.csv.gz"
    if demo.exists():
        return str(demo)

    sr = ROOT / "scenario_runs"
    if sr.exists():
        files = list(sr.glob("*.csv.gz")) + list(sr.glob("*.csv"))
        if files:
            newest = max(files, key=lambda p: p.stat().st_mtime)
            return str(newest)

    return None

def pick_default_model() -> str | None:
    """
    Prefer model.joblib at repo root; otherwise newest in scenario_runs/.
    """
    root_model = ROOT / "model.joblib"
    if root_model.exists():
        return str(root_model)

    sr = ROOT / "scenario_runs"
    if sr.exists():
        files = list(sr.glob("model_*.joblib")) + list(sr.glob("*.joblib"))
        if files:
            newest = max(files, key=lambda p: p.stat().st_mtime)
            return str(newest)

    return None

def pick_default_metrics() -> str | None:
    """
    Prefer metrics.json at repo root; otherwise newest in scenario_runs/.
    """
    root_metrics = ROOT / "metrics.json"
    if root_metrics.exists():
        return str(root_metrics)

    sr = ROOT / "scenario_runs"
    if sr.exists():
        files = list(sr.glob("metrics_*.json")) + list(sr.glob("*.json"))
        if files:
            newest = max(files, key=lambda p: p.stat().st_mtime)
            return str(newest)

    return None

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    d["cycle_start_date"] = pd.to_datetime(d["cycle_start_date"], errors="coerce")
    return d

# -----------------------------
# Resolve paths (works locally AND on Streamlit Cloud)
# -----------------------------
data_path = ARGS.data or pick_default_data()
model_path = ARGS.model or pick_default_model()
metrics_path = ARGS.metrics or pick_default_metrics()

if not model_path:
    st.error("No model found. Add model.joblib to repo root (recommended) or pass --model.")
    st.stop()

# Load model
model = joblib.load(model_path)

# Load metrics (optional)
if metrics_path and Path(metrics_path).exists():
    metrics = load_json(metrics_path)
    FEATURES = metrics.get("features", FEATURE_FALLBACK)
    TARGETS = metrics.get("targets", TARGET_FALLBACK)
    MAPE_BY_TARGET = metrics.get("mape_by_target", {})
else:
    metrics = None
    FEATURES = FEATURE_FALLBACK
    TARGETS = TARGET_FALLBACK
    MAPE_BY_TARGET = {
        "households_served": 0.14,
        "total_food_lbs_distributed": 0.18,
        "volunteers_recommended": 0.16
    }

# Load data
if not data_path:
    st.error(
        "No data found.\n\n"
        "✅ For Streamlit Cloud: add a small demo file named **demo_data.csv.gz** to your repo.\n"
        "Or run locally with: `streamlit run dashboard.py -- --data <file> --model <file> --metrics <file>`"
    )
    st.stop()

df = load_data(data_path)

# -----------------------------
# UI
# -----------------------------
st.title("Community Food Pantry Predictive Model")
st.caption(f"Data: {data_path}")
st.caption(f"Model: {model_path}")
if metrics_path:
    st.caption(f"Metrics: {metrics_path}")

# ---------- Choose pantry + build “latest row” ----------
pantry_ids = sorted(df["pantry_id"].dropna().unique().tolist())
if not pantry_ids:
    st.error("No pantry_id values found in the dataset.")
    st.stop()

default_pid = pantry_ids[0]

left, right = st.columns([1.15, 1])

with left:
    st.subheader("Auto-loaded inputs (latest cycle → predict next cycle)")

    pantry_id = st.selectbox(
        "Pantry ID",
        pantry_ids,
        index=pantry_ids.index(default_pid) if default_pid in pantry_ids else 0
    )

    dfp = df[df["pantry_id"] == pantry_id].sort_values("cycle_start_date")
    latest = dfp.iloc[-1].copy()

    # predict next cycle date (biweekly)
    next_date = latest["cycle_start_date"] + timedelta(days=14)

    # derive next cycle calendar fields
    year = int(next_date.year)
    month = int(next_date.month)
    weekofyear = int(next_date.isocalendar().week)
    season = season_from_month(month)

    # default carry-over signals (staff can override)
    holiday_flag = int(latest.get("holiday_flag", 0))
    school_break_flag = int(latest.get("school_break_flag", 0))
    severe_weather_flag = int(latest.get("severe_weather_flag", 0))
    crisis_flag = int(latest.get("crisis_flag", 0))
    food_drive_flag = int(latest.get("food_drive_flag", 0))

    avg_temp_f = float(latest.get("avg_temp_f", 70.0))
    precip_in = float(latest.get("precip_in", 0.8))
    econ_pressure_index = float(latest.get("econ_pressure_index", 0.55))
    donations_lbs = float(latest.get("donations_lbs", 2500.0))
    hardship_index = float(latest.get("hardship_index", 0.5))
    immigrant_share = float(latest.get("immigrant_share", 0.35))
    capacity_households = int(latest.get("capacity_households", 600))

    st.write(f"**Latest cycle in data:** {latest['cycle_start_date'].date()}")
    st.write(f"**Forecasting next cycle:** {next_date.date()}")

    st.markdown("---")
    st.subheader("Optional staff overrides")

    c1, c2, c3 = st.columns(3)
    with c1:
        holiday_flag = st.checkbox("Vacation / Holiday?", value=bool(holiday_flag))
        school_break_flag = st.checkbox("School break?", value=bool(school_break_flag))
    with c2:
        severe_weather_flag = st.checkbox("Severe weather expected?", value=bool(severe_weather_flag))
        crisis_flag = st.checkbox("Emergency / Crisis active?", value=bool(crisis_flag))
    with c3:
        food_drive_flag = st.checkbox("Food drive / aid week?", value=bool(food_drive_flag))

    c4, c5, c6 = st.columns(3)
    with c4:
        avg_temp_f = st.number_input("Avg temp (°F)", value=float(avg_temp_f))
        precip_in = st.number_input("Precip (inches)", value=float(precip_in))
    with c5:
        donations_lbs = st.number_input("Expected donations (lbs)", min_value=0.0, value=float(donations_lbs))
        econ_pressure_index = st.number_input(
            "Economic pressure (0–1.5)",
            min_value=0.0, max_value=1.5,
            value=float(econ_pressure_index)
        )
    with c6:
        hardship_index = st.number_input("Hardship index (0–1)", min_value=0.0, max_value=1.0, value=float(hardship_index))
        immigrant_share = st.number_input("Immigrant share (0–1)", min_value=0.0, max_value=1.0, value=float(immigrant_share))

    capacity_households = st.number_input("Capacity (households per cycle)", min_value=20, value=int(capacity_households))
    distribution_days = st.number_input("Distribution days in cycle", min_value=1, max_value=31, value=5)

    run = st.button("Generate Forecast", type="primary")


def build_X() -> pd.DataFrame:
    row = {
        "pantry_id": int(pantry_id),
        "year": int(year),
        "month": int(month),
        "weekofyear": int(weekofyear),
        "season": str(season),

        "holiday_flag": int(holiday_flag),
        "school_break_flag": int(school_break_flag),
        "severe_weather_flag": int(severe_weather_flag),

        "avg_temp_f": float(avg_temp_f),
        "precip_in": float(precip_in),
        "econ_pressure_index": float(econ_pressure_index),
        "donations_lbs": float(donations_lbs),

        "hardship_index": float(hardship_index),
        "immigrant_share": float(immigrant_share),
        "capacity_households": int(capacity_households),

        "crisis_flag": int(crisis_flag),
        "food_drive_flag": int(food_drive_flag),
    }

    X = pd.DataFrame([row]).reindex(columns=FEATURES, fill_value=0)
    if "season" in X.columns:
        X["season"] = X["season"].astype(str)
    return X


def predict() -> dict:
    X = build_X()
    yhat = model.predict(X)[0]
    return {t: float(yhat[i]) for i, t in enumerate(TARGETS)}


with right:
    st.subheader("Forecast Output")

    if not run:
        st.info("Adjust inputs, then click **Generate Forecast**.")
        st.stop()

    preds = predict()

    hh = preds.get("households_served", np.nan)
    food = preds.get("total_food_lbs_distributed", np.nan)
    vol = preds.get("volunteers_recommended", np.nan)

    hh_lo, hh_hi = pred_range(hh, MAPE_BY_TARGET.get("households_served", 0.15))
    food_lo, food_hi = pred_range(food, MAPE_BY_TARGET.get("total_food_lbs_distributed", 0.18))
    vol_lo, vol_hi = pred_range(vol, MAPE_BY_TARGET.get("volunteers_recommended", 0.16))

    days = max(1, int(distribution_days))
    hh_daily = hh / days
    vol_daily = vol / days

    st.markdown(
        f"""
        **Distribution cycle:** {next_date.date()}  ·  **Season:** {season.title()}  ·
        **Holiday:** {"Yes" if holiday_flag else "No"}  ·  **Crisis:** {"Yes" if crisis_flag else "No"}
        """
    )

    cA, cB = st.columns(2)

    with cA:
        st.markdown("### Turnout Forecast")
        st.write(f"**Expected households served:** {fmt_int(hh)}")
        st.write(f"**Range:** {fmt_int(hh_lo)} – {fmt_int(hh_hi)}")
        st.write(f"**Households expected per day:** {fmt_float(hh_daily, 1)}")

        st.markdown("### Volunteer Need Forecast")
        st.write(f"**Volunteers recommended (cycle):** {fmt_float(vol, 1)}")
        st.write(f"**Range:** {fmt_float(vol_lo, 1)} – {fmt_float(vol_hi, 1)}")
        st.write(f"**Volunteers recommended per day:** {fmt_float(vol_daily, 1)}")

    with cB:
        st.markdown("### Food Demand Forecast")
        st.write(f"**Total (lbs):** {fmt_lbs(food)}")
        st.write(f"**Range:** {fmt_lbs(food_lo)} – {fmt_lbs(food_hi)}")

        st.markdown("#### Category split")
        shares = {
            "Produce": 0.35,
            "Protein": 0.22,
            "Grains": 0.17,
            "Dairy": 0.13,
            "Shelf-stable": 0.13,
        }
        rows = [{"Category": k, "Rec. Amount (lbs)": int(round(food * v))} for k, v in shares.items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.markdown("---")
    st.subheader("Model Error Summary (quick-look)")

    for t in ["households_served", "total_food_lbs_distributed", "volunteers_recommended"]:
        mean_val = float({"households_served": hh, "total_food_lbs_distributed": food, "volunteers_recommended": vol}[t])
        m = float(MAPE_BY_TARGET.get(t, 0.0))
        typical = m * mean_val

        if t == "households_served":
            st.write(f"**{t}**")
            st.write(f"mean ≈ {fmt_int(mean_val)} households")
            st.write(f"MAPE ≈ {m*100:.1f}%")
            st.write(f"typical miss ≈ {m:.3f} × {mean_val:,.1f} ≈ {fmt_int(typical)} households")
        elif t == "total_food_lbs_distributed":
            st.write(f"**{t}**")
            st.write(f"mean ≈ {int(round(mean_val)):,} lbs")
            st.write(f"MAPE ≈ {m*100:.1f}%")
            st.write(f"typical miss ≈ {m:.3f} × {mean_val:,.1f} ≈ {int(round(typical)):,} lbs")
        else:
            st.write(f"**{t}**")
            st.write(f"mean ≈ {fmt_float(mean_val,1)} volunteers")
            st.write(f"MAPE ≈ {m*100:.1f}%")
            st.write(f"typical miss ≈ {m:.3f} × {mean_val:,.1f} ≈ {fmt_float(typical,1)} volunteers")