# dashboard_ipad.py
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
# Polished styling (cards + color accents)
# Change the HEX codes below to match your paper.
# -----------------------------
ACCENT_RED = "#cb0000"
ACCENT_BLUE = "#b28256"
ACCENT_GREEN = "#305939"
SOFT_BG = "#a7c9ae"

st.markdown(
    f"""
<style>
.block-container {{ padding-top: 1.2rem; padding-bottom: 1.5rem; }}
h1, h2, h3 {{ letter-spacing: -0.02em; }}

.card {{
  background: white;
  border: 1px solid rgba(17,24,39,0.10);
  border-radius: 16px;
  padding: 16px 16px;
  box-shadow: 0 8px 22px rgba(17,24,39,0.06);
  margin-bottom: 14px;
}}

.card-soft {{
  background: linear-gradient(180deg, {SOFT_BG} 0%, rgba(255,255,255,1) 100%);
}}

.accent-red {{ border-left: 6px solid {ACCENT_RED}; }}
.accent-blue {{ border-left: 6px solid {ACCENT_BLUE}; }}
.accent-green {{ border-left: 6px solid {ACCENT_GREEN}; }}

.pill {{
  display:inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  background: color-mix(in srgb, {ACCENT_RED} 12%, white);
  color: {ACCENT_RED};
  font-weight: 700;
  font-size: 0.85rem;
}}

.smallcap {{
  font-size: 0.85rem;
  opacity: 0.78;
}}

hr {{ margin: 0.6rem 0 0.9rem 0; }}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# CLI args (optional)
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


def fmt_int(x):
    return f"{int(round(float(x))):,}"


def fmt_lbs(x):
    return f"{int(round(float(x))):,} lbs"


def fmt_float(x, d=1):
    return f"{float(x):.{d}f}"


FEATURE_FALLBACK = [
    "pantry_id",
    "year",
    "month",
    "weekofyear",
    "season",
    "holiday_flag",
    "school_break_flag",
    "severe_weather_flag",
    "avg_temp_f",
    "precip_in",
    "econ_pressure_index",
    "donations_lbs",
    "hardship_index",
    "immigrant_share",
    "capacity_households",
    "crisis_flag",
    "food_drive_flag",
]
TARGET_FALLBACK = [
    "households_served",
    "total_food_lbs_distributed",
    "volunteers_recommended",
]


def pick_default_data() -> str | None:
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


@st.cache_resource
def load_model(path: str):
    return joblib.load(path)


# -----------------------------
# Resolve assets (Cloud or local)
# -----------------------------
data_path = ARGS.data or pick_default_data()
model_path = ARGS.model or pick_default_model()
metrics_path = ARGS.metrics or pick_default_metrics()

if not model_path:
    st.error("No model found. Add model.joblib to repo root or pass --model.")
    st.stop()

model = load_model(model_path)

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
        "volunteers_recommended": 0.16,
    }

if not data_path:
    st.error(
        "No data found.\n\n"
        "For Streamlit Cloud, add a small demo dataset named **demo_data.csv.gz** to your repo."
    )
    st.stop()

df = load_data(data_path)

# -----------------------------
# Header
# -----------------------------
st.title("Community Food Pantry Predictive Model Sample - DECA")
st.markdown("<span class='pill'>Staff-ready forecast</span>", unsafe_allow_html=True)
st.caption(f"Data: {Path(data_path).name}   ·   Model: {Path(model_path).name}")

# -----------------------------
# Layout
# -----------------------------
left, right = st.columns([1.10, 1.0])

# -------- Left: Inputs --------
pantry_ids = sorted(df["pantry_id"].dropna().unique().tolist())
if not pantry_ids:
    st.error("No pantry_id values found in the dataset.")
    st.stop()

with left:
    st.markdown('<div class="card card-soft accent-blue">', unsafe_allow_html=True)
    st.subheader("Auto-loaded inputs (latest cycle → predict next cycle)")

    pantry_id = st.selectbox("Pantry ID", pantry_ids, index=0)

    dfp = df[df["pantry_id"] == pantry_id].sort_values("cycle_start_date")
    latest = dfp.iloc[-1].copy()

    next_date = latest["cycle_start_date"] + timedelta(days=14)
    year = int(next_date.year)
    month = int(next_date.month)
    weekofyear = int(next_date.isocalendar().week)
    season = season_from_month(month)

    st.write(f"**Latest cycle in data:** {latest['cycle_start_date'].date()}")
    st.write(f"**Forecasting next cycle:** {next_date.date()}")
    st.markdown("</div>", unsafe_allow_html=True)

    # defaults carry over (staff can override)
    holiday_flag = bool(int(latest.get("holiday_flag", 0)))
    school_break_flag = bool(int(latest.get("school_break_flag", 0)))
    severe_weather_flag = bool(int(latest.get("severe_weather_flag", 0)))
    crisis_flag = bool(int(latest.get("crisis_flag", 0)))
    food_drive_flag = bool(int(latest.get("food_drive_flag", 0)))

    avg_temp_f = float(latest.get("avg_temp_f", 70.0))
    precip_in = float(latest.get("precip_in", 0.8))
    econ_pressure_index = float(latest.get("econ_pressure_index", 0.55))
    donations_lbs = float(latest.get("donations_lbs", 2500.0))
    hardship_index = float(latest.get("hardship_index", 0.5))
    immigrant_share = float(latest.get("immigrant_share", 0.35))
    capacity_households = int(latest.get("capacity_households", 600))

    st.markdown('<div class="card card-soft accent-red">', unsafe_allow_html=True)
    st.subheader("Optional staff overrides")

    with st.expander("Open overrides", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            holiday_flag = st.checkbox("Vacation / Holiday?", value=holiday_flag)
            school_break_flag = st.checkbox("School break?", value=school_break_flag)
        with c2:
            severe_weather_flag = st.checkbox("Severe weather expected?", value=severe_weather_flag)
            crisis_flag = st.checkbox("Emergency / Crisis active?", value=crisis_flag)
        with c3:
            food_drive_flag = st.checkbox("Food drive / aid week?", value=food_drive_flag)

        c4, c5, c6 = st.columns(3)
        with c4:
            avg_temp_f = st.number_input("Avg temp (°F)", value=float(avg_temp_f))
            precip_in = st.number_input("Precip (inches)", value=float(precip_in))
        with c5:
            donations_lbs = st.number_input(
                "Expected donations (lbs)", min_value=0.0, value=float(donations_lbs)
            )
            econ_pressure_index = st.number_input(
                "Economic pressure (0–1.5)",
                min_value=0.0,
                max_value=1.5,
                value=float(econ_pressure_index),
            )
        with c6:
            hardship_index = st.number_input(
                "Hardship index (0–1)", min_value=0.0, max_value=1.0, value=float(hardship_index)
            )
            immigrant_share = st.number_input(
                "Immigrant share (0–1)", min_value=0.0, max_value=1.0, value=float(immigrant_share)
            )

        capacity_households = st.number_input(
            "Capacity (households per cycle)", min_value=20, value=int(capacity_households)
        )

    run = st.button("Generate Forecast", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


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


# -------- Right: Outputs --------
with right:
    st.markdown('<div class="card card-soft accent-green">', unsafe_allow_html=True)
    st.subheader("Forecast Output")

    st.caption(
        f"Cycle start: {next_date.date()} · Season: {season.title()} · "
        f"Holiday: {'Yes' if holiday_flag else 'No'} · Crisis: {'Yes' if crisis_flag else 'No'}"
    )

    if not run:
        st.info("Set overrides (optional), then click **Generate Forecast**.")
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    preds = predict()

    hh = preds.get("households_served", np.nan)
    food = preds.get("total_food_lbs_distributed", np.nan)
    vol = preds.get("volunteers_recommended", np.nan)

    hh_lo, hh_hi = pred_range(hh, MAPE_BY_TARGET.get("households_served", 0.15))
    food_lo, food_hi = pred_range(food, MAPE_BY_TARGET.get("total_food_lbs_distributed", 0.18))
    vol_lo, vol_hi = pred_range(vol, MAPE_BY_TARGET.get("volunteers_recommended", 0.16))

    # KPI tiles
    k1, k2, k3 = st.columns(3)
    k1.metric("Expected households (two-week total)", fmt_int(hh), f"{fmt_int(hh_lo)}–{fmt_int(hh_hi)}")
    k2.metric("Total food needed (lbs)", fmt_lbs(food), f"{fmt_lbs(food_lo)}–{fmt_lbs(food_hi)}")
    k3.metric("Volunteers (two-week total)", fmt_float(vol, 1), f"{fmt_float(vol_lo,1)}–{fmt_float(vol_hi,1)}")

    # Per calendar day (assumption: open all 14 days)
    CYCLE_DAYS = 14
    st.write("")
    d1, d2 = st.columns(2)
    d1.markdown(
        f"<div class='smallcap'>Households per day (assuming open all 14 days)</div><b>{fmt_float(hh/CYCLE_DAYS,1)}</b>",
        unsafe_allow_html=True,
    )
    d2.markdown(
        f"<div class='smallcap'>Volunteers per day (assuming open all 14 days)</div><b>{fmt_float(vol/CYCLE_DAYS,2)}</b>",
        unsafe_allow_html=True,
    )

    st.markdown("<hr/>", unsafe_allow_html=True)

    # Category split
    st.markdown("### Category split")
    shares = {
        "Produce": 0.35,
        "Protein": 0.22,
        "Grains": 0.17,
        "Dairy": 0.13,
        "Shelf-stable": 0.13,
    }
    rows = [{"Category": k, "Rec. Amount (lbs)": int(round(food * v))} for k, v in shares.items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)