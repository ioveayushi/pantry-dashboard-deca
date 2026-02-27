from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def pretty_summary(df: pd.DataFrame, metrics: dict) -> None:
    means = df[TARGETS].mean(numeric_only=True)
    mape_dict = metrics["mape_by_target"]

    def fmt_mean(name: str, val: float) -> str:
        if name == "total_food_lbs_distributed":
            return f"{val:,.0f} lbs"
        if name == "households_served":
            return f"{val:.0f} households"
        if name == "volunteers_recommended":
            return f"{val:.1f} volunteers"
        return f"{val:.2f}"

    def fmt_miss(name: str, val: float) -> str:
        if name in ["households_served", "total_food_lbs_distributed"]:
            return f"{val:,.0f}"
        return f"{val:.1f}"

    for t in TARGETS:
        mean_val = float(means[t])
        mape = float(mape_dict[t])          # fraction, like 0.137
        typical = mape * mean_val

        print(t)
        print(f"mean ≈ {fmt_mean(t, mean_val)}")
        print(f"MAPE ≈ {mape*100:.1f}%")
        print(f"typical miss ≈ {mape:.3f} × {mean_val:,.1f} ≈ {fmt_miss(t, typical)}")
        print()

FEATURES = [
    "crisis_flag",
    "food_drive_flag",
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
]

TARGETS = [
    "households_served",
    "total_food_lbs_distributed",
    "volunteers_recommended",
]


def build_pipeline(numeric: list[str], categorical: list[str], alpha: float) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    model = MultiOutputRegressor(Ridge(alpha=alpha, random_state=0))
    return Pipeline([("preprocess", pre), ("model", model)])


def mape_by_target(y_true: np.ndarray, y_pred: np.ndarray, names: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for i, name in enumerate(names):
        denom = np.where(y_true[:, i] == 0, 1.0, y_true[:, i])
        out[name] = float(np.mean(np.abs((y_true[:, i] - y_pred[:, i]) / denom)))
    return out


def tune_alpha(X: pd.DataFrame, y: pd.DataFrame, numeric: list[str], categorical: list[str]) -> float:
    alphas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    seeds = list(range(10))

    best_alpha = alphas[0]
    best_score = float("inf")

    for a in alphas:
        scores = []
        for s in seeds:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=s)
            pipe = build_pipeline(numeric, categorical, alpha=a)
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            scores.append(mean_absolute_percentage_error(y_test, y_pred))
        avg = float(np.mean(scores))
        if avg < best_score:
            best_score = avg
            best_alpha = a

    return best_alpha


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--out-model", type=str, default="model.joblib")
    p.add_argument("--out-metrics", type=str, default="metrics.json")
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--tune", action="store_true")
    args = p.parse_args()

    df = pd.read_csv(args.data)
    X = df[FEATURES].copy()
    y = df[TARGETS].copy()

    # force season to be categorical
    X["season"] = X["season"].astype(str)

    categorical = ["season"]
    numeric = [c for c in FEATURES if c not in categorical]

    alpha = float(args.alpha) if args.alpha is not None else (tune_alpha(X, y, numeric, categorical) if args.tune else 1.0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    pipe = build_pipeline(numeric, categorical, alpha=alpha)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    metrics = {
        "alpha": alpha,
        "overall_mape": float(mean_absolute_percentage_error(y_test, y_pred)),
        "mape_by_target": mape_by_target(y_test.to_numpy(), y_pred, TARGETS),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features": FEATURES,
        "targets": TARGETS,
    }
    
    pretty_summary(df, metrics)

    with open(args.out_metrics, "w") as f:
        json.dump(metrics, f, indent=2)

    joblib.dump(pipe, args.out_model)

    print(f"Saved model → {args.out_model}")
    print(f"Saved metrics → {args.out_metrics}")
    print(json.dumps(metrics, indent=2)[:1200])


if __name__ == "__main__":
    main()
