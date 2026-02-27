# run_pipeline.py
import argparse
import subprocess
import sys
import time
from pathlib import Path

SCENARIO_CHOICES = {
    "1": "baseline",
    "2": "extreme_weather",
    "3": "extreme_hardship",
    "4": "low_hardship",
}

CRISIS_CHOICES = {
    "1": "low",
    "2": "med",
    "3": "high",
}

def pick_scenario():
    print("\nPick a scenario:")
    print("  1) baseline (normal)")
    print("  2) extreme_weather")
    print("  3) extreme_hardship")
    print("  4) low_hardship\n")
    choice = input("Type 1/2/3/4 and press Enter: ").strip()
    return SCENARIO_CHOICES.get(choice, "baseline")

def pick_crisis():
    print("\nPick crisis level (how often crisis_flag=1):")
    print("  1) low")
    print("  2) med")
    print("  3) high\n")
    choice = input("Type 1/2/3 and press Enter: ").strip()
    return CRISIS_CHOICES.get(choice, "med")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", choices=list(SCENARIO_CHOICES.values()), default=None)
    p.add_argument("--rows", type=int, default=500_000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--noise", type=float, default=0.025)
    p.add_argument("--randomness", type=float, default=2.5)
    p.add_argument("--crisis-level", choices=["low", "med", "high"], default=None)
    p.add_argument("--no-dashboard", action="store_true", help="Don't launch dashboard automatically")
    args = p.parse_args()

    scenario = args.scenario or pick_scenario()
    crisis_level = args.crisis_level or pick_crisis()

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path("scenario_runs")
    out_dir.mkdir(exist_ok=True)

    data_path = out_dir / f"pantry_{scenario}_crisis{crisis_level}_rows{args.rows}_seed{args.seed}_noise{args.noise}_rand{args.randomness}_{ts}.csv.gz"
    model_path = out_dir / f"model_{scenario}_crisis{crisis_level}_seed{args.seed}_rand{args.randomness}_{ts}.joblib"
    metrics_path = out_dir / f"metrics_{scenario}_crisis{crisis_level}_seed{args.seed}_rand{args.randomness}_{ts}.json"

    # 1) Generate data
    gen_cmd = [
        sys.executable, "generate_pantry_data.py",
        "--scenario", scenario,
        "--rows", str(args.rows),
        "--seed", str(args.seed),
        "--noise", str(args.noise),
        "--randomness", str(args.randomness),
        "--crisis-level", crisis_level,
        "--out", str(data_path),
    ]
    print("\nGenerating:", " ".join(gen_cmd))
    subprocess.check_call(gen_cmd)

    # 2) Train model
    train_cmd = [
        sys.executable, "model_training.py",
        "--data", str(data_path),
        "--out-model", str(model_path),
        "--out-metrics", str(metrics_path),
    ]
    print("\nTraining:", " ".join(train_cmd))
    subprocess.check_call(train_cmd)

    print("\n✅ Done!")
    print("Data   :", data_path)
    print("Model  :", model_path)
    print("Metrics:", metrics_path)

    # 3) Launch dashboard
    if not args.no_dashboard:
        dash_cmd = [
            sys.executable, "-m", "streamlit", "run", "dashboard.py",
            "--",
            "--data", str(data_path),
            "--model", str(model_path),
            "--metrics", str(metrics_path),
        ]
        print("\nLaunching dashboard:", " ".join(dash_cmd))
        subprocess.Popen(dash_cmd)  # keeps server running after pipeline finishes

if __name__ == "__main__":
    main()