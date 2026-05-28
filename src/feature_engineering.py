"""
feature_engineering.py
----------------------
Builds a rich model-ready dataset from all raw CSVs.

New features vs v1:
  - dnf_rate_5          : driver DNF rate over last 5 races (reliability signal)
  - teammate_gap_3      : driver avg finish vs teammate avg finish (last 3 races)
  - points_momentum     : points scored in last 3 races (hot/cold streak)
  - con_development     : constructor points change over last 3 rounds (car improving?)
  - circuit_win_rate    : driver win rate at this specific circuit
  - front_row           : binary flag, qualifying P1 or P2

Usage:
    python src/feature_engineering.py

Output:
    data/processed/model_dataset.csv
    data/processed/feature_summary.txt
"""

import os
import pandas as pd
import numpy as np

RAW_DIR       = "data/raw"
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)


def load_raw_data() -> dict:
    print("Loading raw CSVs...")
    data = {
        "results":    pd.read_csv(f"{RAW_DIR}/race_results.csv"),
        "qualifying": pd.read_csv(f"{RAW_DIR}/qualifying.csv"),
        "driver_std": pd.read_csv(f"{RAW_DIR}/driver_standings.csv"),
        "con_std":    pd.read_csv(f"{RAW_DIR}/constructor_standings.csv"),
        "pit_stops":  pd.read_csv(f"{RAW_DIR}/pit_stops.csv"),
        "lap_times":  pd.read_csv(f"{RAW_DIR}/lap_times.csv"),
    }
    for name, df in data.items():
        print(f"  {name:15s} — {len(df):,} rows")
    print()
    return data


def add_qualifying(base, qualifying):
    print("  Adding qualifying positions...")
    quali = qualifying[["season", "round", "driver_id", "quali_position"]].copy()
    df = base.merge(quali, on=["season", "round", "driver_id"], how="left")
    df["quali_position"] = df["quali_position"].fillna(20).astype(int)
    df["front_row"] = (df["quali_position"] <= 2).astype(int)
    return df


def add_championship_standings(base, driver_std, con_std):
    print("  Adding championship standings...")
    d = driver_std.copy()
    d["round"] = d["round"] + 1
    d = d.rename(columns={"champ_position": "driver_champ_pos_pre",
                           "points": "driver_points_pre",
                           "wins": "driver_wins_pre"})
    c = con_std.copy()
    c["round"] = c["round"] + 1
    c = c.rename(columns={"champ_position": "con_champ_pos_pre",
                           "points": "con_points_pre",
                           "wins": "con_wins_pre"})
    df = base.merge(
        d[["season", "round", "driver_id", "driver_champ_pos_pre",
           "driver_points_pre", "driver_wins_pre"]],
        on=["season", "round", "driver_id"], how="left"
    )
    df = df.merge(
        c[["season", "round", "constructor_id", "con_champ_pos_pre",
           "con_points_pre", "con_wins_pre"]],
        on=["season", "round", "constructor_id"], how="left"
    )
    df["driver_champ_pos_pre"] = df["driver_champ_pos_pre"].fillna(10)
    df["driver_points_pre"]    = df["driver_points_pre"].fillna(0)
    df["driver_wins_pre"]      = df["driver_wins_pre"].fillna(0)
    df["con_champ_pos_pre"]    = df["con_champ_pos_pre"].fillna(5)
    df["con_points_pre"]       = df["con_points_pre"].fillna(0)
    df["con_wins_pre"]         = df["con_wins_pre"].fillna(0)
    return df


def add_rolling_form(base):
    print("  Adding rolling form and DNF rate...")
    df = base.sort_values(["driver_id", "season", "round"]).copy()
    df["finish_pos_filled"] = df["finish_position"].fillna(20)

    df["rolling_avg_3"] = (
        df.groupby("driver_id")["finish_pos_filled"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )
    df["rolling_avg_5"] = (
        df.groupby("driver_id")["finish_pos_filled"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )
    # Points momentum: sum of points in last 3 races
    df["points_momentum"] = (
        df.groupby("driver_id")["points"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).sum())
    )
    # DNF rate over last 5 races
    df["dnf_rate_5"] = (
        df.groupby("driver_id")["dnf"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )
    df = df.drop(columns=["finish_pos_filled"])
    return df


def add_teammate_gap(base):
    """
    How does this driver's recent form compare to their teammate's?
    Negative = driver finishing better than teammate on average.
    """
    print("  Adding teammate gap...")
    df = base.sort_values(["driver_id", "season", "round"]).copy()
    df["finish_pos_filled"] = df["finish_position"].fillna(20)

    driver_avg = (
        df.groupby("driver_id")["finish_pos_filled"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )
    df["driver_rolling_3"] = driver_avg

    # Get constructor's average finish across both drivers (last 3)
    con_avg = (
        df.groupby(["constructor_id", "season", "round"])["finish_pos_filled"]
        .mean().reset_index().rename(columns={"finish_pos_filled": "con_avg_finish"})
    )
    # Shift con_avg by 1 round to avoid leakage
    con_avg = con_avg.sort_values(["constructor_id", "season", "round"])
    con_avg["con_avg_finish_pre"] = (
        con_avg.groupby("constructor_id")["con_avg_finish"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )

    df = df.merge(
        con_avg[["constructor_id", "season", "round", "con_avg_finish_pre"]],
        on=["constructor_id", "season", "round"], how="left"
    )
    # teammate_gap: driver rolling avg minus constructor avg (negative = better than teammate)
    df["teammate_gap_3"] = df["driver_rolling_3"] - df["con_avg_finish_pre"]
    df["teammate_gap_3"] = df["teammate_gap_3"].fillna(0)
    df = df.drop(columns=["finish_pos_filled", "driver_rolling_3", "con_avg_finish_pre"])
    return df


def add_constructor_development(base, con_std):
    """Constructor points change over last 3 rounds — is the car improving?"""
    print("  Adding constructor development trend...")
    c = con_std.sort_values(["constructor_id", "season", "round"]).copy()
    c["con_pts_momentum"] = (
        c.groupby(["constructor_id", "season"])["points"]
        .transform(lambda x: x.diff(3))
    )
    c["con_pts_momentum"] = c["con_pts_momentum"].fillna(0)
    df = base.merge(
        c[["season", "round", "constructor_id", "con_pts_momentum"]],
        on=["season", "round", "constructor_id"], how="left"
    )
    df["con_pts_momentum"] = df["con_pts_momentum"].fillna(0)
    return df


def add_circuit_history(base):
    print("  Adding circuit history...")
    df = base.sort_values(["driver_id", "circuit_id", "season", "round"]).copy()
    df["finish_pos_filled"] = df["finish_position"].fillna(20)

    df["circuit_avg_finish"] = (
        df.groupby(["driver_id", "circuit_id"])["finish_pos_filled"]
        .transform(lambda x: x.shift(1).expanding().mean())
    ).fillna(10)

    df["circuit_win_rate"] = (
        df.groupby(["driver_id", "circuit_id"])["podium"]
        .transform(lambda x: x.shift(1).expanding().mean())
    ).fillna(0)

    df = df.drop(columns=["finish_pos_filled"])
    return df


def add_pit_stop_features(base, pit_stops):
    print("  Adding pit stop features...")
    pit_counts = (
        pit_stops.groupby(["season", "round", "driver_id"])
        .size().reset_index(name="num_pit_stops")
    )
    df = base.merge(pit_counts, on=["season", "round", "driver_id"], how="left")
    df["num_pit_stops"] = df["num_pit_stops"].fillna(2).astype(int)
    return df


def add_fastest_lap_features(base, lap_times):
    print("  Adding fastest lap features...")
    lap = lap_times[["season", "round", "driver_id",
                     "fastest_lap_rank", "avg_speed_kph"]].copy()
    df = base.merge(lap, on=["season", "round", "driver_id"], how="left")
    df["fastest_lap_rank"] = df["fastest_lap_rank"].fillna(10).astype(int)
    df["avg_speed_kph"]    = df["avg_speed_kph"].fillna(df["avg_speed_kph"].median())
    return df


def add_home_race(base):
    print("  Adding home race indicator...")
    home_circuits = {
        "hamilton":   ["silverstone"], "russell":    ["silverstone"],
        "norris":     ["silverstone"], "alonso":     ["catalunya"],
        "sainz":      ["catalunya"],   "leclerc":    ["monaco"],
        "vettel":     ["hockenheim"],  "verstappen": ["zandvoort"],
        "gasly":      ["paul_ricard"], "ocon":       ["paul_ricard"],
        "stroll":     ["villeneuve"],  "tsunoda":    ["suzuka"],
        "ricciardo":  ["albert_park"], "hulkenberg": ["hockenheim"],
        "antonelli":  ["monza"],       "hamilton":   ["silverstone"],
    }
    base["home_race"] = base.apply(
        lambda r: int(r["circuit_id"] in home_circuits.get(r["driver_id"], [])), axis=1
    )
    return base


def build_features() -> pd.DataFrame:
    data = load_raw_data()
    print("Building features...")
    df = data["results"].copy()
    df["season"] = df["season"].astype(int)
    df["round"]  = df["round"].astype(int)
    df["podium"] = df["podium"].astype(int)
    df["dnf"]    = df["dnf"].astype(int)

    df = add_qualifying(df, data["qualifying"])
    df = add_championship_standings(df, data["driver_std"], data["con_std"])
    df = add_rolling_form(df)
    df = add_teammate_gap(df)
    df = add_constructor_development(df, data["con_std"])
    df = add_circuit_history(df)
    df = add_pit_stop_features(df, data["pit_stops"])
    df = add_fastest_lap_features(df, data["lap_times"])
    df = add_home_race(df)
    print()

    before = len(df)
    df = df.dropna(subset=["finish_position"])
    print(f"  Dropped {before - len(df)} rows with no finish position")

    df = df.sort_values(["season", "round", "finish_position"]).reset_index(drop=True)
    out_path = f"{PROCESSED_DIR}/model_dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df):,} rows × {df.shape[1]} cols → {out_path}\n")
    return df


def print_summary(df):
    lines = [
        "=" * 60, "F1 Podium Predictor — Feature Summary", "=" * 60,
        f"Rows:          {len(df):,}",
        f"Features:      {df.shape[1]}",
        f"Seasons:       {df['season'].min()}–{df['season'].max()}",
        f"Races:         {df.groupby(['season','round']).ngroups}",
        f"Drivers:       {df['driver_id'].nunique()}",
        f"Circuits:      {df['circuit_id'].nunique()}",
        "",
        f"Podium rate:   {df['podium'].mean():.1%}  "
        f"({df['podium'].sum():,} podiums / {len(df):,} rows)",
        "",
        "Null counts per feature:",
    ]
    skip = {"race_name", "status", "points", "finish_position", "podium", "dnf"}
    for col in df.columns:
        if col not in skip:
            n = df[col].isnull().sum()
            if n > 0:
                lines.append(f"  {col:35s} {n}")
    summary = "\n".join(lines)
    print(summary)
    with open(f"{PROCESSED_DIR}/feature_summary.txt", "w") as f:
        f.write(summary)
    print(f"\nSummary → {PROCESSED_DIR}/feature_summary.txt")


if __name__ == "__main__":
    df = build_features()
    print_summary(df)
    print("\nNext step: run model.py")