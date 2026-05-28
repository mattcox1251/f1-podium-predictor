"""
feature_engineering.py
----------------------
Combines all raw CSVs into a single model-ready dataset.

Each row represents one driver in one race, with features:
  - Qualifying position
  - Championship standings going into the race
  - Rolling form (avg finish over last 3 and 5 races)
  - Circuit history (avg finish at this specific track)
  - Pit stop strategy (number of stops)
  - Fastest lap rank and average speed
  - Constructor strength (constructor champ position)
  - Home race indicator
  - Target label: podium (1 = top 3 finish, 0 = did not podium)

Usage:
    python src/feature_engineering.py

Output:
    data/processed/model_dataset.csv
    data/processed/feature_summary.txt
"""

import os
import pandas as pd
import numpy as np

# ── Configuration ─────────────────────────────────────────────────────────────

RAW_DIR       = "data/raw"
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ── Load raw data ─────────────────────────────────────────────────────────────

def load_raw_data() -> dict:
    print("Loading raw CSVs...")
    data = {
        "results":      pd.read_csv(f"{RAW_DIR}/race_results.csv"),
        "qualifying":   pd.read_csv(f"{RAW_DIR}/qualifying.csv"),
        "driver_std":   pd.read_csv(f"{RAW_DIR}/driver_standings.csv"),
        "con_std":      pd.read_csv(f"{RAW_DIR}/constructor_standings.csv"),
        "pit_stops":    pd.read_csv(f"{RAW_DIR}/pit_stops.csv"),
        "lap_times":    pd.read_csv(f"{RAW_DIR}/lap_times.csv"),
    }
    for name, df in data.items():
        print(f"  {name:15s} — {len(df):,} rows, {df.shape[1]} cols")
    print()
    return data


# ── Feature builders ──────────────────────────────────────────────────────────

def add_qualifying(base: pd.DataFrame, qualifying: pd.DataFrame) -> pd.DataFrame:
    """Merge qualifying position onto the base race results."""
    print("  Adding qualifying positions...")
    quali = qualifying[["season", "round", "driver_id", "quali_position"]].copy()
    df = base.merge(quali, on=["season", "round", "driver_id"], how="left")

    # Fill missing quali positions (e.g. late entries) with last place (20)
    df["quali_position"] = df["quali_position"].fillna(20).astype(int)
    return df


def add_championship_standings(
    base: pd.DataFrame,
    driver_std: pd.DataFrame,
    con_std: pd.DataFrame
) -> pd.DataFrame:
    """
    Add driver and constructor championship positions GOING INTO each race.
    We use standings from round-1 of the same season to avoid data leakage
    (we can't use standings that include the current race's points).
    """
    print("  Adding championship standings...")

    # Shift standings by one round so we use pre-race standings
    driver_pre = driver_std.copy()
    driver_pre["round"] = driver_pre["round"] + 1
    driver_pre = driver_pre.rename(columns={
        "champ_position": "driver_champ_pos_pre",
        "points":         "driver_points_pre",
        "wins":           "driver_wins_pre",
    })

    con_pre = con_std.copy()
    con_pre["round"] = con_pre["round"] + 1
    con_pre = con_pre.rename(columns={
        "champ_position": "con_champ_pos_pre",
        "points":         "con_points_pre",
        "wins":           "con_wins_pre",
    })

    df = base.merge(
        driver_pre[["season", "round", "driver_id", "driver_champ_pos_pre",
                    "driver_points_pre", "driver_wins_pre"]],
        on=["season", "round", "driver_id"],
        how="left"
    )
    df = df.merge(
        con_pre[["season", "round", "constructor_id", "con_champ_pos_pre",
                 "con_points_pre", "con_wins_pre"]],
        on=["season", "round", "constructor_id"],
        how="left"
    )

    # For round 1 of each season there's no prior round — fill with neutral values
    df["driver_champ_pos_pre"] = df["driver_champ_pos_pre"].fillna(10)
    df["driver_points_pre"]    = df["driver_points_pre"].fillna(0)
    df["driver_wins_pre"]      = df["driver_wins_pre"].fillna(0)
    df["con_champ_pos_pre"]    = df["con_champ_pos_pre"].fillna(5)
    df["con_points_pre"]       = df["con_points_pre"].fillna(0)
    df["con_wins_pre"]         = df["con_wins_pre"].fillna(0)

    return df


def add_rolling_form(base: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling average finish position over last 3 and 5 races per driver.
    Sorted by season + round to ensure correct temporal ordering.
    Uses shift(1) to avoid including the current race in the average.
    """
    print("  Adding rolling form (last 3 and 5 races)...")

    df = base.sort_values(["driver_id", "season", "round"]).copy()

    # Replace DNF/DNS (null finish_position) with 20 for rolling calc
    df["finish_pos_filled"] = df["finish_position"].fillna(20)

    df["rolling_avg_3"] = (
        df.groupby("driver_id")["finish_pos_filled"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )
    df["rolling_avg_5"] = (
        df.groupby("driver_id")["finish_pos_filled"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )

    df = df.drop(columns=["finish_pos_filled"])
    return df


def add_circuit_history(base: pd.DataFrame) -> pd.DataFrame:
    """
    Add each driver's historical average finish position at this specific circuit.
    Uses only past races (shift to avoid leakage).
    """
    print("  Adding circuit history...")

    df = base.sort_values(["driver_id", "circuit_id", "season", "round"]).copy()
    df["finish_pos_filled"] = df["finish_position"].fillna(20)

    df["circuit_avg_finish"] = (
        df.groupby(["driver_id", "circuit_id"])["finish_pos_filled"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    # Fill NaN for first appearance at a circuit with a neutral value (10th)
    df["circuit_avg_finish"] = df["circuit_avg_finish"].fillna(10)
    df = df.drop(columns=["finish_pos_filled"])
    return df


def add_pit_stop_features(base: pd.DataFrame, pit_stops: pd.DataFrame) -> pd.DataFrame:
    """
    Add number of pit stops per driver per race as a strategy proxy.
    Fewer stops = aggressive strategy, more stops = conservative.
    """
    print("  Adding pit stop features...")

    pit_counts = (
        pit_stops.groupby(["season", "round", "driver_id"])
        .size()
        .reset_index(name="num_pit_stops")
    )

    df = base.merge(pit_counts, on=["season", "round", "driver_id"], how="left")

    # Fill missing pit stop data with median (2 stops)
    df["num_pit_stops"] = df["num_pit_stops"].fillna(2).astype(int)
    return df


def add_fastest_lap_features(base: pd.DataFrame, lap_times: pd.DataFrame) -> pd.DataFrame:
    """
    Add fastest lap rank and average speed per driver per race.
    Fastest lap rank = 1 means this driver set the fastest lap of the race.
    """
    print("  Adding fastest lap features...")

    lap = lap_times[["season", "round", "driver_id", "fastest_lap_rank", "avg_speed_kph"]].copy()

    df = base.merge(lap, on=["season", "round", "driver_id"], how="left")

    # Fill missing with neutral values (mid-field rank, median speed)
    df["fastest_lap_rank"] = df["fastest_lap_rank"].fillna(10).astype(int)
    median_speed = df["avg_speed_kph"].median()
    df["avg_speed_kph"] = df["avg_speed_kph"].fillna(median_speed)
    return df


def add_home_race(base: pd.DataFrame) -> pd.DataFrame:
    """
    Flag whether a driver is racing in their home country.
    Uses a manually maintained mapping of driver_id → nationality → home circuits.
    """
    print("  Adding home race indicator...")

    # Driver nationality → circuit_ids in their home country
    home_circuits = {
        "hamilton":     ["silverstone"],
        "russell":      ["silverstone"],
        "norris":       ["silverstone"],
        "alonso":       ["catalunya"],
        "sainz":        ["catalunya"],
        "leclerc":      ["monaco"],
        "vettel":       ["hockenheim", "nurburgring"],
        "schumacher":   ["hockenheim", "nurburgring"],
        "verstappen":   ["zandvoort"],
        "gasly":        ["paul_ricard"],
        "ocon":         ["paul_ricard"],
        "stroll":       ["montreal"],
        "latifi":       ["montreal"],
        "tsunoda":      ["suzuka"],
        "ricciardo":    ["albert_park"],
        "hulkenberg":   ["hockenheim", "nurburgring"],
        "bottas":       ["finland"],    # no Finnish GP currently
        "raikkonen":    ["finland"],
        "zhou":         ["shanghai"],
        "magnussen":    ["denmark"],    # no Danish GP currently
    }

    def is_home(row):
        circuits = home_circuits.get(row["driver_id"], [])
        return int(row["circuit_id"] in circuits)

    base["home_race"] = base.apply(is_home, axis=1)
    return base


# ── Main pipeline ─────────────────────────────────────────────────────────────

def build_features() -> pd.DataFrame:
    data = load_raw_data()

    print("Building features...")

    # Start with race results as the base — one row per driver per race
    df = data["results"].copy()

    # Ensure correct types
    df["season"] = df["season"].astype(int)
    df["round"]  = df["round"].astype(int)
    df["podium"] = df["podium"].astype(int)

    # Apply each feature builder in sequence
    df = add_qualifying(df, data["qualifying"])
    df = add_championship_standings(df, data["driver_std"], data["con_std"])
    df = add_rolling_form(df)
    df = add_circuit_history(df)
    df = add_pit_stop_features(df, data["pit_stops"])
    df = add_fastest_lap_features(df, data["lap_times"])
    df = add_home_race(df)

    print()

    # ── Final cleanup ─────────────────────────────────────────────────────────

    # Drop rows where finish_position is null (retirements we can't label)
    before = len(df)
    df = df.dropna(subset=["finish_position"])
    print(f"  Dropped {before - len(df)} rows with no finish position")

    # Sort for readability
    df = df.sort_values(["season", "round", "finish_position"]).reset_index(drop=True)

    # Save
    out_path = f"{PROCESSED_DIR}/model_dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df):,} rows × {df.shape[1]} cols → {out_path}\n")

    return df


def print_summary(df: pd.DataFrame):
    """Print a human-readable summary of the final dataset."""
    summary_lines = []
    summary_lines.append("=" * 60)
    summary_lines.append("F1 Podium Predictor — Feature Summary")
    summary_lines.append("=" * 60)
    summary_lines.append(f"Total rows:       {len(df):,}")
    summary_lines.append(f"Total features:   {df.shape[1]}")
    summary_lines.append(f"Seasons covered:  {df['season'].min()}–{df['season'].max()}")
    summary_lines.append(f"Races covered:    {df.groupby(['season','round']).ngroups}")
    summary_lines.append(f"Drivers:          {df['driver_id'].nunique()}")
    summary_lines.append(f"Circuits:         {df['circuit_id'].nunique()}")
    summary_lines.append("")
    summary_lines.append("Target distribution:")
    podium_rate = df["podium"].mean() * 100
    summary_lines.append(f"  Podium (1):     {df['podium'].sum():,} ({podium_rate:.1f}%)")
    summary_lines.append(f"  No podium (0):  {(df['podium']==0).sum():,} ({100-podium_rate:.1f}%)")
    summary_lines.append("")
    summary_lines.append("Feature columns:")
    feature_cols = [c for c in df.columns if c not in [
        "race_name", "status", "points", "finish_position", "podium"
    ]]
    for col in feature_cols:
        nulls = df[col].isnull().sum()
        summary_lines.append(f"  {col:35s} nulls: {nulls}")
    summary_lines.append("")
    summary_lines.append("Sample rows (first race, sorted by finish):")
    sample = df[df["round"] == df["round"].min()][
        ["driver_code", "quali_position", "driver_champ_pos_pre",
         "rolling_avg_3", "num_pit_stops", "podium"]
    ].head(5).to_string(index=False)
    summary_lines.append(sample)
    summary_lines.append("=" * 60)

    summary = "\n".join(summary_lines)
    print(summary)

    summary_path = f"{PROCESSED_DIR}/feature_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"\nSummary saved → {summary_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = build_features()
    print_summary(df)
    print("\nNext step: run model.py")