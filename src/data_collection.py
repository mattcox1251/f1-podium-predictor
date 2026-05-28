"""
data_collection.py
------------------
Collects and saves F1 data from the Jolpica API:
  - Historical race results, qualifying, driver/constructor standings
  - Pit stops, fastest lap times per driver per race

FastF1 is not used here as its live timing endpoints are blocked in
Codespaces. All data is pulled from the Jolpica API instead.

Usage:
    python src/data_collection.py

Output:
    data/raw/race_results.csv
    data/raw/qualifying.csv
    data/raw/driver_standings.csv
    data/raw/constructor_standings.csv
    data/raw/pit_stops.csv
    data/raw/weather.csv
    data/raw/lap_times.csv
"""

import os
import time
import requests
import pandas as pd

# ── Configuration ────────────────────────────────────────────────────────────

# Years to collect historical data for (2019+ has reliable data)
SEASONS = list(range(2019, 2025))   # 2019–2024 inclusive

# Output directory
RAW_DIR = "data/raw"
os.makedirs(RAW_DIR, exist_ok=True)

# Jolpica base URL (drop-in replacement for the old Ergast API)
JOLPICA_BASE = "https://api.jolpi.ca/ergast/f1"


# ── Jolpica helpers ──────────────────────────────────────────────────────────

def jolpica_get(endpoint: str, limit: int = 100) -> list:
    """
    Fetch all pages from a Jolpica endpoint and return the inner results list.
    Handles pagination automatically.
    """
    results = []
    offset = 0

    while True:
        url = f"{JOLPICA_BASE}/{endpoint}.json?limit={limit}&offset={offset}"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"  [WARNING] Request failed for {url}: {e}")
            break

        data = response.json().get("MRData", {})
        total = int(data.get("total", 0))

        # Dig into whichever table key is present
        table = data.get("RaceTable") or data.get("StandingsTable") or {}
        items = (
            table.get("Races")
            or table.get("StandingsLists")
            or []
        )
        results.extend(items)

        offset += limit
        if offset >= total:
            break

        time.sleep(1.0)   # be polite to the API

    return results


def collect_race_results(seasons: list) -> pd.DataFrame:
    """Fetch finish positions, points, and status for every race."""
    print("Collecting race results...")
    rows = []

    for season in seasons:
        print(f"  Season {season}")
        races = jolpica_get(f"{season}/results")

        for race in races:
            circuit = race["Circuit"]["circuitId"]
            race_name = race["raceName"]
            round_no = int(race["round"])

            for result in race.get("Results", []):
                rows.append({
                    "season":           season,
                    "round":            round_no,
                    "race_name":        race_name,
                    "circuit_id":       circuit,
                    "driver_id":        result["Driver"]["driverId"],
                    "driver_code":      result["Driver"].get("code", ""),
                    "constructor_id":   result["Constructor"]["constructorId"],
                    "grid_position":    int(result.get("grid", 0)),
                    "finish_position":  int(result["position"]) if result["position"].isdigit() else None,
                    "points":           float(result.get("points", 0)),
                    "status":           result.get("status", ""),
                    "podium":           int(result["position"]) <= 3 if result["position"].isdigit() else False,
                })

        time.sleep(2.0)

    df = pd.DataFrame(rows)
    path = f"{RAW_DIR}/race_results.csv"
    df.to_csv(path, index=False)
    print(f"  Saved {len(df)} rows → {path}\n")
    return df


def collect_qualifying(seasons: list) -> pd.DataFrame:
    """Fetch qualifying positions for every race."""
    print("Collecting qualifying results...")
    rows = []

    for season in seasons:
        print(f"  Season {season}")
        races = jolpica_get(f"{season}/qualifying")

        for race in races:
            round_no = int(race["round"])
            circuit = race["Circuit"]["circuitId"]

            for result in race.get("QualifyingResults", []):
                rows.append({
                    "season":       season,
                    "round":        round_no,
                    "circuit_id":   circuit,
                    "driver_id":    result["Driver"]["driverId"],
                    "driver_code":  result["Driver"].get("code", ""),
                    "quali_position": int(result["position"]),
                    "q1":           result.get("Q1", None),
                    "q2":           result.get("Q2", None),
                    "q3":           result.get("Q3", None),
                })

        time.sleep(2.0)

    df = pd.DataFrame(rows)
    path = f"{RAW_DIR}/qualifying.csv"
    df.to_csv(path, index=False)
    print(f"  Saved {len(df)} rows → {path}\n")
    return df


def collect_standings(seasons: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Derive driver and constructor championship standings directly from
    race_results.csv instead of the Jolpica standings endpoints (which
    are blocked in Codespaces). Computes cumulative points and position
    after each round, matching what the API would have returned.
    """
    print("Deriving standings from race_results.csv...")

    results_path = f"{RAW_DIR}/race_results.csv"
    if not os.path.exists(results_path):
        raise FileNotFoundError(
            f"Cannot find {results_path} — run collect_race_results() first."
        )

    df = pd.read_csv(results_path)
    driver_rows = []
    constructor_rows = []

    for season in sorted(df["season"].unique()):
        season_df = df[df["season"] == season].copy()

        for round_no in sorted(season_df["round"].unique()):
            round_df = season_df[season_df["round"] <= round_no]

            # ── Driver standings ───────────────────────────────────────────
            driver_pts = (
                round_df.groupby(["driver_id", "driver_code"])
                .agg(points=("points", "sum"), wins=("finish_position", lambda x: (x == 1).sum()))
                .reset_index()
                .sort_values("points", ascending=False)
                .reset_index(drop=True)
            )
            driver_pts["champ_position"] = driver_pts.index + 1
            driver_pts["season"] = season
            driver_pts["round"] = round_no
            driver_rows.append(driver_pts)

            # ── Constructor standings ──────────────────────────────────────
            con_pts = (
                round_df.groupby("constructor_id")
                .agg(points=("points", "sum"), wins=("finish_position", lambda x: (x == 1).sum()))
                .reset_index()
                .sort_values("points", ascending=False)
                .reset_index(drop=True)
            )
            con_pts["champ_position"] = con_pts.index + 1
            con_pts["season"] = season
            con_pts["round"] = round_no
            constructor_rows.append(con_pts)

        print(f"  Season {season} — {round_no} rounds processed")

    driver_df = pd.concat(driver_rows, ignore_index=True)
    con_df = pd.concat(constructor_rows, ignore_index=True)

    # Reorder columns to match original schema
    driver_df = driver_df[["season", "round", "driver_id", "driver_code", "points", "wins", "champ_position"]]
    con_df = con_df[["season", "round", "constructor_id", "points", "wins", "champ_position"]]

    driver_df.to_csv(f"{RAW_DIR}/driver_standings.csv", index=False)
    con_df.to_csv(f"{RAW_DIR}/constructor_standings.csv", index=False)
    print(f"  Saved {len(driver_df)} driver standing rows → {RAW_DIR}/driver_standings.csv")
    print(f"  Saved {len(con_df)} constructor standing rows → {RAW_DIR}/constructor_standings.csv\n")
    return driver_df, con_df


# ── Jolpica pit stop & lap time helpers ─────────────────────────────────────

def collect_fastf1_data(seasons: list) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Pull lap times, pit stops, and a weather placeholder from Jolpica API.
    FastF1 live timing endpoints are blocked in Codespaces, so we use
    Jolpica for all data. This gives us:
      - Fastest lap per driver (rank, time, avg speed)
      - Pit stop count, lap number, and duration per driver
    Weather is not available via Jolpica — an empty placeholder is saved
    so the rest of the pipeline doesn't break.
    """
    print("Collecting lap/pit data via Jolpica API...")
    pit_rows = []
    lap_rows = []

    for season in seasons:
        print(f"  Season {season}")
        races = jolpica_get(f"{season}/results")

        for race in races:
            round_no = int(race["round"])
            circuit = race["Circuit"]["circuitId"]

            # ── Pit stops ──────────────────────────────────────────────────
            pit_data = jolpica_get(f"{season}/{round_no}/pitstops")
            for pit_race in pit_data:
                for stop in pit_race.get("PitStops", []):
                    pit_rows.append({
                        "season":       season,
                        "round":        round_no,
                        "circuit_id":   circuit,
                        "driver_id":    stop["driverId"],
                        "stop_number":  int(stop["stop"]),
                        "lap_number":   int(stop["lap"]),
                        "duration":     stop.get("duration", None),
                    })

            # ── Fastest lap per driver from race results ───────────────────
            for result in race.get("Results", []):
                fastest = result.get("FastestLap", {})
                if fastest:
                    lap_rows.append({
                        "season":           season,
                        "round":            round_no,
                        "circuit_id":       circuit,
                        "driver_id":        result["Driver"]["driverId"],
                        "driver_code":      result["Driver"].get("code", ""),
                        "fastest_lap_rank": int(fastest.get("rank", 0)),
                        "fastest_lap_time": fastest.get("Time", {}).get("time", None),
                        "fastest_lap_no":   int(fastest.get("lap", 0)),
                        "avg_speed_kph":    float(fastest.get("AverageSpeed", {}).get("speed", 0) or 0),
                    })

            time.sleep(1.5)

        time.sleep(2.0)

    pit_df = pd.DataFrame(pit_rows)
    lap_df = pd.DataFrame(lap_rows)

    # Weather isn't available via Jolpica — create an empty placeholder
    # so the rest of the pipeline doesn't break
    weather_df = pd.DataFrame(columns=[
        "season", "round", "air_temp_avg", "track_temp_avg",
        "humidity_avg", "rainfall", "wind_speed_avg"
    ])

    pit_df.to_csv(f"{RAW_DIR}/pit_stops.csv", index=False)
    lap_df.to_csv(f"{RAW_DIR}/lap_times.csv", index=False)
    weather_df.to_csv(f"{RAW_DIR}/weather.csv", index=False)

    print(f"  Saved {len(pit_df)} pit stop rows → {RAW_DIR}/pit_stops.csv")
    print(f"  Saved {len(lap_df)} fastest lap rows → {RAW_DIR}/lap_times.csv")
    print(f"  Saved empty weather placeholder → {RAW_DIR}/weather.csv\n")
    return lap_df, pit_df, weather_df


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("F1 Podium Predictor — Data Collection")
    print("=" * 60)
    print(f"Seasons: {SEASONS[0]}–{SEASONS[-1]}")
    print(f"Output:  {RAW_DIR}/\n")

    # ✅ Already collected successfully — uncomment to re-fetch if needed
    # collect_race_results(SEASONS)
    # collect_qualifying(SEASONS)
    # collect_fastf1_data(SEASONS)

    # Derived from race_results.csv — no API calls needed
    collect_standings(SEASONS)

    print("=" * 60)
    print("Data collection complete!")
    print(f"Check your {RAW_DIR}/ folder for all CSV files.")
    print("Next step: run feature_engineering.py")
    print("=" * 60)