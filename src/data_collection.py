"""
data_collection.py
------------------
Collects F1 data from the Jolpica API for 2019-2025 (historical training)
and 2026 (live current season).

Usage:
    python src/data_collection.py

Output:
    data/raw/race_results.csv
    data/raw/qualifying.csv
    data/raw/driver_standings.csv
    data/raw/constructor_standings.csv
    data/raw/pit_stops.csv
    data/raw/lap_times.csv
    data/raw/weather.csv         (placeholder)
"""

import os
import time
import requests
import pandas as pd

# ── Configuration ─────────────────────────────────────────────────────────────

SEASONS      = list(range(2019, 2026))   # 2019–2025 historical
LIVE_SEASON  = 2026                       # current season fetched live
RAW_DIR      = "data/raw"
JOLPICA_BASE = "https://api.jolpi.ca/ergast/f1"

os.makedirs(RAW_DIR, exist_ok=True)


# ── Jolpica helpers ───────────────────────────────────────────────────────────

def jolpica_get(endpoint: str, limit: int = 100) -> list:
    results, offset = [], 0
    while True:
        url = f"{JOLPICA_BASE}/{endpoint}.json?limit={limit}&offset={offset}"
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"  [WARNING] {url}: {e}")
            break
        data  = r.json().get("MRData", {})
        total = int(data.get("total", 0))
        table = data.get("RaceTable") or data.get("StandingsTable") or {}
        items = table.get("Races") or table.get("StandingsLists") or []
        results.extend(items)
        offset += limit
        if offset >= total:
            break
        time.sleep(1.0)
    return results


# ── Collectors ────────────────────────────────────────────────────────────────

def collect_race_results(seasons: list) -> pd.DataFrame:
    print("Collecting race results...")
    rows = []
    for season in seasons:
        print(f"  Season {season}")
        for race in jolpica_get(f"{season}/results"):
            circuit   = race["Circuit"]["circuitId"]
            race_name = race["raceName"]
            round_no  = int(race["round"])
            for result in race.get("Results", []):
                pos = result["position"]
                rows.append({
                    "season":          season,
                    "round":           round_no,
                    "race_name":       race_name,
                    "circuit_id":      circuit,
                    "driver_id":       result["Driver"]["driverId"],
                    "driver_code":     result["Driver"].get("code", ""),
                    "constructor_id":  result["Constructor"]["constructorId"],
                    "grid_position":   int(result.get("grid", 0)),
                    "finish_position": int(pos) if pos.isdigit() else None,
                    "points":          float(result.get("points", 0)),
                    "status":          result.get("status", ""),
                    "podium":          int(pos) <= 3 if pos.isdigit() else False,
                    "dnf":             0 if result.get("status", "") == "Finished"
                                         or result.get("status", "").startswith("+")
                                         else 1,
                })
        time.sleep(2.0)
    df = pd.DataFrame(rows)
    df.to_csv(f"{RAW_DIR}/race_results.csv", index=False)
    print(f"  Saved {len(df):,} rows\n")
    return df


def collect_qualifying(seasons: list) -> pd.DataFrame:
    print("Collecting qualifying results...")
    rows = []
    for season in seasons:
        print(f"  Season {season}")
        for race in jolpica_get(f"{season}/qualifying"):
            round_no = int(race["round"])
            circuit  = race["Circuit"]["circuitId"]
            for result in race.get("QualifyingResults", []):
                rows.append({
                    "season":         season,
                    "round":          round_no,
                    "circuit_id":     circuit,
                    "driver_id":      result["Driver"]["driverId"],
                    "driver_code":    result["Driver"].get("code", ""),
                    "quali_position": int(result["position"]),
                    "q1":             result.get("Q1", None),
                    "q2":             result.get("Q2", None),
                    "q3":             result.get("Q3", None),
                })
        time.sleep(2.0)
    df = pd.DataFrame(rows)
    df.to_csv(f"{RAW_DIR}/qualifying.csv", index=False)
    print(f"  Saved {len(df):,} rows\n")
    return df


def collect_standings(seasons: list) -> tuple:
    """Derive standings from race_results.csv (Jolpica standings blocked in Codespaces)."""
    print("Deriving standings from race_results.csv...")
    df = pd.read_csv(f"{RAW_DIR}/race_results.csv")
    driver_rows, constructor_rows = [], []

    for season in sorted(df["season"].unique()):
        s_df = df[df["season"] == season]
        for round_no in sorted(s_df["round"].unique()):
            r_df = s_df[s_df["round"] <= round_no]

            d_pts = (
                r_df.groupby(["driver_id", "driver_code"])
                .agg(points=("points", "sum"), wins=("finish_position", lambda x: (x == 1).sum()))
                .reset_index().sort_values("points", ascending=False).reset_index(drop=True)
            )
            d_pts["champ_position"] = d_pts.index + 1
            d_pts["season"], d_pts["round"] = season, round_no
            driver_rows.append(d_pts)

            c_pts = (
                r_df.groupby("constructor_id")
                .agg(points=("points", "sum"), wins=("finish_position", lambda x: (x == 1).sum()))
                .reset_index().sort_values("points", ascending=False).reset_index(drop=True)
            )
            c_pts["champ_position"] = c_pts.index + 1
            c_pts["season"], c_pts["round"] = season, round_no
            constructor_rows.append(c_pts)

        print(f"  Season {season} — {round_no} rounds")

    driver_df = pd.concat(driver_rows, ignore_index=True)[
        ["season", "round", "driver_id", "driver_code", "points", "wins", "champ_position"]]
    con_df    = pd.concat(constructor_rows, ignore_index=True)[
        ["season", "round", "constructor_id", "points", "wins", "champ_position"]]

    driver_df.to_csv(f"{RAW_DIR}/driver_standings.csv", index=False)
    con_df.to_csv(f"{RAW_DIR}/constructor_standings.csv", index=False)
    print(f"  Saved {len(driver_df):,} driver rows, {len(con_df):,} constructor rows\n")
    return driver_df, con_df


def collect_pit_and_lap(seasons: list) -> tuple:
    print("Collecting pit stop & lap time data...")
    pit_rows, lap_rows = [], []
    for season in seasons:
        print(f"  Season {season}")
        for race in jolpica_get(f"{season}/results"):
            round_no = int(race["round"])
            circuit  = race["Circuit"]["circuitId"]

            for pit_race in jolpica_get(f"{season}/{round_no}/pitstops"):
                for stop in pit_race.get("PitStops", []):
                    pit_rows.append({
                        "season":      season, "round": round_no,
                        "circuit_id":  circuit,
                        "driver_id":   stop["driverId"],
                        "stop_number": int(stop["stop"]),
                        "lap_number":  int(stop["lap"]),
                        "duration":    stop.get("duration", None),
                    })

            for result in race.get("Results", []):
                fl = result.get("FastestLap", {})
                if fl:
                    lap_rows.append({
                        "season":           season, "round": round_no,
                        "circuit_id":       circuit,
                        "driver_id":        result["Driver"]["driverId"],
                        "driver_code":      result["Driver"].get("code", ""),
                        "fastest_lap_rank": int(fl.get("rank", 0)),
                        "fastest_lap_time": fl.get("Time", {}).get("time", None),
                        "fastest_lap_no":   int(fl.get("lap", 0)),
                        "avg_speed_kph":    float(fl.get("AverageSpeed", {}).get("speed", 0) or 0),
                    })

            time.sleep(1.5)
        time.sleep(2.0)

    pit_df = pd.DataFrame(pit_rows)
    lap_df = pd.DataFrame(lap_rows)
    weather_df = pd.DataFrame(columns=[
        "season", "round", "air_temp_avg", "track_temp_avg",
        "humidity_avg", "rainfall", "wind_speed_avg"
    ])
    pit_df.to_csv(f"{RAW_DIR}/pit_stops.csv", index=False)
    lap_df.to_csv(f"{RAW_DIR}/lap_times.csv", index=False)
    weather_df.to_csv(f"{RAW_DIR}/weather.csv", index=False)
    print(f"  Saved {len(pit_df):,} pit rows, {len(lap_df):,} lap rows\n")
    return pit_df, lap_df


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("F1 Podium Predictor — Data Collection")
    print(f"Seasons: {SEASONS[0]}–{SEASONS[-1]} + {LIVE_SEASON} live")
    print("=" * 60 + "\n")

    all_seasons = SEASONS + [LIVE_SEASON]

    collect_race_results(all_seasons)
    collect_qualifying(all_seasons)
    collect_standings(all_seasons)
    collect_pit_and_lap(all_seasons)

    print("=" * 60)
    print("Data collection complete! Next: feature_engineering.py")
    print("=" * 60)