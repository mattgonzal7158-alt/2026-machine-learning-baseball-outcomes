import json
import os
from datetime import date, datetime, timedelta

import pandas as pd
from pybaseball import statcast

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

STATE_PATH = os.path.join(DATA_DIR, "update_state.json")

def today_local():
    return date.today()

def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, "r") as f:
            return json.load(f)
    return {}

def save_state(state):
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)

def default_start_for_year(year: int):
    # Statcast regular season typically starts late March/early April
    # We'll just start Mar 1 to be safe (spring training may be sparse).
    return date(year, 3, 1)

def safe_date_parse(s: str):
    return datetime.strptime(s, "%Y-%m-%d").date()

def main():
    state = load_state()

    year = today_local().year
    ytd_path = os.path.join(DATA_DIR, f"statcast_{year}_ytd.parquet")

    # Determine start date
    if "last_date" in state and state.get("year") == year:
        start_dt = safe_date_parse(state["last_date"]) + timedelta(days=1)
    else:
        start_dt = default_start_for_year(year)

    end_dt = today_local()

    # If already up to date
    if start_dt > end_dt:
        print(f"[update] Already up to date through {state.get('last_date')}.")
        return

    # Statcast endpoint expects strings
    start_s = start_dt.strftime("%Y-%m-%d")
    end_s = end_dt.strftime("%Y-%m-%d")

    print(f"[update] Downloading Statcast {year} data from {start_s} to {end_s} ...")
    new_df = statcast(start_dt=start_s, end_dt=end_s)

    if new_df is None or len(new_df) == 0:
        print("[update] No new rows returned (off day or API empty). Updating last_date anyway.")
        state["year"] = year
        state["last_date"] = end_s
        save_state(state)
        return

    print(f"[update] New rows: {new_df.shape}")

    # Append to existing parquet (dedupe)
    if os.path.exists(ytd_path):
        old_df = pd.read_parquet(ytd_path)
        combined = pd.concat([old_df, new_df], axis=0, ignore_index=True)
        # Use a strong-ish dedupe key if present
        # pitch_id exists on many Statcast pulls; if not, fallback to core identifiers.
        if "pitch_id" in combined.columns:
            combined = combined.drop_duplicates(subset=["pitch_id"])
        else:
            key_cols = [c for c in ["game_pk", "at_bat_number", "pitch_number", "pitch_type", "batter", "pitcher", "inning", "balls", "strikes"] if c in combined.columns]
            if key_cols:
                combined = combined.drop_duplicates(subset=key_cols)
        combined.to_parquet(ytd_path, index=False)
        print(f"[update] Updated parquet: {ytd_path} -> {combined.shape}")
    else:
        new_df.to_parquet(ytd_path, index=False)
        print(f"[update] Created parquet: {ytd_path} -> {new_df.shape}")

    # Save state
    state["year"] = year
    state["last_date"] = end_s
    save_state(state)
    print(f"[update] Saved state: {STATE_PATH} (last_date={end_s})")

if __name__ == "__main__":
    main()