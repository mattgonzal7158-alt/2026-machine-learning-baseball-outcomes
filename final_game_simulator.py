#!/usr/bin/env python
# coding: utf-8

# In[20]:


# ============================================================
# CELL 1: imports
# ============================================================

import requests
import pandas as pd
import numpy as np
import ipywidgets as widgets

from datetime import datetime
from IPython.display import display, HTML, clear_output

# In[21]:


# ============================================================
# CORE SIMULATOR DEPENDENCY BLOCK
# ============================================================

import os
import json
import unicodedata
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from pybaseball import playerid_lookup

warnings.filterwarnings("ignore")

# ----------------------------
# Paths
# ----------------------------
BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

STATCAST_MAIN_PATH = DATA_DIR / "statcast_2021_2025.parquet"
STATCAST_YTD_PATH = DATA_DIR / "statcast_2026_ytd.parquet"
PA_MODEL_PATH = MODELS_DIR / "pregame_pa_model.cbm"

# optional meta
POSSIBLE_META_PATHS = [
    MODELS_DIR / "pregame_pa_model_meta.json",
    MODELS_DIR / "model_meta.json",
    MODELS_DIR / "winloss_pa_model_meta.json"
]

PA_META_PATH = None
for p in POSSIBLE_META_PATHS:
    if p.exists():
        PA_META_PATH = p
        break

# ----------------------------
# Load model
# ----------------------------
pa_model = None
model_feature_names = []

if PA_MODEL_PATH.exists():
    try:
        pa_model = CatBoostClassifier()
        pa_model.load_model(str(PA_MODEL_PATH))
        model_feature_names = pa_model.feature_names_
        print("Loaded CatBoost PA model.")
    except Exception as e:
        print("Could not load CatBoost model:", e)
        pa_model = None
else:
    print("PA model not found. Falling back to stabilized historical rates only.")

if PA_META_PATH is not None:
    try:
        with open(PA_META_PATH, "r") as f:
            pa_meta = json.load(f)
    except Exception:
        pa_meta = {}
else:
    pa_meta = {}

# ----------------------------
# Load data
# ----------------------------
if not STATCAST_MAIN_PATH.exists():
    raise FileNotFoundError(f"Missing required file: {STATCAST_MAIN_PATH}")

df_main = pd.read_parquet(STATCAST_MAIN_PATH)

if STATCAST_YTD_PATH.exists():
    df_ytd = pd.read_parquet(STATCAST_YTD_PATH)
    df = pd.concat([df_main, df_ytd], ignore_index=True)
else:
    df = df_main.copy()

df = df.copy()
if "game_date" in df.columns:
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")

# ----------------------------
# Event mapping
# ----------------------------
PA_CLASSES = ["out", "strikeout", "walk", "single", "double", "triple", "home_run"]

BB_EVENTS = {"walk", "intent_walk"}
K_EVENTS = {"strikeout", "strikeout_double_play"}

def map_event_to_pa_class(event):
    if pd.isna(event):
        return "out"

    event = str(event).lower().strip()

    if event in BB_EVENTS:
        return "walk"
    if event in K_EVENTS:
        return "strikeout"
    if event == "single":
        return "single"
    if event == "double":
        return "double"
    if event == "triple":
        return "triple"
    if event == "home_run":
        return "home_run"

    return "out"

# ----------------------------
# Build true PA dataset
# ----------------------------
pa_sort_cols = [c for c in ["game_pk", "at_bat_number", "pitch_number"] if c in df.columns]
pa_work = df.copy()

if pa_sort_cols:
    pa_work = pa_work.sort_values(pa_sort_cols)

if "game_pk" in pa_work.columns and "at_bat_number" in pa_work.columns:
    pa_df = (
        pa_work.groupby(["game_pk", "at_bat_number"], as_index=False)
        .tail(1)
        .copy()
    )
else:
    pa_df = pa_work[pa_work["events"].notna()].copy()

pa_df = pa_df[
    pa_df["batter"].notna() &
    pa_df["pitcher"].notna()
].copy()

pa_df["pa_class_simple"] = pa_df["events"].apply(map_event_to_pa_class)

# ----------------------------
# League rates
# ----------------------------
league_rates = pa_df["pa_class_simple"].value_counts(normalize=True).to_dict()
for c in PA_CLASSES:
    league_rates[c] = league_rates.get(c, 0.0)

# ----------------------------
# Stabilized batter/pitcher tables
# ----------------------------
batter_counts = (
    pa_df.groupby("batter")["pa_class_simple"]
    .value_counts()
    .unstack(fill_value=0)
)
for c in PA_CLASSES:
    if c not in batter_counts.columns:
        batter_counts[c] = 0

batter_pa = batter_counts.sum(axis=1)
BATTER_PRIOR = 200
batter_stats = batter_counts.copy()
for c in PA_CLASSES:
    batter_stats[c] = (batter_counts[c] + BATTER_PRIOR * league_rates[c]) / (batter_pa + BATTER_PRIOR)
batter_stats = batter_stats.reset_index()

pitcher_counts = (
    pa_df.groupby("pitcher")["pa_class_simple"]
    .value_counts()
    .unstack(fill_value=0)
)
for c in PA_CLASSES:
    if c not in pitcher_counts.columns:
        pitcher_counts[c] = 0

pitcher_pa_counts = pitcher_counts.sum(axis=1)
PITCHER_PRIOR = 250
pitcher_stats = pitcher_counts.copy()
for c in PA_CLASSES:
    pitcher_stats[c] = (pitcher_counts[c] + PITCHER_PRIOR * league_rates[c]) / (pitcher_pa_counts + PITCHER_PRIOR)
pitcher_stats = pitcher_stats.reset_index()

batter_stats_indexed = batter_stats.set_index("batter")
pitcher_stats_indexed = pitcher_stats.set_index("pitcher")

# ----------------------------
# Hitter/pitcher feature tables for CatBoost
# ----------------------------
if "stand" not in pa_df.columns:
    pa_df["stand"] = "R"
if "p_throws" not in pa_df.columns:
    pa_df["p_throws"] = "R"

hitter_pa = (
    pa_df.groupby("batter", as_index=False)
    .agg(
        batter_pa=("pa_class_simple", "size"),
        batter_k_rate=("pa_class_simple", lambda s: (s == "strikeout").mean()),
        batter_bb_rate=("pa_class_simple", lambda s: (s == "walk").mean()),
        batter_1b_rate=("pa_class_simple", lambda s: (s == "single").mean()),
        batter_2b_rate=("pa_class_simple", lambda s: (s == "double").mean()),
        batter_3b_rate=("pa_class_simple", lambda s: (s == "triple").mean()),
        batter_hr_rate=("pa_class_simple", lambda s: (s == "home_run").mean()),
        batter_out_rate=("pa_class_simple", lambda s: (s == "out").mean()),
        stand=("stand", lambda s: s.dropna().mode().iloc[0] if len(s.dropna()) else "R")
    )
)

pitcher_pa = (
    pa_df.groupby("pitcher", as_index=False)
    .agg(
        pitcher_pa=("pa_class_simple", "size"),
        pitcher_k_rate=("pa_class_simple", lambda s: (s == "strikeout").mean()),
        pitcher_bb_rate=("pa_class_simple", lambda s: (s == "walk").mean()),
        pitcher_1b_rate=("pa_class_simple", lambda s: (s == "single").mean()),
        pitcher_2b_rate=("pa_class_simple", lambda s: (s == "double").mean()),
        pitcher_3b_rate=("pa_class_simple", lambda s: (s == "triple").mean()),
        pitcher_hr_rate=("pa_class_simple", lambda s: (s == "home_run").mean()),
        pitcher_out_rate=("pa_class_simple", lambda s: (s == "out").mean()),
        p_throws=("p_throws", lambda s: s.dropna().mode().iloc[0] if len(s.dropna()) else "R")
    )
)

hitter_split = (
    pa_df.groupby(["batter", "p_throws"], as_index=False)
    .agg(
        split_pa=("pa_class_simple", "size"),
        split_k_rate=("pa_class_simple", lambda s: (s == "strikeout").mean()),
        split_bb_rate=("pa_class_simple", lambda s: (s == "walk").mean()),
        split_1b_rate=("pa_class_simple", lambda s: (s == "single").mean()),
        split_2b_rate=("pa_class_simple", lambda s: (s == "double").mean()),
        split_3b_rate=("pa_class_simple", lambda s: (s == "triple").mean()),
        split_hr_rate=("pa_class_simple", lambda s: (s == "home_run").mean()),
        split_out_rate=("pa_class_simple", lambda s: (s == "out").mean())
    )
)

pitcher_split = (
    pa_df.groupby(["pitcher", "stand"], as_index=False)
    .agg(
        p_split_pa=("pa_class_simple", "size"),
        p_split_k_rate=("pa_class_simple", lambda s: (s == "strikeout").mean()),
        p_split_bb_rate=("pa_class_simple", lambda s: (s == "walk").mean()),
        p_split_1b_rate=("pa_class_simple", lambda s: (s == "single").mean()),
        p_split_2b_rate=("pa_class_simple", lambda s: (s == "double").mean()),
        p_split_3b_rate=("pa_class_simple", lambda s: (s == "triple").mean()),
        p_split_hr_rate=("pa_class_simple", lambda s: (s == "home_run").mean()),
        p_split_out_rate=("pa_class_simple", lambda s: (s == "out").mean())
    )
)

hitter_pa_idx = hitter_pa.set_index("batter")
pitcher_pa_idx = pitcher_pa.set_index("pitcher")
hitter_split_idx = hitter_split.set_index(["batter", "p_throws"])
pitcher_split_idx = pitcher_split.set_index(["pitcher", "stand"])

# ----------------------------
# Name lookup
# ----------------------------
MANUAL_ID_MAP = {
    "teoscar hernandez": 606192,
    "kike hernandez": 571771,
    "enrique hernandez": 571771,
}

def normalize_name(name):
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("utf-8")
    name = name.lower().strip()
    return " ".join(name.split())

def get_player_id_from_full_name(full_name):
    clean = normalize_name(full_name)

    if clean in MANUAL_ID_MAP:
        return MANUAL_ID_MAP[clean]

    parts = full_name.strip().split()
    if len(parts) < 2:
        return None

    first = parts[0]
    last = parts[-1]

    try:
        res = playerid_lookup(last, first)
    except Exception:
        return None

    if len(res) == 0:
        return None

    if "mlb_played_last" in res.columns:
        res = res.sort_values("mlb_played_last", ascending=False)

    return int(res.iloc[0]["key_mlbam"])

# ----------------------------
# Validation
# ----------------------------
def validate_lineup_ids(lineup_ids, team_name="Team"):
    bad = [x for x in lineup_ids if x is None]
    if bad:
        raise ValueError(f"{team_name} lineup still has missing player IDs: {lineup_ids}")
    if len(lineup_ids) != 9:
        raise ValueError(f"{team_name} lineup must have 9 hitters. Got {len(lineup_ids)}")
    return True

# ----------------------------
# Probability helpers
# ----------------------------
def safe_prob_vector(prob_dict):
    arr = np.array([prob_dict.get(c, 0.0) for c in PA_CLASSES], dtype=float)
    if arr.sum() <= 0:
        arr = np.array([league_rates[c] for c in PA_CLASSES], dtype=float)
    arr = np.clip(arr, 1e-9, None)
    arr = arr / arr.sum()
    return arr

def get_batter_probs(batter_id):
    if batter_id in batter_stats_indexed.index:
        row = batter_stats_indexed.loc[batter_id, PA_CLASSES]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        return safe_prob_vector(row.to_dict())
    return safe_prob_vector(league_rates)

def get_pitcher_probs(pitcher_id):
    if pitcher_id in pitcher_stats_indexed.index:
        row = pitcher_stats_indexed.loc[pitcher_id, PA_CLASSES]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        return safe_prob_vector(row.to_dict())
    return safe_prob_vector(league_rates)

def get_matchup_probs(batter_id, pitcher_id):
    b_probs = get_batter_probs(batter_id)
    p_probs = get_pitcher_probs(pitcher_id)
    l_probs = np.array([league_rates[c] for c in PA_CLASSES], dtype=float)

    probs = l_probs * (b_probs / l_probs) * (p_probs / l_probs)
    probs = np.clip(probs, 1e-9, None)
    probs = probs / probs.sum()
    return probs

def get_row_or_default(indexed_df, key):
    if key in indexed_df.index:
        row = indexed_df.loc[key]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        return row.to_dict()
    return {}

def build_model_feature_row(batter_id, pitcher_id):
    row = {}
    row["batter"] = batter_id
    row["pitcher"] = pitcher_id

    h = get_row_or_default(hitter_pa_idx, batter_id)
    p = get_row_or_default(pitcher_pa_idx, pitcher_id)
    row.update(h)
    row.update(p)

    stand = row.get("stand", "R")
    p_throws = row.get("p_throws", "R")

    hs = get_row_or_default(hitter_split_idx, (batter_id, p_throws))
    ps = get_row_or_default(pitcher_split_idx, (pitcher_id, stand))
    row.update(hs)
    row.update(ps)

    neutral_defaults = {
        "balls": 0,
        "strikes": 0,
        "outs_when_up": 0,
        "on_1b": 0,
        "on_2b": 0,
        "on_3b": 0,
        "inning": 1,
        "bat_score": 0,
        "fld_score": 0,
        "home_team": "UNK",
        "away_team": "UNK",
        "inning_topbot": "Top",
        "if_fielding_alignment": "Standard",
        "of_fielding_alignment": "Standard",
    }

    for k, v in neutral_defaults.items():
        if k not in row:
            row[k] = v

    final_row = {}
    for feat in model_feature_names:
        if feat in row:
            final_row[feat] = row[feat]
        else:
            if feat == "stand":
                final_row[feat] = stand
            elif feat == "p_throws":
                final_row[feat] = p_throws
            elif feat == "inning_topbot":
                final_row[feat] = "Top"
            elif feat in ["if_fielding_alignment", "of_fielding_alignment"]:
                final_row[feat] = "Standard"
            elif feat in ["home_team", "away_team"]:
                final_row[feat] = "UNK"
            else:
                final_row[feat] = 0

    return pd.DataFrame([final_row])

matchup_prob_cache = {}

def get_catboost_matchup_probs(batter_id, pitcher_id):
    key = (int(batter_id), int(pitcher_id))

    if key in matchup_prob_cache:
        return matchup_prob_cache[key]

    if pa_model is None or len(model_feature_names) == 0:
        arr = get_matchup_probs(batter_id, pitcher_id)
        matchup_prob_cache[key] = arr
        return arr

    try:
        X = build_model_feature_row(batter_id, pitcher_id)
        probs = pa_model.predict_proba(X)[0]

        if hasattr(pa_model, "classes_") and pa_model.classes_ is not None:
            model_classes = list(pa_model.classes_)
        else:
            model_classes = PA_CLASSES

        prob_map = {cls: prob for cls, prob in zip(model_classes, probs)}
        arr = np.array([prob_map.get(c, 0.0) for c in PA_CLASSES], dtype=float)
        arr = np.clip(arr, 1e-9, None)
        arr = arr / arr.sum()

        matchup_prob_cache[key] = arr
        return arr

    except Exception:
        arr = get_matchup_probs(batter_id, pitcher_id)
        matchup_prob_cache[key] = arr
        return arr

# ----------------------------
# Run environment calibration
# ----------------------------
RUN_ENV_MULTIPLIERS = {
    "out": 1.08,
    "strikeout": 1.03,
    "walk": 0.94,
    "single": 0.93,
    "double": 0.92,
    "triple": 0.90,
    "home_run": 0.88
}

def apply_run_environment_adjustment(probs):
    adjusted = np.array(probs, dtype=float).copy()
    for i, cls in enumerate(PA_CLASSES):
        adjusted[i] *= RUN_ENV_MULTIPLIERS.get(cls, 1.0)
    adjusted = np.clip(adjusted, 1e-9, None)
    adjusted = adjusted / adjusted.sum()
    return adjusted

USE_CATBOOST_PA_MODEL = True

def simulate_pa(batter_id, pitcher_id):
    if USE_CATBOOST_PA_MODEL:
        probs = get_catboost_matchup_probs(batter_id, pitcher_id)
    else:
        probs = get_matchup_probs(batter_id, pitcher_id)

    probs = apply_run_environment_adjustment(probs)
    return np.random.choice(PA_CLASSES, p=probs)

# ----------------------------
# Base advancement
# ----------------------------
def advance_runners(bases, outcome):
    on_1b, on_2b, on_3b = bases
    runs = 0
    outs = 0

    if outcome in ["out", "strikeout"]:
        outs = 1
        return [on_1b, on_2b, on_3b], runs, outs

    if outcome == "walk":
        if on_1b and on_2b and on_3b:
            runs += 1
            return [1, 1, 1], runs, outs
        elif on_1b and on_2b:
            return [1, 1, 1], runs, outs
        elif on_1b:
            return [1, 1, on_3b], runs, outs
        else:
            return [1, on_2b, on_3b], runs, outs

    if outcome == "single":
        runs += on_3b
        return [1, on_1b, on_2b], runs, outs

    if outcome == "double":
        runs += on_3b + on_2b
        return [0, 1, on_1b], runs, outs

    if outcome == "triple":
        runs += on_1b + on_2b + on_3b
        return [0, 0, 1], runs, outs

    if outcome == "home_run":
        runs += on_1b + on_2b + on_3b + 1
        return [0, 0, 0], runs, outs

    outs = 1
    return [on_1b, on_2b, on_3b], runs, outs

# ----------------------------
# Starter / reliever usage
# ----------------------------
work = df.copy()

if "inning_topbot" in work.columns and "home_team" in work.columns and "away_team" in work.columns:
    work["pitching_team"] = np.where(
        work["inning_topbot"].astype(str).str.lower().eq("top"),
        work["home_team"],
        work["away_team"]
    )
else:
    work["pitching_team"] = None

sort_cols = [c for c in ["game_date", "game_pk", "inning", "at_bat_number", "pitch_number"] if c in work.columns]
if sort_cols:
    work = work.sort_values(sort_cols)

starter_rows = (
    work.groupby(["game_pk", "pitching_team"], as_index=False)
    .first()[["game_pk", "pitching_team", "pitcher"]]
    .rename(columns={"pitcher": "starter_pitcher_id"})
)

work = work.merge(starter_rows, on=["game_pk", "pitching_team"], how="left")
work["is_starter_pitcher"] = work["pitcher"] == work["starter_pitcher_id"]

starter_df = work[work["is_starter_pitcher"]].copy()
relief_df = work[~work["is_starter_pitcher"]].copy()

starter_pitch_counts = (
    starter_df.groupby(["game_pk", "pitcher"])
    .size()
    .reset_index(name="starter_pitches")
)

starter_bf = (
    starter_df.groupby(["game_pk", "pitcher"])["batter"]
    .nunique()
    .reset_index(name="batters_faced")
)

starter_innings = (
    starter_df.groupby(["game_pk", "pitcher"])["inning"]
    .nunique()
    .reset_index(name="innings_seen")
)

starter_game_summary = starter_pitch_counts.merge(starter_bf, on=["game_pk", "pitcher"])
starter_game_summary = starter_game_summary.merge(starter_innings, on=["game_pk", "pitcher"])

starter_usage = (
    starter_game_summary.groupby("pitcher", as_index=False)
    .agg(
        starts=("game_pk", "nunique"),
        avg_batters_faced=("batters_faced", "mean"),
        avg_innings_seen=("innings_seen", "mean"),
        avg_pitches=("starter_pitches", "mean"),
        med_pitches=("starter_pitches", "median"),
        max_pitches=("starter_pitches", "max")
    )
)

reliever_pitch_counts = (
    relief_df.groupby(["game_pk", "pitcher"])
    .size()
    .reset_index(name="relief_pitches")
)

reliever_bf = (
    relief_df.groupby(["game_pk", "pitcher"])["batter"]
    .nunique()
    .reset_index(name="batters_faced")
)

reliever_innings = (
    relief_df.groupby(["game_pk", "pitcher"])
    .agg(
        innings_seen=("inning", "nunique"),
        first_inning_entered=("inning", "min")
    )
    .reset_index()
)

reliever_game_summary = reliever_pitch_counts.merge(reliever_bf, on=["game_pk", "pitcher"])
reliever_game_summary = reliever_game_summary.merge(reliever_innings, on=["game_pk", "pitcher"])

reliever_usage = (
    reliever_game_summary.groupby("pitcher", as_index=False)
    .agg(
        relief_apps=("game_pk", "nunique"),
        avg_relief_innings=("innings_seen", "mean"),
        avg_relief_bf=("batters_faced", "mean"),
        avg_relief_pitches=("relief_pitches", "mean"),
        avg_entry_inning=("first_inning_entered", "mean"),
        median_entry_inning=("first_inning_entered", "median")
    )
)

def assign_reliever_role(avg_entry_inning):
    if pd.isna(avg_entry_inning):
        return "middle"
    if avg_entry_inning >= 8.5:
        return "closer"
    if avg_entry_inning >= 7.5:
        return "setup"
    if avg_entry_inning <= 5.0:
        return "long"
    return "middle"

reliever_usage["role_guess"] = reliever_usage["avg_entry_inning"].apply(assign_reliever_role)

# ----------------------------
# Starter leash / bullpen
# ----------------------------
starter_usage_idx = starter_usage.set_index("pitcher")
reliever_usage_work = reliever_usage[reliever_usage["relief_apps"] >= 5].copy()
reliever_usage_idx = reliever_usage_work.set_index("pitcher")

def get_starter_leash(pitcher_id):
    default_pitch_cap = 95
    default_innings_cap = 5.5

    if pitcher_id not in starter_usage_idx.index:
        return {
            "pitch_cap": default_pitch_cap,
            "innings_cap": default_innings_cap,
            "avg_pitches": default_pitch_cap,
            "avg_innings": default_innings_cap,
            "starts": 0
        }

    row = starter_usage_idx.loc[pitcher_id]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]

    starts = float(row.get("starts", 0))
    avg_pitches = float(row.get("avg_pitches", default_pitch_cap))
    avg_innings = float(row.get("avg_innings_seen", default_innings_cap))

    weight = min(starts / 15.0, 1.0)

    pitch_cap = (weight * avg_pitches) + ((1 - weight) * default_pitch_cap)
    innings_cap = (weight * avg_innings) + ((1 - weight) * default_innings_cap)

    pitch_cap = min(max(pitch_cap + 5, 75), 110)
    innings_cap = min(max(innings_cap + 0.3, 4.0), 7.5)

    return {
        "pitch_cap": float(pitch_cap),
        "innings_cap": float(innings_cap),
        "avg_pitches": float(avg_pitches),
        "avg_innings": float(avg_innings),
        "starts": int(starts)
    }

def get_team_pitcher_pool(team_code):
    if "pitching_team" not in pa_df.columns:
        pa_df["pitching_team"] = np.where(
            pa_df["inning_topbot"].astype(str).str.lower().eq("top"),
            pa_df["home_team"],
            pa_df["away_team"]
        )

    return pa_df.loc[pa_df["pitching_team"] == team_code, "pitcher"].dropna().unique().tolist()

def build_team_bullpen(team_code, starter_id):
    team_pitchers = set(get_team_pitcher_pool(team_code))

    bullpen = reliever_usage_work[
        reliever_usage_work["pitcher"].isin(team_pitchers) &
        (reliever_usage_work["pitcher"] != starter_id)
    ].copy()

    if bullpen.empty:
        return bullpen

    bullpen = bullpen.sort_values(
        ["role_guess", "relief_apps", "avg_relief_pitches"],
        ascending=[True, False, False]
    ).reset_index(drop=True)

    return bullpen

# ----------------------------
# Pitch counts / pitcher state
# ----------------------------
PITCH_ESTIMATES = {
    "out": 3.6,
    "strikeout": 4.4,
    "walk": 5.2,
    "single": 3.5,
    "double": 3.7,
    "triple": 4.0,
    "home_run": 3.8
}

def estimate_pitches_for_pa(outcome):
    base = PITCH_ESTIMATES.get(outcome, 3.8)
    val = np.random.normal(loc=base, scale=0.8)
    return max(1, int(round(val)))

def init_pitcher_state(pitcher_id, role="starter"):
    if role == "starter":
        leash = get_starter_leash(pitcher_id)
        pitch_cap = leash["pitch_cap"]
        inning_cap = leash["innings_cap"]
    else:
        if pitcher_id in reliever_usage_idx.index:
            row = reliever_usage_idx.loc[pitcher_id]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            avg_relief_pitches = float(row.get("avg_relief_pitches", 18))
        else:
            avg_relief_pitches = 18

        pitch_cap = min(max(avg_relief_pitches + 5, 15), 35)
        inning_cap = 1.3

    return {
        "pitcher_id": pitcher_id,
        "role": role,
        "pitch_count": 0,
        "batters_faced": 0,
        "outs_recorded": 0,
        "runs_allowed": 0,
        "current_inning_runs": 0,
        "pitch_cap": float(pitch_cap),
        "inning_cap": float(inning_cap)
    }

def choose_next_reliever(bullpen_df, used_pitchers, inning, score_diff):
    if bullpen_df is None or bullpen_df.empty:
        return None

    available = bullpen_df[~bullpen_df["pitcher"].isin(used_pitchers)].copy()
    if available.empty:
        return None

    low_leverage = abs(score_diff) >= 6

    if inning >= 9 and not low_leverage:
        pref_roles = ["closer", "setup", "middle", "long"]
    elif inning == 8 and not low_leverage:
        pref_roles = ["setup", "closer", "middle", "long"]
    elif inning <= 5:
        pref_roles = ["long", "middle", "setup", "closer"]
    else:
        pref_roles = ["middle", "setup", "long", "closer"]

    if low_leverage:
        pref_roles = ["long", "middle", "setup", "closer"]

    for role in pref_roles:
        candidates = available[available["role_guess"] == role].copy()
        if not candidates.empty:
            candidates = candidates.sort_values(
                ["relief_apps", "avg_relief_pitches"],
                ascending=[False, False]
            )
            return int(candidates.iloc[0]["pitcher"])

    return int(available.iloc[0]["pitcher"])

def should_remove_pitcher(p_state, inning, half, score_diff, just_allowed_runs=0):
    role = p_state["role"]
    innings_completed_est = p_state["outs_recorded"] / 3.0

    if just_allowed_runs > 0:
        p_state["current_inning_runs"] += just_allowed_runs

    if role == "starter":
        if p_state["pitch_count"] >= p_state["pitch_cap"]:
            return True
        if innings_completed_est >= p_state["inning_cap"] and inning >= 5:
            return True
        if inning >= 6 and p_state["pitch_count"] >= max(p_state["pitch_cap"] - 10, 75):
            return True
        if p_state["current_inning_runs"] >= 4:
            return True
        if p_state["runs_allowed"] >= 6 and inning <= 5:
            return True
        return False

    else:
        if p_state["pitch_count"] >= p_state["pitch_cap"]:
            return True
        if innings_completed_est >= p_state["inning_cap"] and p_state["pitch_count"] >= 12:
            return True
        if p_state["current_inning_runs"] >= 3:
            return True
        if p_state["batters_faced"] >= 6 and inning >= 6:
            return True
        return False

# ----------------------------
# Inning / game simulation
# ----------------------------
def simulate_half_inning_with_pitching(
    lineup_ids,
    lineup_pos,
    pitching_team_code,
    batting_team_runs,
    pitching_team_runs,
    inning,
    half,
    current_pitcher_state,
    bullpen_df,
    used_pitchers
):
    outs = 0
    runs = 0
    bases = [0, 0, 0]

    current_pitcher_state["current_inning_runs"] = 0

    while outs < 3:
        batter_id = lineup_ids[lineup_pos % 9]
        pitcher_id = current_pitcher_state["pitcher_id"]

        outcome = simulate_pa(batter_id, pitcher_id)
        est_pitches = estimate_pitches_for_pa(outcome)

        current_pitcher_state["pitch_count"] += est_pitches
        current_pitcher_state["batters_faced"] += 1

        bases, new_runs, new_outs = advance_runners(bases, outcome)

        runs += new_runs
        outs += new_outs

        current_pitcher_state["outs_recorded"] += new_outs
        current_pitcher_state["runs_allowed"] += new_runs

        lineup_pos += 1

        score_diff = pitching_team_runs - (batting_team_runs + runs)

        if outs < 3 and should_remove_pitcher(
            current_pitcher_state,
            inning=inning,
            half=half,
            score_diff=score_diff,
            just_allowed_runs=new_runs
        ):
            next_pitcher_id = choose_next_reliever(
                bullpen_df=bullpen_df,
                used_pitchers=used_pitchers,
                inning=inning,
                score_diff=score_diff
            )

            if next_pitcher_id is not None:
                used_pitchers.add(next_pitcher_id)
                current_pitcher_state = init_pitcher_state(next_pitcher_id, role="reliever")
                current_pitcher_state["current_inning_runs"] = 0

    return runs, lineup_pos, current_pitcher_state, used_pitchers

def simulate_game_with_pitching(
    away_lineup_ids,
    home_lineup_ids,
    away_pitcher_id,
    home_pitcher_id,
    away_team_code="AWAY",
    home_team_code="HOME",
    max_extra_innings=6
):
    away_runs = 0
    home_runs = 0
    away_lineup_pos = 0
    home_lineup_pos = 0

    away_bullpen = build_team_bullpen(away_team_code, away_pitcher_id)
    home_bullpen = build_team_bullpen(home_team_code, home_pitcher_id)

    used_away_pitchers = {away_pitcher_id}
    used_home_pitchers = {home_pitcher_id}

    away_pitcher_state = init_pitcher_state(away_pitcher_id, role="starter")
    home_pitcher_state = init_pitcher_state(home_pitcher_id, role="starter")

    for inning in range(1, 10):
        top_runs, away_lineup_pos, home_pitcher_state, used_home_pitchers = simulate_half_inning_with_pitching(
            lineup_ids=away_lineup_ids,
            lineup_pos=away_lineup_pos,
            pitching_team_code=home_team_code,
            batting_team_runs=away_runs,
            pitching_team_runs=home_runs,
            inning=inning,
            half="top",
            current_pitcher_state=home_pitcher_state,
            bullpen_df=home_bullpen,
            used_pitchers=used_home_pitchers
        )
        away_runs += top_runs

        bot_runs, home_lineup_pos, away_pitcher_state, used_away_pitchers = simulate_half_inning_with_pitching(
            lineup_ids=home_lineup_ids,
            lineup_pos=home_lineup_pos,
            pitching_team_code=away_team_code,
            batting_team_runs=home_runs,
            pitching_team_runs=away_runs,
            inning=inning,
            half="bottom",
            current_pitcher_state=away_pitcher_state,
            bullpen_df=away_bullpen,
            used_pitchers=used_away_pitchers
        )
        home_runs += bot_runs

        if should_remove_pitcher(home_pitcher_state, inning, "end_top", home_runs - away_runs, 0):
            next_pitcher_id = choose_next_reliever(home_bullpen, used_home_pitchers, inning + 1, home_runs - away_runs)
            if next_pitcher_id is not None and next_pitcher_id != home_pitcher_state["pitcher_id"]:
                used_home_pitchers.add(next_pitcher_id)
                home_pitcher_state = init_pitcher_state(next_pitcher_id, role="reliever")

        if should_remove_pitcher(away_pitcher_state, inning, "end_bottom", away_runs - home_runs, 0):
            next_pitcher_id = choose_next_reliever(away_bullpen, used_away_pitchers, inning + 1, away_runs - home_runs)
            if next_pitcher_id is not None and next_pitcher_id != away_pitcher_state["pitcher_id"]:
                used_away_pitchers.add(next_pitcher_id)
                away_pitcher_state = init_pitcher_state(next_pitcher_id, role="reliever")

    extras_used = 0
    inning = 10
    while away_runs == home_runs and extras_used < max_extra_innings:
        top_runs, away_lineup_pos, home_pitcher_state, used_home_pitchers = simulate_half_inning_with_pitching(
            lineup_ids=away_lineup_ids,
            lineup_pos=away_lineup_pos,
            pitching_team_code=home_team_code,
            batting_team_runs=away_runs,
            pitching_team_runs=home_runs,
            inning=inning,
            half="top",
            current_pitcher_state=home_pitcher_state,
            bullpen_df=home_bullpen,
            used_pitchers=used_home_pitchers
        )
        away_runs += top_runs

        bot_runs, home_lineup_pos, away_pitcher_state, used_away_pitchers = simulate_half_inning_with_pitching(
            lineup_ids=home_lineup_ids,
            lineup_pos=home_lineup_pos,
            pitching_team_code=away_team_code,
            batting_team_runs=home_runs,
            pitching_team_runs=away_runs,
            inning=inning,
            half="bottom",
            current_pitcher_state=away_pitcher_state,
            bullpen_df=away_bullpen,
            used_pitchers=used_away_pitchers
        )
        home_runs += bot_runs

        inning += 1
        extras_used += 1

    return {
        "away_runs": away_runs,
        "home_runs": home_runs
    }

def simulate_matchup_with_pitching(
    away_lineup_ids,
    home_lineup_ids,
    away_pitcher_id,
    home_pitcher_id,
    away_team_code,
    home_team_code,
    n_sims=150
):
    results = []

    for _ in range(n_sims):
        g = simulate_game_with_pitching(
            away_lineup_ids=away_lineup_ids,
            home_lineup_ids=home_lineup_ids,
            away_pitcher_id=away_pitcher_id,
            home_pitcher_id=home_pitcher_id,
            away_team_code=away_team_code,
            home_team_code=home_team_code
        )
        results.append(g)

    results_df = pd.DataFrame(results)

    return {
        "results_df": results_df,
        "away_avg_runs": results_df["away_runs"].mean(),
        "home_avg_runs": results_df["home_runs"].mean(),
        "away_win_pct": (results_df["away_runs"] > results_df["home_runs"]).mean(),
        "home_win_pct": (results_df["home_runs"] > results_df["away_runs"]).mean()
    }

print("Core simulator dependency block loaded.")
print("PA rows:", len(pa_df))
print("League rates:", {k: round(v, 4) for k, v in league_rates.items()})



# In[22]:


# ============================================================
# CELL 2: MLB API helpers
# ============================================================

BASE_MLB_API = "https://statsapi.mlb.com/api"

TEAM_ABBR_TO_ID = {
    "ARI": 109, "ATL": 144, "BAL": 110, "BOS": 111, "CHC": 112, "CWS": 145,
    "CIN": 113, "CLE": 114, "COL": 115, "DET": 116, "HOU": 117, "KC": 118,
    "LAA": 108, "LAD": 119, "MIA": 146, "MIL": 158, "MIN": 142, "NYM": 121,
    "NYY": 147, "ATH": 133, "PHI": 143, "PIT": 134, "SD": 135, "SEA": 136,
    "SF": 137, "STL": 138, "TB": 139, "TEX": 140, "TOR": 141, "WSH": 120
}

TEAM_ID_TO_ABBR = {v: k for k, v in TEAM_ABBR_TO_ID.items()}

def mlb_get(path, params=None, timeout=20):
    r = requests.get(f"{BASE_MLB_API}{path}", params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

def safe_name(x):
    if not x:
        return None
    return x.get("fullName") or x.get("name")

def team_abbr_from_id(team_id, fallback_name=None):
    return TEAM_ID_TO_ABBR.get(team_id, fallback_name)

# In[23]:


# ============================================================
# CELL 3: today's games
# ============================================================

def get_daily_games(game_date=None):
    if game_date is None:
        game_date = datetime.now().strftime("%Y-%m-%d")

    data = mlb_get(
        "/v1/schedule",
        params={
            "sportId": 1,
            "date": game_date,
            "hydrate": "probablePitcher,linescore"
        }
    )

    rows = []

    for date_block in data.get("dates", []):
        for g in date_block.get("games", []):
            away = g.get("teams", {}).get("away", {})
            home = g.get("teams", {}).get("home", {})
            away_team = away.get("team", {})
            home_team = home.get("team", {})

            away_id = away_team.get("id")
            home_id = home_team.get("id")

            away_abbr = away_team.get("abbreviation") or team_abbr_from_id(away_id, away_team.get("name"))
            home_abbr = home_team.get("abbreviation") or team_abbr_from_id(home_id, home_team.get("name"))

            rows.append({
                "gamePk": g.get("gamePk"),
                "gameDate": g.get("gameDate"),
                "status": g.get("status", {}).get("detailedState"),
                "away_name": away_team.get("name"),
                "away_abbr": away_abbr,
                "away_id": away_id,
                "away_probable_pitcher": safe_name(away.get("probablePitcher")),
                "home_name": home_team.get("name"),
                "home_abbr": home_abbr,
                "home_id": home_id,
                "home_probable_pitcher": safe_name(home.get("probablePitcher")),
            })

    games_df = pd.DataFrame(rows)

    if not games_df.empty:
        games_df["label"] = (
            games_df["away_abbr"].astype(str) + " @ " +
            games_df["home_abbr"].astype(str) + " | " +
            games_df["status"].fillna("Unknown").astype(str) + " | " +
            games_df["gameDate"].astype(str)
        )

    return games_df

# In[24]:


# ============================================================
# CELL 4: active roster + game snapshot
# ============================================================

def get_team_active_roster(team_id, game_date=None):
    if game_date is None:
        game_date = datetime.now().strftime("%Y-%m-%d")

    data = mlb_get(
        f"/v1/teams/{team_id}/roster",
        params={
            "rosterType": "active",
            "date": game_date
        }
    )

    rows = []

    for p in data.get("roster", []):
        person = p.get("person", {})
        pos = p.get("position", {})

        rows.append({
            "player_id": person.get("id"),
            "player_name": person.get("fullName"),
            "position_code": pos.get("abbreviation"),
            "position_type": pos.get("type")
        })

    return pd.DataFrame(rows)

def get_game_snapshot(game_pk):
    feed = mlb_get(f"/v1.1/game/{game_pk}/feed/live")

    game_data = feed.get("gameData", {})
    live_data = feed.get("liveData", {})
    boxscore = live_data.get("boxscore", {})
    linescore = live_data.get("linescore", {})

    away_team = game_data.get("teams", {}).get("away", {})
    home_team = game_data.get("teams", {}).get("home", {})

    offense = linescore.get("offense", {}) or {}
    defense = linescore.get("defense", {}) or {}

    return {
        "gamePk": game_pk,
        "away_name": away_team.get("name"),
        "away_abbr": away_team.get("abbreviation") or team_abbr_from_id(away_team.get("id"), away_team.get("name")),
        "away_id": away_team.get("id"),
        "home_name": home_team.get("name"),
        "home_abbr": home_team.get("abbreviation") or team_abbr_from_id(home_team.get("id"), home_team.get("name")),
        "home_id": home_team.get("id"),
        "away_probable_pitcher": safe_name(away_team.get("probablePitcher")),
        "home_probable_pitcher": safe_name(home_team.get("probablePitcher")),
        "status": game_data.get("status", {}).get("detailedState"),
        "inning": linescore.get("currentInning"),
        "inning_half": linescore.get("inningHalf"),
        "outs": linescore.get("outs"),
        "away_score": linescore.get("teams", {}).get("away", {}).get("runs"),
        "home_score": linescore.get("teams", {}).get("home", {}).get("runs"),
        "runner_on_1b": offense.get("first", {}).get("id") if offense.get("first") else None,
        "runner_on_2b": offense.get("second", {}).get("id") if offense.get("second") else None,
        "runner_on_3b": offense.get("third", {}).get("id") if offense.get("third") else None,
        "current_batter": safe_name(offense.get("batter")),
        "current_pitcher": safe_name(defense.get("pitcher")),
        "feed_json": feed,
        "away_box": boxscore.get("teams", {}).get("away", {}),
        "home_box": boxscore.get("teams", {}).get("home", {})
    }

# In[25]:


# ============================================================
# CELL 5: lineup resolution
# ============================================================

def extract_confirmed_lineup(team_box):
    players = team_box.get("players", {})
    rows = []

    for _, p in players.items():
        batting_order = p.get("battingOrder")
        if batting_order:
            rows.append({
                "player_id": p.get("person", {}).get("id"),
                "player_name": p.get("person", {}).get("fullName"),
                "position_code": p.get("position", {}).get("abbreviation"),
                "bat_order": batting_order
            })

    lineup_df = pd.DataFrame(rows)

    if not lineup_df.empty:
        lineup_df = lineup_df.sort_values("bat_order").head(9).reset_index(drop=True)

    return lineup_df

def build_projected_lineup_from_roster(roster_df):
    hitters = roster_df[
        ~roster_df["position_type"].astype(str).str.contains("Pitcher", case=False, na=False)
    ].copy()

    hitters = hitters.head(9).reset_index(drop=True)
    hitters["bat_order"] = [(i + 1) * 100 for i in range(len(hitters))]

    return hitters[["player_id", "player_name", "position_code", "bat_order"]]

def resolve_game_lineups(snapshot, game_date=None):
    away_lineup = extract_confirmed_lineup(snapshot["away_box"])
    home_lineup = extract_confirmed_lineup(snapshot["home_box"])

    away_source = "confirmed"
    home_source = "confirmed"

    if away_lineup.empty:
        away_lineup = build_projected_lineup_from_roster(
            get_team_active_roster(snapshot["away_id"], game_date=game_date)
        )
        away_source = "projected_from_active_roster"

    if home_lineup.empty:
        home_lineup = build_projected_lineup_from_roster(
            get_team_active_roster(snapshot["home_id"], game_date=game_date)
        )
        home_source = "projected_from_active_roster"

    return {
        "away_lineup": away_lineup,
        "home_lineup": home_lineup,
        "away_source": away_source,
        "home_source": home_source
    }

def lineup_df_to_ids(lineup_df):
    return lineup_df["player_id"].dropna().astype(int).tolist()

# In[26]:


# ============================================================
# CELL 6: live-state helpers
# ============================================================

def probable_pitcher_id_from_name(name):
    if not name:
        return None
    return get_player_id_from_full_name(name)

def count_team_pas_from_feed(feed_json):
    all_plays = feed_json.get("liveData", {}).get("plays", {}).get("allPlays", [])
    away_pa = 0
    home_pa = 0

    for play in all_plays:
        is_top = play.get("about", {}).get("isTopInning")
        if is_top is True:
            away_pa += 1
        elif is_top is False:
            home_pa += 1

    return away_pa, home_pa

def next_lineup_positions_from_feed(snapshot, away_lineup_ids, home_lineup_ids):
    away_pa, home_pa = count_team_pas_from_feed(snapshot["feed_json"])
    away_pos = away_pa % max(len(away_lineup_ids), 1)
    home_pos = home_pa % max(len(home_lineup_ids), 1)
    return away_pos, home_pos

def normalize_next_half_inning_state(snapshot):
    inning = snapshot.get("inning") or 1
    half = str(snapshot.get("inning_half") or "Top").lower()
    outs = snapshot.get("outs")
    away_score = int(snapshot.get("away_score") or 0)
    home_score = int(snapshot.get("home_score") or 0)

    start_inning = inning
    start_half = half

    if outs == 3:
        if half == "top":
            start_half = "bottom"
        else:
            start_half = "top"
            start_inning = inning + 1

    return {
        "start_inning": int(start_inning),
        "start_half": start_half,
        "away_score": away_score,
        "home_score": home_score
    }

def get_boxscore_pitcher_stats_for_team(snapshot, side="away"):
    team_box = snapshot["away_box"] if side == "away" else snapshot["home_box"]
    out = {}

    for _, p in team_box.get("players", {}).items():
        pid = p.get("person", {}).get("id")
        stats = p.get("stats", {}).get("pitching")

        if pid is None or stats is None:
            continue

        out[int(pid)] = {
            "player_name": p.get("person", {}).get("fullName"),
            "numberOfPitches": stats.get("numberOfPitches", 0),
            "outs": stats.get("outs", 0),
            "runs": stats.get("runs", 0),
            "isCurrentPitcher": p.get("gameStatus", {}).get("isCurrentPitcher", False)
        }

    return out

def get_current_or_fallback_pitcher_id(snapshot, side="away"):
    box = get_boxscore_pitcher_stats_for_team(snapshot, side)

    for pid, info in box.items():
        if info.get("isCurrentPitcher", False):
            return int(pid)

    fallback_name = snapshot.get("away_probable_pitcher") if side == "away" else snapshot.get("home_probable_pitcher")
    return probable_pitcher_id_from_name(fallback_name)

def get_used_pitchers_from_snapshot(snapshot, side="away"):
    return set(get_boxscore_pitcher_stats_for_team(snapshot, side).keys())

def init_live_pitcher_state(snapshot, pitcher_id, side="away"):
    box = get_boxscore_pitcher_stats_for_team(snapshot, side)
    info = box.get(int(pitcher_id), {})

    probable_name = snapshot.get("away_probable_pitcher") if side == "away" else snapshot.get("home_probable_pitcher")
    probable_id = probable_pitcher_id_from_name(probable_name)

    role = "starter" if probable_id == pitcher_id else "reliever"

    p_state = init_pitcher_state(pitcher_id, role=role)
    p_state["pitch_count"] = int(info.get("numberOfPitches", 0) or 0)
    p_state["outs_recorded"] = int(info.get("outs", 0) or 0)
    p_state["runs_allowed"] = int(info.get("runs", 0) or 0)

    return p_state

# In[27]:


# ============================================================
# CELL 7: pregame projection
# ============================================================

def run_pregame_projection(game_pk, n_sims=150):
    snapshot = get_game_snapshot(game_pk)
    lineup_info = resolve_game_lineups(snapshot)

    away_lineup_ids = lineup_df_to_ids(lineup_info["away_lineup"])
    home_lineup_ids = lineup_df_to_ids(lineup_info["home_lineup"])

    validate_lineup_ids(away_lineup_ids, "Away")
    validate_lineup_ids(home_lineup_ids, "Home")

    away_pitcher_id = probable_pitcher_id_from_name(snapshot.get("away_probable_pitcher"))
    home_pitcher_id = probable_pitcher_id_from_name(snapshot.get("home_probable_pitcher"))

    if away_pitcher_id is None:
        away_pitcher_id = get_current_or_fallback_pitcher_id(snapshot, side="away")
    if home_pitcher_id is None:
        home_pitcher_id = get_current_or_fallback_pitcher_id(snapshot, side="home")

    if away_pitcher_id is None or home_pitcher_id is None:
        raise ValueError("Could not resolve one or both pitchers for pregame projection.")

    result = simulate_matchup_with_pitching(
        away_lineup_ids=away_lineup_ids,
        home_lineup_ids=home_lineup_ids,
        away_pitcher_id=away_pitcher_id,
        home_pitcher_id=home_pitcher_id,
        away_team_code=snapshot["away_abbr"],
        home_team_code=snapshot["home_abbr"],
        n_sims=n_sims
    )

    away_proj = int(round(result["away_avg_runs"]))
    home_proj = int(round(result["home_avg_runs"]))

    print("=== PREGAME PROJECTED SCORE ===")
    print(f"{snapshot['away_abbr']} @ {snapshot['home_abbr']}")
    print(f"Projected score: {snapshot['away_abbr']} {away_proj} - {snapshot['home_abbr']} {home_proj}")
    print(f"Average runs: {snapshot['away_abbr']} {result['away_avg_runs']:.2f} | {snapshot['home_abbr']} {result['home_avg_runs']:.2f}")
    print(f"Win %: {snapshot['away_abbr']} {result['away_win_pct']*100:.1f}% | {snapshot['home_abbr']} {result['home_win_pct']*100:.1f}%")
    print(f"Away lineup source: {lineup_info['away_source']}")
    print(f"Home lineup source: {lineup_info['home_source']}")

    return {
        "snapshot": snapshot,
        "lineup_info": lineup_info,
        "result": result
    }

# In[28]:


# ============================================================
# CELL 8: live projection
# ============================================================

def simulate_remaining_game_from_state(
    away_lineup_ids,
    home_lineup_ids,
    away_team_code,
    home_team_code,
    away_pitcher_state,
    home_pitcher_state,
    away_used_pitchers,
    home_used_pitchers,
    away_score_start,
    home_score_start,
    start_inning,
    start_half,
    away_lineup_pos,
    home_lineup_pos,
    max_extra_innings=6
):
    away_runs = int(away_score_start)
    home_runs = int(home_score_start)

    away_bullpen = build_team_bullpen(away_team_code, away_pitcher_state["pitcher_id"])
    home_bullpen = build_team_bullpen(home_team_code, home_pitcher_state["pitcher_id"])

    inning = int(start_inning)
    half = str(start_half).lower()

    def play_top(i):
        nonlocal away_runs, away_lineup_pos, home_pitcher_state, home_used_pitchers
        r, away_lineup_pos, home_pitcher_state, home_used_pitchers = simulate_half_inning_with_pitching(
            lineup_ids=away_lineup_ids,
            lineup_pos=away_lineup_pos,
            pitching_team_code=home_team_code,
            batting_team_runs=away_runs,
            pitching_team_runs=home_runs,
            inning=i,
            half="top",
            current_pitcher_state=home_pitcher_state,
            bullpen_df=home_bullpen,
            used_pitchers=home_used_pitchers
        )
        away_runs += r

    def play_bottom(i):
        nonlocal home_runs, home_lineup_pos, away_pitcher_state, away_used_pitchers
        r, home_lineup_pos, away_pitcher_state, away_used_pitchers = simulate_half_inning_with_pitching(
            lineup_ids=home_lineup_ids,
            lineup_pos=home_lineup_pos,
            pitching_team_code=away_team_code,
            batting_team_runs=home_runs,
            pitching_team_runs=away_runs,
            inning=i,
            half="bottom",
            current_pitcher_state=away_pitcher_state,
            bullpen_df=away_bullpen,
            used_pitchers=away_used_pitchers
        )
        home_runs += r

    if half == "top":
        play_top(inning)
        play_bottom(inning)
        inning += 1
    else:
        play_bottom(inning)
        inning += 1

    while inning <= 9:
        play_top(inning)
        play_bottom(inning)
        inning += 1

    extras_used = 0
    while away_runs == home_runs and extras_used < max_extra_innings:
        play_top(inning)
        play_bottom(inning)
        inning += 1
        extras_used += 1

    return {
        "away_runs": away_runs,
        "home_runs": home_runs
    }

def run_live_projection(game_pk, n_sims=150):
    snapshot = get_game_snapshot(game_pk)
    lineup_info = resolve_game_lineups(snapshot)

    away_lineup_ids = lineup_df_to_ids(lineup_info["away_lineup"])
    home_lineup_ids = lineup_df_to_ids(lineup_info["home_lineup"])

    validate_lineup_ids(away_lineup_ids, "Away")
    validate_lineup_ids(home_lineup_ids, "Home")

    norm = normalize_next_half_inning_state(snapshot)
    away_lineup_pos, home_lineup_pos = next_lineup_positions_from_feed(
        snapshot,
        away_lineup_ids,
        home_lineup_ids
    )

    away_pitcher_id = get_current_or_fallback_pitcher_id(snapshot, side="away")
    home_pitcher_id = get_current_or_fallback_pitcher_id(snapshot, side="home")

    if away_pitcher_id is None or home_pitcher_id is None:
        raise ValueError("Could not resolve one or both current pitchers for live projection.")

    away_pitcher_state = init_live_pitcher_state(snapshot, away_pitcher_id, side="away")
    home_pitcher_state = init_live_pitcher_state(snapshot, home_pitcher_id, side="home")

    away_used_pitchers = get_used_pitchers_from_snapshot(snapshot, side="away")
    home_used_pitchers = get_used_pitchers_from_snapshot(snapshot, side="home")

    results = []

    for _ in range(n_sims):
        results.append(
            simulate_remaining_game_from_state(
                away_lineup_ids=away_lineup_ids,
                home_lineup_ids=home_lineup_ids,
                away_team_code=snapshot["away_abbr"],
                home_team_code=snapshot["home_abbr"],
                away_pitcher_state=away_pitcher_state.copy(),
                home_pitcher_state=home_pitcher_state.copy(),
                away_used_pitchers=set(away_used_pitchers),
                home_used_pitchers=set(home_used_pitchers),
                away_score_start=norm["away_score"],
                home_score_start=norm["home_score"],
                start_inning=norm["start_inning"],
                start_half=norm["start_half"],
                away_lineup_pos=away_lineup_pos,
                home_lineup_pos=home_lineup_pos
            )
        )

    df = pd.DataFrame(results)
    away_avg = df["away_runs"].mean()
    home_avg = df["home_runs"].mean()

    print("=== LIVE UPDATED FINAL SCORE ===")
    print(f"{snapshot['away_abbr']} @ {snapshot['home_abbr']}")
    print(f"Current score: {snapshot['away_abbr']} {snapshot.get('away_score')} - {snapshot['home_abbr']} {snapshot.get('home_score')}")
    print(f"Update point: inning {norm['start_inning']} {norm['start_half']}")
    print(f"Projected final: {snapshot['away_abbr']} {int(round(away_avg))} - {snapshot['home_abbr']} {int(round(home_avg))}")
    print(f"Average final runs: {snapshot['away_abbr']} {away_avg:.2f} | {snapshot['home_abbr']} {home_avg:.2f}")
    print(f"Win % from here: {snapshot['away_abbr']} {(df['away_runs'] > df['home_runs']).mean()*100:.1f}% | {snapshot['home_abbr']} {(df['home_runs'] > df['away_runs']).mean()*100:.1f}%")

    return {
        "snapshot": snapshot,
        "lineup_info": lineup_info,
        "result": {
            "results_df": df,
            "away_avg_runs": away_avg,
            "home_avg_runs": home_avg
        }
    }

# In[29]:


# ============================================================
# CELL 9: selectors
# ============================================================

def launch_pregame_projection_selector(game_date=None, default_sims=150):
    games_df = get_daily_games(game_date)

    dropdown = widgets.Dropdown(
        options=[(str(r["label"]), int(r["gamePk"])) for _, r in games_df.iterrows()],
        description="Pregame:",
        layout=widgets.Layout(width="900px")
    )

    sims_box = widgets.IntText(
        value=default_sims,
        description="Sims:",
        layout=widgets.Layout(width="180px")
    )

    btn = widgets.Button(description="Run Pregame Projection", button_style="primary")
    out = widgets.Output()

    def on_click(_):
        with out:
            clear_output()
            try:
                run_pregame_projection(dropdown.value, n_sims=int(sims_box.value))
            except Exception as e:
                print("Pregame projection failed:")
                print(e)

    btn.on_click(on_click)
    display(widgets.VBox([widgets.HBox([dropdown, sims_box]), btn, out]))

def launch_live_projection_selector(game_date=None, default_sims=150):
    games_df = get_daily_games(game_date)

    dropdown = widgets.Dropdown(
        options=[(str(r["label"]), int(r["gamePk"])) for _, r in games_df.iterrows()],
        description="Live:",
        layout=widgets.Layout(width="900px")
    )

    sims_box = widgets.IntText(
        value=default_sims,
        description="Sims:",
        layout=widgets.Layout(width="180px")
    )

    btn = widgets.Button(description="Run Live Projection", button_style="success")
    out = widgets.Output()

    def on_click(_):
        with out:
            clear_output()
            try:
                run_live_projection(dropdown.value, n_sims=int(sims_box.value))
            except Exception as e:
                print("Live projection failed:")
                print(e)

    btn.on_click(on_click)
    display(widgets.VBox([widgets.HBox([dropdown, sims_box]), btn, out]))

# In[30]:


# ============================================================
# CELL 10: launch selectors
# ============================================================

launch_pregame_projection_selector()
launch_live_projection_selector()
