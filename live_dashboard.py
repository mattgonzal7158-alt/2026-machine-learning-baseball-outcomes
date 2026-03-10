import time
import json
import importlib.util
from pathlib import Path
from datetime import datetime

import requests
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from catboost import CatBoostClassifier

# ============================================================
# Live MLB Matchup Odds Dashboard
# Includes:
# - Scorebug with matchup inside
# - Live PA outcome / ball type snapshot
# - Compact game simulator box on left:
#   - Pregame win %
#   - Pregame projected score
#   - Live win %
#   - Live projected final score
#
# IMPORTANT:
# This expects:
#   final_game_simulator.py
# in the same folder as this file.
# ============================================================

st.set_page_config(page_title="Live MLB Matchup Odds", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.0rem; padding-bottom: 2rem;}
    h1, h2, h3 {letter-spacing: -0.2px;}

    [data-testid="stMetricLabel"] {font-size: 0.9rem;}
    [data-testid="stMetricValue"] {font-size: 1.55rem;}

    .stDataFrame {border-radius: 12px; overflow: hidden;}

    .pill {
      display:inline-block;
      padding:4px 10px;
      border-radius:999px;
      border:1px solid rgba(255,255,255,0.14);
      background: rgba(20,22,30,0.55);
      font-size: 12px;
      opacity: 0.98;
      color: rgba(255,255,255,0.95);
    }

    .probrow {
      display:flex;
      align-items:center;
      justify-content:space-between;
      gap:12px;
      padding:10px 12px;
      border-radius:12px;
      border:1px solid rgba(255,255,255,0.08);
      background: rgba(20,22,30,0.35);
      margin-bottom:8px;
    }
    .barwrap {
      flex:1;
      height:10px;
      border-radius:999px;
      background: rgba(255,255,255,0.08);
      overflow:hidden;
    }
    .barfill {
      height:100%;
      border-radius:999px;
      background: rgba(250,204,21,0.85);
      width: 0%;
    }
    .label {min-width:120px; font-weight:700;}
    .pct {min-width:78px; text-align:right; font-weight:800;}

    .sim-wrap{
        margin-top:10px;
        padding:12px;
        border-radius:16px;
        border:1px solid rgba(255,255,255,0.10);
        background:rgba(20,22,30,0.55);
    }
    .sim-title{
        font-size:13px;
        font-weight:800;
        color:rgba(255,255,255,0.96);
        margin-bottom:8px;
    }
    .sim-sub{
        font-size:11px;
        color:rgba(255,255,255,0.70);
        margin-bottom:10px;
    }
    .sim-grid{
        display:grid;
        grid-template-columns:1fr 1fr;
        gap:10px;
    }
    .sim-card{
        border-radius:14px;
        padding:10px 11px;
        border:1px solid rgba(255,255,255,0.08);
        background:rgba(0,0,0,0.14);
    }
    .sim-label{
        font-size:11px;
        color:rgba(255,255,255,0.65);
        margin-bottom:6px;
        text-transform:uppercase;
        letter-spacing:0.2px;
    }
    .sim-value{
        font-size:18px;
        font-weight:900;
        color:rgba(255,255,255,0.97);
        line-height:1.1;
    }
    .sim-value-sm{
        font-size:14px;
        font-weight:800;
        color:rgba(255,255,255,0.97);
        line-height:1.2;
    }
    .sim-foot{
        margin-top:8px;
        font-size:11px;
        color:rgba(255,255,255,0.62);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("⚾ Live MLB Matchup Odds (Your Model) — Upgraded")


# ----------------------------
# Config
# ----------------------------
MODEL_PA_PATH = "models/model_pa.cbm"
MODEL_BT_PATH = "models/model_balltype.cbm"
DF_PATH = "models/model_df.parquet"
META_PATH = "models/model_meta.json"

SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule?sportId=1"
LIVE_URL_TMPL = "https://statsapi.mlb.com/api/v1.1/game/{gamePk}/feed/live"

SCHEDULE_REFRESH_SECONDS = 600
GAME_SIM_MODULE_PATH = "final_game_simulator.py"


# ----------------------------
# Cached loaders
# ----------------------------
@st.cache_resource
def load_models():
    m1 = CatBoostClassifier()
    m1.load_model(MODEL_PA_PATH)
    m2 = CatBoostClassifier()
    m2.load_model(MODEL_BT_PATH)
    return m1, m2

@st.cache_data
def load_meta():
    with open(META_PATH, "r") as f:
        return json.load(f)

@st.cache_data
def load_df():
    return pd.read_parquet(DF_PATH)

model_pa, model_balltype = load_models()
meta = load_meta()
df = load_df()

features = meta["features"]
cat_cols = meta["cat_cols"]

missing_feats = [c for c in features if c not in df.columns]
if missing_feats:
    st.error(f"model_df.parquet is missing required feature columns: {missing_feats}")
    st.stop()


# ----------------------------
# Helpers
# ----------------------------
def safe_get(dct, keys, default=None):
    cur = dct
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

def current_base_state(offense_obj):
    r1 = 1 if offense_obj.get("first") is not None else 0
    r2 = 1 if offense_obj.get("second") is not None else 0
    r3 = 1 if offense_obj.get("third") is not None else 0
    return r1, r2, r3

def runners_text(r1, r2, r3):
    bases = []
    if int(r1):
        bases.append("first")
    if int(r2):
        bases.append("second")
    if int(r3):
        bases.append("third")
    if not bases:
        return "Bases empty"
    if len(bases) == 1:
        return f"Runner on {bases[0]}"
    if len(bases) == 2:
        return f"Runners on {bases[0]} and {bases[1]}"
    return "Bases loaded"

def lead_text(away_team, home_team, away_runs, home_runs):
    away_runs = int(away_runs)
    home_runs = int(home_runs)
    if away_runs == home_runs:
        return f"Tied {away_runs}–{home_runs}"
    if away_runs > home_runs:
        return f"{away_team} lead by {away_runs - home_runs} ({away_runs}–{home_runs})"
    return f"{home_team} lead by {home_runs - away_runs} ({home_runs}–{away_runs})"

def _mode(series):
    vc = series.value_counts(dropna=True)
    return vc.index[0] if len(vc) else None

def get_now_et():
    return pd.Timestamp.now(tz="America/New_York")

def get_today_et_str():
    return get_now_et().strftime("%Y-%m-%d")

def status_priority(status: str) -> int:
    s = str(status).lower()
    if any(x in s for x in ["in progress", "manager challenge", "review", "delayed", "suspended"]):
        return 0
    if any(x in s for x in ["pre-game", "scheduled", "warmup", "pregame"]):
        return 1
    if any(x in s for x in ["final", "game over", "completed"]):
        return 3
    return 2

def build_matchup_label(row):
    return f"{row['away']} @ {row['home']} ({row['status']})"

def scorebug_html(
    away_team, home_team, away_runs, home_runs,
    inning, is_top, balls, strikes, outs,
    r1, r2, r3, status,
    pitcher_name, pitcher_hand,
    batter_name, batter_hand
):
    text = "rgba(255,255,255,0.95)"
    sub = "rgba(255,255,255,0.70)"
    border = "rgba(255,255,255,0.10)"
    bg = "rgba(20,22,30,0.80)"

    half = "TOP" if is_top else "BOT"
    cnt = f"{balls}-{strikes}"

    on = "#facc15"
    off = "#2a2f3a"
    b1 = on if int(r1) else off
    b2 = on if int(r2) else off
    b3 = on if int(r3) else off

    o_on = "#f97316"
    o_off = "#2a2f3a"
    o1 = o_on if outs >= 1 else o_off
    o2 = o_on if outs >= 2 else o_off

    status_txt = status or ""
    status_badge = f"""
      <span style="
        padding:4px 10px; border-radius:999px;
        border:1px solid {border};
        background: rgba(0,0,0,0.18);
        font-size:12px; color:{text};">
        {status_txt}
      </span>
    """ if status_txt else ""

    p_hand = pitcher_hand if pitcher_hand in ["L", "R"] else "?"
    b_hand = batter_hand if batter_hand in ["L", "R"] else "?"

    return f"""
    <div style="
        display:flex; align-items:center; justify-content:space-between;
        padding:12px 14px; border-radius:16px;
        background:{bg};
        border: 1px solid {border};
        backdrop-filter: blur(6px);
        margin: 6px 0 10px 0;
        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial;
        color:{text};
        gap:14px;
    ">

      <div style="min-width:260px;">
        <div style="display:flex; align-items:center; gap:10px;">
          <div style="font-size:12px; color:{sub};">SCORE</div>
          {status_badge}
        </div>
        <div style="display:flex; flex-direction:column; gap:6px; margin-top:8px;">
          <div style="display:flex; justify-content:space-between; gap:14px;">
            <div style="font-weight:800; color:{text};">{away_team}</div>
            <div style="font-weight:900; color:{text};">{away_runs}</div>
          </div>
          <div style="display:flex; justify-content:space-between; gap:14px;">
            <div style="font-weight:800; color:{text};">{home_team}</div>
            <div style="font-weight:900; color:{text};">{home_runs}</div>
          </div>
        </div>
      </div>

      <div style="
        flex:1;
        padding:10px 12px;
        border-radius:14px;
        border:1px solid {border};
        background: rgba(0,0,0,0.12);
        min-width: 280px;
      ">
        <div style="font-size:12px; color:{sub}; margin-bottom:6px;">MATCHUP</div>
        <div style="display:flex; align-items:center; justify-content:space-between; gap:14px;">
          <div style="flex:1;">
            <div style="font-weight:900; font-size:14px; color:{text}; line-height:1.15;">
              {pitcher_name}
            </div>
            <div style="font-size:12px; color:{sub};">Pitcher · throws {p_hand}</div>
          </div>
          <div style="
            font-weight:900;
            font-size:12px;
            padding:6px 10px;
            border-radius:999px;
            border:1px solid {border};
            background: rgba(0,0,0,0.18);
            color:{text};
          ">VS</div>
          <div style="flex:1; text-align:right;">
            <div style="font-weight:900; font-size:14px; color:{text}; line-height:1.15;">
              {batter_name}
            </div>
            <div style="font-size:12px; color:{sub};">Batter · bats {b_hand}</div>
          </div>
        </div>
      </div>

      <div style="display:flex; align-items:center; gap:18px;">
        <div style="text-align:right;">
          <div style="font-size:12px; color:{sub};">INN · COUNT · OUTS</div>
          <div style="margin-top:8px; display:flex; align-items:center; gap:12px; justify-content:flex-end;">
            <div style="font-size:14px; font-weight:900; color:{text};">{half} {inning}</div>
            <div style="font-size:14px; font-weight:900; color:{text};">{cnt}</div>
            <div style="display:flex; gap:6px; align-items:center;">
              <div style="width:10px; height:10px; border-radius:999px; background:{o1};"></div>
              <div style="width:10px; height:10px; border-radius:999px; background:{o2};"></div>
            </div>
          </div>
        </div>

        <div style="display:flex; flex-direction:column; align-items:flex-end; gap:8px;">
          <div style="font-size:12px; color:{sub};">BASES</div>
          <div style="position:relative; width:72px; height:56px;">
            <div style="position:absolute; left:28px; top:4px; width:16px; height:16px; transform: rotate(45deg);
                        background:{b2}; border:1px solid rgba(255,255,255,0.22); border-radius:3px;"></div>
            <div style="position:absolute; left:8px; top:24px; width:16px; height:16px; transform: rotate(45deg);
                        background:{b3}; border:1px solid rgba(255,255,255,0.22); border-radius:3px;"></div>
            <div style="position:absolute; left:48px; top:24px; width:16px; height:16px; transform: rotate(45deg);
                        background:{b1}; border:1px solid rgba(255,255,255,0.22); border-radius:3px;"></div>
            <div style="position:absolute; left:28px; top:44px; width:16px; height:12px;
                        background:#111827; border:1px solid rgba(255,255,255,0.18);
                        clip-path: polygon(50% 0%, 100% 35%, 100% 100%, 0% 100%, 0% 35%); border-radius:2px;"></div>
          </div>
        </div>
      </div>
    </div>
    """

def approx_rbi_prob(outcome_probs_pct, r1, r2, r3, outs_when_up):
    runners = int(r1) + int(r2) + int(r3)
    p = {k: v / 100.0 for k, v in outcome_probs_pct.items()}
    if runners == 0:
        return round(float(outcome_probs_pct.get("hr", 0.0)), 2)

    pr_hr = p.get("hr", 0.0)
    pr_walk_rbi = p.get("walk", 0.0) * (1.0 if (r1 and r2 and r3) else 0.0)
    pr_triple_rbi = p.get("triple", 0.0) * 0.95

    if r2 or r3:
        dbl_rate = 0.85
    elif r1:
        dbl_rate = 0.45
    else:
        dbl_rate = 0.0
    pr_double_rbi = p.get("double", 0.0) * dbl_rate

    if r3:
        sng_rate = 0.80
    elif r2:
        sng_rate = 0.35
    elif r1:
        sng_rate = 0.05
    else:
        sng_rate = 0.0
    pr_single_rbi = p.get("single", 0.0) * sng_rate

    if outs_when_up >= 2:
        out_rate = 0.0
    else:
        if r3:
            out_rate = 0.07 if outs_when_up == 0 else 0.10
        else:
            out_rate = 0.01 if outs_when_up == 0 else 0.015
    pr_out_rbi = p.get("out", 0.0) * out_rate

    pr_rbi = pr_hr + pr_walk_rbi + pr_triple_rbi + pr_double_rbi + pr_single_rbi + pr_out_rbi
    return round(min(pr_rbi, 1.0) * 100.0, 2)

def get_context_pool(pitcher_id, balls, strikes, base_state, outs_when_up):
    pool = df[
        (df["pitcher"] == pitcher_id) &
        (df["balls"] == balls) &
        (df["strikes"] == strikes) &
        (df["base_state"] == base_state) &
        (df["outs_when_up"] == outs_when_up)
    ]
    if len(pool) >= 80:
        return pool, "exact (count+base+outs)"
    pool = df[
        (df["pitcher"] == pitcher_id) &
        (df["balls"] == balls) &
        (df["strikes"] == strikes) &
        (df["base_state"] == base_state)
    ]
    if len(pool) >= 80:
        return pool, "backoff (count+base)"
    pool = df[
        (df["pitcher"] == pitcher_id) &
        (df["balls"] == balls) &
        (df["strikes"] == strikes)
    ]
    if len(pool) >= 80:
        return pool, "backoff (count)"
    return df[df["pitcher"] == pitcher_id], "backoff (pitcher only)"

def summarize_pitch_guess_mix(pool: pd.DataFrame, top_n=6):
    if "pitch_guess" not in pool.columns or len(pool) == 0:
        return pd.DataFrame(columns=["Pitch", "Pct"])
    vc = pool["pitch_guess"].value_counts(normalize=True).head(top_n) * 100
    out = vc.reset_index()
    out.columns = ["Pitch", "Pct"]
    out["Pct"] = out["Pct"].round(1)
    return out

def top2_pitch_blend(pool: pd.DataFrame):
    if "pitch_guess" not in pool.columns or len(pool) == 0:
        return [("UNK", 1.0)]
    vc = pool["pitch_guess"].value_counts(normalize=True)
    if len(vc) == 0:
        return [("UNK", 1.0)]
    top = vc.head(2)
    pitches = top.index.tolist()
    weights = top.values.tolist()
    s = float(sum(weights)) if sum(weights) > 0 else 1.0
    weights = [float(w) / s for w in weights]
    return list(zip(pitches, weights))

def get_recent_form_snapshot(batter_id, pitcher_id):
    b = df[df["batter"] == batter_id]
    p = df[df["pitcher"] == pitcher_id]

    def last_val(series, default=0.0):
        try:
            if series is None or len(series) == 0:
                return default
            v = series.dropna()
            return float(v.iloc[-1]) if len(v) else default
        except Exception:
            return default

    snap = {
        "p_recent_k_rate": last_val(p.get("p_recent_k_rate", pd.Series([], dtype=float))),
        "p_recent_bb_rate": last_val(p.get("p_recent_bb_rate", pd.Series([], dtype=float))),
        "p_recent_hr_rate": last_val(p.get("p_recent_hr_rate", pd.Series([], dtype=float))),
        "b_recent_k_rate": last_val(b.get("b_recent_k_rate", pd.Series([], dtype=float))),
        "b_recent_bb_rate": last_val(b.get("b_recent_bb_rate", pd.Series([], dtype=float))),
        "b_recent_hr_rate": last_val(b.get("b_recent_hr_rate", pd.Series([], dtype=float))),
    }
    for k in snap:
        snap[k] = round(snap[k] * 100.0, 1)
    return snap

def top_label_pct(probs_dict):
    if not probs_dict:
        return ("NA", 0.0)
    k = max(probs_dict, key=lambda x: probs_dict[x])
    return (k, float(probs_dict[k]))

def fmt_pitch_blend(blend):
    if not blend:
        return "UNK"
    if len(blend) == 1:
        p, w = blend[0]
        return f"{p} ({int(round(w * 100))}%)"
    p1, w1 = blend[0]
    p2, w2 = blend[1]
    return f"{p1} ({int(round(w1 * 100))}%) + {p2} ({int(round(w2 * 100))}%)"

def render_top_prob_rows(title, probs_dict, top_n=5):
    st.markdown(f"### {title}")
    if not probs_dict:
        st.write("No probabilities.")
        return
    items = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    for label, pct in items:
        width = max(0.0, min(100.0, float(pct)))
        st.markdown(
            f"""
            <div class="probrow">
              <div class="label">{label}</div>
              <div class="barwrap"><div class="barfill" style="width:{width}%;"></div></div>
              <div class="pct">{pct:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ----------------------------
# Final game simulator bridge
# ----------------------------
@st.cache_resource
def load_game_simulator_module():
    path = Path(GAME_SIM_MODULE_PATH)
    if not path.exists():
        return None

    spec = importlib.util.spec_from_file_location("final_game_simulator_module", str(path))
    if spec is None or spec.loader is None:
        return None

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def compute_live_win_pcts_from_results_df(results_df):
    if results_df is None or len(results_df) == 0:
        return None, None

    away_win = float((results_df["away_runs"] > results_df["home_runs"]).mean() * 100.0)
    home_win = float((results_df["home_runs"] > results_df["away_runs"]).mean() * 100.0)

    tie_pct = float((results_df["away_runs"] == results_df["home_runs"]).mean() * 100.0)
    if tie_pct > 0:
        away_win += tie_pct / 2.0
        home_win += tie_pct / 2.0

    return away_win, home_win

@st.cache_data(ttl=60)
def get_game_simulator_panel_data(game_pk: int, n_sims: int = 150):
    mod = load_game_simulator_module()
    if mod is None:
        return {
            "loaded": False,
            "error": "final_game_simulator.py not found",
            "pregame": None,
            "live": None
        }

    if not hasattr(mod, "run_pregame_projection") or not hasattr(mod, "run_live_projection"):
        return {
            "loaded": False,
            "error": "run_pregame_projection / run_live_projection not found in final_game_simulator.py",
            "pregame": None,
            "live": None
        }

    out = {
        "loaded": True,
        "error": None,
        "pregame": None,
        "live": None
    }

    try:
        pre = mod.run_pregame_projection(game_pk, n_sims=n_sims)
        pre_result = pre.get("result", {}) if isinstance(pre, dict) else {}
        pre_snap = pre.get("snapshot", {}) if isinstance(pre, dict) else {}

        away_wp = safe_float(pre_result.get("away_win_pct"))
        home_wp = safe_float(pre_result.get("home_win_pct"))

        out["pregame"] = {
            "away_abbr": pre_snap.get("away_abbr"),
            "home_abbr": pre_snap.get("home_abbr"),
            "away_score": safe_float(pre_result.get("away_avg_runs")),
            "home_score": safe_float(pre_result.get("home_avg_runs")),
            "away_win_pct": away_wp * 100.0 if away_wp is not None and away_wp <= 1.0 else away_wp,
            "home_win_pct": home_wp * 100.0 if home_wp is not None and home_wp <= 1.0 else home_wp,
        }
    except Exception as e:
        out["pregame"] = {"error": str(e)}

    try:
        live_proj = mod.run_live_projection(game_pk, n_sims=n_sims)
        live_result = live_proj.get("result", {}) if isinstance(live_proj, dict) else {}
        live_snap = live_proj.get("snapshot", {}) if isinstance(live_proj, dict) else {}

        results_df = live_result.get("results_df")
        away_live_win, home_live_win = compute_live_win_pcts_from_results_df(results_df)

        out["live"] = {
            "away_abbr": live_snap.get("away_abbr"),
            "home_abbr": live_snap.get("home_abbr"),
            "away_score": safe_float(live_result.get("away_avg_runs")),
            "home_score": safe_float(live_result.get("home_avg_runs")),
            "away_win_pct": away_live_win,
            "home_win_pct": home_live_win,
            "current_away_score": live_snap.get("away_score"),
            "current_home_score": live_snap.get("home_score"),
        }
    except Exception as e:
        out["live"] = {"error": str(e)}

    return out

def fmt_sim_pct(x):
    if x is None:
        return "—"
    return f"{float(x):.1f}%"

def fmt_sim_score(a, h):
    if a is None or h is None:
        return "—"
    return f"{float(a):.1f} - {float(h):.1f}"

def sim_leader_text(away_label, home_label, away_pct, home_pct):
    if away_pct is None or home_pct is None:
        return "—"
    if away_pct >= home_pct:
        return f"{away_label} favored"
    return f"{home_label} favored"

def render_simulator_box(sim_data, away_team, home_team):
    if not sim_data.get("loaded", False):
        st.markdown(
            f"""
            <div class="sim-wrap">
              <div class="sim-title">📊 Game Simulator</div>
              <div class="sim-sub">{away_team} @ {home_team}</div>
              <div class="sim-foot">{sim_data.get("error", "Simulator not loaded")}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        return

    pre = sim_data.get("pregame", {}) or {}
    livep = sim_data.get("live", {}) or {}

    pre_away_abbr = pre.get("away_abbr") or "AWAY"
    pre_home_abbr = pre.get("home_abbr") or "HOME"
    live_away_abbr = livep.get("away_abbr") or "AWAY"
    live_home_abbr = livep.get("home_abbr") or "HOME"

    pre_best_pct = None
    if pre.get("away_win_pct") is not None and pre.get("home_win_pct") is not None:
        pre_best_pct = max(pre["away_win_pct"], pre["home_win_pct"])

    live_best_pct = None
    if livep.get("away_win_pct") is not None and livep.get("home_win_pct") is not None:
        live_best_pct = max(livep["away_win_pct"], livep["home_win_pct"])

    pre_leader = sim_leader_text(
        pre_away_abbr,
        pre_home_abbr,
        pre.get("away_win_pct"),
        pre.get("home_win_pct")
    )

    live_leader = sim_leader_text(
        live_away_abbr,
        live_home_abbr,
        livep.get("away_win_pct"),
        livep.get("home_win_pct")
    )

    pre_score = fmt_sim_score(pre.get("away_score"), pre.get("home_score"))
    live_score = fmt_sim_score(livep.get("away_score"), livep.get("home_score"))

    pre_pct = fmt_sim_pct(pre_best_pct)
    live_pct = fmt_sim_pct(live_best_pct)

    html = f"""
    <div class="sim-wrap">
      <div class="sim-title">📊 Game Simulator</div>
      <div class="sim-sub">{away_team} @ {home_team}</div>

      <div class="sim-grid">
        <div class="sim-card">
          <div class="sim-label">Pregame Win%</div>
          <div class="sim-value">{pre_pct}</div>
          <div class="sim-foot">{pre_leader}</div>
        </div>

        <div class="sim-card">
          <div class="sim-label">Pregame Score</div>
          <div class="sim-value-sm">{pre_score}</div>
          <div class="sim-foot">{pre_away_abbr} - {pre_home_abbr}</div>
        </div>

        <div class="sim-card">
          <div class="sim-label">Live Win%</div>
          <div class="sim-value">{live_pct}</div>
          <div class="sim-foot">{live_leader}</div>
        </div>

        <div class="sim-card">
          <div class="sim-label">Live Final Score</div>
          <div class="sim-value-sm">{live_score}</div>
          <div class="sim-foot">updates during game</div>
        </div>
      </div>
    </div>
    """

    st.markdown(html, unsafe_allow_html=True)


# ----------------------------
# Model simulation (pitch-blend)
# ----------------------------
def build_situation_samples(
    batter_id,
    pitcher_id,
    balls,
    strikes,
    outs_when_up,
    inning,
    runner_1b,
    runner_2b,
    runner_3b,
    run_diff,
    n=1000
):
    base_state = f"{int(runner_1b)}{int(runner_2b)}{int(runner_3b)}"
    count_id = f"{int(balls)}-{int(strikes)}"

    ctx_pool, ctx_level = get_context_pool(pitcher_id, balls, strikes, base_state, outs_when_up)
    mix_table = summarize_pitch_guess_mix(ctx_pool, top_n=6)
    blend = top2_pitch_blend(ctx_pool)

    pool = df[(df["pitcher"] == pitcher_id) & (df["balls"] == balls) & (df["strikes"] == strikes)]
    if len(pool) < 300:
        pool = df[df["pitcher"] == pitcher_id]
    if len(pool) < 300:
        raise ValueError("Not enough historical samples for this pitcher in model_df.parquet")

    samp = pool.sample(n=min(n, len(pool)), replace=(len(pool) < n), random_state=42).copy()

    samp["batter"] = batter_id
    samp["pitcher"] = pitcher_id
    samp["balls"] = balls
    samp["strikes"] = strikes
    samp["outs_when_up"] = outs_when_up
    samp["inning"] = inning
    samp["runner_1b"] = runner_1b
    samp["runner_2b"] = runner_2b
    samp["runner_3b"] = runner_3b
    samp["run_diff"] = run_diff

    if "count_id" in samp.columns:
        samp["count_id"] = count_id
    if "base_state" in samp.columns:
        samp["base_state"] = base_state

    b_hist = df[df["batter"] == batter_id]
    p_hist = df[df["pitcher"] == pitcher_id]
    b_stand = _mode(b_hist["stand"]) if len(b_hist) else None
    p_throw = _mode(p_hist["p_throws"]) if len(p_hist) else None
    if b_stand is not None and "stand" in samp.columns:
        samp["stand"] = b_stand
    if p_throw is not None and "p_throws" in samp.columns:
        samp["p_throws"] = p_throw

    samp = samp.fillna(0)
    for c in cat_cols:
        if c in samp.columns:
            samp[c] = samp[c].astype("string").fillna("NA")

    return samp[features], blend, ctx_level, mix_table, b_stand, p_throw, base_state, count_id

def predict_live_state_blended(
    batter_id,
    pitcher_id,
    balls,
    strikes,
    outs_when_up,
    inning,
    runner_1b,
    runner_2b,
    runner_3b,
    run_diff,
    n_sims=1000
):
    Xbase, blend, ctx_level, mix_table, b_stand, p_throw, base_state, count_id = build_situation_samples(
        batter_id=batter_id,
        pitcher_id=pitcher_id,
        balls=balls,
        strikes=strikes,
        outs_when_up=outs_when_up,
        inning=inning,
        runner_1b=runner_1b,
        runner_2b=runner_2b,
        runner_3b=runner_3b,
        run_diff=run_diff,
        n=n_sims
    )

    pa_accum = None
    bt_accum = None

    for pitch_guess, w in blend:
        Xsim = Xbase.copy()
        if "pitch_guess" in Xsim.columns:
            Xsim["pitch_guess"] = str(pitch_guess)
        for c in cat_cols:
            if c in Xsim.columns:
                Xsim[c] = Xsim[c].astype("string").fillna("NA")
        pa_probs = model_pa.predict_proba(Xsim).mean(axis=0)
        bt_probs = model_balltype.predict_proba(Xsim).mean(axis=0)
        if pa_accum is None:
            pa_accum = w * pa_probs
            bt_accum = w * bt_probs
        else:
            pa_accum += w * pa_probs
            bt_accum += w * bt_probs

    out_probs = pd.Series(pa_accum, index=model_pa.classes_).sort_values(ascending=False)
    out_probs_pct = (out_probs * 100).round(2).to_dict()

    bt_probs = pd.Series(bt_accum, index=model_balltype.classes_).sort_values(ascending=False)
    bt_probs_pct = (bt_probs * 100).round(2).to_dict()

    rbi_pct = approx_rbi_prob(out_probs_pct, runner_1b, runner_2b, runner_3b, outs_when_up)
    return out_probs_pct, bt_probs_pct, rbi_pct, blend, ctx_level, mix_table, b_stand, p_throw, base_state, count_id


# ----------------------------
# API
# ----------------------------
@st.cache_data(ttl=300)
def get_schedule_for_date(date_str: str):
    r = requests.get(f"{SCHEDULE_URL}&date={date_str}", timeout=20)
    r.raise_for_status()
    data = r.json()

    games = []
    for date_block in data.get("dates", []):
        for g in date_block.get("games", []):
            games.append({
                "gamePk": int(g["gamePk"]),
                "away": g["teams"]["away"]["team"]["name"],
                "home": g["teams"]["home"]["team"]["name"],
                "status": g["status"]["detailedState"],
                "gameDate": g.get("gameDate", ""),
            })

    out = pd.DataFrame(games)
    if out.empty:
        return out

    out["status_rank"] = out["status"].apply(status_priority)
    out["matchup"] = out.apply(build_matchup_label, axis=1)
    out = out.sort_values(
        by=["status_rank", "gameDate", "away", "home"],
        ascending=[True, True, True, True]
    ).reset_index(drop=True)
    return out

def refresh_schedule_if_needed(force=False):
    today_et = get_today_et_str()
    now_ts = time.time()

    if "schedule_date_et" not in st.session_state:
        st.session_state.schedule_date_et = ""
    if "schedule_last_loaded_ts" not in st.session_state:
        st.session_state.schedule_last_loaded_ts = 0.0
    if "sched_df" not in st.session_state:
        st.session_state.sched_df = pd.DataFrame()
    if "selected_gamePk" not in st.session_state:
        st.session_state.selected_gamePk = None

    needs_new_day = st.session_state.schedule_date_et != today_et
    stale_intraday = (now_ts - float(st.session_state.schedule_last_loaded_ts)) >= SCHEDULE_REFRESH_SECONDS
    empty_sched = st.session_state.sched_df.empty

    if force:
        get_schedule_for_date.clear()

    if force or needs_new_day or stale_intraday or empty_sched:
        sched = get_schedule_for_date(today_et)
        st.session_state.sched_df = sched.copy()
        st.session_state.schedule_date_et = today_et
        st.session_state.schedule_last_loaded_ts = now_ts

        if not sched.empty:
            valid_ids = set(sched["gamePk"].astype(int).tolist())
            if st.session_state.selected_gamePk not in valid_ids:
                st.session_state.selected_gamePk = int(sched.iloc[0]["gamePk"])

    return st.session_state.sched_df.copy()

def get_live_game(gamePk: int):
    url = LIVE_URL_TMPL.format(gamePk=gamePk)
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.json()

def extract_state(live_json):
    count = safe_get(live_json, ["liveData", "plays", "currentPlay", "count"], {}) or {}
    balls = int(count.get("balls", 0))
    strikes = int(count.get("strikes", 0))

    linescore = safe_get(live_json, ["liveData", "linescore"], {}) or {}
    inning = int(linescore.get("currentInning", 1))
    outs = int(linescore.get("outs", 0))

    offense = linescore.get("offense", {}) if isinstance(linescore, dict) else {}
    r1, r2, r3 = current_base_state(offense)

    matchup = safe_get(live_json, ["liveData", "plays", "currentPlay", "matchup"], {}) or {}
    batter = matchup.get("batter", {}) if isinstance(matchup, dict) else {}
    pitcher = matchup.get("pitcher", {}) if isinstance(matchup, dict) else {}

    batter_name = batter.get("fullName", "Unknown")
    pitcher_name = pitcher.get("fullName", "Unknown")
    batter_id = batter.get("id", None)
    pitcher_id = pitcher.get("id", None)

    about = safe_get(live_json, ["liveData", "plays", "currentPlay", "about"], {}) or {}
    is_top = bool(about.get("isTopInning", True))

    away_runs = int(safe_get(linescore, ["teams", "away", "runs"], 0) or 0)
    home_runs = int(safe_get(linescore, ["teams", "home", "runs"], 0) or 0)

    balls = max(0, min(3, balls))
    strikes = max(0, min(2, strikes))
    outs = max(0, min(2, outs))

    return {
        "batter_name": batter_name,
        "pitcher_name": pitcher_name,
        "batter_id": batter_id,
        "pitcher_id": pitcher_id,
        "balls": balls,
        "strikes": strikes,
        "outs": outs,
        "inning": inning,
        "r1": r1,
        "r2": r2,
        "r3": r3,
        "away_runs": away_runs,
        "home_runs": home_runs,
        "is_top": is_top
    }


# ----------------------------
# UI
# ----------------------------
left, right = st.columns([1, 2])

with left:
    st.subheader("Controls")
    refresh_sec = st.number_input("Auto-refresh seconds", min_value=2, max_value=20, value=5, step=1)
    auto = st.checkbox("Auto refresh", value=True)
    n_sims = st.slider("Simulation samples (n_sims)", min_value=200, max_value=5000, value=1000, step=200)
    st.caption("Higher n_sims = smoother probs, slower refresh.")

    st.divider()
    st.subheader("Today's Games")

    manual = st.button("Refresh now")
    sched = refresh_schedule_if_needed(force=manual)

    if sched.empty:
        st.warning("No games returned from schedule API for today's ET slate.")
        st.stop()

    current_game_ids = sched["gamePk"].astype(int).tolist()
    if st.session_state.selected_gamePk not in current_game_ids:
        st.session_state.selected_gamePk = int(sched.iloc[0]["gamePk"])

    selected_index = current_game_ids.index(int(st.session_state.selected_gamePk))

    game_choice = st.selectbox(
        "Pick a game",
        options=current_game_ids,
        index=selected_index,
        format_func=lambda gpk: sched.loc[sched["gamePk"] == gpk, "matchup"].iloc[0]
    )
    st.session_state.selected_gamePk = int(game_choice)

    row = sched.loc[sched["gamePk"] == st.session_state.selected_gamePk].iloc[0]
    gamePk = int(row["gamePk"])
    away_team = str(row["away"])
    home_team = str(row["home"])
    status = str(row["status"])

    schedule_loaded_at = datetime.fromtimestamp(st.session_state.schedule_last_loaded_ts).strftime("%I:%M:%S %p")
    st.caption(
        f"Schedule date: {st.session_state.schedule_date_et} ET · "
        f"last schedule refresh: {schedule_loaded_at} · "
        f"auto schedule refresh: every {SCHEDULE_REFRESH_SECONDS // 60} min or when ET date changes"
    )

    st.divider()

    sim_n_sims = st.slider(
        "Game simulator sims",
        min_value=50,
        max_value=500,
        value=150,
        step=25
    )

    sim_data = get_game_simulator_panel_data(
        game_pk=gamePk,
        n_sims=int(sim_n_sims)
    )
    render_simulator_box(sim_data, away_team, home_team)

with right:
    try:
        live = get_live_game(gamePk)
        state = extract_state(live)

        batter_hand = "?"
        pitcher_hand = "?"
        if state["batter_id"] is not None and "stand" in df.columns:
            b_hist = df[df["batter"] == int(state["batter_id"])]
            batter_hand = _mode(b_hist["stand"]) if len(b_hist) else "?"
        if state["pitcher_id"] is not None and "p_throws" in df.columns:
            p_hist = df[df["pitcher"] == int(state["pitcher_id"])]
            pitcher_hand = _mode(p_hist["p_throws"]) if len(p_hist) else "?"

        bug = scorebug_html(
            away_team=away_team,
            home_team=home_team,
            away_runs=state["away_runs"],
            home_runs=state["home_runs"],
            inning=state["inning"],
            is_top=state["is_top"],
            balls=state["balls"],
            strikes=state["strikes"],
            outs=state["outs"],
            r1=state["r1"],
            r2=state["r2"],
            r3=state["r3"],
            status=status,
            pitcher_name=state["pitcher_name"],
            pitcher_hand=pitcher_hand,
            batter_name=state["batter_name"],
            batter_hand=batter_hand
        )
        components.html(bug, height=140)

        lead_line = lead_text(away_team, home_team, state["away_runs"], state["home_runs"])
        runner_line = runners_text(state["r1"], state["r2"], state["r3"])
        st.markdown(f"**{lead_line}** · <span class='pill'>{runner_line}</span>", unsafe_allow_html=True)

        is_finalish = ("Final" in status) or ("Game Over" in status)
        if is_finalish:
            st.info("This game is final — live pitch-state modeling is disabled here. Pick an in-progress game for live odds.")
        else:
            if state["batter_id"] is None or state["pitcher_id"] is None:
                st.info("Not in a live pitch state yet (pregame / between innings). Try another game or wait.")
            else:
                if state["is_top"]:
                    bat_score = state["away_runs"]
                    fld_score = state["home_runs"]
                else:
                    bat_score = state["home_runs"]
                    fld_score = state["away_runs"]
                run_diff = int(bat_score - fld_score)

                out_probs_pct, bt_probs_pct, rbi_pct, blend, ctx_level, mix_table, b_stand, p_throw, base_state, count_id = predict_live_state_blended(
                    batter_id=int(state["batter_id"]),
                    pitcher_id=int(state["pitcher_id"]),
                    balls=int(state["balls"]),
                    strikes=int(state["strikes"]),
                    outs_when_up=int(state["outs"]),
                    inning=int(state["inning"]),
                    runner_1b=int(state["r1"]),
                    runner_2b=int(state["r2"]),
                    runner_3b=int(state["r3"]),
                    run_diff=run_diff,
                    n_sims=int(n_sims)
                )

                top_outcome, top_outcome_pct = top_label_pct(out_probs_pct)
                top_bt, top_bt_pct = top_label_pct(bt_probs_pct)
                pitch_blend_str = fmt_pitch_blend(blend)

                st.markdown("## 🎯 Pitch Context")
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Expected Pitch", pitch_blend_str)
                    st.caption(f"Context: {ctx_level}")
                with c2:
                    st.metric("Expected Outcome", f"{top_outcome}", f"{top_outcome_pct:.2f}%")
                with c3:
                    st.metric("Expected Ball Type", f"{top_bt}", f"{top_bt_pct:.2f}%")
                with c4:
                    st.metric("RBI Odds (approx)", f"{rbi_pct:.2f}%")

                d1, d2, d3, d4 = st.columns(4)
                d1.metric("Count", count_id)
                d2.metric("Base State", base_state)
                topbot = "Top" if state["is_top"] else "Bot"
                d3.metric("Inning / Outs", f"{topbot} {state['inning']} / {state['outs']}")
                platoon_txt = "✅" if (b_stand in ["L", "R"] and p_throw in ["L", "R"] and b_stand != p_throw) else "❌"
                d4.metric("Bats / Throws", f"{b_stand or 'NA'} / {p_throw or 'NA'}")
                d4.caption(f"Platoon: {platoon_txt}")

                st.markdown("### 🔥 Recent Form (rolling, %)")
                form = get_recent_form_snapshot(int(state["batter_id"]), int(state["pitcher_id"]))
                rf1, rf2, rf3, rf4, rf5, rf6 = st.columns(6)
                rf1.metric("P K%", f"{form['p_recent_k_rate']:.1f}")
                rf2.metric("P BB%", f"{form['p_recent_bb_rate']:.1f}")
                rf3.metric("P HR%", f"{form['p_recent_hr_rate']:.1f}")
                rf4.metric("B K%", f"{form['b_recent_k_rate']:.1f}")
                rf5.metric("B BB%", f"{form['b_recent_bb_rate']:.1f}")
                rf6.metric("B HR%", f"{form['b_recent_hr_rate']:.1f}")

                st.divider()

                colA, colB = st.columns([1.25, 1])
                with colA:
                    render_top_prob_rows("PA Outcome (Top)", out_probs_pct, top_n=5)
                    with st.expander("Full PA outcome table"):
                        out_df = pd.DataFrame({
                            "Outcome": list(out_probs_pct.keys()),
                            "Percent": list(out_probs_pct.values())
                        })
                        out_df = out_df.sort_values("Percent", ascending=False).reset_index(drop=True)
                        st.dataframe(out_df, use_container_width=True, height=260)

                with colB:
                    render_top_prob_rows("Ball Type (Top)", bt_probs_pct, top_n=5)
                    with st.expander("Pitch mix breakdown (this situation)"):
                        st.dataframe(mix_table, use_container_width=True, height=220)
                    with st.expander("Full ball type table"):
                        bt_df = pd.DataFrame({
                            "BallType": list(bt_probs_pct.keys()),
                            "Percent": list(bt_probs_pct.values())
                        })
                        bt_df = bt_df.sort_values("Percent", ascending=False).reset_index(drop=True)
                        st.dataframe(bt_df, use_container_width=True, height=220)

    except Exception as e:
        st.error(f"Error loading live game or running model: {e}")

if auto and not manual:
    time.sleep(float(refresh_sec))
    st.rerun()

# Run:
# python -m streamlit run live_dashboard.py