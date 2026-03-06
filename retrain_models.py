import json
import os
from datetime import date

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

from catboost import CatBoostClassifier


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

HIST_PATH = os.path.join(DATA_DIR, "statcast_2021_2025.parquet")

YEAR = date.today().year
YTD_PATH = os.path.join(DATA_DIR, f"statcast_{YEAR}_ytd.parquet")

MODEL_PA_PATH = os.path.join(MODELS_DIR, "model_pa.cbm")
MODEL_BT_PATH = os.path.join(MODELS_DIR, "model_balltype.cbm")
MODEL_DF_PATH = os.path.join(MODELS_DIR, "model_df.parquet")
META_PATH = os.path.join(MODELS_DIR, "model_meta.json")


def pick_first_existing(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None


def map_pa_outcome(events, des):
    ev = None if pd.isna(events) else str(events)
    d = "" if pd.isna(des) else str(des).lower()

    if ev == "single": return "single"
    if ev == "double": return "double"
    if ev == "triple": return "triple"
    if ev == "home_run": return "hr"

    if ev in ["walk", "intent_walk", "hit_by_pitch"]:
        return "walk"

    if "strikeout" in d or ev in ["strikeout", "strikeout_double_play"]:
        return "strikeout"

    if ev is not None:
        return "out"

    return "none"


def map_ball_type(bb_type, pa_outcome):
    if pa_outcome in ["walk", "strikeout"]:
        return "not_in_play"
    if bb_type is None or pd.isna(bb_type):
        return "not_in_play"
    bt = str(bb_type).lower().strip()
    if bt in ["ground_ball", "fly_ball", "line_drive", "popup"]:
        return bt
    return "not_in_play"


def _mode_series(s: pd.Series):
    vc = s.value_counts(dropna=True)
    return vc.index[0] if len(vc) else None


def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    # count_id + flags
    df["count_id"] = df["balls"].astype(int).astype(str) + "-" + df["strikes"].astype(int).astype(str)
    df["is_two_strike"] = (df["strikes"] == 2).astype(np.int8)
    df["is_three_ball"] = (df["balls"] == 3).astype(np.int8)

    # hitter / putaway buckets
    df["is_hitter_count"] = ((df["balls"] >= 2) & (df["strikes"] <= 1)).astype(np.int8)
    df["is_putaway_count"] = ((df["strikes"] == 2) & (df["balls"] <= 2)).astype(np.int8)

    # base state as categorical
    df["base_state"] = (
        df["runner_1b"].astype(int).astype(str)
        + df["runner_2b"].astype(int).astype(str)
        + df["runner_3b"].astype(int).astype(str)
    )

    # handedness interactions
    # stand: "L"/"R"/"S"; p_throws: "L"/"R"
    df["same_side"] = ((df["stand"] == df["p_throws"]) & df["stand"].isin(["L", "R"])).astype(np.int8)
    df["platoon_adv"] = ((df["stand"].isin(["L", "R"])) & (df["p_throws"].isin(["L", "R"])) & (df["stand"] != df["p_throws"])).astype(np.int8)

    return df


def add_pitch_guess(df: pd.DataFrame) -> pd.DataFrame:
    """
    pitch_guess = most common pitch thrown by that pitcher in that exact context:
      (pitcher, balls, strikes, base_state, outs_when_up)
    Backoff inside transform: if group too small, we fall back later in Streamlit.
    For training, we compute it using the full df (simple and effective).
    """
    keys = ["pitcher", "balls", "strikes", "base_state", "outs_when_up"]
    # Mode pitch_type per group
    # Compute with groupby apply on pitch_type values (fast enough for typical sizes)
    mode_map = (
        df.groupby(keys)["pitch_type"]
        .agg(lambda s: _mode_series(s))
        .reset_index()
        .rename(columns={"pitch_type": "pitch_guess"})
    )
    df = df.merge(mode_map, on=keys, how="left")
    # If missing (rare), fallback to pitcher overall mode
    pit_mode = (
        df.groupby(["pitcher"])["pitch_type"]
        .agg(lambda s: _mode_series(s))
        .reset_index()
        .rename(columns={"pitch_type": "pitch_guess_pit"})
    )
    df = df.merge(pit_mode, on=["pitcher"], how="left")
    df["pitch_guess"] = df["pitch_guess"].fillna(df["pitch_guess_pit"]).fillna("UNK")
    df.drop(columns=["pitch_guess_pit"], inplace=True)
    return df


def add_pitch_mix_priors(df: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    """
    Adds pitcher pitch-mix priors:
      - mix_pit_<PT>: overall pitch fractions by pitcher
      - mix_cnt_<PT>: pitch fractions by (pitcher,count_id,base_state,outs_when_up)
    Limited to top_k pitch types to avoid exploding columns.
    """
    # choose top pitch types globally
    top_pts = df["pitch_type"].value_counts().head(top_k).index.tolist()

    # overall pitcher mix
    pit_counts = df[df["pitch_type"].isin(top_pts)].groupby(["pitcher", "pitch_type"]).size().unstack(fill_value=0)
    pit_tot = pit_counts.sum(axis=1).replace(0, 1)
    pit_mix = (pit_counts.div(pit_tot, axis=0)).add_prefix("mix_pit_").reset_index()

    df = df.merge(pit_mix, on=["pitcher"], how="left")
    for pt in top_pts:
        col = f"mix_pit_{pt}"
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    # context mix (pitcher + count + base + outs)
    ctx_keys = ["pitcher", "count_id", "base_state", "outs_when_up"]
    ctx_counts = df[df["pitch_type"].isin(top_pts)].groupby(ctx_keys + ["pitch_type"]).size().unstack(fill_value=0)
    ctx_tot = ctx_counts.sum(axis=1).replace(0, 1)
    ctx_mix = (ctx_counts.div(ctx_tot, axis=0)).add_prefix("mix_ctx_").reset_index()

    df = df.merge(ctx_mix, on=ctx_keys, how="left")
    for pt in top_pts:
        col = f"mix_ctx_{pt}"
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    return df


def add_recent_form(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lightweight rolling 'form' features.
    Uses PA labels already attached to each pitch row (pa_outcome).
    We'll compute rolling rates by pitcher and batter over the last N PAs (proxy).
    """
    # These are terminal outcomes per PA, but repeated across pitches in the PA.
    # We'll approximate by using every row; it still helps as a signal.
    df = df.copy()

    # indicators
    df["is_k"] = (df["pa_outcome"] == "strikeout").astype(np.int8)
    df["is_bb"] = (df["pa_outcome"] == "walk").astype(np.int8)
    df["is_hr"] = (df["pa_outcome"] == "hr").astype(np.int8)

    # Sort stable
    sort_cols = ["game_pk", "at_bat_number"]
    # if pitch order exists, keep it; but after merge we might not have it
    if "pitch_number" in df.columns:
        sort_cols.append("pitch_number")
    elif "pitch_num" in df.columns:
        sort_cols.append("pitch_num")
    df = df.sort_values(sort_cols)

    # Rolling over last 200 rows as a rough "recent" window
    # (fast + avoids heavy groupby on full PAs)
    window = 200

    def roll_rate(g, col):
        return g[col].rolling(window=window, min_periods=30).mean().shift(1)

    # pitcher form
    df["p_recent_k_rate"] = df.groupby("pitcher", group_keys=False).apply(lambda g: roll_rate(g, "is_k"))
    df["p_recent_bb_rate"] = df.groupby("pitcher", group_keys=False).apply(lambda g: roll_rate(g, "is_bb"))
    df["p_recent_hr_rate"] = df.groupby("pitcher", group_keys=False).apply(lambda g: roll_rate(g, "is_hr"))

    # batter form
    df["b_recent_k_rate"] = df.groupby("batter", group_keys=False).apply(lambda g: roll_rate(g, "is_k"))
    df["b_recent_bb_rate"] = df.groupby("batter", group_keys=False).apply(lambda g: roll_rate(g, "is_bb"))
    df["b_recent_hr_rate"] = df.groupby("batter", group_keys=False).apply(lambda g: roll_rate(g, "is_hr"))

    # Fill NaNs
    for c in [
        "p_recent_k_rate","p_recent_bb_rate","p_recent_hr_rate",
        "b_recent_k_rate","b_recent_bb_rate","b_recent_hr_rate"
    ]:
        df[c] = df[c].fillna(df[c].median() if df[c].notna().any() else 0.0)

    # cleanup helper cols
    df.drop(columns=["is_k","is_bb","is_hr"], inplace=True)

    return df


def main():
    if not os.path.exists(HIST_PATH):
        raise FileNotFoundError(f"Missing historical parquet: {HIST_PATH}")

    if not os.path.exists(YTD_PATH):
        raise FileNotFoundError(f"Missing current-year parquet: {YTD_PATH}. Run update_statcast_ytd.py first.")

    print("[train] Loading parquet files...")
    hist = pd.read_parquet(HIST_PATH)
    ytd = pd.read_parquet(YTD_PATH)

    raw = pd.concat([hist, ytd], axis=0, ignore_index=True)
    print("[train] raw:", raw.shape)

    cols = set(raw.columns)

    sort_col = pick_first_existing(cols, ["pitch_number", "pitch_num"])
    if sort_col is None:
        raise ValueError("Need 'pitch_number' (or 'pitch_num') for ordering pitches within PA.")

    des_col = pick_first_existing(cols, ["des", "description"])
    if des_col is None:
        raise ValueError("Need 'des' or 'description' column in statcast data.")

    base_required = [
        "game_pk", "at_bat_number", sort_col,
        "batter", "pitcher",
        "pitch_type", "balls", "strikes", "outs_when_up", "inning",
        "stand", "p_throws",
        "on_1b", "on_2b", "on_3b",
        "release_speed", "release_spin_rate", "pfx_x", "pfx_z", "zone",
        "bat_score", "fld_score",
        "events", des_col
    ]
    missing = [c for c in base_required if c not in cols]
    if missing:
        raise ValueError(f"Missing columns in combined raw data: {missing}")

    has_bb_type = "bb_type" in cols
    keep = base_required + (["bb_type"] if has_bb_type else [])
    tmp = raw[keep].copy()

    tmp["runner_1b"] = tmp["on_1b"].notna().astype(np.int8)
    tmp["runner_2b"] = tmp["on_2b"].notna().astype(np.int8)
    tmp["runner_3b"] = tmp["on_3b"].notna().astype(np.int8)
    tmp.drop(columns=["on_1b","on_2b","on_3b"], inplace=True)

    tmp["run_diff"] = (tmp["bat_score"].fillna(0) - tmp["fld_score"].fillna(0)).astype(np.int16)
    tmp.drop(columns=["bat_score","fld_score"], inplace=True)

    tmp = tmp.sort_values(["game_pk","at_bat_number", sort_col])

    last_pitch = tmp.groupby(["game_pk","at_bat_number"], as_index=False).tail(1).copy()
    last_pitch["pa_outcome"] = [map_pa_outcome(e, d) for e, d in zip(last_pitch["events"], last_pitch[des_col])]
    last_pitch = last_pitch[last_pitch["pa_outcome"] != "none"].copy()

    if has_bb_type:
        last_pitch["ball_type"] = [map_ball_type(bt, oc) for bt, oc in zip(last_pitch["bb_type"], last_pitch["pa_outcome"])]
    else:
        last_pitch["ball_type"] = "not_in_play"

    labels = last_pitch[["game_pk","at_bat_number","pa_outcome","ball_type"]]
    tmp = tmp.merge(labels, on=["game_pk","at_bat_number"], how="inner")

    drop_cols = ["events", des_col]
    if has_bb_type:
        drop_cols.append("bb_type")
    tmp.drop(columns=drop_cols, inplace=True)

    for c in ["release_speed","release_spin_rate","pfx_x","pfx_z","zone"]:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce")

    # drop rows with missing key inputs
    tmp = tmp.dropna(subset=["pitch_type","stand","p_throws","release_speed","pfx_x","pfx_z","zone"]).copy()

    # ---- Feature engineering ----
    tmp = add_context_features(tmp)
    tmp = add_pitch_guess(tmp)
    tmp = add_pitch_mix_priors(tmp, top_k=10)
    tmp = add_recent_form(tmp)

    # IMPORTANT: for live realism, we train on pitch_guess, not pitch_type
    # (we still keep pitch_type inside df for building pitch_guess + pitch mix features)
    features = [
        "pitcher","batter",
        "pitch_guess",          # <-- live-knowable pitch proxy
        "balls","strikes",
        "count_id","base_state",
        "is_two_strike","is_three_ball","is_hitter_count","is_putaway_count",
        "outs_when_up","inning",
        "stand","p_throws","same_side","platoon_adv",
        "release_speed","release_spin_rate","pfx_x","pfx_z","zone",
        "runner_1b","runner_2b","runner_3b",
        "run_diff",
        # recent form
        "p_recent_k_rate","p_recent_bb_rate","p_recent_hr_rate",
        "b_recent_k_rate","b_recent_bb_rate","b_recent_hr_rate",
    ]

    # add the mix columns dynamically (created by add_pitch_mix_priors)
    mix_cols = [c for c in tmp.columns if c.startswith("mix_pit_") or c.startswith("mix_ctx_")]
    features = features + sorted(mix_cols)

    # modeling df
    df = tmp[features + ["pa_outcome","ball_type"]].copy()
    print("[train] modeling df:", df.shape)
    print("[train] pa_outcome dist (%):")
    print(df["pa_outcome"].value_counts(normalize=True).mul(100).round(2).head(10))
    print("[train] ball_type dist (%):")
    print(df["ball_type"].value_counts(normalize=True).mul(100).round(2).head(10))

    df.to_parquet(MODEL_DF_PATH, index=False)
    print(f"[train] saved: {MODEL_DF_PATH}")

    # categorical columns
    cat_cols = [
        "pitcher","batter",
        "pitch_guess",
        "count_id","base_state",
        "stand","p_throws","zone"
    ]

    with open(META_PATH, "w") as f:
        json.dump({"features": features, "cat_cols": cat_cols}, f, indent=2)
    print(f"[train] saved: {META_PATH}")

    # -----------------------
    # Train PA outcome model
    # -----------------------
    X = df[features].copy()
    y = df["pa_outcome"].copy()

    for c in cat_cols:
        X[c] = X[c].astype("string").fillna("NA")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    def train_catboost_multiclass(Xtr, ytr, Xva, yva, cat_features, iterations, lr, depth, use_gpu=True):
        params = dict(
            loss_function="MultiClass",
            iterations=iterations,
            learning_rate=lr,
            depth=depth,
            eval_metric="MultiClass",
            verbose=100,
            # early stopping (saves best iteration)
            use_best_model=True,
            od_type="Iter",
            od_wait=60,
        )
        if use_gpu:
            params.update(dict(task_type="GPU", devices="0"))

        m = CatBoostClassifier(**params)
        m.fit(
            Xtr, ytr,
            cat_features=cat_features,
            eval_set=(Xva, yva),
        )
        return m

    print("[train] training PA outcome model...")
    try:
        model_pa = train_catboost_multiclass(X_train, y_train, X_test, y_test, cat_cols, iterations=1400, lr=0.05, depth=8, use_gpu=True)
    except Exception as e:
        print("[train] GPU train failed, retrying CPU. Error was:", str(e))
        model_pa = train_catboost_multiclass(X_train, y_train, X_test, y_test, cat_cols, iterations=1400, lr=0.05, depth=8, use_gpu=False)

    probs = model_pa.predict_proba(X_test)
    print("[train] PA logloss:", log_loss(y_test, probs, labels=model_pa.classes_))
    print("[train] PA classes:", list(model_pa.classes_))
    model_pa.save_model(MODEL_PA_PATH)
    print(f"[train] saved: {MODEL_PA_PATH}")

    # -----------------------
    # Train ball type model
    # -----------------------
    Xb = df[features].copy()
    yb = df["ball_type"].copy()

    for c in cat_cols:
        Xb[c] = Xb[c].astype("string").fillna("NA")

    Xb_train, Xb_test, yb_train, yb_test = train_test_split(
        Xb, yb, test_size=0.2, random_state=42, stratify=yb
    )

    print("[train] training Ball Type model...")
    try:
        model_bt = train_catboost_multiclass(Xb_train, yb_train, Xb_test, yb_test, cat_cols, iterations=1000, lr=0.06, depth=8, use_gpu=True)
    except Exception as e:
        print("[train] GPU train failed, retrying CPU. Error was:", str(e))
        model_bt = train_catboost_multiclass(Xb_train, yb_train, Xb_test, yb_test, cat_cols, iterations=1000, lr=0.06, depth=8, use_gpu=False)

    probs_bt = model_bt.predict_proba(Xb_test)
    print("[train] BT logloss:", log_loss(yb_test, probs_bt, labels=model_bt.classes_))
    print("[train] BT classes:", list(model_bt.classes_))
    model_bt.save_model(MODEL_BT_PATH)
    print(f"[train] saved: {MODEL_BT_PATH}")

    print("\n[train] DONE. Streamlit will use artifacts in /models.")
    print(" - model_pa.cbm")
    print(" - model_balltype.cbm")
    print(" - model_df.parquet")
    print(" - model_meta.json")


if __name__ == "__main__":
    main()