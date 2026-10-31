"""
IPL chase win-probability pipeline.

This module is the reusable core extracted from the original notebook
(IPL_Match_Winning_Prediction_ANN.ipynb). It exposes functions that can be
called from a script (train.py), a notebook, or a web app (app.py):

    * data:        download_cricsheet(), load_matches(), build_snapshots(), clean_snapshots()
    * training:    train_and_save()
    * artifacts:   save/load via load_artifacts()
    * prediction:  predict_chase_win_probability(), predict_probabilities()

TensorFlow is imported lazily so that importing this module stays fast.
"""
from __future__ import annotations

import json
import os
import random
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # keep TF logs quiet

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
SEED = 42
BASE_DIR = Path(__file__).resolve().parent
DATA_URL = "https://cricsheet.org/downloads/ipl_json.zip"
DEFAULT_DATA_DIR = BASE_DIR / "data"
DEFAULT_ARTIFACT_DIR = BASE_DIR / "artifacts"

MODEL_FILE = "ipl_ann_model.keras"
PREPROCESSOR_FILE = "ipl_preprocessor.joblib"
FEATURES_FILE = "ipl_model_features.json"
DATASET_FILE = "ipl_model_dataset.csv"
REPORT_FILE = "ipl_training_report.json"

NUMERIC_FEATURES = [
    "target", "current_score", "wickets_lost", "legal_balls_bowled",
    "balls_remaining", "runs_remaining", "current_run_rate", "required_run_rate",
]
CATEGORICAL_FEATURES = ["season", "venue", "batting_team", "bowling_team"]
FEATURES = ["season", "venue", "batting_team", "bowling_team",
            "target", "current_score", "wickets_lost",
            "legal_balls_bowled", "balls_remaining", "runs_remaining",
            "current_run_rate", "required_run_rate"]
TARGET = "chase_win"
MAX_BALLS = 120

# Franchise renames -> a single current name, so one team is not split in two.
TEAM_ALIASES = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
}
# Common venue spelling variants. Extend this after inspecting the venue list
# in your downloaded data (see `venue_report()`).
VENUE_ALIASES = {
    "M. Chinnaswamy Stadium": "M Chinnaswamy Stadium",
    "MA Chidambaram Stadium, Chepauk": "MA Chidambaram Stadium",
    "Feroz Shah Kotla": "Arun Jaitley Stadium",
    "Punjab Cricket Association IS Bindra Stadium": "Punjab Cricket Association Stadium",
    "Punjab Cricket Association IS Bindra Stadium, Mohali": "Punjab Cricket Association Stadium",
    "Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh": "Punjab Cricket Association Stadium",
    "Sardar Patel Stadium, Motera": "Narendra Modi Stadium",
    "Rajiv Gandhi International Stadium, Uppal": "Rajiv Gandhi International Stadium",
}

ProgressFn = Callable[[str], None]


def _noop(_: str) -> None:
    pass


def normalise_team(name: str) -> str:
    return TEAM_ALIASES.get(name, name)


def normalise_venue(name: str) -> str:
    """Apply explicit aliases, then drop a trailing ', City' suffix."""
    if name in VENUE_ALIASES:
        return VENUE_ALIASES[name]
    base = name.split(",")[0].strip() if isinstance(name, str) else name
    return VENUE_ALIASES.get(base, base)


def set_seeds(seed: int = SEED) -> None:
    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# --------------------------------------------------------------------------- #
# 1. Data download and parsing
# --------------------------------------------------------------------------- #
def download_cricsheet(data_dir: Path = DEFAULT_DATA_DIR, force: bool = False,
                       progress: ProgressFn = _noop) -> Path:
    """Download and extract the Cricsheet IPL JSON archive. Returns the JSON folder."""
    data_dir = Path(data_dir)
    json_dir = data_dir / "ipl_json"
    zip_path = data_dir / "ipl_json.zip"
    data_dir.mkdir(parents=True, exist_ok=True)

    if force or not zip_path.exists():
        progress("Downloading IPL data from Cricsheet ...")
        resp = requests.get(DATA_URL, timeout=180)
        resp.raise_for_status()
        zip_path.write_bytes(resp.content)

    json_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(json_dir)
    return json_dir


def load_matches(json_dir: Path, progress: ProgressFn = _noop
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse Cricsheet JSON files into (matches_df, deliveries_df)."""
    json_files = sorted(Path(json_dir).glob("*.json"))
    if not json_files:
        raise RuntimeError(f"No JSON files found in {json_dir}.")
    progress(f"Parsing {len(json_files)} match files ...")

    match_rows, delivery_rows = [], []
    for fp in json_files:
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue  # skip unreadable files

        info = data.get("info", {})
        teams = [normalise_team(t) for t in info.get("teams", [])]
        if len(teams) != 2:
            continue

        match_id = fp.stem
        outcome = info.get("outcome", {})
        winner = outcome.get("winner")
        winner = normalise_team(winner) if winner else None
        season = str(info.get("season", "Unknown"))
        venue = normalise_venue(info.get("venue", "Unknown"))

        match_rows.append({
            "match_id": match_id, "season": season, "venue": venue,
            "team1": teams[0], "team2": teams[1], "winner": winner,
            "outcome_method": outcome.get("method"),
        })

        for innings_no, innings in enumerate(data.get("innings", []), start=1):
            if innings_no > 2:  # ignore super overs
                continue
            batting_team = normalise_team(innings.get("team", "Unknown"))
            bowling_team = next((t for t in teams if t != batting_team), "Unknown")
            target_runs = (innings.get("target", {}) or {}).get("runs")

            for over_obj in innings.get("overs", []):
                for ball_no, d in enumerate(over_obj.get("deliveries", []), start=1):
                    runs = d.get("runs", {})
                    extras = d.get("extras", {}) or {}
                    delivery_rows.append({
                        "match_id": match_id, "innings": innings_no,
                        "season": season, "venue": venue,
                        "batting_team": batting_team, "bowling_team": bowling_team,
                        "over": int(over_obj.get("over", 0)), "ball_in_over": ball_no,
                        "total_runs": int(runs.get("total", 0)),
                        "wide": int(extras.get("wides", 0) or 0),
                        "noballs": int(extras.get("noballs", 0) or 0),
                        "wicket_count": len(d.get("wickets", []) or []),
                        "target_from_source": target_runs,
                    })

    return pd.DataFrame(match_rows), pd.DataFrame(delivery_rows)


def venue_report(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Match counts per (normalised) venue - handy for spotting remaining aliases."""
    return matches_df["venue"].value_counts().rename_axis("venue").reset_index(name="matches")


# --------------------------------------------------------------------------- #
# 2. Match-state snapshots (vectorised version of the notebook loop)
# --------------------------------------------------------------------------- #
def build_snapshots(matches_df: pd.DataFrame, deliveries_df: pd.DataFrame) -> pd.DataFrame:
    """One row per second-innings delivery, labelled with whether the chase won."""
    m = matches_df.copy()
    m = m[m["winner"].notna()]
    m = m[~m["outcome_method"].astype(str).str.upper().isin({"D/L", "DL"})]
    winner_map = m.set_index("match_id")["winner"].to_dict()

    d = deliveries_df[deliveries_df["match_id"].isin(winner_map)].copy()
    first_totals = (d[d["innings"] == 1].groupby("match_id")["total_runs"].sum())
    s = d[d["innings"] == 2].copy()
    s = s[s["match_id"].isin(first_totals.index)]
    if s.empty:
        raise RuntimeError("No labelled chase snapshots were created. Inspect the raw data/parser.")

    s["is_legal"] = ((s["wide"] == 0) & (s["noballs"] == 0)).astype(int)
    g = s.groupby("match_id", sort=False)
    s["current_score"] = g["total_runs"].cumsum()
    s["wickets_lost"] = g["wicket_count"].cumsum().clip(upper=10)
    s["legal_balls_bowled"] = g["is_legal"].cumsum()

    source_target = g["target_from_source"].transform("first")  # first non-null
    s["target"] = source_target.fillna(s["match_id"].map(first_totals) + 1).astype(int)

    s["balls_remaining"] = (MAX_BALLS - s["legal_balls_bowled"]).clip(lower=0)
    s["runs_remaining"] = (s["target"] - s["current_score"]).clip(lower=0)
    overs_done = s["legal_balls_bowled"] / 6
    s["current_run_rate"] = np.where(overs_done > 0, s["current_score"] / overs_done.replace(0, np.nan), 0.0)
    s["required_run_rate"] = np.where(
        s["balls_remaining"] > 0,
        s["runs_remaining"] / (s["balls_remaining"].replace(0, np.nan) / 6), 0.0)
    s["chase_win"] = (s["match_id"].map(winner_map) == s["batting_team"]).astype(int)

    cols = ["match_id", "season", "venue", "batting_team", "bowling_team", "target",
            "current_score", "wickets_lost", "legal_balls_bowled", "balls_remaining",
            "runs_remaining", "current_run_rate", "required_run_rate", "chase_win"]
    return s[cols].reset_index(drop=True)


def clean_snapshots(model_df: pd.DataFrame) -> pd.DataFrame:
    df = model_df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET])
    df = df.drop_duplicates(subset=["match_id", "legal_balls_bowled", "current_score"])
    df = df[(df["target"] > 0) & (df["current_score"] >= 0)
            & df["wickets_lost"].between(0, 10)
            & df["legal_balls_bowled"].between(0, MAX_BALLS)].copy()
    df[TARGET] = df[TARGET].astype(int)
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 3. Model definition
# --------------------------------------------------------------------------- #
def make_preprocessor() -> ColumnTransformer:
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # older scikit-learn
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer(
        [("num", StandardScaler(), NUMERIC_FEATURES),
         ("cat", encoder, CATEGORICAL_FEATURES)],
        remainder="drop",
    )


def build_model(input_dim: int):
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Input

    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation="relu"), Dropout(0.30),
        Dense(64, activation="relu"), Dropout(0.20),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )
    return model


# --------------------------------------------------------------------------- #
# 4. Training + evaluation + saving
# --------------------------------------------------------------------------- #
def _evaluate(y_true: pd.Series, prob: np.ndarray, x_test: pd.DataFrame) -> dict:
    pred = (prob >= 0.5).astype(int)
    out = {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "brier_score": brier_score_loss(y_true, prob),
        "log_loss": log_loss(y_true, np.clip(prob, 1e-7, 1 - 1e-7), labels=[0, 1]),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=[0, 1]).tolist(),
    }
    out["roc_auc"] = roc_auc_score(y_true, prob) if y_true.nunique() == 2 else None

    # Performance by match phase (later phases carry more information).
    phases = pd.cut(x_test["legal_balls_bowled"], bins=[-1, 35, 89, MAX_BALLS],
                    labels=["Overs 1-6", "Overs 7-15", "Overs 16-20"])
    rows = []
    for label in phases.cat.categories:
        mask = (phases == label).to_numpy()
        if mask.sum() == 0:
            continue
        yt, pp = y_true.to_numpy()[mask], prob[mask]
        rows.append({"phase": label, "rows": int(mask.sum()),
                     "accuracy": float(accuracy_score(yt, pp >= 0.5)),
                     "roc_auc": float(roc_auc_score(yt, pp)) if len(set(yt)) == 2 else None})
    out["by_phase"] = rows
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in out.items()}


def train_and_save(
    data_dir: Path = DEFAULT_DATA_DIR,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    epochs: int = 40,
    batch_size: int = 256,
    download: bool = True,
    force_download: bool = False,
    json_dir: Optional[Path] = None,
    progress: ProgressFn = _noop,
    extra_callbacks: Optional[list] = None,
) -> dict:
    """Run the full pipeline and write all artifacts. Returns the training report."""
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    set_seeds()
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    if json_dir is None:
        json_dir = (download_cricsheet(data_dir, force_download, progress) if download
                    else Path(data_dir) / "ipl_json")

    matches_df, deliveries_df = load_matches(json_dir, progress)
    progress("Building second-innings snapshots ...")
    model_df = clean_snapshots(build_snapshots(matches_df, deliveries_df))
    progress(f"Dataset ready: {len(model_df):,} snapshots from {model_df['match_id'].nunique()} matches")

    X, y, groups = model_df[FEATURES], model_df[TARGET], model_df["match_id"]

    # Split by match so deliveries from one match never straddle partitions.
    tv_idx, test_idx = next(GroupShuffleSplit(1, test_size=0.20, random_state=SEED).split(X, y, groups))
    X_tv, y_tv, g_tv = X.iloc[tv_idx], y.iloc[tv_idx], groups.iloc[tv_idx]
    tr_idx, val_idx = next(GroupShuffleSplit(1, test_size=0.20, random_state=SEED).split(X_tv, y_tv, g_tv))
    X_train, y_train = X_tv.iloc[tr_idx], y_tv.iloc[tr_idx]
    X_val, y_val = X_tv.iloc[val_idx], y_tv.iloc[val_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    preprocessor = make_preprocessor()
    Xtr = preprocessor.fit_transform(X_train).astype("float32")
    Xva = preprocessor.transform(X_val).astype("float32")
    Xte = preprocessor.transform(X_test).astype("float32")

    model = build_model(Xtr.shape[1])
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        *(extra_callbacks or []),
    ]
    progress("Training the ANN ...")
    history = model.fit(Xtr, y_train, validation_data=(Xva, y_val), epochs=epochs,
                        batch_size=batch_size, callbacks=callbacks, verbose=0)

    progress("Evaluating on held-out test matches ...")
    test_prob = model.predict(Xte, verbose=0).ravel()
    metrics = _evaluate(y_test, test_prob, X_test)

    report = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": {"train": len(X_train), "validation": len(X_val), "test": len(X_test)},
        "matches": {"train": int(g_tv.iloc[tr_idx].nunique()),
                    "validation": int(g_tv.iloc[val_idx].nunique()),
                    "test": int(groups.iloc[test_idx].nunique())},
        "test_match_ids": sorted(groups.iloc[test_idx].unique().tolist()),
        "metrics": metrics,
        "history": {k: [float(x) for x in v] for k, v in history.history.items()},
    }

    model.save(artifact_dir / MODEL_FILE)
    joblib.dump(preprocessor, artifact_dir / PREPROCESSOR_FILE)
    (artifact_dir / FEATURES_FILE).write_text(json.dumps(FEATURES, indent=4))
    (artifact_dir / REPORT_FILE).write_text(json.dumps(report, indent=2))
    model_df.to_csv(artifact_dir / DATASET_FILE, index=False)
    progress("Saved model, preprocessor, dataset and report.")
    return report


# --------------------------------------------------------------------------- #
# 5. Loading artifacts and predicting
# --------------------------------------------------------------------------- #
@dataclass
class Artifacts:
    model: object
    preprocessor: ColumnTransformer
    options: dict = field(default_factory=dict)  # seasons / venues / teams the model knows
    report: Optional[dict] = None


def artifacts_exist(artifact_dir: Path = DEFAULT_ARTIFACT_DIR) -> bool:
    d = Path(artifact_dir)
    return (d / MODEL_FILE).exists() and (d / PREPROCESSOR_FILE).exists()


def _season_key(s: str) -> tuple:
    try:
        return (int(str(s)[:4]), str(s))
    except ValueError:
        return (0, str(s))


def load_artifacts(artifact_dir: Path = DEFAULT_ARTIFACT_DIR) -> Artifacts:
    """Load model + preprocessor (+ optional report). Dropdown options are read from the
    fitted encoder, so they always match what the model was trained on."""
    import tensorflow as tf

    d = Path(artifact_dir)
    if not artifacts_exist(d):
        raise FileNotFoundError(
            f"Model files not found in {d}. Run `python train.py` first "
            f"(or copy the notebook's saved files there).")

    model = tf.keras.models.load_model(d / MODEL_FILE)
    preprocessor = joblib.load(d / PREPROCESSOR_FILE)

    cats = dict(zip(CATEGORICAL_FEATURES, preprocessor.named_transformers_["cat"].categories_))
    options = {
        "seasons": sorted((str(x) for x in cats["season"]), key=_season_key),
        "venues": sorted(str(x) for x in cats["venue"]),
        "teams": sorted({str(x) for x in cats["batting_team"]} | {str(x) for x in cats["bowling_team"]}),
    }
    report = None
    if (d / REPORT_FILE).exists():
        report = json.loads((d / REPORT_FILE).read_text())
    return Artifacts(model, preprocessor, options, report)


def derive_state(target: int, current_score: int, wickets_lost: int, legal_balls_bowled: int) -> dict:
    balls_remaining = max(0, MAX_BALLS - int(legal_balls_bowled))
    runs_remaining = max(0, int(target) - int(current_score))
    overs_done = int(legal_balls_bowled) / 6
    crr = float(current_score) / overs_done if overs_done > 0 else 0.0
    rrr = runs_remaining / (balls_remaining / 6) if balls_remaining > 0 else 0.0
    return {"balls_remaining": balls_remaining, "runs_remaining": runs_remaining,
            "current_run_rate": crr, "required_run_rate": rrr}


def validate_inputs(batting_team, bowling_team, target, current_score,
                    wickets_lost, legal_balls_bowled) -> None:
    if batting_team == bowling_team:
        raise ValueError("Batting and bowling teams must be different.")
    if int(target) <= 0:
        raise ValueError("Target must be a positive number.")
    if int(current_score) < 0:
        raise ValueError("Current score cannot be negative.")
    if not 0 <= int(wickets_lost) <= 10:
        raise ValueError("Wickets lost must be between 0 and 10.")
    if not 0 <= int(legal_balls_bowled) <= MAX_BALLS:
        raise ValueError(f"Legal balls bowled must be between 0 and {MAX_BALLS}.")


def build_feature_frame(rows: list[dict]) -> pd.DataFrame:
    """rows: dicts with season, venue, batting_team, bowling_team, target,
    current_score, wickets_lost, legal_balls_bowled -> model-ready DataFrame."""
    out = []
    for r in rows:
        st = derive_state(r["target"], r["current_score"], r["wickets_lost"], r["legal_balls_bowled"])
        out.append({
            "season": str(r["season"]), "venue": str(r["venue"]),
            "batting_team": str(r["batting_team"]), "bowling_team": str(r["bowling_team"]),
            "target": float(r["target"]), "current_score": float(r["current_score"]),
            "wickets_lost": float(r["wickets_lost"]),
            "legal_balls_bowled": float(r["legal_balls_bowled"]),
            **{k: float(v) for k, v in st.items()},
        })
    return pd.DataFrame(out)[FEATURES]


def predict_probabilities(artifacts: Artifacts, frame: pd.DataFrame) -> np.ndarray:
    """Chasing-team win probability (0-1) for each row of a FEATURES-shaped frame."""
    processed = artifacts.preprocessor.transform(frame[FEATURES]).astype("float32")
    return np.asarray(artifacts.model(processed, training=False)).ravel()


def predict_chase_win_probability(
    artifacts: Artifacts, batting_team: str, bowling_team: str, venue: str, season: str,
    target: int, current_score: int, wickets_lost: int, legal_balls_bowled: int,
) -> dict:
    """Predict the chasing team's win probability for one live match state.

    Raises ValueError for impossible inputs. Already-decided states (target reached,
    all out, or innings over) are resolved by the rules of the game instead of the ANN.
    """
    validate_inputs(batting_team, bowling_team, target, current_score, wickets_lost, legal_balls_bowled)

    state = derive_state(target, current_score, wickets_lost, legal_balls_bowled)
    warnings = []
    known = artifacts.options
    if season not in known.get("seasons", []):
        warnings.append(f"Season '{season}' was not in the training data.")
    if venue not in known.get("venues", []):
        warnings.append(f"Venue '{venue}' was not in the training data.")
    for team in (batting_team, bowling_team):
        if team not in known.get("teams", []):
            warnings.append(f"Team '{team}' was not in the training data.")

    if state["runs_remaining"] == 0:
        status, prob = "chase_won", 1.0
    elif int(wickets_lost) >= 10 or state["balls_remaining"] == 0:
        status, prob = "chase_failed", 0.0
    else:
        status = "in_progress"
        frame = build_feature_frame([dict(
            season=season, venue=venue, batting_team=batting_team, bowling_team=bowling_team,
            target=target, current_score=current_score, wickets_lost=wickets_lost,
            legal_balls_bowled=legal_balls_bowled)])
        prob = float(predict_probabilities(artifacts, frame)[0])

    return {
        "status": status,
        "batting_team_win_probability": round(prob * 100, 2),
        "bowling_team_win_probability": round((1 - prob) * 100, 2),
        "runs_remaining": state["runs_remaining"],
        "balls_remaining": state["balls_remaining"],
        "current_run_rate": round(state["current_run_rate"], 2),
        "required_run_rate": round(state["required_run_rate"], 2),
        "warnings": warnings,
    }
