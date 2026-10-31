"""Streamlit app: IPL chase win-probability predictor.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ipl_pipeline as ip

st.set_page_config(page_title="IPL Chase Win Predictor", page_icon="🏏", layout="wide")

GREEN, RED = "#2e9e5b", "#d9534f"


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
def _artifact_stamp() -> float:
    """Changes whenever the model file changes, so the cache refreshes after retraining."""
    f = ip.DEFAULT_ARTIFACT_DIR / ip.MODEL_FILE
    return f.stat().st_mtime if f.exists() else 0.0


@st.cache_resource(show_spinner="Loading model ...")
def get_artifacts(stamp: float) -> ip.Artifacts:
    return ip.load_artifacts(ip.DEFAULT_ARTIFACT_DIR)


@st.cache_data(show_spinner="Loading match history ...")
def get_dataset(stamp: float) -> pd.DataFrame | None:
    path = ip.DEFAULT_ARTIFACT_DIR / ip.DATASET_FILE
    return pd.read_csv(path) if path.exists() else None


# --------------------------------------------------------------------------- #
# In-app training (used only if no trained model exists yet)
# --------------------------------------------------------------------------- #
def training_screen() -> None:
    st.title("🏏 IPL Chase Win Predictor")
    st.warning("No trained model found yet.")
    st.markdown(
        "Train one from the terminal with `python train.py`, or use the button below "
        "(downloads the Cricsheet IPL data, then trains the ANN - this can take several minutes)."
    )
    epochs = st.slider("Max training epochs", 5, 60, 40)

    if st.button("Download data and train model", type="primary"):
        import tensorflow as tf

        bar = st.progress(0.0, text="Starting ...")

        class EpochProgress(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                bar.progress((epoch + 1) / epochs,
                             text=f"Epoch {epoch + 1}/{epochs} - val_loss {logs.get('val_loss', 0):.4f}")

        with st.status("Training ...", expanded=True) as status:
            try:
                ip.train_and_save(epochs=epochs, progress=st.write,
                                  extra_callbacks=[EpochProgress()])
            except Exception as exc:  # network errors, parsing problems, etc.
                status.update(label="Training failed", state="error")
                st.error(f"{type(exc).__name__}: {exc}")
                return
            status.update(label="Training complete", state="complete")
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def gauge(prob_pct: float, team: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_pct,
        number={"suffix": "%", "valueformat": ".1f"},
        title={"text": f"{team} win probability"},
        gauge={"axis": {"range": [0, 100]},
               "bar": {"color": GREEN if prob_pct >= 50 else RED},
               "threshold": {"line": {"color": "gray", "width": 3}, "value": 50}},
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=10))
    return fig


def wicket_sensitivity(arts: ip.Artifacts, base: dict) -> go.Figure:
    rows = [{**base, "wickets_lost": w} for w in range(0, 11)]
    probs = ip.predict_probabilities(arts, ip.build_feature_frame(rows)) * 100
    cur = base["wickets_lost"]
    fig = go.Figure(go.Bar(
        x=list(range(11)), y=probs,
        marker_color=[GREEN if w == cur else "#9aa5b1" for w in range(11)],
        hovertemplate="Wickets lost: %{x}<br>Win probability: %{y:.1f}%<extra></extra>"))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10),
                      title="Same situation, different wickets lost",
                      xaxis_title="Wickets lost", yaxis_title="Chasing team win %",
                      yaxis_range=[0, 100])
    return fig


# --------------------------------------------------------------------------- #
# Tab 1: live prediction
# --------------------------------------------------------------------------- #
def live_tab(arts: ip.Artifacts) -> None:
    opts = arts.options
    teams, venues, seasons = opts["teams"], opts["venues"], opts["seasons"]

    def idx(items, wanted, default=0):
        return items.index(wanted) if wanted in items else default

    st.sidebar.header("Match situation")
    batting = st.sidebar.selectbox("Chasing (batting) team", teams,
                                   index=idx(teams, "Chennai Super Kings"))
    bowling_options = [t for t in teams if t != batting]
    bowling = st.sidebar.selectbox("Bowling team", bowling_options,
                                   index=idx(bowling_options, "Mumbai Indians"))
    venue = st.sidebar.selectbox("Venue", venues, index=idx(venues, "Wankhede Stadium"))
    season = st.sidebar.selectbox("Season", seasons, index=len(seasons) - 1)

    target = st.sidebar.number_input("Target (runs to win)", 1, 300, 181, step=1)
    score = st.sidebar.number_input("Current score", 0, 400, 145, step=1)
    wickets = st.sidebar.slider("Wickets lost", 0, 10, 4)

    st.sidebar.markdown("**Overs bowled** (legal deliveries)")
    c1, c2 = st.sidebar.columns(2)
    overs = c1.number_input("Overs", 0, 20, 15, step=1)
    balls = c2.number_input("Balls", 0, 5, 0, step=1)
    legal_balls = int(overs) * 6 + int(balls)
    if legal_balls > ip.MAX_BALLS:
        st.sidebar.warning("An innings has at most 20 overs - using 20.0.")
        legal_balls = ip.MAX_BALLS

    try:
        res = ip.predict_chase_win_probability(
            arts, batting, bowling, venue, season, int(target), int(score), int(wickets), legal_balls)
    except ValueError as exc:
        st.error(str(exc))
        return

    st.subheader(f"{batting} chasing {int(target)} vs {bowling}")
    if res["status"] == "chase_won":
        st.success(f"Target reached - {batting} have won.")
    elif res["status"] == "chase_failed":
        st.error(f"{batting} cannot win from here (all out or overs finished).")
    for w in res["warnings"]:
        st.warning(w)

    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(gauge(res["batting_team_win_probability"], batting), width="stretch")
        st.caption(f"{bowling} win probability: **{res['bowling_team_win_probability']:.1f}%**")
    with right:
        a, b = st.columns(2)
        a.metric("Runs needed", res["runs_remaining"])
        b.metric("Balls left", res["balls_remaining"])
        c, d = st.columns(2)
        c.metric("Current run rate", f"{res['current_run_rate']:.2f}")
        d.metric("Required run rate", f"{res['required_run_rate']:.2f}")
        st.metric("Wickets in hand", 10 - int(wickets))

    if res["status"] == "in_progress":
        base = dict(season=season, venue=venue, batting_team=batting, bowling_team=bowling,
                    target=int(target), current_score=int(score), wickets_lost=int(wickets),
                    legal_balls_bowled=legal_balls)
        st.plotly_chart(wicket_sensitivity(arts, base), width="stretch")

    st.caption("Estimates come from a model trained on historical IPL data. It does not know "
               "about player form, injuries, pitch or weather, so treat it as a guide, not a guarantee.")


# --------------------------------------------------------------------------- #
# Tab 2: replay a historical chase
# --------------------------------------------------------------------------- #
def replay_tab(arts: ip.Artifacts) -> None:
    data = get_dataset(_artifact_stamp())
    if data is None:
        st.info(f"`{ip.DATASET_FILE}` not found in the artifacts folder, so match replay is unavailable.")
        return

    test_ids = set((arts.report or {}).get("test_match_ids", []))
    season = st.selectbox("Season", sorted(data["season"].unique(), key=ip._season_key, reverse=True),
                          key="replay_season")
    summary = (data[data["season"] == season]
               .groupby("match_id", sort=False)
               .agg(batting_team=("batting_team", "first"), bowling_team=("bowling_team", "first"),
                    venue=("venue", "first"), target=("target", "first"), won=("chase_win", "first"))
               .reset_index())
    if summary.empty:
        st.info("No matches for that season.")
        return

    labels = {r.match_id: f"{r.batting_team} chasing {r.target} vs {r.bowling_team}  ({r.venue})"
              for r in summary.itertuples()}
    match_id = st.selectbox("Match", list(labels), format_func=labels.get)
    row = summary[summary["match_id"] == match_id].iloc[0]
    m = data[data["match_id"] == match_id].sort_values("legal_balls_bowled", kind="stable")

    probs = ip.predict_probabilities(arts, m) * 100
    overs_x = m["legal_balls_bowled"] / 6
    fig = go.Figure(go.Scatter(
        x=overs_x, y=probs, mode="lines", line=dict(width=3, color=GREEN),
        customdata=m[["current_score", "wickets_lost"]].to_numpy(),
        hovertemplate="Over %{x:.1f}<br>Score %{customdata[0]}/%{customdata[1]}"
                      "<br>Win probability %{y:.1f}%<extra></extra>"))
    fig.add_hline(y=50, line_dash="dot", line_color="gray")
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10),
                      title=f"{row.batting_team} win probability through the chase",
                      xaxis_title="Overs bowled", yaxis_title="Win %", yaxis_range=[0, 100])
    st.plotly_chart(fig, width="stretch")

    outcome = f"{row.batting_team} won" if row.won == 1 else f"{row.bowling_team} won"
    st.markdown(f"**Actual result:** {outcome}")
    if test_ids:
        if match_id in test_ids:
            st.caption("This match was held out of training, so the curve is a genuine out-of-sample prediction.")
        else:
            st.caption("This match was part of the training data, so the model has seen its outcome; "
                       "pick a held-out match for an honest test.")


# --------------------------------------------------------------------------- #
# Tab 3: model performance
# --------------------------------------------------------------------------- #
def performance_tab(arts: ip.Artifacts) -> None:
    rep = arts.report
    if not rep:
        st.info(f"`{ip.REPORT_FILE}` not found. Re-run `python train.py` to generate the metrics report.")
        return

    m = rep["metrics"]
    st.caption(f"Evaluated on {rep['matches']['test']} held-out matches "
               f"({rep['rows']['test']:,} delivery snapshots). Trained {rep['trained_at']}.")
    cols = st.columns(6)
    for col, (label, key) in zip(cols, [("Accuracy", "accuracy"), ("Precision", "precision"),
                                        ("Recall", "recall"), ("F1", "f1"),
                                        ("ROC-AUC", "roc_auc"), ("Brier", "brier_score")]):
        val = m.get(key)
        col.metric(label, "n/a" if val is None else f"{val:.3f}")

    left, right = st.columns(2)
    with left:
        hist = pd.DataFrame(rep["history"])
        fig = go.Figure()
        fig.add_scatter(y=hist["loss"], name="Training loss")
        fig.add_scatter(y=hist["val_loss"], name="Validation loss")
        fig.update_layout(title="Loss per epoch", height=330, margin=dict(l=10, r=10, t=40, b=10),
                          xaxis_title="Epoch", yaxis_title="Binary cross-entropy")
        st.plotly_chart(fig, width="stretch")
    with right:
        cm = m["confusion_matrix"]
        fig = go.Figure(go.Heatmap(z=cm, x=["Pred: lose", "Pred: win"], y=["Actual: lose", "Actual: win"],
                                   text=cm, texttemplate="%{text}", colorscale="Blues", showscale=False))
        fig.update_layout(title="Confusion matrix (test)", height=330, margin=dict(l=10, r=10, t=40, b=10),
                          yaxis_autorange="reversed")
        st.plotly_chart(fig, width="stretch")

    if m.get("by_phase"):
        st.markdown("**Accuracy by phase of the chase** - predictions get easier as the game unfolds.")
        phase = pd.DataFrame(m["by_phase"]).rename(
            columns={"phase": "Phase", "rows": "Snapshots", "accuracy": "Accuracy", "roc_auc": "ROC-AUC"})
        st.dataframe(phase, hide_index=True, width="stretch")


def about_tab() -> None:
    st.markdown(
        """
**What this does** - estimates the chasing team's probability of winning, ball by ball in the
second innings, using an Artificial Neural Network trained on Cricsheet ball-by-ball IPL data.

**Inputs** - teams, venue, season, target, current score, wickets lost and legal balls bowled.
Run rates and balls/runs remaining are derived automatically.

**Good to know**
- Matches decided by D/L, and super overs, are excluded from training.
- No player-level, toss, weather or pitch information is used.
- Probabilities should be calibrated and validated before being used for anything beyond exploration.
- Data source: [Cricsheet](https://cricsheet.org/) - check their terms before redistributing the data.
        """
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    if not ip.artifacts_exist():
        training_screen()
        return

    try:
        arts = get_artifacts(_artifact_stamp())
    except Exception as exc:
        st.error(f"Could not load the model: {type(exc).__name__}: {exc}")
        st.stop()

    st.title("🏏 IPL Chase Win Predictor")
    tabs = st.tabs(["Live prediction", "Match replay", "Model performance", "About"])
    with tabs[0]:
        live_tab(arts)
    with tabs[1]:
        replay_tab(arts)
    with tabs[2]:
        performance_tab(arts)
    with tabs[3]:
        about_tab()


main()
