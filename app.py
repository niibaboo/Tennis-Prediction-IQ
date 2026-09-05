"""
ATP & WTA Tennis Match Predictor
Match winner + Total games (Over/Under)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from pathlib import Path
from scipy.stats import norm

st.set_page_config(page_title="Tennis Predictor", page_icon="🎾", layout="centered", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stMetric { background-color: rgba(28,131,225,0.08); border: 1px solid rgba(28,131,225,0.2);
                padding: 12px 16px; border-radius: 10px; }
    div[data-testid="stMetricValue"] { font-size: 1.7rem; }
    .stButton > button { border-radius: 8px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    base = Path(__file__).parent
    return {
        "ATP": {
            "winner": joblib.load(base / "tennis_xgb_model.joblib"),
            "games": joblib.load(base / "tennis_atp_games_model.joblib"),
        },
        "WTA": {
            "winner": joblib.load(base / "tennis_wta_model.joblib"),
            "games": joblib.load(base / "tennis_wta_games_model.joblib"),
        },
    }

models = load_models()

def get_players(tour):
    elo = models[tour]["winner"]["elo_ratings"]
    return sorted(elo.keys(), key=lambda p: -elo[p])

def get_elo(tour, player, source="winner"):
    return models[tour][source]["elo_ratings"].get(player, 1500.0)

def get_serve_avg(tour, player, source="winner"):
    hist = models[tour][source].get("serve_history", {}).get(player)
    window = models[tour][source].get("serve_window", 15)
    if not hist:
        return None
    recent = hist[-window:]
    return {k: float(np.nanmean([h[k] for h in recent])) for k in recent[0]}

def predict_winner(tour, p1, p2, surface="Hard", best_of=3):
    data = models[tour]["winner"]
    model, best_iter, feat_cols = data["xgb_model"], data["best_iteration"], data["feat_cols"]
    surface = surface if surface in ("Hard","Clay","Grass") else "Hard"
    e1, e2 = get_elo(tour, p1), get_elo(tour, p2)
    s1, s2 = get_serve_avg(tour, p1), get_serve_avg(tour, p2)
    row = {c: 0.0 for c in feat_cols}
    row["elo_diff"] = e1 - e2
    row["rank_diff"] = row["log_rank_diff"] = 0.0
    row["best_of"] = float(best_of)
    for s in ("Clay","Grass","Hard"):
        if f"surf_{s}" in row:
            row[f"surf_{s}"] = 1.0 if surface == s else 0.0
    if s1 and s2:
        for k in ["1st_pct","1stWon_pct","2ndWon_pct","ace_rate","df_rate","bp_save"]:
            col = f"{k}_diff"
            if col in row:
                row[col] = s1.get(k,0) - s2.get(k,0)
    X = pd.DataFrame([row])[feat_cols].astype(np.float32)
    dmat = xgb.DMatrix(X, feature_names=feat_cols)
    prob = float(model.predict(dmat, iteration_range=(0, best_iter+1))[0])
    return {"prob_p1": prob, "prob_p2": 1-prob, "elo1": round(e1,1), "elo2": round(e2,1),
            "has_serve": s1 is not None and s2 is not None}

def predict_total_games(tour, p1, p2, surface="Hard", best_of=3):
    data = models[tour]["games"]
    model, best_iter, feat_cols = data["games_model"], data["best_iteration"], data["feat_cols"]
    residual_std = data.get("residual_std", 6.0)
    surface = surface if surface in ("Hard","Clay","Grass") else "Hard"
    e1 = get_elo(tour, p1, "games")
    e2 = get_elo(tour, p2, "games")
    s1 = get_serve_avg(tour, p1, "games")
    s2 = get_serve_avg(tour, p2, "games")
    row = {c: 0.0 for c in feat_cols}
    row["elo_diff_abs"] = abs(e1 - e2)
    row["elo_sum"] = e1 + e2
    row["rank_diff_abs"] = 0.0
    row["best_of"] = float(best_of)
    for s in ("Clay","Grass","Hard"):
        if f"surf_{s}" in row:
            row[f"surf_{s}"] = 1.0 if surface == s else 0.0
    if s1 and s2:
        for k in ["1st_pct","1stWon_pct","2ndWon_pct","ace_rate","df_rate"]:
            if f"{k}_avg" in row:
                row[f"{k}_avg"] = (s1.get(k,0) + s2.get(k,0)) / 2
            if f"{k}_diff_abs" in row:
                row[f"{k}_diff_abs"] = abs(s1.get(k,0) - s2.get(k,0))
    X = pd.DataFrame([row])[feat_cols].astype(np.float32)
    dmat = xgb.DMatrix(X, feature_names=feat_cols)
    expected = float(model.predict(dmat, iteration_range=(0, best_iter+1))[0])
    # Approximate P(Over line) assuming residual ~ Normal(0, residual_std)
    lines = [21.5, 22.5, 23.5, 24.5, 25.5, 26.5]
    ou = {}
    for line in lines:
        p_over = 1 - norm.cdf(line, loc=expected, scale=residual_std)
        ou[line] = {"over": p_over, "under": 1 - p_over}
    return {"expected_games": expected, "ou": ou, "std": residual_std}

EXAMPLES = {
    "ATP": [("Jannik Sinner","Carlos Alcaraz","Hard"), ("Novak Djokovic","Alexander Zverev","Clay"),
            ("Jannik Sinner","Taylor Fritz","Hard"), ("Carlos Alcaraz","Jack Draper","Grass")],
    "WTA": [("Aryna Sabalenka","Iga Swiatek","Hard"), ("Coco Gauff","Elena Rybakina","Hard"),
            ("Iga Swiatek","Jessica Pegula","Clay"), ("Aryna Sabalenka","Coco Gauff","Hard")],
}

if "tour" not in st.session_state: st.session_state.tour = "ATP"
if "p1_idx" not in st.session_state: st.session_state.p1_idx = 0
if "p2_idx" not in st.session_state: st.session_state.p2_idx = 1
if "surface" not in st.session_state: st.session_state.surface = "Hard"

def set_example(tour, p1n, p2n, surf):
    st.session_state.tour = tour
    pl = get_players(tour)
    st.session_state.p1_idx = pl.index(p1n) if p1n in pl else 0
    st.session_state.p2_idx = pl.index(p2n) if p2n in pl else 1
    st.session_state.surface = surf

st.title("🎾 Tennis Match Predictor")
st.markdown("ATP & WTA · Match winner + **Total games** Over/Under")

with st.sidebar:
    st.header("Tour")
    tour = st.radio("Select tour", ["ATP","WTA"], index=0 if st.session_state.tour=="ATP" else 1, horizontal=True)
    st.session_state.tour = tour
    st.markdown("---")
    st.header("Quick examples")
    for p1e,p2e,surfe in EXAMPLES[tour]:
        label = f"{p1e.split()[-1]} vs {p2e.split()[-1]} ({surfe})"
        if st.button(label, key=f"ex_{tour}_{p1e}_{p2e}", use_container_width=True):
            set_example(tour, p1e, p2e, surfe)
            st.rerun()
    st.markdown("---")
    st.caption("Educational use only · Not for real-money betting")

PLAYERS = get_players(tour)
st.subheader(f"Select {tour} matchup")
c1, c2 = st.columns(2)
with c1:
    p1 = st.selectbox("Player 1", PLAYERS, index=min(st.session_state.p1_idx, len(PLAYERS)-1), key=f"p1_{tour}")
with c2:
    p2 = st.selectbox("Player 2", PLAYERS, index=min(st.session_state.p2_idx, len(PLAYERS)-1), key=f"p2_{tour}")
st.session_state.p1_idx = PLAYERS.index(p1)
st.session_state.p2_idx = PLAYERS.index(p2)

c3, c4 = st.columns(2)
with c3:
    opts = ["Hard","Clay","Grass"]
    surface = st.selectbox("Surface", opts, index=opts.index(st.session_state.surface) if st.session_state.surface in opts else 0, key=f"surf_{tour}")
    st.session_state.surface = surface
with c4:
    best_of = st.radio("Best of", [3,5], index=0, horizontal=True, key=f"bo_{tour}")

if st.button("Predict", type="primary", use_container_width=True):
    if p1 == p2:
        st.warning("Select two different players.")
    else:
        win = predict_winner(tour, p1, p2, surface, best_of)
        games = predict_total_games(tour, p1, p2, surface, best_of)

        st.markdown("---")
        st.subheader(f"{tour} · Match Winner")
        m1, m2 = st.columns(2)
        with m1:
            st.metric(p1, f"{win['prob_p1']*100:.1f} %", delta=f"Elo {win['elo1']}")
        with m2:
            st.metric(p2, f"{win['prob_p2']*100:.1f} %", delta=f"Elo {win['elo2']}")
        st.progress(win["prob_p1"], text=f"{p1} → {win['prob_p1']*100:.1f}%")
        fav = p1 if win["prob_p1"] >= 0.5 else p2
        conf = max(win["prob_p1"], win["prob_p2"])
        if conf >= 0.70:
            st.success(f"**Clear favourite:** {fav} ({conf*100:.1f}%)")
        elif conf >= 0.58:
            st.info(f"**Slight favourite:** {fav} ({conf*100:.1f}%)")
        else:
            st.warning(f"**Toss-up** — edge to {fav} ({conf*100:.1f}%)")

        # Decimal odds
        with st.expander("Match odds (decimal)"):
            o1 = 1 / win["prob_p1"] if win["prob_p1"] > 0.02 else 50
            o2 = 1 / win["prob_p2"] if win["prob_p2"] > 0.02 else 50
            st.write({p1: f"{o1:.2f}", p2: f"{o2:.2f}"})

        st.markdown("---")
        st.subheader("Total Games")
        exp = games["expected_games"]
        st.metric("Expected total games", f"{exp:.1f}")

        st.markdown("**Over / Under probabilities**")
        ou_rows = []
        for line, probs in games["ou"].items():
            ou_rows.append({
                "Line": line,
                "Over %": f"{probs['over']*100:.0f}%",
                "Under %": f"{probs['under']*100:.0f}%",
            })
        st.table(pd.DataFrame(ou_rows))

        st.caption(f"Model residual std ≈ {games['std']:.1f} games · MAE on hold-out ≈ 5 games")

st.markdown("---")
st.caption("Models trained on public ATP/WTA data 2018–2025. Educational use only.")
