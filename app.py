"""
ATP & WTA Tennis Match Predictor
Streamlit app powered by tuned XGBoost models
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from pathlib import Path

st.set_page_config(
    page_title="Tennis Match Predictor",
    page_icon="🎾",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stMetric {
        background-color: rgba(28, 131, 225, 0.08);
        border: 1px solid rgba(28, 131, 225, 0.2);
        padding: 12px 16px;
        border-radius: 10px;
    }
    div[data-testid="stMetricValue"] { font-size: 1.8rem; }
    .stButton > button { border-radius: 8px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Load models
# -------------------------------------------------
@st.cache_resource
def load_models():
    base = Path(__file__).parent
    atp = joblib.load(base / "tennis_xgb_model.joblib")
    wta = joblib.load(base / "tennis_wta_model.joblib")
    return {"ATP": atp, "WTA": wta}

models = load_models()

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def get_players(tour: str):
    elo = models[tour]["elo_ratings"]
    return sorted(elo.keys(), key=lambda p: -elo[p])

def get_elo(tour, player):
    return models[tour]["elo_ratings"].get(player, 1500.0)

def get_serve_avg(tour, player):
    hist = models[tour].get("serve_history", {}).get(player)
    window = models[tour].get("serve_window", 15)
    if not hist:
        return None
    recent = hist[-window:]
    keys = recent[0].keys()
    return {k: float(np.nanmean([h[k] for h in recent])) for k in keys}

def predict_match(tour: str, p1: str, p2: str, surface: str = "Hard", best_of: int = 3) -> dict:
    data = models[tour]
    model = data["xgb_model"]
    best_iter = data["best_iteration"]
    feat_cols = data["feat_cols"]

    surface = surface.capitalize()
    if surface not in ("Hard", "Clay", "Grass"):
        surface = "Hard"

    e1 = get_elo(tour, p1)
    e2 = get_elo(tour, p2)
    s1 = get_serve_avg(tour, p1)
    s2 = get_serve_avg(tour, p2)

    row = {c: 0.0 for c in feat_cols}
    row["elo_diff"] = e1 - e2
    row["rank_diff"] = 0.0
    row["log_rank_diff"] = 0.0
    row["best_of"] = float(best_of)

    for s in ("Clay", "Grass", "Hard"):
        col = f"surf_{s}"
        if col in row:
            row[col] = 1.0 if surface == s else 0.0

    if s1 and s2:
        for k in ["1st_pct", "1stWon_pct", "2ndWon_pct", "ace_rate", "df_rate", "bp_save"]:
            col = f"{k}_diff"
            if col in row:
                row[col] = s1.get(k, 0.0) - s2.get(k, 0.0)

    X = pd.DataFrame([row])[feat_cols].astype(np.float32)
    dmat = xgb.DMatrix(X, feature_names=feat_cols)
    prob = float(model.predict(dmat, iteration_range=(0, best_iter + 1))[0])

    return {
        "player1": p1, "player2": p2, "surface": surface, "best_of": best_of,
        "prob_p1": prob, "prob_p2": 1.0 - prob,
        "elo1": round(e1, 1), "elo2": round(e2, 1),
        "has_serve": s1 is not None and s2 is not None,
        "tour": tour,
    }

# Example matchups
EXAMPLES = {
    "ATP": [
        ("Jannik Sinner", "Carlos Alcaraz", "Hard"),
        ("Novak Djokovic", "Alexander Zverev", "Clay"),
        ("Jannik Sinner", "Taylor Fritz", "Hard"),
        ("Carlos Alcaraz", "Jack Draper", "Grass"),
    ],
    "WTA": [
        ("Aryna Sabalenka", "Iga Swiatek", "Hard"),
        ("Coco Gauff", "Elena Rybakina", "Hard"),
        ("Iga Swiatek", "Jessica Pegula", "Clay"),
        ("Aryna Sabalenka", "Coco Gauff", "Hard"),
    ],
}

# Session state
if "tour" not in st.session_state:
    st.session_state.tour = "ATP"
if "p1_idx" not in st.session_state:
    st.session_state.p1_idx = 0
if "p2_idx" not in st.session_state:
    st.session_state.p2_idx = 1
if "surface" not in st.session_state:
    st.session_state.surface = "Hard"

def set_example(tour, p1_name, p2_name, surf):
    st.session_state.tour = tour
    players = get_players(tour)
    st.session_state.p1_idx = players.index(p1_name) if p1_name in players else 0
    st.session_state.p2_idx = players.index(p2_name) if p2_name in players else 1
    st.session_state.surface = surf

# -------------------------------------------------
# UI
# -------------------------------------------------
st.title("🎾 Tennis Match Predictor")
st.markdown("ATP & WTA singles predictions powered by tuned **XGBoost** models (Elo + serve stats).")

# Sidebar
with st.sidebar:
    st.header("Tour")
    tour = st.radio("Select tour", ["ATP", "WTA"], index=0 if st.session_state.tour == "ATP" else 1, horizontal=True)
    st.session_state.tour = tour

    st.markdown("---")
    st.header("Quick examples")
    for p1e, p2e, surfe in EXAMPLES[tour]:
        label = f"{p1e.split()[-1]} vs {p2e.split()[-1]} ({surfe})"
        if st.button(label, key=f"ex_{tour}_{p1e}_{p2e}", use_container_width=True):
            set_example(tour, p1e, p2e, surfe)
            st.rerun()

    st.markdown("---")
    st.header("About")
    st.markdown(
        f"""
        **{tour} model**  
        - Hold-out ≈ 64 % accuracy  
        - Brier ≈ 0.217  
        - Elo + ranking + surface + rolling serve stats  
        - {len(models[tour]['elo_ratings'])} players
        """
    )
    st.caption("Educational use only · Not for betting")

# Players for current tour
PLAYERS = get_players(tour)

st.subheader(f"Select {tour} matchup")

col1, col2 = st.columns(2)
with col1:
    p1 = st.selectbox(
        "Player 1", options=PLAYERS,
        index=min(st.session_state.p1_idx, len(PLAYERS)-1),
        key=f"p1_{tour}", help="Type to search",
    )
with col2:
    p2 = st.selectbox(
        "Player 2", options=PLAYERS,
        index=min(st.session_state.p2_idx, len(PLAYERS)-1),
        key=f"p2_{tour}", help="Type to search",
    )

st.session_state.p1_idx = PLAYERS.index(p1)
st.session_state.p2_idx = PLAYERS.index(p2)

c3, c4 = st.columns(2)
with c3:
    surface_opts = ["Hard", "Clay", "Grass"]
    surface = st.selectbox(
        "Surface", surface_opts,
        index=surface_opts.index(st.session_state.surface) if st.session_state.surface in surface_opts else 0,
        key=f"surf_{tour}",
    )
    st.session_state.surface = surface
with c4:
    best_of = st.radio("Best of", [3, 5], index=0, horizontal=True, key=f"bo_{tour}")

if st.button("Predict winner", type="primary", use_container_width=True):
    if p1 == p2:
        st.warning("Please select two different players.")
    else:
        result = predict_match(tour, p1, p2, surface, best_of)

        st.markdown("---")
        st.subheader(f"{tour} Prediction")

        m1, m2 = st.columns(2)
        with m1:
            st.metric(result["player1"], f"{result['prob_p1']*100:.1f} %", delta=f"Elo {result['elo1']}")
        with m2:
            st.metric(result["player2"], f"{result['prob_p2']*100:.1f} %", delta=f"Elo {result['elo2']}")

        st.progress(result["prob_p1"], text=f"{result['player1']}  →  {result['prob_p1']*100:.1f} %")

        fav = result["player1"] if result["prob_p1"] >= 0.5 else result["player2"]
        conf = max(result["prob_p1"], result["prob_p2"])

        if conf >= 0.70:
            st.success(f"**Clear favourite:** {fav}  ({conf*100:.1f} %)")
        elif conf >= 0.58:
            st.info(f"**Slight favourite:** {fav}  ({conf*100:.1f} %)")
        else:
            st.warning(f"**Toss-up** — edge to {fav}  ({conf*100:.1f} %)")

        with st.expander("Model details"):
            st.write({
                "Tour": result["tour"],
                "Surface": result["surface"],
                "Best of": result["best_of"],
                "Elo difference (P1 − P2)": round(result["elo1"] - result["elo2"], 1),
                "Serve history used": "Yes" if result["has_serve"] else "No",
            })

st.markdown("---")
st.caption("Models trained on public ATP/WTA match data (2018–2025). Educational use only.")
