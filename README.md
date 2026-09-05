# 🎾 ATP & WTA Tennis Match Predictor

Streamlit web app that predicts the winner of professional singles matches for both **ATP** (men) and **WTA** (women).

**Models**
- Tuned XGBoost
- Features: Elo rating difference, ranking, surface, rolling 15-match serve statistics
- Hold-out performance (2024–25): ≈ 64 % accuracy, Brier ≈ 0.217 (both tours)

---

## App features
- Switch between **ATP** and **WTA**
- Searchable player dropdowns
- Quick example matchups in the sidebar
- Surface & best-of selection
- Clear probability display and favourite callout

---

## Quick start (local)

```bash
git clone <your-repo-url>
cd <repo-name>
pip install -r requirements.txt
streamlit run app.py
```

---

## Deploy to Streamlit Community Cloud (free)

1. Create a **public** GitHub repository
2. Upload these files:
   - `app.py`
   - `requirements.txt`
   - `tennis_xgb_model.joblib`   (ATP)
   - `tennis_wta_model.joblib`   (WTA)
   - `README.md`
3. Go to [https://share.streamlit.io](https://share.streamlit.io)
4. Sign in with GitHub → **New app**
5. Select your repo, main file path: `app.py`
6. Click **Deploy**

Live URL will be ready in 1–2 minutes.

---

## Notes
- Educational / demonstration project only — **not** for real-money betting
- Player names follow the training data format
