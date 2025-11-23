import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from PIL import Image


# ---------------------------------
# Custom Manchester United Theme (Red + Black)
# ---------------------------------
st.markdown("""
    <style>
        body {
            background-color: #1A1A1A;
        }
        .main {
            background-color: #1A1A1A;
        }
        h1, h2, h3, h4, h5, h6, label, .stMarkdown {
            color: #FFFFFF !important;
        }
        .css-1oe5cao {
            color: white !important;
        }
        .stButton > button {
            background-color: #DA291C !important;
            color: white !important;
            border-radius: 8px !important;
            padding: 0.5rem 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------
# Page Config
# ---------------------------------
st.set_page_config(
    page_title="Manchester United Match Predictor",
    page_icon="⚽",
    layout="centered",
)

# ---------------------------------
# Load model + encoder + features
# ---------------------------------
model = pickle.load(open("model.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))
feature_cols = pickle.load(open("features.pkl", "rb"))

df = pd.read_csv("Manchester United.csv")
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.sort_values("Date").reset_index(drop=True)

# ---------------------------------
# Logo + Heading
# ---------------------------------
logo = Image.open("man_united_logo.webp")

# Center using Streamlit columns
col1, col2, col3 = st.columns([2, 2, 1])
with col2:
    st.image(logo, width=130)

st.markdown(
    "<h1 style='text-align:center;margin-top:10px;'>Manchester United : Pre Match Predictor</h1>",
    unsafe_allow_html=True
)
# =================================
# INPUT SECTION
# =================================
st.header("Enter Match Details")

opponents = list(encoder.classes_)
opponent_name = st.selectbox("Opponent", opponents)
opponent_encoded = int(encoder.transform([opponent_name])[0])

venue = st.radio("Venue", ["Home", "Away"], horizontal=True)
is_home = 1 if venue == "Home" else 0

season = st.number_input("Season (Year)", min_value=2010, max_value=2035, value=2024)
month = st.selectbox("Month", list(range(1, 13)))

days = {
    1: "Monday", 2: "Tuesday", 3: "Wednesday",
    4: "Thursday", 5: "Friday", 6: "Saturday", 7: "Sunday"
}
day_of_week = st.selectbox("Day of Week", list(days.keys()), format_func=lambda x: days[x])

st.subheader("Recent Man United Form")
last5_goals = st.slider("Last 5 Avg Goals Scored", 0.0, 5.0, 1.6, step=0.1)
last5_winrate = st.slider("Last 5 Win Rate", 0.0, 1.0, 0.5, step=0.05)

input_df = pd.DataFrame(
    [{
        "Opponent": opponent_encoded,
        "Is_Home": is_home,
        "Season": season,
        "Month": month,
        "Day_of_Week": day_of_week,
        "Last5_Avg_Goals": last5_goals,
        "Last5_Win_Rate": last5_winrate
    }]
)[feature_cols]

predict_btn = st.button("🔮 Predict Result")

st.markdown("---")

# =================================
# Helper Functions
# =================================
def get_h2h_and_recent(df_all, opp_name):
    df_opp = df_all[df_all["Opponent"] == opp_name].copy()
    total = len(df_opp)
    if total == 0:
        return {"total": 0, "wins": 0, "draws": 0, "losses": 0, "avg_for": 0, "avg_against": 0, "recent": pd.DataFrame(), "win_rate": 0.5}

    wins = (df_opp["Result"] == 1).sum()
    draws = (df_opp["Result"] == 0).sum()
    losses = (df_opp["Result"] == -1).sum()
    avg_for = df_opp["Goals"].mean()
    avg_against = df_opp["Opponent_Goals"].mean()
    recent = df_opp.sort_values("Date", ascending=False).head(5)
    win_rate = wins / total

    return {"total": total, "wins": wins, "draws": draws, "losses": losses, "avg_for": avg_for, "avg_against": avg_against, "recent": recent, "win_rate": win_rate}

def difficulty_from_win_rate(win_rate):
    if win_rate >= 0.7: return "Easy", "🟢"
    elif win_rate >= 0.4: return "Medium", "🟡"
    else: return "Hard", "🔴"

# =================================
# OUTPUT SECTION
# =================================
st.header("Prediction & Insights")

if predict_btn:
    pred = model.predict(input_df)[0]
    probs = model.predict_proba(input_df)[0]

    class_to_prob = {cls: p for cls, p in zip(model.classes_, probs)}
    win_p = class_to_prob.get(1, 0) * 100
    draw_p = class_to_prob.get(0, 0) * 100
    loss_p = class_to_prob.get(-1, 0) * 100

    result_text = {1: "WIN 🎉", 0: "DRAW ⚪", -1: "LOSS ❌"}[pred]

    st.subheader(f"Predicted Result: **{result_text}**")

    # 🔥 Add Win%, Draw%, Loss% directly below result
    st.markdown(
        f"""
        
            <h5 style="color:white;">Win Probability: {win_p:.2f}%</h4>
            <h5 style="color:white;">Draw Probability: {draw_p:.2f}%</h4>
            <h5 style="color:white;">Loss Probability: {loss_p:.2f}%</h4>
        
        """,
        unsafe_allow_html=True
    )

    # Probability bar chart
    prob_df = pd.DataFrame(
        {"Outcome": ["Loss", "Draw", "Win"], "Probability": [loss_p, draw_p, win_p]}
    ).set_index("Outcome")
    st.bar_chart(prob_df)

    st.markdown("---")

    # Head-to-head
    h2h = get_h2h_and_recent(df, opponent_name)
    st.subheader(f"Head-to-Head vs {opponent_name}")
    st.write(f"**Record:** {h2h['wins']}W - {h2h['draws']}D - {h2h['losses']}L")
    st.write(f"**Avg Goals Scored:** {h2h['avg_for']:.2f}")
    st.write(f"**Avg Goals Conceded:** {h2h['avg_against']:.2f}")

    diff_label, diff_icon = difficulty_from_win_rate(h2h["win_rate"])
    st.write(f"### Difficulty: {diff_icon} {diff_label}")

    st.markdown("---")

    st.subheader("Last 5 Meetings")
    if not h2h["recent"].empty:
        table = h2h["recent"][["Date", "Is_Home", "Goals", "Opponent_Goals", "Result"]].copy()
        table["Is_Home"] = table["Is_Home"].map({1: "Home", 0: "Away"})
        table["Result"] = table["Result"].map({1: "Win", 0: "Draw", -1: "Loss"})
        st.dataframe(table, use_container_width=True)
    else:
        st.info("No recent meetings found.")

    st.markdown("---")

   

else:
    st.info("Fill the details above and click Predict Result.")

