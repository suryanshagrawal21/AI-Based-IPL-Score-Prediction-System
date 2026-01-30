import streamlit as st
import numpy as np
import pandas as pd
import pickle
import altair as alt
import os

# --- Configuration ---
st.set_page_config(
    page_title="AI-Based IPL Score Prediction System",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CSS: Glassmorphism & Background ---
st.markdown("""
<style>
/* Background Image */
.stApp {
    background-image: url("https://wallpaperaccess.com/full/1088620.jpg");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}

/* Glassmorphism Logic */
.block-container {
    background-color: rgba(0, 0, 0, 0.65); /* Dark transparent overlay */
    padding: 2rem;
    border-radius: 15px;
    backdrop-filter: blur(5px);
    margin-top: 50px;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.1);
}

/* Typography */
h1, h2, h3, h4, h5, h6 {
    color: #ffffff !important;
    font-family: 'Segoe UI', sans-serif;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
}
p, span, label {
    color: #e0e0e0 !important; 
    font-weight: 500;
}

/* Metrics */
div[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    padding: 10px;
    border-radius: 10px;
    text-align: center;
    transition: transform 0.2s;
}
div[data-testid="stMetric"]:hover {
    transform: scale(1.05);
    background: rgba(255, 255, 255, 0.1);
}
div[data-testid="stMetricLabel"] { color: #cfcfcf !important; }
div[data-testid="stMetricValue"] { color: #ffeb3b !important; text-shadow: 0 0 10px rgba(255, 235, 59, 0.5); }

/* Buttons */
.stButton > button {
    background: linear-gradient(45deg, #FF4B2B, #FF416C);
    color: white;
    border: none;
    border-radius: 25px;
    font-size: 18px;
    font-weight: bold;
    padding: 10px 30px;
    box-shadow: 0 5px 15px rgba(255, 75, 43, 0.4);
    transition: all 0.3s ease;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 7px 20px rgba(255, 75, 43, 0.6);
}

/* Inputs */
.stSelectbox, .stNumberInput {
    background-color: transparent !important;
}

/* Custom Header */
.team-vs-header {
    display: flex;
    justify-content: space-around;
    align-items: center;
    background: linear-gradient(90deg, rgba(0, 0, 0, 0.0) 0%, rgba(255,255,255,0.1) 50%, rgba(0,0,0,0.0) 100%);
    padding: 20px;
    border-radius: 20px;
    margin-bottom: 30px;
}
.team-logo-img {
    width: 80px;
    height: 80px;
    object-fit: contain;
    filter: drop-shadow(0 0 10px rgba(255,255,255,0.3));
    transition: transform 0.3s;
}
.team-logo-img:hover { transform: scale(1.2); }
.vs-badge {
    font-size: 32px;
    font-weight: 900;
    color: #ffffff;
    padding: 0 20px;
    text-shadow: 0 0 20px #ff0000;
}
</style>
""", unsafe_allow_html=True)

# --- Team Data ---
TEAM_CONFIG = {
    "Chennai Super Kings": {"logo": "https://upload.wikimedia.org/wikipedia/en/thumb/2/2b/Chennai_Super_Kings_Logo.svg/1200px-Chennai_Super_Kings_Logo.svg.png"},
    "Delhi Daredevils": {"logo": "https://upload.wikimedia.org/wikipedia/en/thumb/f/f5/Delhi_Daredevils_Logo.svg/1200px-Delhi_Daredevils_Logo.svg.png"},
    "Kings XI Punjab": {"logo": "https://upload.wikimedia.org/wikipedia/en/thumb/8/81/Kings_XI_Punjab_logo_2011.svg/1200px-Kings_XI_Punjab_logo_2011.svg.png"},
    "Kolkata Knight Riders": {"logo": "https://upload.wikimedia.org/wikipedia/en/thumb/4/4c/Kolkata_Knight_Riders_Logo.svg/1200px-Kolkata_Knight_Riders_Logo.svg.png"},
    "Mumbai Indians": {"logo": "https://upload.wikimedia.org/wikipedia/en/thumb/c/cd/Mumbai_Indians_Logo.svg/1200px-Mumbai_Indians_Logo.svg.png"},
    "Rajasthan Royals": {"logo": "https://upload.wikimedia.org/wikipedia/en/thumb/6/60/Rajasthan_Royals_Logo.svg/1200px-Rajasthan_Royals_Logo.svg.png"},
    "Royal Challengers Bangalore": {"logo": "https://upload.wikimedia.org/wikipedia/en/thumb/2/2a/Royal_Challengers_Bangalore_2020.svg/1200px-Royal_Challengers_Bangalore_2020.svg.png"},
    "Sunrisers Hyderabad": {"logo": "https://upload.wikimedia.org/wikipedia/en/thumb/8/81/Sunrisers_Hyderabad.svg/1200px-Sunrisers_Hyderabad.svg.png"}
}
DEFAULT_LOGO = "https://cdn-icons-png.flaticon.com/512/1099/1099672.png"

# --- Load Model (Simpler Random Forest) ---
@st.cache_resource
def load_resources():
    try:
        # Load logic specific to the new Scikit-Learn pipeline
        model = pickle.load(open("ipl_score_model.pkl", "rb"))
        team_encoder = pickle.load(open("team_encoder.pkl", "rb"))
        venue_encoder = pickle.load(open("venue_encoder.pkl", "rb"))
        scaler = pickle.load(open("scaler.pkl", "rb"))
        return model, team_encoder, venue_encoder, scaler
    except:
        return None, None, None, None

model, team_encoder, venue_encoder, scaler = load_resources()

if not model:
    st.error("⚠️ **System Not Ready**: Please run `python ipl_score_prediction.py` to train the simplified model first.")
    st.stop()

# --- UI Layout ---
st.title("🏆 AI-Based IPL Score Prediction System")
st.markdown("### ⚡ Fast & Accurate Match Forecasting")

teams = team_encoder.classes_.tolist()
teams = [t for t in teams if t in TEAM_CONFIG] # Filter valid teams
venues = venue_encoder.classes_.tolist()

# 1. Match Setup
with st.container():
    c1, c2, c3 = st.columns([1,1,2])
    with c1:
        bat_team = st.selectbox("Batting Team", teams)
    with c2:
        bowl_team = st.selectbox("Bowling Team", teams, index=1)
    with c3:
        venue = st.selectbox("Stadium", venues)

# VS Badge Visual
bat_logo = TEAM_CONFIG.get(bat_team, {}).get("logo", DEFAULT_LOGO)
bowl_logo = TEAM_CONFIG.get(bowl_team, {}).get("logo", DEFAULT_LOGO)

st.markdown(f"""
    <div class="team-vs-header">
        <div style="text-align:center">
            <img src="{bat_logo}" class="team-logo-img">
            <h4 style="margin-top:10px">{bat_team}</h4>
        </div>
        <div class="vs-badge">VS</div>
        <div style="text-align:center">
            <img src="{bowl_logo}" class="team-logo-img">
            <h4 style="margin-top:10px">{bowl_team}</h4>
        </div>
    </div>
""", unsafe_allow_html=True)

# 2. Score Input
st.markdown("##### 🏏 Live Match Status")
col1, col2, col3, col4 = st.columns(4)
runs = col1.number_input("Current Runs", 0, 300, step=1)
wickets = col2.number_input("Wickets Down", 0, 9, step=1)
overs = col3.number_input("Overs Done", 0, 19, step=1)
balls = col4.number_input("Balls (This Over)", 0, 5, step=1)

total_overs = overs + (balls/6)

# 3. Recent Stats
st.markdown("##### 🔥 Recent Form (Last 5 Overs)")
if total_overs >= 5:
    rc1, rc2 = st.columns(2)
    runs_last_5 = rc1.number_input("Runs Scored", 0, 100, step=1)
    wickets_last_5 = rc2.number_input("Wickets Lost", 0, 10, step=1)
else:
    st.info("Match is in early stages (< 5 overs). Recent form ignored.")
    runs_last_5 = 0
    wickets_last_5 = 0

# 4. Prediction Trigger
st.write("")
if st.button("🔮 PREDICT FINAL SCORE"):
    # Validation
    if bat_team == bowl_team:
        st.error("🚫 Teams must be different!")
        st.stop()
    if total_overs == 0:
        st.error("🚫 Match hasn't started yet!")
        st.stop()
        
    # Processing
    input_data = np.array([[
        team_encoder.transform([bat_team])[0],
        team_encoder.transform([bowl_team])[0],
        venue_encoder.transform([venue])[0],
        runs, wickets, total_overs, runs_last_5, wickets_last_5
    ]])
    
    # Scale & Predict
    input_scaled = scaler.transform(input_data)
    prediction = int(model.predict(input_scaled)[0])
    
    # Range
    lower = prediction - 5
    upper = prediction + 5
    
    # Display Result
    st.markdown("---")
    st.markdown(f"""
        <div style="text-align: center; background: rgba(0,255,0,0.1); padding: 20px; border-radius: 15px; border: 1px solid #4CAF50;">
            <h2 style="color: #4CAF50 !important; margin:0;">PROJECTED SCORE</h2>
            <h1 style="font-size: 80px; margin: 0; color: #fff !important; text-shadow: 0 0 20px #4CAF50;">{prediction}</h1>
            <h4 style="color: #bbb !important;">Expected Range: {lower} - {upper}</h4>
        </div>
    """, unsafe_allow_html=True)
    
    # Metrics
    crr = runs/total_overs
    proj_crr = int(crr * 20)
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Current Run Rate", f"{crr:.2f}")
    m2.metric("Projected (at CRR)", f"{proj_crr}")
    m3.metric("RPO Required (to reach {prediction})", f"{(prediction-runs)/(20-total_overs):.2f}" if total_overs < 20 else "N/A")

st.markdown("<br><center style='color: #666; font-size: 12px;'>Simple Random Forest Model • Scikit-Learn</center>", unsafe_allow_html=True)
