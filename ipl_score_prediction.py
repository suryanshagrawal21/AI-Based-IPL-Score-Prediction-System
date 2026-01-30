# ipl_score_prediction.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pickle

# --- Step 1: Load Data ---
df = pd.read_csv("ipl.csv")

# --- Step 2: Clean Data ---
columns_to_drop = ["date", "mid", "batsman", "bowler", "striker", "non-striker"]
df.drop(columns=columns_to_drop, errors="ignore", inplace=True)

# --- Step 3: Encode Teams ---
all_teams = pd.concat([df["bat_team"], df["bowl_team"]]).unique()
team_encoder = LabelEncoder()
team_encoder.fit(all_teams)

df["bat_team"] = team_encoder.transform(df["bat_team"])
df["bowl_team"] = team_encoder.transform(df["bowl_team"])

# --- Step 4: Encode Venues ---
venue_encoder = LabelEncoder()
df["venue"] = venue_encoder.fit_transform(df["venue"])

# --- Step 5: Save Encoders ---
with open("team_encoder.pkl", "wb") as f:
    pickle.dump(team_encoder, f)

with open("venue_encoder.pkl", "wb") as f:
    pickle.dump(venue_encoder, f)

# --- Step 6: Prepare Features ---
feature_columns = ["bat_team", "bowl_team", "venue", "runs", "wickets", "overs", "runs_last_5", "wickets_last_5"]
X = df[feature_columns]
y = df["total"]

# --- Step 7: Scale Features ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# --- Step 8: Train Model (Random Forest) ---
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print("🚀 Training Random Forest Model... (This is faster and more accurate for this data)")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Step 9: Evaluate ---
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"✅ Training Complete. RMSE: {rmse:.2f}")

# --- Step 10: Save Model ---
with open("ipl_score_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("💾 Model saved as 'ipl_score_model.pkl'")
