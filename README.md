# 🏏 AI-Based IPL Score Prediction System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Framework](https://img.shields.io/badge/Framework-Streamlit-red)
![ML Algorithm](https://img.shields.io/badge/Algorithm-Random%20Forest-green)
![License](https://img.shields.io/badge/License-MIT-green)

This project is a **Machine Learning-based web application** designed to predict the total runs a batting team is likely to score in an IPL (Indian Premier League) match. 

It uses a **Random Forest Regressor** to analyze historical match data and predict accurate scores. The application features a **Premium Glassmorphism UI** built with Streamlit for a modern user experience.

---

## 🚀 Key Features

-   **Advanced Machine Learning**: Uses `RandomForestRegressor` from **Scikit-Learn** for high accuracy and fast performance.
-   **Premium UI Design**: Features a **Cricket Stadium Background** and **Glassmorphism** (transparent glass effect) styling.
-   **Real-time Prediction**: Instantly forecasts scores based on current match conditions (Runs, Wickets, Overs).
-   **Dynamic Visuals**: Auto-updating team logos and a specialized "VS" badge interactively.
-   **Smart Validation**: Prevents invalid inputs (e.g., same team selection, excessive wickets).

---

## 🛠️ Technology Stack

-   **Frontend**: Streamlit (Python Web Framework)
-   **Machine Learning**: Scikit-Learn (Random Forest)
-   **Data Processing**: Pandas, NumPy
-   **Persistence**: Pickle (Model serialization)

---

## 📂 Project Structure

```bash
IPL-Score-Prediction/
├── app.py                    # Main Streamlit App (Glassmorphism UI)
├── ipl_score_prediction.py   # Training Script (Random Forest)
├── ipl.csv                   # Dataset (Matches up to 2017)
├── requirements.txt          # Lightweight Dependencies
├── ipl_score_model.pkl       # Trained Model Artifact
├── team_encoder.pkl          # Team Label Encoder
├── venue_encoder.pkl         # Venue Label Encoder
├── scaler.pkl                # Feature Scaler
└── README.md                 # Project Documentation
```

---

## 🔧 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/suryanshagrawal21/AI-Based-IPL-Score-Prediction-System.git
cd AI-Based-IPL-Score-Prediction-System
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Model (Optional)
The model represents historical patterns. To retrain:
```bash
python ipl_score_prediction.py
```

### 4. Run the Application
Start the premium web interface:
```bash
streamlit run app.py
```
Access the app at: `http://localhost:8501`

---

## 📊 Dataset & Model Details

The model is trained on the `ipl.csv` dataset containing ball-by-ball IPL match data from seasons up to **2017**.

-   **Algorithm**: Random Forest Regressor (n_estimators=100)
-   **Features Used**: Batting Team, Bowling Team, Venue, Current Runs, Wickets, Overs, Runs in Last 5 Overs, Wickets in Last 5 Overs.
-   **Note**: The dataset includes legacy teams (Delhi Daredevils, Kings XI Punjab). Predictions reflect historical scoring patterns.

---

## 👨‍💻 Author

**Suryansh Agrawal**
-   GitHub: [suryanshagrawal21](https://github.com/suryanshagrawal21)

---

## 📜 License

This project is licensed under the MIT License.
