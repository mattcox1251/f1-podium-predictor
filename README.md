# 🏎 F1 Podium Predictor

A machine learning project that predicts Formula 1 podium finishes using historical race data, engineered features, and an optimised ensemble model — with an interactive Streamlit dashboard including live 2026 season data.

---

## 🎯 Results

| Model | ROC-AUC | Avg Precision | Podium Accuracy |
|---|---|---|---|
| Logistic Regression | 0.947 | 0.790 | 70.8% |
| Random Forest (tuned) | 0.957 | 0.788 | 72.2% |
| **XGBoost (tuned)** | **0.958** | **0.796** | **72.9%** |
| Stacked Ensemble | 0.949 | 0.791 | 72.9% |

**Podium Accuracy** = for each race, pick the top 3 drivers by predicted probability and measure how many actual podium finishers were correctly identified. Random chance ≈ 45%.

---

## 📦 Project Structure

```
f1-podium-predictor/
├── src/
│   ├── data_collection.py       # Fetch data from Jolpica API
│   ├── feature_engineering.py   # Build model-ready dataset
│   ├── model.py                 # Train + evaluate models
│   └── dashboard.py             # Streamlit dashboard
├── data/
│   ├── raw/                     # Raw CSVs from API
│   └── processed/               # Engineered feature dataset + plots
├── models/                      # Saved model + feature list
├── notebooks/
│   └── eda.ipynb                # Exploratory data analysis
├── .streamlit/
│   └── config.toml              # Streamlit theme config
└── requirements.txt
```

---

## 🛠 Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/f1-podium-predictor
cd f1-podium-predictor

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Running the Pipeline

Run these scripts in order:

```bash
# 1. Collect data (2019–2026, ~25 min due to API rate limits)
python src/data_collection.py

# 2. Engineer features
python src/feature_engineering.py

# 3. Train and evaluate models (~15 min for Optuna tuning)
python src/model.py

# 4. Launch the dashboard
streamlit run src/dashboard.py
```

---

## 📊 Features

21 engineered features covering:

| Category | Features |
|---|---|
| Qualifying | `quali_position`, `front_row` |
| Championship form | `driver_champ_pos_pre`, `driver_points_pre`, `driver_wins_pre` |
| Constructor | `con_champ_pos_pre`, `con_points_pre`, `con_wins_pre`, `con_pts_momentum` |
| Rolling form | `rolling_avg_3`, `rolling_avg_5`, `points_momentum` |
| Reliability | `dnf_rate_5` |
| Teammate | `teammate_gap_3` |
| Circuit | `circuit_avg_finish`, `circuit_win_rate` |
| Strategy | `num_pit_stops`, `fastest_lap_rank`, `avg_speed_kph` |
| Other | `home_race`, `grid_position` |

**Key design decision:** all rolling and standings features use `shift(1)` to avoid data leakage — the model only sees information available before each race.

---

## 🧠 Model

- **Best model:** XGBoost, hyperparameter-tuned with Optuna (40 trials)
- **Validation:** TimeSeriesSplit cross-validation (temporal ordering preserved)
- **Class imbalance:** handled via `scale_pos_weight` (tuned by Optuna)
- **Ensemble:** Stacked classifier (XGBoost + Random Forest + Logistic Regression → Logistic meta-learner)
- **Explainability:** SHAP values via TreeExplainer (beeswarm, waterfall, dependence plots)

---

## 📱 Dashboard Pages

| Page | Description |
|---|---|
| 🏁 Podium Predictor | Pick any race, see predicted podium + all driver probabilities |
| 🔴 2026 Live Season | Live standings, race winners, next race prediction |
| 📊 Model Performance | Metrics, feature importance, confusion matrix |
| 🔍 Driver Deep Dive | Per-driver stats, form over time, circuit history |
| 📈 Season Analysis | Race-by-race and circuit-by-circuit prediction accuracy |
| 🧠 SHAP Explainability | Global beeswarm, single prediction waterfall, dependence plots |

---

## 🌐 Deployment

Deployed on Streamlit Community Cloud: **[ADD YOUR URL HERE]**

To deploy your own:
1. Push this repo to GitHub (ensure `data/` and `models/` are committed)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set main file path to `src/dashboard.py`
5. Deploy

---

## 📚 Data Source

All data from the [Jolpica F1 API](https://github.com/jolpica/jolpica-f1) — a community-maintained successor to the Ergast API, covering F1 data from 1950 to present.

---

## 🎓 Built As

A learning project before starting a Data Science program. Covers the full DS workflow:
data collection → cleaning → EDA → feature engineering → modelling → evaluation → deployment.

**Stack:** Python · pandas · scikit-learn · XGBoost · Optuna · SHAP · Streamlit · Plotly