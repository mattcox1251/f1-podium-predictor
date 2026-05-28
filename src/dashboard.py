"""
dashboard.py
------------
Streamlit dashboard for the F1 Podium Predictor.

Pages:
  1. 🏁 Podium Predictor  — select a race, see predicted podium + probabilities
  2. 📊 Model Performance — evaluation metrics, confusion matrix, feature importance
  3. 🔍 Driver Deep Dive  — driver stats, circuit history, rolling form
  4. 📈 Season Analysis   — race-by-race prediction accuracy across the test seasons

Usage:
    streamlit run src/dashboard.py
"""

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="F1 Podium Predictor",
    page_icon="🏎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --f1-red:    #E10600;
    --f1-dark:   #0C0C0C;
    --f1-card:   #1A1A1A;
    --f1-border: #2A2A2A;
    --f1-text:   #FFFFFF;
    --f1-muted:  #888888;
    --f1-gold:   #FFD700;
    --f1-silver: #C0C0C0;
    --f1-bronze: #CD7F32;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: var(--f1-dark);
    color: var(--f1-text);
}

/* Header */
.f1-header {
    background: linear-gradient(135deg, #0C0C0C 0%, #1a0000 50%, #0C0C0C 100%);
    border-bottom: 2px solid var(--f1-red);
    padding: 1.5rem 2rem;
    margin: -1rem -1rem 2rem -1rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.f1-title {
    font-family: 'Orbitron', monospace;
    font-size: 2rem;
    font-weight: 900;
    color: var(--f1-text);
    letter-spacing: 0.05em;
    margin: 0;
}
.f1-title span { color: var(--f1-red); }
.f1-subtitle {
    font-size: 0.8rem;
    color: var(--f1-muted);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin: 0;
}

/* Metric cards */
.metric-card {
    background: var(--f1-card);
    border: 1px solid var(--f1-border);
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.metric-value {
    font-family: 'Orbitron', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: var(--f1-red);
}
.metric-label {
    font-size: 0.75rem;
    color: var(--f1-muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.3rem;
}

/* Podium */
.podium-container {
    display: flex;
    justify-content: center;
    align-items: flex-end;
    gap: 1rem;
    margin: 2rem 0;
}
.podium-block {
    text-align: center;
    border-radius: 8px 8px 0 0;
    padding: 1rem;
    min-width: 140px;
}
.podium-1 { background: linear-gradient(180deg, #2a2400 0%, #1a1600 100%); border: 1px solid var(--f1-gold); height: 180px; }
.podium-2 { background: linear-gradient(180deg, #1a1a1a 0%, #141414 100%); border: 1px solid var(--f1-silver); height: 150px; }
.podium-3 { background: linear-gradient(180deg, #1a1200 0%, #140e00 100%); border: 1px solid var(--f1-bronze); height: 120px; }
.podium-pos { font-family: 'Orbitron', monospace; font-size: 2rem; font-weight: 900; }
.podium-1 .podium-pos { color: var(--f1-gold); }
.podium-2 .podium-pos { color: var(--f1-silver); }
.podium-3 .podium-pos { color: var(--f1-bronze); }
.podium-driver { font-family: 'Orbitron', monospace; font-size: 0.9rem; font-weight: 700; margin: 0.5rem 0 0.3rem; }
.podium-prob { font-size: 0.8rem; color: var(--f1-muted); }

/* Section headers */
.section-header {
    font-family: 'Orbitron', monospace;
    font-size: 0.85rem;
    font-weight: 700;
    color: var(--f1-red);
    text-transform: uppercase;
    letter-spacing: 0.15em;
    border-left: 3px solid var(--f1-red);
    padding-left: 0.75rem;
    margin: 1.5rem 0 1rem;
}

/* Driver tag */
.driver-tag {
    display: inline-block;
    background: var(--f1-card);
    border: 1px solid var(--f1-border);
    border-radius: 4px;
    padding: 0.2rem 0.6rem;
    font-family: 'Orbitron', monospace;
    font-size: 0.75rem;
    margin: 0.2rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #111111;
    border-right: 1px solid var(--f1-border);
}
</style>
""", unsafe_allow_html=True)

# ── Load artifacts ────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    with open("models/best_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_feature_cols():
    with open("models/feature_cols.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/model_dataset.csv")
    df["rolling_avg_3"] = df["rolling_avg_3"].fillna(10)
    df["rolling_avg_5"] = df["rolling_avg_5"].fillna(10)
    return df

@st.cache_data
def load_model_comparison():
    return pd.read_csv("models/model_comparison.csv")

# ── Prediction helper ─────────────────────────────────────────────────────────

def predict_race(model, df: pd.DataFrame, feature_cols: list,
                 season: int, round_no: int) -> pd.DataFrame:
    race = df[(df["season"] == season) & (df["round"] == round_no)].copy()
    if race.empty:
        return pd.DataFrame()
    probs = model.predict_proba(race[feature_cols])[:, 1]
    race["podium_prob"] = probs
    race = race.sort_values("podium_prob", ascending=False).reset_index(drop=True)
    race["predicted_position"] = range(1, len(race) + 1)
    return race

# ── Plotly theme ──────────────────────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#1A1A1A",
    plot_bgcolor="#1A1A1A",
    font=dict(color="#FFFFFF", family="Inter"),
    margin=dict(l=40, r=40, t=50, b=40),
)

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="f1-header">
    <div>
        <p class="f1-title">F1 <span>PODIUM</span> PREDICTOR</p>
        <p class="f1-subtitle">Machine Learning · 2019–2024 · Random Forest</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────

try:
    model        = load_model()
    feature_cols = load_feature_cols()
    df           = load_data()
    comparison   = load_model_comparison()
except FileNotFoundError as e:
    st.error(f"Missing file: {e}. Make sure you've run data_collection.py, feature_engineering.py, and model.py first.")
    st.stop()

# ── Sidebar navigation ────────────────────────────────────────────────────────

st.sidebar.markdown("""
<p style='font-family: Orbitron, monospace; font-size: 0.8rem; color: #E10600;
   letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 1rem;'>
   Navigation
</p>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigation",
    ["🏁 Podium Predictor", "📊 Model Performance", "🔍 Driver Deep Dive", "📈 Season Analysis"],
    label_visibility="collapsed"
)

st.sidebar.divider()
st.sidebar.markdown("""
<p style='font-size: 0.7rem; color: #555; margin-top: 1rem;'>
Data: Jolpica API · 2019–2024<br>
Model: Random Forest · ROC-AUC 0.943<br>
Podium Accuracy: 68.1%
</p>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PODIUM PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════

if page == "🏁 Podium Predictor":

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown('<p class="section-header">Select Race</p>', unsafe_allow_html=True)

        seasons = sorted(df["season"].unique(), reverse=True)
        season  = st.selectbox("Season", seasons)

        rounds = df[df["season"] == season][["round", "race_name"]].drop_duplicates().sort_values("round")
        round_labels = {row["round"]: f"R{row['round']} — {row['race_name']}" for _, row in rounds.iterrows()}
        round_no = st.selectbox("Race", options=list(round_labels.keys()), format_func=lambda x: round_labels[x])

    race_df = predict_race(model, df, feature_cols, season, round_no)

    if race_df.empty:
        st.warning("No data found for this race.")
        st.stop()

    top3    = race_df.head(3)
    actual  = race_df[race_df["podium"] == 1]["driver_id"].tolist()
    correct = len(set(top3["driver_id"].tolist()) & set(actual))

    with col2:
        st.markdown('<p class="section-header">Predicted Podium</p>', unsafe_allow_html=True)

        p1, p2, p3 = top3.iloc[0], top3.iloc[1], top3.iloc[2]

        st.markdown(f"""
        <div class="podium-container">
            <div class="podium-block podium-2">
                <div class="podium-pos">2</div>
                <div class="podium-driver">{p2['driver_code']}</div>
                <div class="podium-prob">{p2['podium_prob']:.1%}</div>
            </div>
            <div class="podium-block podium-1">
                <div class="podium-pos">1</div>
                <div class="podium-driver">{p1['driver_code']}</div>
                <div class="podium-prob">{p1['podium_prob']:.1%}</div>
            </div>
            <div class="podium-block podium-3">
                <div class="podium-pos">3</div>
                <div class="podium-driver">{p3['driver_code']}</div>
                <div class="podium-prob">{p3['podium_prob']:.1%}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if actual:
            result_color = "#00C851" if correct == 3 else "#FFD700" if correct >= 1 else "#E10600"
            st.markdown(f"""
            <div style='text-align:center; padding: 0.5rem;
                background: #1A1A1A; border-radius: 6px; border: 1px solid #2A2A2A;'>
                <span style='color:{result_color}; font-family: Orbitron, monospace;
                    font-size: 1.1rem; font-weight: 700;'>
                    {correct}/3 Correct
                </span>
                <span style='color:#888; font-size: 0.8rem; margin-left: 0.5rem;'>
                    Actual podium: {' · '.join([r.upper() for r in actual])}
                </span>
            </div>
            """, unsafe_allow_html=True)

    # Full probability chart
    st.markdown('<p class="section-header">All Driver Probabilities</p>', unsafe_allow_html=True)

    race_plot = race_df.sort_values("podium_prob")
    colors = ["#E10600" if d in actual else "#2A2A2A" for d in race_plot["driver_id"]]

    fig = go.Figure(go.Bar(
        x=race_plot["podium_prob"],
        y=race_plot["driver_code"],
        orientation="h",
        marker_color=colors,
        text=[f"{p:.1%}" for p in race_plot["podium_prob"]],
        textposition="outside",
        textfont=dict(size=11),
    ))
    fig.add_vline(x=0.5, line_dash="dash", line_color="#555")
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=500,
        xaxis=dict(title="Predicted Podium Probability", tickformat=".0%", gridcolor="#2A2A2A"),
        yaxis=dict(title=""),
        showlegend=False,
    )
    st.plotly_chart(fig, width='stretch')
    st.caption("🔴 Red bars = actual podium finishers")

    # Key race features table
    st.markdown('<p class="section-header">Race Features</p>', unsafe_allow_html=True)
    feature_display = race_df[[
        "driver_code", "quali_position", "driver_champ_pos_pre",
        "con_champ_pos_pre", "rolling_avg_3", "circuit_avg_finish",
        "num_pit_stops", "podium_prob"
    ]].copy()
    feature_display.columns = [
        "Driver", "Quali Pos", "Driver Champ Pos",
        "Constructor Pos", "Form (Last 3)", "Circuit Avg",
        "Pit Stops", "Podium Prob"
    ]
    feature_display["Podium Prob"] = feature_display["Podium Prob"].map("{:.1%}".format)
    feature_display["Form (Last 3)"] = feature_display["Form (Last 3)"].map("{:.1f}".format)
    feature_display["Circuit Avg"] = feature_display["Circuit Avg"].map("{:.1f}".format)
    st.dataframe(feature_display, width='stretch', hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Model Performance":

    st.markdown('<p class="section-header">Model Comparison</p>', unsafe_allow_html=True)

    # Metric cards
    best = comparison.loc[comparison["podium_accuracy"].idxmax()]
    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        ("ROC-AUC", f"{best['roc_auc']:.3f}"),
        ("Avg Precision", f"{best['avg_precision']:.3f}"),
        ("Podium Accuracy", f"{best['podium_accuracy']:.1%}"),
        ("Best Model", best['model'].split()[0]),
    ]
    for col, (label, value) in zip([c1, c2, c3, c4], metrics):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # Model comparison bar chart
    fig = go.Figure()
    metric_names  = ["ROC-AUC", "Avg Precision", "Podium Accuracy"]
    metric_cols   = ["roc_auc", "avg_precision", "podium_accuracy"]
    bar_colors    = ["#E10600", "#FF6B35", "#FFD700"]

    for metric, col, color in zip(metric_names, metric_cols, bar_colors):
        fig.add_trace(go.Bar(
            name=metric,
            x=comparison["model"],
            y=comparison[col],
            marker_color=color,
            text=[f"{v:.3f}" for v in comparison[col]],
            textposition="outside",
        ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        barmode="group",
        height=380,
        yaxis=dict(range=[0, 1.1], gridcolor="#2A2A2A", title="Score"),
        xaxis=dict(title=""),
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig, width='stretch')

    # Feature importance
    st.markdown('<p class="section-header">Feature Importance (Random Forest)</p>', unsafe_allow_html=True)

    clf = model.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        fi = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=True)
        fig2 = go.Figure(go.Bar(
            x=fi.values,
            y=fi.index,
            orientation="h",
            marker_color="#E10600",
            marker_line_color="#FF4444",
            marker_line_width=1,
        ))
        fig2.update_layout(
            **PLOTLY_LAYOUT,
            height=480,
            xaxis=dict(title="Importance", gridcolor="#2A2A2A"),
            yaxis=dict(title=""),
        )
        st.plotly_chart(fig2, width='stretch')

    # Confusion matrix
    st.markdown('<p class="section-header">Confusion Matrix (Test Set 2023–2024)</p>', unsafe_allow_html=True)

    test_df_page = df[df["season"].isin([2023, 2024])].copy()
    y_pred  = model.predict(test_df_page[feature_cols])
    y_true  = test_df_page["podium"].values

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)

    fig3 = go.Figure(go.Heatmap(
        z=cm,
        x=["Predicted: No Podium", "Predicted: Podium"],
        y=["Actual: No Podium", "Actual: Podium"],
        text=cm,
        texttemplate="%{text}",
        textfont=dict(size=20),
        colorscale=[[0, "#1A1A1A"], [1, "#E10600"]],
        showscale=False,
    ))
    fig3.update_layout(**PLOTLY_LAYOUT, height=300)
    st.plotly_chart(fig3, width='stretch')


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — DRIVER DEEP DIVE
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔍 Driver Deep Dive":

    st.markdown('<p class="section-header">Select Driver</p>', unsafe_allow_html=True)

    drivers = sorted(df["driver_code"].dropna().unique())
    driver  = st.selectbox("Driver", drivers)

    driver_df = df[df["driver_code"] == driver].sort_values(["season", "round"])

    if driver_df.empty:
        st.warning("No data for this driver.")
        st.stop()

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    stats = [
        ("Races", str(len(driver_df))),
        ("Podiums", str(driver_df["podium"].sum())),
        ("Podium Rate", f"{driver_df['podium'].mean():.1%}"),
        ("Avg Finish", f"{driver_df['finish_position'].mean():.1f}"),
    ]
    for col, (label, value) in zip([c1, c2, c3, c4], stats):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # Rolling form chart
    st.markdown('<p class="section-header">Finishing Position Over Time</p>', unsafe_allow_html=True)

    driver_df["race_label"] = driver_df["season"].astype(str) + " R" + driver_df["round"].astype(str)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=driver_df["race_label"],
        y=driver_df["finish_position"],
        mode="lines+markers",
        name="Finish Position",
        line=dict(color="#E10600", width=2),
        marker=dict(
            color=["#FFD700" if p else "#E10600" for p in driver_df["podium"]],
            size=8,
        ),
    ))
    fig.add_trace(go.Scatter(
        x=driver_df["race_label"],
        y=driver_df["rolling_avg_5"],
        mode="lines",
        name="5-Race Rolling Avg",
        line=dict(color="#888888", width=1.5, dash="dot"),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=350,
        yaxis=dict(title="Finish Position", autorange="reversed", gridcolor="#2A2A2A"),
        xaxis=dict(title="", tickangle=45, tickfont=dict(size=9)),
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig, width='stretch')
    st.caption("🟡 Gold markers = podium finishes")

    # Circuit history
    st.markdown('<p class="section-header">Circuit History</p>', unsafe_allow_html=True)

    circuit_stats = (
        driver_df.groupby("circuit_id")
        .agg(
            races=("finish_position", "count"),
            avg_finish=("finish_position", "mean"),
            podiums=("podium", "sum"),
        )
        .reset_index()
        .sort_values("avg_finish")
    )

    fig2 = go.Figure(go.Bar(
        x=circuit_stats["circuit_id"],
        y=circuit_stats["avg_finish"],
        marker_color=circuit_stats["avg_finish"],
        marker_colorscale=[[0, "#E10600"], [0.5, "#FF8C00"], [1, "#2A2A2A"]],
        text=circuit_stats["podiums"].apply(lambda x: f"🏆{x}" if x > 0 else ""),
        textposition="outside",
    ))
    fig2.update_layout(
        **PLOTLY_LAYOUT,
        height=350,
        yaxis=dict(title="Avg Finish Position", autorange="reversed", gridcolor="#2A2A2A"),
        xaxis=dict(title="Circuit", tickangle=45, tickfont=dict(size=9)),
    )
    st.plotly_chart(fig2, width='stretch')
    st.caption("Lower = better. 🏆 = podium finishes at that circuit")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — SEASON ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📈 Season Analysis":

    st.markdown('<p class="section-header">Race-by-Race Prediction Accuracy</p>', unsafe_allow_html=True)

    test_df_sa = df[df["season"].isin([2023, 2024])].copy()
    probs = model.predict_proba(test_df_sa[feature_cols])[:, 1]
    test_df_sa["podium_prob"] = probs

    race_results = []
    for (season, round_no), race in test_df_sa.groupby(["season", "round"]):
        predicted = set(race.nlargest(3, "podium_prob")["driver_id"].values)
        actual    = set(race[race["podium"] == 1]["driver_id"].values)
        correct   = len(predicted & actual)
        race_results.append({
            "season":    season,
            "round":     round_no,
            "race_name": race["race_name"].iloc[0],
            "correct":   correct,
            "label":     f"{int(season)} R{int(round_no)}",
        })

    race_acc = pd.DataFrame(race_results)
    overall  = race_acc["correct"].sum() / (len(race_acc) * 3)

    # Summary stats
    c1, c2, c3, c4 = st.columns(4)
    summary_stats = [
        ("Overall Accuracy", f"{overall:.1%}"),
        ("Perfect Predictions", str((race_acc["correct"] == 3).sum())),
        ("Races Analyzed",  str(len(race_acc))),
        ("Avg Correct/Race", f"{race_acc['correct'].mean():.1f}/3"),
    ]
    for col, (label, value) in zip([c1, c2, c3, c4], summary_stats):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # Race-by-race bar chart
    color_map = {0: "#E10600", 1: "#FF8C00", 2: "#FFD700", 3: "#00C851"}
    bar_colors = [color_map[c] for c in race_acc["correct"]]

    fig = go.Figure(go.Bar(
        x=race_acc["label"],
        y=race_acc["correct"],
        marker_color=bar_colors,
        text=race_acc["correct"].apply(lambda x: f"{x}/3"),
        textposition="outside",
    ))
    fig.add_hline(
        y=race_acc["correct"].mean(),
        line_dash="dash", line_color="#888",
        annotation_text=f"Avg: {race_acc['correct'].mean():.1f}",
        annotation_position="right",
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=420,
        yaxis=dict(title="Correct Podium Picks", range=[0, 4], dtick=1, gridcolor="#2A2A2A"),
        xaxis=dict(title="", tickangle=45, tickfont=dict(size=9)),
    )
    st.plotly_chart(fig, width='stretch')
    st.caption("🟢 3/3 Perfect · 🟡 2/3 · 🟠 1/3 · 🔴 0/3")

    # Accuracy by circuit
    st.markdown('<p class="section-header">Accuracy by Circuit</p>', unsafe_allow_html=True)

    circuit_acc = (
        race_acc.merge(
            test_df_sa[["season", "round", "circuit_id"]].drop_duplicates(),
            on=["season", "round"]
        )
        .groupby("circuit_id")["correct"]
        .mean()
        .sort_values(ascending=True)
        .reset_index()
    )

    fig2 = go.Figure(go.Bar(
        x=circuit_acc["correct"],
        y=circuit_acc["circuit_id"],
        orientation="h",
        marker_color=circuit_acc["correct"],
        marker_colorscale=[[0, "#E10600"], [0.5, "#FFD700"], [1, "#00C851"]],
        text=[f"{v:.1f}/3" for v in circuit_acc["correct"]],
        textposition="outside",
    ))
    fig2.update_layout(
        **PLOTLY_LAYOUT,
        height=500,
        xaxis=dict(title="Avg Correct Podium Picks", range=[0, 4], gridcolor="#2A2A2A"),
        yaxis=dict(title=""),
    )
    st.plotly_chart(fig2, width='stretch')