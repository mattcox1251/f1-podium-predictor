"""
dashboard.py  (v2)
------------------
Streamlit F1 Podium Predictor dashboard with 5 pages:
  1. 🏁  Podium Predictor    — historical race predictions
  2. 🔴  2026 Live Season    — current season live data + next race prediction
  3. 📊  Model Performance   — metrics, feature importance, confusion matrix
  4. 🔍  Driver Deep Dive    — per-driver stats and circuit history
  5. 📈  Season Analysis     — race-by-race accuracy across test seasons

Usage:
    streamlit run src/dashboard.py
"""

import os
import pickle
import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="F1 Podium Predictor",
    page_icon="🏎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;500;600&display=swap');
:root {
    --red:#E10600; --dark:#0C0C0C; --card:#1A1A1A; --border:#2A2A2A;
    --gold:#FFD700; --silver:#C0C0C0; --bronze:#CD7F32; --muted:#888;
}
html,body,[class*="css"]{font-family:'Inter',sans-serif;background:var(--dark);color:#fff;}
.f1-header{background:linear-gradient(135deg,#0C0C0C,#1a0000,#0C0C0C);
  border-bottom:2px solid var(--red);padding:1.2rem 2rem;margin:-1rem -1rem 1.5rem -1rem;}
.f1-title{font-family:'Orbitron',monospace;font-size:1.8rem;font-weight:900;margin:0;}
.f1-title span{color:var(--red);}
.f1-sub{font-size:.75rem;color:var(--muted);letter-spacing:.2em;text-transform:uppercase;margin:0;}
.metric-card{background:var(--card);border:1px solid var(--border);border-radius:8px;
  padding:1rem 1.2rem;text-align:center;}
.metric-value{font-family:'Orbitron',monospace;font-size:1.8rem;font-weight:700;color:var(--red);}
.metric-label{font-size:.72rem;color:var(--muted);text-transform:uppercase;letter-spacing:.1em;margin-top:.2rem;}
.sec{font-family:'Orbitron',monospace;font-size:.8rem;font-weight:700;color:var(--red);
  text-transform:uppercase;letter-spacing:.15em;border-left:3px solid var(--red);
  padding-left:.7rem;margin:1.2rem 0 .8rem;}
.podium-wrap{display:flex;justify-content:center;align-items:flex-end;gap:1rem;margin:1.5rem 0;}
.pb{text-align:center;border-radius:8px 8px 0 0;padding:1rem;min-width:130px;}
.p1{background:linear-gradient(180deg,#2a2400,#1a1600);border:1px solid var(--gold);height:170px;}
.p2{background:linear-gradient(180deg,#1a1a1a,#141414);border:1px solid var(--silver);height:140px;}
.p3{background:linear-gradient(180deg,#1a1200,#140e00);border:1px solid var(--bronze);height:110px;}
.ppos{font-family:'Orbitron',monospace;font-size:1.8rem;font-weight:900;}
.p1 .ppos{color:var(--gold);} .p2 .ppos{color:var(--silver);} .p3 .ppos{color:var(--bronze);}
.pdrv{font-family:'Orbitron',monospace;font-size:.85rem;font-weight:700;margin:.4rem 0 .2rem;}
.pprob{font-size:.75rem;color:var(--muted);}
.live-badge{display:inline-block;background:#E10600;color:#fff;font-size:.65rem;
  font-weight:700;padding:.15rem .5rem;border-radius:3px;letter-spacing:.1em;
  text-transform:uppercase;margin-left:.5rem;vertical-align:middle;}
section[data-testid="stSidebar"]{background:#111;border-right:1px solid var(--border);}
</style>
""", unsafe_allow_html=True)

# ── Loaders ───────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    with open("models/best_model.pkl", "rb") as f: return pickle.load(f)

@st.cache_resource
def load_feature_cols():
    with open("models/feature_cols.pkl", "rb") as f: return pickle.load(f)

@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/model_dataset.csv")
    for col in ["rolling_avg_3","rolling_avg_5","points_momentum",
                "dnf_rate_5","teammate_gap_3","con_pts_momentum","circuit_win_rate"]:
        if col in df.columns:
            df[col] = df[col].fillna(0 if "momentum" in col or "dnf" in col
                                      or "gap" in col or "win" in col else 10)
    return df

@st.cache_data
def load_comparison():
    return pd.read_csv("models/model_comparison.csv")

JOLPICA = "https://api.jolpi.ca/ergast/f1"

def jolpica_get(endpoint, limit=100):
    results, offset = [], 0
    while True:
        url = f"{JOLPICA}/{endpoint}.json?limit={limit}&offset={offset}"
        try:
            r = requests.get(url, timeout=8)
            r.raise_for_status()
        except Exception:
            break
        data  = r.json().get("MRData", {})
        total = int(data.get("total", 0))
        table = data.get("RaceTable") or data.get("StandingsTable") or {}
        items = table.get("Races") or table.get("StandingsLists") or []
        results.extend(items)
        offset += limit
        if offset >= total: break
        time.sleep(0.5)
    return results

@st.cache_data(ttl=3600)
def fetch_2026_results():
    races = jolpica_get("2026/results")
    rows = []
    for race in races:
        for r in race.get("Results", []):
            pos = r["position"]
            rows.append({
                "round": int(race["round"]), "race_name": race["raceName"],
                "circuit_id": race["Circuit"]["circuitId"],
                "driver_id": r["Driver"]["driverId"],
                "driver_code": r["Driver"].get("code",""),
                "constructor_id": r["Constructor"]["constructorId"],
                "grid_position": int(r.get("grid",0)),
                "finish_position": int(pos) if pos.isdigit() else None,
                "points": float(r.get("points",0)),
                "status": r.get("status",""),
                "podium": int(pos)<=3 if pos.isdigit() else False,
                "dnf": 0 if r.get("status","")=="Finished" or r.get("status","").startswith("+") else 1,
            })
    return pd.DataFrame(rows)

@st.cache_data(ttl=3600)
def fetch_2026_quali():
    races = jolpica_get("2026/qualifying")
    rows = []
    for race in races:
        for r in race.get("QualifyingResults",[]):
            rows.append({
                "round": int(race["round"]),
                "circuit_id": race["Circuit"]["circuitId"],
                "driver_id": r["Driver"]["driverId"],
                "driver_code": r["Driver"].get("code",""),
                "quali_position": int(r["position"]),
            })
    return pd.DataFrame(rows)

@st.cache_data(ttl=3600)
def fetch_2026_schedule():
    """Get full 2026 schedule including future races."""
    races = jolpica_get("2026")
    rows = []
    for race in races:
        rows.append({
            "round": int(race["round"]),
            "race_name": race["raceName"],
            "circuit_id": race["Circuit"]["circuitId"],
            "circuit_name": race["Circuit"]["circuitName"],
            "country": race["Circuit"]["Location"]["country"],
            "date": race.get("date",""),
        })
    return pd.DataFrame(rows)

# ── Helpers ───────────────────────────────────────────────────────────────────

def predict_race(model, df, feature_cols, season, round_no):
    race = df[(df["season"]==season)&(df["round"]==round_no)].copy()
    if race.empty: return pd.DataFrame()
    probs = model.predict_proba(race[feature_cols])[:,1]
    race["podium_prob"] = probs
    return race.sort_values("podium_prob", ascending=False).reset_index(drop=True)

def podium_html(p1,p2,p3):
    return f"""
    <div class="podium-wrap">
      <div class="pb p2"><div class="ppos">2</div>
        <div class="pdrv">{p2['driver_code']}</div>
        <div class="pprob">{p2['podium_prob']:.1%}</div></div>
      <div class="pb p1"><div class="ppos">1</div>
        <div class="pdrv">{p1['driver_code']}</div>
        <div class="pprob">{p1['podium_prob']:.1%}</div></div>
      <div class="pb p3"><div class="ppos">3</div>
        <div class="pdrv">{p3['driver_code']}</div>
        <div class="pprob">{p3['podium_prob']:.1%}</div></div>
    </div>"""

LAYOUT = dict(paper_bgcolor="#1A1A1A", plot_bgcolor="#1A1A1A",
              font=dict(color="#FFF",family="Inter"),
              margin=dict(l=40,r=40,t=50,b=40))

# ── Load ──────────────────────────────────────────────────────────────────────

try:
    model        = load_model()
    feature_cols = load_feature_cols()
    df           = load_data()
    comparison   = load_comparison()
except FileNotFoundError as e:
    st.error(f"Missing file: {e}. Run the pipeline scripts first.")
    st.stop()

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="f1-header">
  <p class="f1-title">F1 <span>PODIUM</span> PREDICTOR</p>
  <p class="f1-sub">Machine Learning · 2019–2026 · Stacked Ensemble</p>
</div>""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.markdown("""<p style='font-family:Orbitron,monospace;font-size:.8rem;
color:#E10600;letter-spacing:.15em;text-transform:uppercase;'>Navigation</p>""",
unsafe_allow_html=True)

page = st.sidebar.radio("Navigation",
    ["🏁 Podium Predictor","🔴 2026 Live Season",
     "📊 Model Performance","🔍 Driver Deep Dive","📈 Season Analysis",
     "🧠 SHAP Explainability"],
    label_visibility="collapsed")

st.sidebar.divider()
st.sidebar.markdown("""<p style='font-size:.7rem;color:#555;'>
Data: Jolpica API · 2019–2026<br>
Model: Stacked Ensemble<br>
Features: 21 engineered signals
</p>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PODIUM PREDICTOR (historical)
# ══════════════════════════════════════════════════════════════════════════════

if page == "🏁 Podium Predictor":
    col1, col2 = st.columns([1,2])
    with col1:
        st.markdown('<p class="sec">Select Race</p>', unsafe_allow_html=True)
        seasons = sorted(df["season"].unique(), reverse=True)
        season  = st.selectbox("Season", seasons)
        rounds  = (df[df["season"]==season][["round","race_name"]]
                   .drop_duplicates().sort_values("round"))
        rlabels = {r["round"]: f"R{r['round']} — {r['race_name']}"
                   for _,r in rounds.iterrows()}
        round_no = st.selectbox("Race", list(rlabels.keys()), format_func=lambda x: rlabels[x])

    race_df = predict_race(model, df, feature_cols, season, round_no)
    if race_df.empty:
        st.warning("No data for this race.")
        st.stop()

    top3   = race_df.head(3)
    actual = race_df[race_df["podium"]==1]["driver_id"].tolist()
    correct= len(set(top3["driver_id"])&set(actual))

    with col2:
        st.markdown('<p class="sec">Predicted Podium</p>', unsafe_allow_html=True)
        st.markdown(podium_html(top3.iloc[0], top3.iloc[1], top3.iloc[2]),
                    unsafe_allow_html=True)
        if actual:
            clr = "#00C851" if correct==3 else "#FFD700" if correct>=1 else "#E10600"
            st.markdown(f"""<div style='text-align:center;padding:.5rem;background:#1A1A1A;
            border-radius:6px;border:1px solid #2A2A2A;'>
            <span style='color:{clr};font-family:Orbitron,monospace;font-size:1.1rem;font-weight:700;'>
            {correct}/3 Correct</span>
            <span style='color:#888;font-size:.8rem;margin-left:.5rem;'>
            Actual: {' · '.join([r.upper() for r in actual])}</span></div>""",
            unsafe_allow_html=True)

    st.markdown('<p class="sec">All Driver Probabilities</p>', unsafe_allow_html=True)
    rp = race_df.sort_values("podium_prob")
    fig = go.Figure(go.Bar(
        x=rp["podium_prob"], y=rp["driver_code"], orientation="h",
        marker_color=["#E10600" if d in actual else "#2A2A2A" for d in rp["driver_id"]],
        text=[f"{p:.1%}" for p in rp["podium_prob"]], textposition="outside",
    ))
    fig.add_vline(x=0.5, line_dash="dash", line_color="#555")
    fig.update_layout(**LAYOUT, height=480,
        xaxis=dict(title="Podium Probability", tickformat=".0%", gridcolor="#2A2A2A"),
        yaxis=dict(title=""), showlegend=False)
    st.plotly_chart(fig, width="stretch")
    st.caption("🔴 Red = actual podium finishers")

    st.markdown('<p class="sec">Race Feature Table</p>', unsafe_allow_html=True)
    disp_cols = ["driver_code","quali_position","driver_champ_pos_pre",
                 "con_champ_pos_pre","rolling_avg_3","points_momentum",
                 "dnf_rate_5","num_pit_stops","podium_prob"]
    disp_cols = [c for c in disp_cols if c in race_df.columns]
    disp = race_df[disp_cols].copy()
    disp.columns = [c.replace("_"," ").title() for c in disp_cols]
    if "Podium Prob" in disp.columns:
        disp["Podium Prob"] = disp["Podium Prob"].map("{:.1%}".format)
    if "Rolling Avg 3" in disp.columns:
        disp["Rolling Avg 3"] = disp["Rolling Avg 3"].map("{:.1f}".format)
    if "Points Momentum" in disp.columns:
        disp["Points Momentum"] = disp["Points Momentum"].map("{:.0f}".format)
    if "Dnf Rate 5" in disp.columns:
        disp["Dnf Rate 5"] = disp["Dnf Rate 5"].map("{:.1%}".format)
    st.dataframe(disp, width="stretch", hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — 2026 LIVE SEASON
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔴 2026 Live Season":

    st.markdown("""<p class="sec">2026 Season <span class="live-badge">LIVE</span></p>""",
                unsafe_allow_html=True)

    with st.spinner("Fetching 2026 live data..."):
        res26   = fetch_2026_results()
        quali26 = fetch_2026_quali()
        sched26 = fetch_2026_schedule()

    if res26.empty:
        st.warning("No 2026 race results available yet. Check back after the first race.")
        st.stop()

    completed_rounds = sorted(res26["round"].unique())
    last_round = max(completed_rounds)
    next_round = last_round + 1

    # ── Season summary metrics ────────────────────────────────────────────────
    total_races = len(completed_rounds)
    winners = res26[res26["finish_position"]==1]["driver_code"].value_counts()
    leader  = winners.index[0] if len(winners) else "TBD"
    leader_wins = int(winners.iloc[0]) if len(winners) else 0

    driver_pts = (res26.groupby(["driver_id","driver_code"])["points"]
                  .sum().reset_index().sort_values("points",ascending=False))
    leader_pts = f"{int(driver_pts.iloc[0]['points'])}pts" if len(driver_pts) else "—"

    c1,c2,c3,c4 = st.columns(4)
    for col,(label,val) in zip([c1,c2,c3,c4],[
        ("Races Complete", str(total_races)),
        ("Championship Leader", leader),
        ("Leader Points", leader_pts),
        ("Leader Wins", str(leader_wins)),
    ]):
        col.markdown(f"""<div class="metric-card">
        <div class="metric-value">{val}</div>
        <div class="metric-label">{label}</div></div>""", unsafe_allow_html=True)

    st.markdown("")

    # ── Driver standings ──────────────────────────────────────────────────────
    st.markdown('<p class="sec">Driver Championship Standings</p>', unsafe_allow_html=True)

    top10 = driver_pts.head(10).sort_values("points")
    fig = go.Figure(go.Bar(
        x=top10["points"], y=top10["driver_code"], orientation="h",
        marker_color=["#FFD700" if i==len(top10)-1 else "#E10600"
                      for i in range(len(top10))],
        text=top10["points"].astype(int), textposition="outside",
    ))
    fig.update_layout(**LAYOUT, height=360,
        xaxis=dict(title="Championship Points", gridcolor="#2A2A2A"),
        yaxis=dict(title=""), showlegend=False)
    st.plotly_chart(fig, width="stretch")

    # ── Race-by-race results ──────────────────────────────────────────────────
    st.markdown('<p class="sec">Race Winners So Far</p>', unsafe_allow_html=True)

    winners_df = (res26[res26["finish_position"]==1]
                  .sort_values("round")[["round","race_name","driver_code","constructor_id"]])
    winners_df.columns = ["Round","Race","Winner","Constructor"]
    st.dataframe(winners_df, width="stretch", hide_index=True)

    # ── Next race prediction ──────────────────────────────────────────────────
    next_race_info = sched26[sched26["round"]==next_round]
    if not next_race_info.empty:
        nri = next_race_info.iloc[0]
        st.markdown(f'<p class="sec">Next Race Prediction — R{next_round}: {nri["race_name"]}</p>',
                    unsafe_allow_html=True)
        st.info(f"🏎 **{nri['race_name']}** · {nri['circuit_name']}, {nri['country']} · {nri['date']}")

        # Build a synthetic feature row for each 2026 driver using historical averages
        # Pull their last-known data from 2025 (most recent historical season in model)
        df_2025 = df[df["season"]==df["season"].max()].copy()
        last_2025 = (df_2025.sort_values(["driver_id","round"])
                     .groupby("driver_id").last().reset_index())

        # Get current 2026 standings
        cur_driver_pts = (res26.groupby(["driver_id","driver_code"])
                          .agg(points=("points","sum"),
                               wins=("finish_position", lambda x:(x==1).sum()))
                          .reset_index().sort_values("points",ascending=False)
                          .reset_index(drop=True))
        cur_driver_pts["driver_champ_pos_pre"] = cur_driver_pts.index + 1

        cur_con_pts = (res26.groupby("constructor_id")
                       .agg(points=("points","sum"))
                       .reset_index().sort_values("points",ascending=False)
                       .reset_index(drop=True))
        cur_con_pts["con_champ_pos_pre"] = cur_con_pts.index + 1

        # 2026 qualifying if available for next round
        next_quali = quali26[quali26["round"]==next_round] if not quali26.empty else pd.DataFrame()

        # Build prediction rows for drivers who have 2026 data
        pred_rows = []
        active_drivers = res26["driver_id"].unique()

        for driver_id in active_drivers:
            d_res = res26[res26["driver_id"]==driver_id]
            d_code = d_res["driver_code"].iloc[0] if len(d_res) else driver_id[:3].upper()
            con_id = d_res["constructor_id"].iloc[-1] if len(d_res) else "unknown"

            # Standings
            ds = cur_driver_pts[cur_driver_pts["driver_id"]==driver_id]
            cs = cur_con_pts[cur_con_pts["constructor_id"]==con_id]

            d_champ  = int(ds["driver_champ_pos_pre"].iloc[0]) if len(ds) else 10
            d_pts    = float(ds["points"].iloc[0]) if len(ds) else 0
            d_wins   = int(ds["wins"].iloc[0]) if len(ds) else 0
            c_champ  = int(cs["con_champ_pos_pre"].iloc[0]) if len(cs) else 5
            c_pts    = float(cur_con_pts[cur_con_pts["constructor_id"]==con_id]["points"].iloc[0]) if len(cs) else 0
            c_wins   = 0

            # Rolling form from 2026
            recent = d_res.sort_values("round").tail(5)
            fp = recent["finish_position"].fillna(20)
            roll3 = fp.tail(3).mean() if len(fp)>=1 else 10.0
            roll5 = fp.tail(5).mean() if len(fp)>=1 else 10.0
            pts_mom = recent["points"].tail(3).sum()
            dnf5 = recent["dnf"].tail(5).mean() if len(recent)>=1 else 0.0

            # Circuit history from historical data
            hist = df[(df["driver_id"]==driver_id)&(df["circuit_id"]==nri["circuit_id"])]
            circ_avg = float(hist["finish_position"].mean()) if len(hist)>0 else 10.0
            circ_win = float(hist["podium"].mean()) if len(hist)>0 else 0.0

            # Qualifying
            if not next_quali.empty:
                qrow = next_quali[next_quali["driver_id"]==driver_id]
                quali_pos = int(qrow["quali_position"].iloc[0]) if len(qrow) else 10
            else:
                quali_pos = int(d_res.sort_values("round")["grid_position"].iloc[-1]) if len(d_res) else 10

            grid_pos = quali_pos

            # Historical teammate gap (use last known)
            hist_d = df[df["driver_id"]==driver_id]
            tm_gap = float(hist_d["teammate_gap_3"].iloc[-1]) if ("teammate_gap_3" in df.columns and len(hist_d)>0) else 0.0
            con_mom = 0.0

            pred_rows.append({
                "driver_id": driver_id, "driver_code": d_code,
                "constructor_id": con_id,
                "quali_position": quali_pos, "front_row": int(quali_pos<=2),
                "driver_champ_pos_pre": d_champ, "driver_points_pre": d_pts,
                "driver_wins_pre": d_wins, "con_champ_pos_pre": c_champ,
                "con_points_pre": c_pts, "con_wins_pre": c_wins,
                "rolling_avg_3": roll3, "rolling_avg_5": roll5,
                "points_momentum": pts_mom, "dnf_rate_5": dnf5,
                "teammate_gap_3": tm_gap, "con_pts_momentum": con_mom,
                "circuit_avg_finish": circ_avg, "circuit_win_rate": circ_win,
                "num_pit_stops": 2, "fastest_lap_rank": 10,
                "avg_speed_kph": 220.0, "home_race": 0,
                "grid_position": grid_pos,
            })

        if pred_rows:
            pred_df = pd.DataFrame(pred_rows)
            # Ensure all feature cols present
            for col in feature_cols:
                if col not in pred_df.columns:
                    pred_df[col] = 0
            probs = model.predict_proba(pred_df[feature_cols])[:,1]
            pred_df["podium_prob"] = probs
            pred_df = pred_df.sort_values("podium_prob", ascending=False).reset_index(drop=True)

            top3p = pred_df.head(3)
            st.markdown(podium_html(top3p.iloc[0], top3p.iloc[1], top3p.iloc[2]),
                        unsafe_allow_html=True)

            st.caption("⚠️ Next race prediction uses 2026 form data + historical circuit averages. "
                       "Accuracy improves once qualifying results are available.")

            pred_plot = pred_df.sort_values("podium_prob")
            fig2 = go.Figure(go.Bar(
                x=pred_plot["podium_prob"], y=pred_plot["driver_code"],
                orientation="h",
                marker_color=["#E10600" if i>=len(pred_plot)-3 else "#2A2A2A"
                              for i in range(len(pred_plot))],
                text=[f"{p:.1%}" for p in pred_plot["podium_prob"]],
                textposition="outside",
            ))
            fig2.add_vline(x=0.5, line_dash="dash", line_color="#555")
            fig2.update_layout(**LAYOUT, height=480,
                xaxis=dict(title="Podium Probability",tickformat=".0%",gridcolor="#2A2A2A"),
                yaxis=dict(title=""), showlegend=False)
            st.plotly_chart(fig2, width="stretch")
    else:
        st.info("No upcoming race data available yet.")

    # ── 2026 podium heatmap ───────────────────────────────────────────────────
    st.markdown('<p class="sec">2026 Podium Heatmap</p>', unsafe_allow_html=True)
    podium26 = res26[res26["finish_position"]<=3].copy()
    if not podium26.empty:
        heat = (podium26.groupby(["driver_code","finish_position"])
                .size().unstack(fill_value=0).reindex(columns=[1,2,3],fill_value=0))
        heat.columns = ["1st","2nd","3rd"]
        heat["Total Podiums"] = heat.sum(axis=1)
        heat = heat.sort_values("Total Podiums", ascending=False)
        st.dataframe(heat, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Model Performance":
    st.markdown('<p class="sec">Model Comparison</p>', unsafe_allow_html=True)

    best_row = comparison.loc[comparison["podium_accuracy"].idxmax()]
    c1,c2,c3,c4 = st.columns(4)
    for col,(label,val) in zip([c1,c2,c3,c4],[
        ("ROC-AUC", f"{best_row['roc_auc']:.3f}"),
        ("Avg Precision", f"{best_row['avg_precision']:.3f}"),
        ("Podium Accuracy", f"{best_row['podium_accuracy']:.1%}"),
        ("Best Model", best_row["model"].split()[0]),
    ]):
        col.markdown(f"""<div class="metric-card">
        <div class="metric-value">{val}</div>
        <div class="metric-label">{label}</div></div>""", unsafe_allow_html=True)

    st.markdown("")

    fig = go.Figure()
    for metric,col,color in [("ROC-AUC","roc_auc","#E10600"),
                               ("Avg Precision","avg_precision","#FF8C00"),
                               ("Podium Accuracy","podium_accuracy","#FFD700")]:
        fig.add_trace(go.Bar(name=metric, x=comparison["model"], y=comparison[col],
                             marker_color=color,
                             text=[f"{v:.3f}" for v in comparison[col]],
                             textposition="outside"))
    fig.update_layout(**LAYOUT, barmode="group", height=380,
        yaxis=dict(range=[0,1.15],gridcolor="#2A2A2A",title="Score"),
        xaxis=dict(title=""),
        legend=dict(orientation="h",y=1.1))
    st.plotly_chart(fig, width="stretch")

    # Feature importance
    st.markdown('<p class="sec">Feature Importance</p>', unsafe_allow_html=True)
    clf_step = None
    if hasattr(model,"named_steps") and "c" in model.named_steps:
        clf_step = model.named_steps["c"]
    elif hasattr(model,"final_estimator_"):
        # stacking — use first base estimator that has importances
        for name, est in model.estimators_:
            inner = est.named_steps.get("c") if hasattr(est,"named_steps") else None
            if inner and hasattr(inner,"feature_importances_"):
                clf_step = inner; break

    if clf_step and hasattr(clf_step,"feature_importances_"):
        cols_used = feature_cols
        if len(clf_step.feature_importances_) == len(cols_used):
            fi = pd.Series(clf_step.feature_importances_, index=cols_used).sort_values(ascending=True)
            fig2 = go.Figure(go.Bar(x=fi.values, y=fi.index, orientation="h",
                                    marker_color="#E10600"))
            fig2.update_layout(**LAYOUT, height=500,
                xaxis=dict(title="Importance",gridcolor="#2A2A2A"), yaxis=dict(title=""))
            st.plotly_chart(fig2, width="stretch")
        else:
            st.info("Feature importance not available for stacked ensemble display.")
    else:
        st.info("Feature importance is displayed for tree-based models. The stacked ensemble "
                "uses a logistic meta-learner — see individual model importances in the plots folder.")

    # Confusion matrix
    st.markdown('<p class="sec">Confusion Matrix (Test Set)</p>', unsafe_allow_html=True)
    test_seasons = [2024, 2025]
    test_df_cm = df[df["season"].isin(test_seasons)].copy()
    if not test_df_cm.empty:
        from sklearn.metrics import confusion_matrix
        y_pred = model.predict(test_df_cm[feature_cols])
        y_true = test_df_cm["podium"].values
        cm = confusion_matrix(y_true, y_pred)
        fig3 = go.Figure(go.Heatmap(
            z=cm,
            x=["Predicted: No Podium","Predicted: Podium"],
            y=["Actual: No Podium","Actual: Podium"],
            text=cm, texttemplate="%{text}", textfont=dict(size=20),
            colorscale=[[0,"#1A1A1A"],[1,"#E10600"]], showscale=False,
        ))
        fig3.update_layout(**LAYOUT, height=300)
        st.plotly_chart(fig3, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — DRIVER DEEP DIVE
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔍 Driver Deep Dive":
    st.markdown('<p class="sec">Select Driver</p>', unsafe_allow_html=True)
    drivers = sorted(df["driver_code"].dropna().unique())
    driver  = st.selectbox("Driver", drivers, label_visibility="collapsed")
    ddf     = df[df["driver_code"]==driver].sort_values(["season","round"])

    if ddf.empty:
        st.warning("No data.")
        st.stop()

    c1,c2,c3,c4 = st.columns(4)
    for col,(label,val) in zip([c1,c2,c3,c4],[
        ("Races", str(len(ddf))),
        ("Podiums", str(ddf["podium"].sum())),
        ("Podium Rate", f"{ddf['podium'].mean():.1%}"),
        ("Avg Finish", f"{ddf['finish_position'].mean():.1f}"),
    ]):
        col.markdown(f"""<div class="metric-card">
        <div class="metric-value">{val}</div>
        <div class="metric-label">{label}</div></div>""", unsafe_allow_html=True)

    st.markdown("")
    st.markdown('<p class="sec">Finishing Position Over Time</p>', unsafe_allow_html=True)
    ddf["race_label"] = ddf["season"].astype(str)+" R"+ddf["round"].astype(str)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ddf["race_label"], y=ddf["finish_position"],
        mode="lines+markers", name="Finish",
        line=dict(color="#E10600",width=2),
        marker=dict(color=["#FFD700" if p else "#E10600" for p in ddf["podium"]], size=7)))
    if "rolling_avg_5" in ddf.columns:
        fig.add_trace(go.Scatter(x=ddf["race_label"], y=ddf["rolling_avg_5"],
            mode="lines", name="5-Race Avg",
            line=dict(color="#888",width=1.5,dash="dot")))
    fig.update_layout(**LAYOUT, height=340,
        yaxis=dict(title="Finish Position",autorange="reversed",gridcolor="#2A2A2A"),
        xaxis=dict(tickangle=45,tickfont=dict(size=8)),
        legend=dict(orientation="h",y=1.1))
    st.plotly_chart(fig, width="stretch")
    st.caption("🟡 Gold = podium finish")

    st.markdown('<p class="sec">Circuit History</p>', unsafe_allow_html=True)
    circ = (ddf.groupby("circuit_id")
            .agg(races=("finish_position","count"),
                 avg_finish=("finish_position","mean"),
                 podiums=("podium","sum"))
            .reset_index().sort_values("avg_finish"))
    fig2 = go.Figure(go.Bar(
        x=circ["circuit_id"], y=circ["avg_finish"],
        marker_color=circ["avg_finish"],
        marker_colorscale=[[0,"#E10600"],[0.5,"#FF8C00"],[1,"#2A2A2A"]],
        text=circ["podiums"].apply(lambda x: f"🏆{x}" if x>0 else ""),
        textposition="outside",
    ))
    fig2.update_layout(**LAYOUT, height=340,
        yaxis=dict(title="Avg Finish",autorange="reversed",gridcolor="#2A2A2A"),
        xaxis=dict(tickangle=45,tickfont=dict(size=8)))
    st.plotly_chart(fig2, width="stretch")
    st.caption("Lower bar = better average. 🏆 = podiums at that circuit.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — SEASON ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📈 Season Analysis":
    st.markdown('<p class="sec">Race-by-Race Accuracy</p>', unsafe_allow_html=True)

    test_seasons = [2024, 2025]
    tdf = df[df["season"].isin(test_seasons)].copy()
    if tdf.empty:
        st.warning("No test-season data available yet.")
        st.stop()

    tdf["podium_prob"] = model.predict_proba(tdf[feature_cols])[:,1]

    rows = []
    for (season, rnd), race in tdf.groupby(["season","round"]):
        pred   = set(race.nlargest(3,"podium_prob")["driver_id"].values)
        actual = set(race[race["podium"]==1]["driver_id"].values)
        correct= len(pred & actual)
        rows.append({"season":season,"round":rnd,
                     "race_name":race["race_name"].iloc[0],
                     "correct":correct,
                     "label":f"{int(season)} R{int(rnd)}"})

    racc = pd.DataFrame(rows)
    overall = racc["correct"].sum() / (len(racc)*3)

    c1,c2,c3,c4 = st.columns(4)
    for col,(label,val) in zip([c1,c2,c3,c4],[
        ("Overall Accuracy", f"{overall:.1%}"),
        ("Perfect (3/3)", str((racc["correct"]==3).sum())),
        ("Races Analyzed", str(len(racc))),
        ("Avg Correct/Race", f"{racc['correct'].mean():.1f}/3"),
    ]):
        col.markdown(f"""<div class="metric-card">
        <div class="metric-value">{val}</div>
        <div class="metric-label">{label}</div></div>""", unsafe_allow_html=True)

    st.markdown("")
    cmap = {0:"#E10600",1:"#FF8C00",2:"#FFD700",3:"#00C851"}
    fig = go.Figure(go.Bar(
        x=racc["label"], y=racc["correct"],
        marker_color=[cmap[c] for c in racc["correct"]],
        text=racc["correct"].apply(lambda x:f"{x}/3"),
        textposition="outside",
    ))
    fig.add_hline(y=racc["correct"].mean(), line_dash="dash", line_color="#888",
                  annotation_text=f"Avg {racc['correct'].mean():.1f}",
                  annotation_position="right")
    fig.update_layout(**LAYOUT, height=420,
        yaxis=dict(title="Correct Podium Picks",range=[0,4],dtick=1,gridcolor="#2A2A2A"),
        xaxis=dict(tickangle=45,tickfont=dict(size=8)))
    st.plotly_chart(fig, width="stretch")
    st.caption("🟢 3/3 Perfect · 🟡 2/3 · 🟠 1/3 · 🔴 0/3")

    # By circuit
    st.markdown('<p class="sec">Accuracy by Circuit</p>', unsafe_allow_html=True)
    cacc = (racc.merge(tdf[["season","round","circuit_id"]].drop_duplicates(),
                       on=["season","round"])
            .groupby("circuit_id")["correct"].mean()
            .sort_values(ascending=True).reset_index())
    fig2 = go.Figure(go.Bar(
        x=cacc["correct"], y=cacc["circuit_id"], orientation="h",
        marker_color=cacc["correct"],
        marker_colorscale=[[0,"#E10600"],[0.5,"#FFD700"],[1,"#00C851"]],
        text=[f"{v:.1f}/3" for v in cacc["correct"]], textposition="outside",
    ))
    fig2.update_layout(**LAYOUT, height=500,
        xaxis=dict(title="Avg Correct Picks",range=[0,4],gridcolor="#2A2A2A"),
        yaxis=dict(title=""))
    st.plotly_chart(fig2, width="stretch")

    # Season comparison
    st.markdown('<p class="sec">Accuracy by Season</p>', unsafe_allow_html=True)
    sacc = racc.groupby("season").agg(
        races=("correct","count"),
        total_correct=("correct","sum"),
    ).reset_index()
    sacc["accuracy"] = sacc["total_correct"] / (sacc["races"]*3)
    fig3 = go.Figure(go.Bar(
        x=sacc["season"].astype(str), y=sacc["accuracy"],
        marker_color="#E10600",
        text=[f"{v:.1%}" for v in sacc["accuracy"]], textposition="outside",
    ))
    fig3.update_layout(**LAYOUT, height=300,
        yaxis=dict(title="Podium Pick Accuracy",tickformat=".0%",
                   range=[0,1.1],gridcolor="#2A2A2A"),
        xaxis=dict(title="Season"))
    st.plotly_chart(fig3, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — SHAP EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🧠 SHAP Explainability":
    import shap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    st.markdown('<p class="sec">SHAP Explainability</p>', unsafe_allow_html=True)
    st.markdown("""
    **SHAP (SHapley Additive exPlanations)** shows *why* the model made each prediction —
    not just which features matter globally, but how each feature pushed a specific
    prediction higher or lower.
    """)

    # Extract the underlying XGBoost classifier from the pipeline
    @st.cache_resource
    def get_shap_explainer(_model, _X_sample):
        """Build SHAP explainer — cached so it only runs once."""
        try:
            clf = _model.named_steps["c"]
            explainer = shap.TreeExplainer(clf)
            return explainer
        except Exception as e:
            return None

    # Use a sample of test data for speed
    test_df_shap = df[df["season"].isin([2024, 2025])].copy()
    for col in ["rolling_avg_3","rolling_avg_5","points_momentum",
                "dnf_rate_5","teammate_gap_3","con_pts_momentum","circuit_win_rate"]:
        if col in test_df_shap.columns:
            test_df_shap[col] = test_df_shap[col].fillna(
                0 if any(x in col for x in ["momentum","dnf","gap","win"]) else 10)

    X_shap = test_df_shap[feature_cols]
    sample_size = min(300, len(X_shap))
    X_sample = X_shap.sample(sample_size, random_state=42)

    # Check model type — SHAP TreeExplainer works on tree models
    clf = model.named_steps.get("c") if hasattr(model, "named_steps") else None
    is_tree = clf is not None and hasattr(clf, "feature_importances_")

    if not is_tree:
        st.info("SHAP TreeExplainer requires a tree-based model (XGBoost or Random Forest). "
                "The current saved model is a stacked ensemble. Reload after saving XGBoost "
                "as best model, or re-run model.py.")
        st.stop()

    with st.spinner("Computing SHAP values (first load takes ~15 seconds)..."):
        explainer = get_shap_explainer(model, X_sample)
        if explainer is None:
            st.error("Could not build SHAP explainer for this model type.")
            st.stop()

        # Scale the sample first (pipeline has StandardScaler)
        scaler    = model.named_steps["s"]
        X_scaled  = pd.DataFrame(scaler.transform(X_sample),
                                  columns=feature_cols, index=X_sample.index)
        shap_vals = explainer(X_scaled)

    # ── Tab layout ────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["🌍 Global Importance", "🎯 Single Race", "📊 Feature Dependence"])

    # ── Tab 1: Global beeswarm ─────────────────────────────────────────────
    with tab1:
        st.markdown('<p class="sec">Global Feature Impact</p>', unsafe_allow_html=True)
        st.markdown("""
        Each dot is one driver-race. **Red = high feature value, Blue = low**.
        Dots to the right pushed the prediction toward podium; left = away from podium.
        """)
        fig, ax = plt.subplots(figsize=(9, 7), facecolor="#1A1A1A")
        shap.plots.beeswarm(shap_vals, max_display=15, show=False, color_bar=True)
        plt.gcf().set_facecolor("#1A1A1A")
        ax = plt.gca()
        ax.set_facecolor("#1A1A1A")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        plt.title("SHAP Beeswarm — Global Feature Impact", color="white", pad=12)
        plt.tight_layout()
        st.pyplot(plt.gcf(), use_container_width=True)
        plt.close()

        st.markdown("""
        **How to read this:**
        - Features are ranked top-to-bottom by overall impact
        - Wide spread = high variability in how this feature affects predictions
        - Tight cluster near zero = feature has consistent but small effect
        """)

    # ── Tab 2: Single race waterfall ──────────────────────────────────────
    with tab2:
        st.markdown('<p class="sec">Single Prediction Breakdown</p>', unsafe_allow_html=True)
        st.markdown("Pick a race and driver to see exactly why the model gave them that podium probability.")

        col1, col2 = st.columns(2)
        with col1:
            shap_seasons = sorted(test_df_shap["season"].unique(), reverse=True)
            sel_season   = st.selectbox("Season", shap_seasons, key="shap_season")
        with col2:
            shap_rounds = (test_df_shap[test_df_shap["season"]==sel_season]
                           [["round","race_name"]].drop_duplicates().sort_values("round"))
            rlabels = {r["round"]: f"R{r['round']} — {r['race_name']}"
                       for _,r in shap_rounds.iterrows()}
            sel_round = st.selectbox("Race", list(rlabels.keys()),
                                     format_func=lambda x: rlabels[x], key="shap_round")

        race_mask = ((test_df_shap["season"]==sel_season) &
                     (test_df_shap["round"]==sel_round))
        race_drivers = test_df_shap[race_mask]["driver_code"].dropna().unique()
        sel_driver   = st.selectbox("Driver", sorted(race_drivers), key="shap_driver")

        driver_mask = race_mask & (test_df_shap["driver_code"]==sel_driver)
        if driver_mask.sum() == 0:
            st.warning("No data for this selection.")
        else:
            driver_idx = test_df_shap[driver_mask].index[0]
            # Find position in X_sample
            if driver_idx in X_sample.index:
                sample_pos = list(X_sample.index).index(driver_idx)
                sv = shap_vals[sample_pos]

                # Actual result
                actual_podium = bool(test_df_shap.loc[driver_idx, "podium"])
                proba = model.predict_proba(X_shap.loc[[driver_idx]])[:,1][0]

                result_color = "#00C851" if actual_podium else "#E10600"
                result_text  = "✅ PODIUM" if actual_podium else "❌ No Podium"
                st.markdown(f"""
                <div style='background:#1A1A1A;border:1px solid #2A2A2A;border-radius:8px;
                padding:.8rem 1.2rem;margin:.5rem 0;display:flex;gap:2rem;align-items:center;'>
                  <span style='font-family:Orbitron,monospace;font-size:1.2rem;font-weight:700;'>
                    {sel_driver}</span>
                  <span style='color:{result_color};font-weight:600;'>{result_text}</span>
                  <span style='color:#888;font-size:.85rem;'>Model probability: {proba:.1%}</span>
                </div>
                """, unsafe_allow_html=True)

                fig2, ax2 = plt.subplots(figsize=(9, 5), facecolor="#1A1A1A")
                shap.plots.waterfall(sv, max_display=12, show=False)
                plt.gcf().set_facecolor("#1A1A1A")
                ax2 = plt.gca()
                ax2.set_facecolor("#1A1A1A")
                ax2.tick_params(colors="white")
                ax2.xaxis.label.set_color("white")
                plt.title(f"SHAP Waterfall — {sel_driver} @ {rlabels[sel_round]}",
                          color="white", pad=10)
                plt.tight_layout()
                st.pyplot(plt.gcf(), use_container_width=True)
                plt.close()

                st.markdown("""
                **How to read this:** Each bar shows how much a feature pushed the prediction
                up (red/right) or down (blue/left) from the base rate. The final value on the
                right is the model's raw SHAP output for this driver.
                """)
            else:
                st.info("This driver-race isn't in the SHAP sample. Try a different selection "
                        "or increase sample_size in the code.")

    # ── Tab 3: Dependence plot ────────────────────────────────────────────
    with tab3:
        st.markdown('<p class="sec">Feature Dependence</p>', unsafe_allow_html=True)
        st.markdown("How does one feature's value affect its SHAP impact? "
                    "The colour shows a second interacting feature.")

        feat = st.selectbox("Feature to explore", feature_cols, key="shap_feat",
                            index=feature_cols.index("quali_position") if "quali_position" in feature_cols else 0)

        feat_idx = feature_cols.index(feat)
        feat_vals  = X_scaled[feat].values
        shap_array = shap_vals.values[:, feat_idx]

        # Color by a correlated feature
        color_feat = "driver_champ_pos_pre"
        color_idx  = feature_cols.index(color_feat) if color_feat in feature_cols else 0
        color_vals = X_scaled[color_feat].values if color_feat in X_scaled.columns else feat_vals

        fig3 = go.Figure(go.Scatter(
            x=feat_vals, y=shap_array, mode="markers",
            marker=dict(color=color_vals, colorscale="RdYlGn_r", size=5,
                        showscale=True,
                        colorbar=dict(title=color_feat.replace("_"," ").title(),
                                      tickfont=dict(color="white"),
                                      titlefont=dict(color="white"))),
            text=[f"SHAP: {s:.3f}" for s in shap_array],
        ))
        fig3.add_hline(y=0, line_dash="dash", line_color="#555")
        fig3.update_layout(
            paper_bgcolor="#1A1A1A", plot_bgcolor="#1A1A1A",
            font=dict(color="#FFF", family="Inter"),
            margin=dict(l=40,r=40,t=50,b=40),
            height=420,
            xaxis=dict(title=feat.replace("_"," ").title(), gridcolor="#2A2A2A"),
            yaxis=dict(title=f"SHAP value for {feat.replace('_',' ').title()}",
                       gridcolor="#2A2A2A"),
            title=f"Dependence Plot: {feat.replace('_',' ').title()}",
        )
        st.plotly_chart(fig3, width="stretch")
        st.markdown("""
        **How to read this:** Points above zero = this feature value pushed the prediction
        toward podium. Points below zero = pushed away. Colour shows the interacting feature.
        A diagonal trend means the feature has a monotonic effect; curves mean non-linear.
        """)