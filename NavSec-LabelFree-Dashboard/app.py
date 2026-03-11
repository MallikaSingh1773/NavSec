import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import os

# ─────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="NavSec — Label-Free Dashboard",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp { background: linear-gradient(135deg, #060a14 0%, #0b1829 60%, #07111f 100%); }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1829 0%, #0f2035 100%);
        border-right: 1px solid #1a3356;
    }
    [data-testid="stSidebar"] * { color: #c8d8e8 !important; }

    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #0d2137, #152d48);
        border: 1px solid #1e4d7a;
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 4px 24px rgba(0,120,200,0.15);
    }
    [data-testid="stMetricValue"] {
        color: #38bdf8 !important;
        font-size: 2.1rem !important;
        font-weight: 800 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #7ba3c8 !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    h1, h2, h3 { color: #e2e8f0 !important; }
    hr { border-color: #1a3356; }
    .stDataFrame { border-radius: 10px; overflow: hidden; }
    .stRadio > div > label {
        background: #0d2137;
        border: 1px solid #1a3356;
        border-radius: 8px;
        padding: 8px 16px;
        cursor: pointer;
        transition: all 0.2s;
        color: #94a3b8 !important;
    }
    .stRadio > div > label:hover {
        background: #152d48;
        border-color: #38bdf8;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  PATHS  (absolute — always resolve)
# ─────────────────────────────────────────
BASE     = r"C:\Users\singh\Downloads\SDP PROJECT MAKKING\NavSec-LabelFree"
DATA_DIR = os.path.join(BASE, 'data_outputs')
VIZ_DIR  = os.path.join(BASE, 'visualization_outputs')

PRED_PATH   = os.path.join(DATA_DIR, 'predictions.csv')
JAMMER_PATH = os.path.join(DATA_DIR, 'jammer_locations.csv')
SIM_PATH    = os.path.join(DATA_DIR, 'simulation_results.csv')

CM_PATH     = os.path.join(VIZ_DIR,  'confusion_matrix.png')
FI_PATH     = os.path.join(VIZ_DIR,  'feature_importance.png')
MAP_PATH    = os.path.join(VIZ_DIR,  'interference_map.html')
HEATMAP_PATH= os.path.join(VIZ_DIR,  'interference_heatmap.html')

# ─────────────────────────────────────────
#  DATA LOADERS
# ─────────────────────────────────────────
@st.cache_data
def load_predictions():
    return pd.read_csv(PRED_PATH) if os.path.exists(PRED_PATH) else pd.DataFrame()

@st.cache_data
def load_jammers():
    return pd.read_csv(JAMMER_PATH) if os.path.exists(JAMMER_PATH) else pd.DataFrame()

@st.cache_data
def load_simulations():
    return pd.read_csv(SIM_PATH) if os.path.exists(SIM_PATH) else pd.DataFrame()


df_pred   = load_predictions()
df_jammers = load_jammers()

CAT_COL = 'Predicted_Category' if 'Predicted_Category' in df_pred.columns else None

COLORS = {
    'Normal GNSS':     '#38bdf8',
    'GNSS Jamming':    '#f87171',
    'GNSS Spoofing':   '#fb923c',
    'Weak GNSS Signal':'#facc15',
}

# ─────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:24px 0 12px 0;'>
        <div style='font-size:3rem;'>🛰️</div>
        <div style='font-size:1.5rem; font-weight:800; color:#38bdf8; letter-spacing:0.04em;'>NavSec</div>
        <div style='display:inline-block; background:linear-gradient(90deg,#0ea5e9,#6366f1);
                    color:white; font-size:0.7rem; font-weight:700; padding:3px 12px;
                    border-radius:20px; margin:6px 0 8px; letter-spacing:0.08em;
                    text-transform:uppercase;'>Label-Free Pipeline</div>
        <div style='font-size:0.72rem; color:#4a6a8a; letter-spacing:0.1em; text-transform:uppercase;'>
            GNSS Interference Detection
        </div>
    </div>
    <hr style='border-color:#1a3356; margin:8px 0 20px 0;'/>
    """, unsafe_allow_html=True)

    section = st.radio("Navigate", [
        "🏠 System Overview",
        "🧠 Pipeline Explainer",
        "🗺️ Interference Map",
        "🔥 Interference Heatmap",
        "⚠️ Attack Simulation",
        "📊 Model Performance",
        "📈 Model Comparison",
        "🔵 Signal Distribution",
        "📍 Jammer Locations",
        "🔍 Dataset Explorer",
    ], label_visibility="collapsed")

    st.markdown("""
    <hr style='border-color:#1a3356; margin:20px 0 10px 0;'/>
    <div style='font-size:0.68rem; color:#1e3a5f; text-align:center;'>
        NavSec &copy; 2026 &mdash; Final Year Project
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────
#  TOP HEADER
# ─────────────────────────────────────────
st.markdown("""
<div style='background:linear-gradient(90deg,#0b1829,#10294a,#081520);
            border-bottom:1px solid #1a3a60; padding:22px 32px 18px;
            margin-bottom:32px; border-radius:0 0 18px 18px;'>
    <div style='display:flex; align-items:center; gap:18px;'>
        <div style='font-size:2.5rem;'>🛰️</div>
        <div>
            <div style='font-size:1.7rem; font-weight:800; color:#e2e8f0; letter-spacing:-0.02em;'>
                NavSec — Label-Free GNSS Interference Detection
            </div>
            <div style='font-size:0.88rem; color:#4a6a8a; margin-top:4px;'>
                Fully Unsupervised-First &nbsp;|&nbsp; K-Means Pseudo-Labels &rarr; XGBoost &nbsp;|&nbsp; No Ground-Truth Labels Required
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════
#  1 — SYSTEM OVERVIEW
# ═══════════════════════════════════
if section == "🏠 System Overview":
    st.markdown("## 🏠 System Overview")
    st.caption("Interference events discovered autonomously — zero manual labels used.")
    st.markdown("---")

    total   = len(df_pred)
    normal  = int((df_pred[CAT_COL] == 'Normal GNSS').sum())   if CAT_COL else 0
    jamming = int((df_pred[CAT_COL] == 'GNSS Jamming').sum())  if CAT_COL else 0
    spoof   = int((df_pred[CAT_COL] == 'GNSS Spoofing').sum()) if CAT_COL else 0
    weak    = int((df_pred[CAT_COL] == 'Weak GNSS Signal').sum()) if CAT_COL else 0

    c1,c2,c3 = st.columns(3)
    c4,c5,c6 = st.columns(3)
    c1.metric("📡 Total Records",    f"{total:,}")
    c2.metric("🟢 Normal GNSS",      f"{normal:,}")
    c3.metric("🔴 GNSS Jamming",     f"{jamming:,}")
    c4.metric("🟠 GNSS Spoofing",    f"{spoof:,}")
    c5.metric("🟡 Weak Signal",      f"{weak:,}")
    c6.metric("📍 Jammer Locations", f"{len(df_jammers)}")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div style='background:linear-gradient(135deg,#0d2137,#0a1a2e);
                border:1px solid #1e4d7a; border-left:4px solid #38bdf8;
                border-radius:12px; padding:18px 22px; margin-bottom:24px;'>
        <div style='color:#38bdf8; font-weight:700; font-size:1rem; margin-bottom:8px;'>
            🧠 Fully Label-Free Approach
        </div>
        <div style='color:#94a3b8; font-size:0.9rem; line-height:1.8;'>
            This system <b style='color:#e2e8f0;'>never used</b> the original label column for training.
            K-Means clustering autonomously discovered 4 interference patterns from physical signal features.
            XGBoost trained on these auto-generated pseudo-labels achieved
            <b style='color:#4ade80;'>F1 Score = 99.39%</b> — with zero manual annotation.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if total > 0 and CAT_COL:
        st.markdown("### Signal Distribution Overview")
        data_ov = pd.DataFrame({
            'Category':   ['Normal GNSS', 'GNSS Jamming', 'GNSS Spoofing', 'Weak GNSS Signal'],
            'Count':      [normal, jamming, spoof, weak],
            'Percentage': [round(v/total*100,1) for v in [normal, jamming, spoof, weak]]
        })
        fig = px.bar(data_ov, x='Category', y='Count', color='Category',
                     color_discrete_sequence=list(COLORS.values()),
                     text='Percentage', template='plotly_dark')
        fig.update_traces(texttemplate='%{text}%', textposition='outside')
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(13,33,55,0.4)',
            showlegend=False, font=dict(color='#94a3b8'), height=380,
            margin=dict(t=20,b=20),
            xaxis=dict(gridcolor='#1a3356'), yaxis=dict(gridcolor='#1a3356'),
        )
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════
#  2 — PIPELINE EXPLAINER
# ═══════════════════════════════════
elif section == "🧠 Pipeline Explainer":
    st.markdown("## 🧠 Label-Free Pipeline — Step by Step")
    st.caption("How the system automatically discovers GNSS interference without any labeled training data.")
    st.markdown("---")

    stages = [
        ("Phase 1", "Preprocessing",
         "Load ADS-B data. Drop label column completely. Impute missing values with column mean. Normalize features.",
         "#38bdf8"),
        ("Phase 2", "Feature Engineering",
         "Add 5 physics-based derived features: altitude_difference, speed_change, trajectory_jump, signal_strength_variation, doppler_shift_anomaly",
         "#818cf8"),
        ("Phase 3", "K-Means Clustering (k=4)",
         "Group flights into 4 clusters using signal physics. Analyze cluster centroids to auto-name clusters: Normal / Jamming / Spoofing / Weak Signal",
         "#c084fc"),
        ("Phase 4", "Pseudo-Label Training",
         "Train Random Forest + XGBoost + Neural Network using K-Means labels as target y. Original label column NEVER touched.",
         "#f472b6"),
        ("Phase 5", "Inference",
         "Best model (XGBoost, F1=99.39%) predicts interference category for every flight record in the dataset",
         "#fb923c"),
        ("Phase 6", "Jammer Localization",
         "DBSCAN clusters Jamming + Spoofing GPS coordinates. Weighted centroid computed per cluster = estimated jammer location.",
         "#f87171"),
        ("Phase 7", "Visualization",
         "Generate confusion matrix, feature importance charts, interactive Folium map, and this dashboard.",
         "#4ade80"),
    ]

    for phase, name, detail, color in stages:
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#0d2137,#0a1a2e);
                    border:1px solid #1a3356; border-left:4px solid {color};
                    border-radius:0 12px 12px 0; padding:14px 20px;
                    margin-bottom:10px; display:flex; align-items:flex-start; gap:16px;'>
            <div style='min-width:80px; color:{color}; font-weight:700; font-size:0.75rem;
                        text-transform:uppercase; letter-spacing:0.07em; padding-top:3px;'>
                {phase}</div>
            <div>
                <div style='color:#e2e8f0; font-weight:600; font-size:0.98rem; margin-bottom:4px;'>{name}</div>
                <div style='color:#7ba3c8; font-size:0.86rem; line-height:1.6;'>{detail}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("💡 The cluster-to-category mapping is guided entirely by **physical signal characteristics** — "
            "no human expert labeling is required to identify interference types.")


# ═══════════════════════════════════
#  3 — INTERFERENCE MAP
# ═══════════════════════════════════
elif section == "🗺️ Interference Map":
    st.markdown("## 🗺️ Interactive Interference Map")
    st.caption("Flight positions colored by auto-detected interference type. Star = estimated jammer.")
    st.markdown("---")

    st.markdown("""
    <div style='display:flex; gap:28px; margin-bottom:16px; flex-wrap:wrap; font-size:0.9rem; color:#94a3b8;'>
        <span><b style='color:#38bdf8;'>●</b> Normal GNSS</span>
        <span><b style='color:#f87171;'>●</b> GNSS Jamming</span>
        <span><b style='color:#fb923c;'>●</b> GNSS Spoofing</span>
        <span><b style='color:#facc15;'>●</b> Weak Signal</span>
        <span><b style='color:#f87171;'>★</b> Estimated Jammer</span>
    </div>
    """, unsafe_allow_html=True)

    if os.path.exists(MAP_PATH):
        with open(MAP_PATH, 'r', encoding='utf-8') as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=630, scrolling=False)
    else:
        st.warning("⚠️ Map not found. Run `python main.py` in NavSec-LabelFree first.")


# ═══════════════════════════════════
#  3.1 — INTERFERENCE HEATMAP
# ═══════════════════════════════════
elif section == "🔥 Interference Heatmap":
    st.markdown("## 🔥 Interference Density Heatmap")
    st.caption("Visualizing the concentration density of GNSS Jamming and Spoofing events.")
    st.markdown("---")

    if os.path.exists(HEATMAP_PATH):
        with open(HEATMAP_PATH, 'r', encoding='utf-8') as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=630, scrolling=False)
    else:
        st.warning("⚠️ Heatmap not found. Run `python main.py` in NavSec-LabelFree first.")


# ═══════════════════════════════════
#  3.2 — ATTACK SIMULATION
# ═══════════════════════════════════
elif section == "⚠️ Attack Simulation":
    st.markdown("## ⚠️ GNSS Attack Simulation")
    st.caption("Inject artificial interference signatures into safe flight profiles and test model detection.")
    st.markdown("---")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.info("The simulation script modifies physical signal features (Doppler anomaly, RSS, trajectory jumps) "
                "on a random subset of flights to mimic real-world attacks. We then pass these through the trained XGBoost model.")
        if st.button("Run GNSS Attack Simulation", type="primary"):
            import subprocess
            with st.spinner("Generating simulated attacks & running model inference..."):
                try:
                    subprocess.run(["python", "simulation.py"], cwd=BASE, check=True)
                    load_simulations.clear() # clear cache to reload
                    st.success("Simulation Complete! Output updated.")
                except Exception as e:
                    st.error(f"Simulation failed: {e}")

    with col2:
        df_sim = load_simulations()
        if not df_sim.empty:
            st.markdown("### Simulation Results Summary")
            
            # Show summary
            summary = []
            for attack in ['Jamming', 'Spoofing', 'Weak']:
                sub = df_sim[df_sim['Simulated_Attack'].str.contains(attack, case=False, na=False)]
                if not sub.empty:
                    top_pred = sub['Predicted_Category'].mode()[0]
                    acc = (sub['Predicted_Category'] == top_pred).mean() * 100
                    summary.append({
                        'Simulated Attack': sub['Simulated_Attack'].iloc[0],
                        'Num Records': len(sub),
                        'Top Model Prediction': top_pred,
                        'Detection Match Rate': f"{acc:.1f}%"
                    })
            
            st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)

    if not df_sim.empty:
        st.markdown("### Full Simulation Dataset")
        # Ensure we only ask for columns that actually exist in the dataframe
        desired_cols = ['Simulated_Attack', 'Predicted_Category', 'lat', 'lon', 'altitude', 'geoaltitude', 'velocity']
        present_cols = [c for c in desired_cols if c in df_sim.columns]
        other_cols   = [c for c in df_sim.columns if c not in present_cols]
        
        st.dataframe(
            df_sim[present_cols + other_cols],
            use_container_width=True, hide_index=True
        )
    else:
        st.warning("Click the button above to run the latest simulation.")


# ═══════════════════════════════════
#  4 — MODEL PERFORMANCE
# ═══════════════════════════════════
elif section == "📊 Model Performance":
    st.markdown("## 📊 Model Performance")
    st.caption("XGBoost evaluation on K-Means pseudo-labels — no original labels were used.")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Confusion Matrix")
        st.caption("Predicted class vs K-Means pseudo-label (used as evaluation ground truth).")
        if os.path.exists(CM_PATH):
            st.image(Image.open(CM_PATH), use_column_width=True)
        else:
            st.warning("confusion_matrix.png not found.")

    with col2:
        st.markdown("### Feature Importance")
        st.caption("Physics features contributing most to interference classification.")
        if os.path.exists(FI_PATH):
            st.image(Image.open(FI_PATH), use_column_width=True)
        else:
            st.warning("feature_importance.png not found.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.success("✅ XGBoost achieved **F1 Score = 99.39%** trained purely on K-Means derived pseudo-labels!")


# ═══════════════════════════════════
#  5 — MODEL COMPARISON
# ═══════════════════════════════════
elif section == "📈 Model Comparison":
    st.markdown("## 📈 Model Comparison")
    st.caption("All three classifiers trained on K-Means pseudo-labels.")
    st.markdown("---")

    model_data = pd.DataFrame({
        'Model':     ['Random Forest', 'XGBoost', 'Neural Network'],
        'Accuracy':  [0.9931, 0.9955, 0.9945],
        'Precision': [0.9931, 0.9957, 0.9957],
        'Recall':    [0.9896, 0.9921, 0.9911],
        'F1 Score':  [0.9914, 0.9939, 0.9933],
    })

    metric_choice = st.selectbox("Select metric:", ['Accuracy','Precision','Recall','F1 Score'])
    colors_m = ['#38bdf8', '#f87171', '#c084fc']

    fig_bar = px.bar(model_data, x='Model', y=metric_choice,
                     color='Model', color_discrete_sequence=colors_m,
                     text_auto='.4f', template='plotly_dark')
    fig_bar.update_traces(textposition='outside')
    fig_bar.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(13,33,55,0.4)',
        showlegend=False, font=dict(color='#94a3b8'), height=420,
        yaxis=dict(range=[0.985,1.0], gridcolor='#1a3356'),
        xaxis=dict(gridcolor='#1a3356'), margin=dict(t=30,b=10),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("### 🕸️ Multi-Metric Radar")
    metrics = ['Accuracy','Precision','Recall','F1 Score']
    fig_radar = go.Figure()
    for i, row in model_data.iterrows():
        vals = [row[m] for m in metrics] + [row[metrics[0]]]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals, theta=metrics+[metrics[0]],
            fill='toself', name=row['Model'],
            line_color=colors_m[i], fillcolor=colors_m[i], opacity=0.25
        ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor='rgba(13,33,55,0.4)',
            radialaxis=dict(visible=True, range=[0.985,1.0], color='#64748b', gridcolor='#1a3356'),
            angularaxis=dict(color='#94a3b8', gridcolor='#1a3356'),
        ),
        paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#94a3b8'), height=420,
        showlegend=True, legend=dict(bgcolor='rgba(13,33,55,0.7)', bordercolor='#1a3356'),
        margin=dict(t=20,b=20),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("---")
    st.markdown("### Raw Numbers")
    st.dataframe(
        model_data.style.format({c: '{:.4f}' for c in ['Accuracy','Precision','Recall','F1 Score']}),
        use_container_width=True, hide_index=True
    )


# ═══════════════════════════════════
#  6 — SIGNAL DISTRIBUTION
# ═══════════════════════════════════
elif section == "🔵 Signal Distribution":
    st.markdown("## 🔵 Signal Distribution")
    st.caption("Auto-discovered interference categories — no manual labeling used.")
    st.markdown("---")

    if CAT_COL:
        dist = df_pred[CAT_COL].value_counts().reset_index()
        dist.columns = ['Category', 'Count']
    else:
        dist = pd.DataFrame({'Category': ['No data'], 'Count': [0]})

    col1, col2 = st.columns([1.2, 1])
    with col1:
        fig_pie = px.pie(dist, values='Count', names='Category',
                         color='Category', color_discrete_map=COLORS,
                         template='plotly_dark', hole=0.45)
        fig_pie.update_traces(textinfo='percent+label', textfont=dict(size=13), pull=[0.04]*len(dist))
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#94a3b8'),
            height=440, margin=dict(t=10,b=10), showlegend=True,
            legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='#1a3356'),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.markdown("### K-Means Derived Labels")
        total_cnt = dist['Count'].sum()
        for _, row in dist.iterrows():
            cat = row['Category']
            cnt = row['Count']
            pct = round(cnt / total_cnt * 100, 2)
            clr = COLORS.get(cat, '#94a3b8')
            st.markdown(f"""
            <div style='background:#0d2137; border-left:4px solid {clr};
                        border-radius:0 8px 8px 0; padding:12px 16px; margin-bottom:10px;'>
                <div style='color:{clr}; font-weight:700;'>{cat}</div>
                <div style='color:#7ba3c8; font-size:0.85rem;'>{cnt:,} records &nbsp;|&nbsp; {pct}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style='background:#071320; border:1px dashed #1a3356; border-radius:8px;
                    padding:12px 16px; margin-top:14px; color:#4a6a8a; font-size:0.82rem;'>
            These categories were derived fully automatically by K-Means clustering
            on physical signal features — no human annotation involved.
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════
#  7 — JAMMER LOCATIONS
# ═══════════════════════════════════
elif section == "📍 Jammer Locations":
    st.markdown("## 📍 Jammer Locations")
    st.caption("Estimated jammer GPS coordinates from DBSCAN clustering of anomalous points.")
    st.markdown("---")

    if not df_jammers.empty:
        st.metric("🎯 Estimated Jammer Sources", len(df_jammers))
        st.markdown("<br>", unsafe_allow_html=True)

        if 'estimated_lat' in df_jammers.columns:
            st.markdown("### Location Map")
            fig_map = px.scatter_mapbox(
                df_jammers, lat='estimated_lat', lon='estimated_lon',
                size='num_points' if 'num_points' in df_jammers.columns else None,
                color_discrete_sequence=['#f87171'],
                hover_data=df_jammers.columns.tolist(), zoom=4, template='plotly_dark',
            )
            fig_map.update_layout(
                mapbox_style="carto-darkmatter", paper_bgcolor='rgba(0,0,0,0)',
                height=420, margin=dict(t=0,b=0,l=0,r=0),
            )
            st.plotly_chart(fig_map, use_container_width=True)

        st.markdown("### Full Data Table")
        
        # Format confidence_score as percentage if exists
        fmt = None
        if 'confidence_score' in df_jammers.columns:
            fmt = {'confidence_score': '{:.2%}'}
            
        if fmt:
            st.dataframe(df_jammers.style.format(fmt), use_container_width=True, hide_index=True)
        else:
            st.dataframe(df_jammers, use_container_width=True, hide_index=True)
    else:
        st.warning("⚠️ jammer_locations.csv not found. Run `python main.py` in NavSec-LabelFree first.")


# ═══════════════════════════════════
#  8 — DATASET EXPLORER
# ═══════════════════════════════════
elif section == "🔍 Dataset Explorer":
    st.markdown("## 🔍 Dataset Explorer")
    st.caption("Browse and filter the predictions dataset interactively.")
    st.markdown("---")

    if df_pred.empty:
        st.warning("predictions.csv not found.")
    else:
        col1, col2 = st.columns([2, 1])
        with col1:
            if CAT_COL:
                options = ['All'] + sorted(df_pred[CAT_COL].dropna().unique().tolist())
                chosen  = st.selectbox("Filter by category:", options)
                df_view = df_pred if chosen == 'All' else df_pred[df_pred[CAT_COL] == chosen]
            else:
                df_view = df_pred
        with col2:
            max_rows = st.slider("Rows to show", 50, min(5000, len(df_view)), 300, step=50)

        st.caption(f"Showing {min(max_rows, len(df_view)):,} of {len(df_view):,} records")
        st.dataframe(df_view.head(max_rows), use_container_width=True, hide_index=True)

        csv_bytes = df_view.to_csv(index=False).encode()
        st.download_button(
            "⬇️ Download Filtered CSV", data=csv_bytes,
            file_name="navsec_labelfree_filtered.csv", mime='text/csv'
        )
