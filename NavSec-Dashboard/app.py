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
    page_title="NavSec Dashboard",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 50%, #0a1628 100%);
    }
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1b2a 0%, #10243a 100%);
        border-right: 1px solid #1e3a5f;
    }
    [data-testid="stSidebar"] * {
        color: #c8d8e8 !important;
    }
    /* Metric cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #0f2235, #162d44);
        border: 1px solid #1e4d7a;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 20px rgba(0,100,180,0.2);
    }
    [data-testid="stMetricValue"] {
        color: #38bdf8 !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    /* Headers */
    h1, h2, h3 { color: #e2e8f0 !important; }
    /* Dividers */
    hr { border-color: #1e3a5f; }
    /* Dataframe */
    .stDataFrame { border-radius: 10px; overflow: hidden; }
    /* Nav radio */
    .stRadio > div { gap: 6px; }
    .stRadio > div > label {
        background: #0f2235;
        border: 1px solid #1e3a5f;
        border-radius: 8px;
        padding: 8px 16px;
        cursor: pointer;
        transition: all 0.2s;
        color: #94a3b8 !important;
    }
    .stRadio > div > label:hover {
        background: #162d44;
        border-color: #38bdf8;
    }
    /* Section containers */
    .section-card {
        background: linear-gradient(135deg, #0f2235, #0d1b2a);
        border: 1px solid #1e3a5f;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    /* Badge colors for interference types */
    .badge-normal   { color: #38bdf8; font-weight: 600; }
    .badge-jamming  { color: #f87171; font-weight: 600; }
    .badge-spoofing { color: #c084fc; font-weight: 600; }
    .badge-weak     { color: #fb923c; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  PATHS — absolute so they always resolve
# ─────────────────────────────────────────
BASE = r"C:\Users\singh\Downloads\SDP PROJECT MAKKING\NavSec-GNSS-Detection"
DATA_DIR   = os.path.join(BASE, 'data_outputs')
VIZ_DIR    = os.path.join(BASE, 'visualization_outputs')

PREDICTIONS_PATH   = os.path.join(DATA_DIR, 'predictions.csv')
UNSUPERVISED_PATH  = os.path.join(DATA_DIR, 'unsupervised_anomalies.csv')
JAMMER_PATH        = os.path.join(DATA_DIR, 'jammer_locations.csv')
CM_PATH            = os.path.join(VIZ_DIR,  'confusion_matrix.png')
FI_PATH            = os.path.join(VIZ_DIR,  'feature_importance.png')
MAP_PATH           = os.path.join(VIZ_DIR,  'interference_map.html')

# ─────────────────────────────────────────
#  DATA LOADERS
# ─────────────────────────────────────────
@st.cache_data
def load_predictions():
    return pd.read_csv(PREDICTIONS_PATH) if os.path.exists(PREDICTIONS_PATH) else pd.DataFrame()

@st.cache_data
def load_unsupervised():
    return pd.read_csv(UNSUPERVISED_PATH) if os.path.exists(UNSUPERVISED_PATH) else pd.DataFrame()

@st.cache_data
def load_jammers():
    return pd.read_csv(JAMMER_PATH) if os.path.exists(JAMMER_PATH) else pd.DataFrame()

df_pred     = load_predictions()
df_unspv    = load_unsupervised()
df_jammers  = load_jammers()

# ─────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 10px 0;'>
        <div style='font-size: 3rem;'>🛰️</div>
        <div style='font-size: 1.4rem; font-weight: 800; color: #38bdf8; letter-spacing: 0.05em;'>NavSec</div>
        <div style='font-size: 0.75rem; color: #64748b; letter-spacing: 0.1em; text-transform: uppercase;'>GNSS Interference Detection</div>
    </div>
    <hr style='border-color:#1e3a5f; margin: 10px 0 20px 0;'/>
    """, unsafe_allow_html=True)

    section = st.radio(
        "Navigate",
        [
            "🏠 System Overview",
            "🗺️ Interactive Map",
            "📊 Model Performance",
            "📈 Model Comparison",
            "🔵 Interference Distribution",
            "📍 Jammer Locations",
            "🔍 Dataset Explorer",
        ],
        label_visibility="collapsed"
    )

    st.markdown("""
    <hr style='border-color:#1e3a5f; margin: 20px 0 10px 0;'/>
    <div style='font-size:0.7rem; color:#334155; text-align:center;'>
        NavSec &copy; 2026 &mdash; Final Year Project
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────
#  TITLE HEADER
# ─────────────────────────────────────────
st.markdown("""
<div style='
    background: linear-gradient(90deg, #0d1b2a 0%, #0f2d4a 60%, #092040 100%);
    border-bottom: 1px solid #1e4d7a;
    padding: 20px 32px 16px 32px;
    margin-bottom: 32px;
    border-radius: 0 0 16px 16px;
'>
    <div style='display:flex; align-items:center; gap:16px;'>
        <div style='font-size:2.4rem;'>🛰️</div>
        <div>
            <div style='font-size:1.8rem; font-weight:800; color:#e2e8f0; letter-spacing:-0.02em;'>
                NavSec GNSS Interference Detection
            </div>
            <div style='font-size:0.9rem; color:#64748b;'>
                AI-powered ADS-B Signal Analysis Dashboard &nbsp;|&nbsp; Supervised + Unsupervised ML
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════
#  SECTION 1: SYSTEM OVERVIEW
# ═══════════════════════════════════════════
if section == "🏠 System Overview":
    st.markdown("## 🏠 System Overview")
    st.caption("Summary of all detected GNSS events across 21,000+ ADS-B flight records.")
    st.markdown("---")

    total = len(df_pred)
    cat_col  = 'Predicted_Category' if 'Predicted_Category' in df_pred.columns else None
    label_col = 'Predicted_Label' if 'Predicted_Label' in df_pred.columns else None

    if cat_col:
        normal   = int((df_pred[cat_col] == 'Normal GNSS').sum())
        jamming  = int((df_pred[cat_col] == 'GNSS Jamming').sum())
        spoofing = int((df_pred[cat_col] == 'GNSS Spoofing').sum())
        weak     = int((df_pred[cat_col] == 'Weak GNSS Signal').sum())
    elif label_col:
        normal   = int((df_pred[label_col] == 0).sum())
        jamming  = int((df_pred[label_col] == 1).sum())
        spoofing = int((df_pred[label_col] == 2).sum())
        weak     = int((df_pred[label_col] == 3).sum())
    else:
        normal = jamming = spoofing = weak = 0

    num_jammers = len(df_jammers) if not df_jammers.empty else 0

    c1, c2, c3 = st.columns(3)
    c4, c5, c6 = st.columns(3)

    c1.metric("📡 Total Records",        f"{total:,}")
    c2.metric("🟢 Normal GNSS",          f"{normal:,}")
    c3.metric("🔴 GNSS Jamming",         f"{jamming:,}")
    c4.metric("🟣 GNSS Spoofing",        f"{spoofing:,}")
    c5.metric("🟠 Weak GNSS Signal",     f"{weak:,}")
    c6.metric("📍 Jammer Locations",     f"{num_jammers}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Timeline-style breakdown using color blocks
    if total > 0:
        st.markdown("### Signal Distribution Overview")
        data_overview = pd.DataFrame({
            'Category': ['Normal GNSS', 'GNSS Jamming', 'GNSS Spoofing', 'Weak GNSS Signal'],
            'Count': [normal, jamming, spoofing, weak],
            'Percentage': [
                round(normal/total*100, 1),
                round(jamming/total*100, 1),
                round(spoofing/total*100, 1),
                round(weak/total*100, 1),
            ]
        })
        colors = ['#38bdf8', '#f87171', '#c084fc', '#fb923c']
        fig = px.bar(
            data_overview, x='Category', y='Count',
            color='Category',
            color_discrete_sequence=colors,
            text='Percentage',
            template='plotly_dark'
        )
        fig.update_traces(texttemplate='%{text}%', textposition='outside')
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15,34,53,0.5)',
            showlegend=False,
            font=dict(color='#94a3b8'),
            height=380,
            margin=dict(t=20, b=20),
            xaxis=dict(gridcolor='#1e3a5f'),
            yaxis=dict(gridcolor='#1e3a5f'),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Unsupervised summary
    if not df_unspv.empty:
        unspv_col = 'Unsupervised_Category' if 'Unsupervised_Category' in df_unspv.columns else None
        if unspv_col:
            st.markdown("### 🔬 Unsupervised K-Means Anomaly Detection Results")
            st.caption("These categories were derived **without labels** using K-Means clustering on physical flight features.")
            unspv_counts = df_unspv[unspv_col].value_counts().reset_index()
            unspv_counts.columns = ['Category', 'Count']
            st.dataframe(unspv_counts, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════
#  SECTION 2: INTERACTIVE MAP
# ═══════════════════════════════════════════
elif section == "🗺️ Interactive Map":
    st.markdown("## 🗺️ Interactive Interference Map")
    st.caption("Live aircraft positions with interference classification. Click/Zoom on the map for details.")
    st.markdown("---")

    if os.path.exists(MAP_PATH):
        with open(MAP_PATH, 'r', encoding='utf-8') as f:
            html_content = f.read()

        st.markdown("""
        <div style='display:flex; gap:24px; margin-bottom:16px; flex-wrap:wrap;'>
            <span>🔵 <span class='badge-normal'>Blue</span> – Normal GNSS</span>
            <span>🔴 <span class='badge-jamming'>Red/Crimson</span> – GNSS Jamming</span>
            <span>🟣 <span class='badge-spoofing'>Purple</span> – GNSS Spoofing</span>
            <span>🟠 <span class='badge-weak'>Orange</span> – Weak Signal</span>
            <span>⭕ Red Ring – Estimated Jammer Location</span>
        </div>
        """, unsafe_allow_html=True)

        st.components.v1.html(html_content, height=620, scrolling=False)
    else:
        st.warning("⚠️ interference_map.html not found.\nRun `python main.py` in the NavSec-GNSS-Detection folder first.")


# ═══════════════════════════════════════════
#  SECTION 3: MODEL PERFORMANCE
# ═══════════════════════════════════════════
elif section == "📊 Model Performance":
    st.markdown("## 📊 Model Performance")
    st.caption("Visual evaluation outputs generated after training the best XGBoost model.")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Confusion Matrix")
        st.caption("Shows how accurately the model classified each interference category.")
        if os.path.exists(CM_PATH):
            img_cm = Image.open(CM_PATH)
            st.image(img_cm, use_column_width=True)
        else:
            st.warning("confusion_matrix.png not found.")

    with col2:
        st.markdown("### Feature Importance")
        st.caption("Which physical features contributed most to the model's decision-making.")
        if os.path.exists(FI_PATH):
            img_fi = Image.open(FI_PATH)
            st.image(img_fi, use_column_width=True)
        else:
            st.warning("feature_importance.png not found.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("💡 **XGBoost** achieved the highest F1-Score of **~98.75%** compared to Random Forest and Neural Network on this dataset.")


# ═══════════════════════════════════════════
#  SECTION 4: MODEL COMPARISON
# ═══════════════════════════════════════════
elif section == "📈 Model Comparison":
    st.markdown("## 📈 Model Comparison")
    st.caption("Performance comparison across all three trained classifiers.")
    st.markdown("---")

    model_data = pd.DataFrame({
        'Model': ['Random Forest', 'XGBoost', 'Neural Network'],
        'Accuracy':  [0.9876, 0.9910, 0.9791],
        'Precision': [0.9852, 0.9888, 0.9731],
        'Recall':    [0.9831, 0.9862, 0.9672],
        'F1 Score':  [0.9841, 0.9875, 0.9701],
    })

    metric_choice = st.selectbox(
        "Select metric to compare:",
        ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    )

    colors = ['#38bdf8', '#f87171', '#c084fc']
    fig_bar = px.bar(
        model_data,
        x='Model', y=metric_choice,
        color='Model',
        color_discrete_sequence=colors,
        text_auto='.4f',
        template='plotly_dark',
    )
    fig_bar.update_traces(textposition='outside')
    fig_bar.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,34,53,0.5)',
        showlegend=False,
        font=dict(color='#94a3b8'),
        height=420,
        yaxis=dict(range=[0.95, 1.0], gridcolor='#1e3a5f'),
        xaxis=dict(gridcolor='#1e3a5f'),
        margin=dict(t=30, b=10),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Radar chart for all metrics at once
    st.markdown("### 🕸️ Multi-Metric Radar Chart")
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    fig_radar = go.Figure()
    radar_colors = ['#38bdf8', '#f87171', '#c084fc']
    for i, row in model_data.iterrows():
        vals = [row[m] for m in metrics]
        vals.append(vals[0])  # close the polygon
        fig_radar.add_trace(go.Scatterpolar(
            r=vals,
            theta=metrics + [metrics[0]],
            fill='toself',
            name=row['Model'],
            line_color=radar_colors[i],
            fillcolor=radar_colors[i],
            opacity=0.25
        ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor='rgba(15,34,53,0.5)',
            radialaxis=dict(visible=True, range=[0.96, 1.0], color='#64748b', gridcolor='#1e3a5f'),
            angularaxis=dict(color='#94a3b8', gridcolor='#1e3a5f'),
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#94a3b8'),
        height=440,
        showlegend=True,
        legend=dict(bgcolor='rgba(15,34,53,0.7)', bordercolor='#1e3a5f'),
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("---")
    st.markdown("### Raw Performance Numbers")
    st.dataframe(
        model_data.style.format({'Accuracy': '{:.4f}', 'Precision': '{:.4f}', 'Recall': '{:.4f}', 'F1 Score': '{:.4f}'}),
        use_container_width=True,
        hide_index=True
    )


# ═══════════════════════════════════════════
#  SECTION 5: INTERFERENCE DISTRIBUTION
# ═══════════════════════════════════════════
elif section == "🔵 Interference Distribution":
    st.markdown("## 🔵 Interference Distribution")
    st.caption("Breakdown of interference type proportions as detected by the XGBoost model.")
    st.markdown("---")

    cat_col  = 'Predicted_Category' if 'Predicted_Category' in df_pred.columns else None
    label_col = 'Predicted_Label' if 'Predicted_Label' in df_pred.columns else None

    if cat_col:
        dist = df_pred[cat_col].value_counts().reset_index()
        dist.columns = ['Category', 'Count']
    elif label_col:
        lmap = {0: 'Normal GNSS', 1: 'GNSS Jamming', 2: 'GNSS Spoofing', 3: 'Weak GNSS Signal'}
        dist = df_pred[label_col].map(lmap).value_counts().reset_index()
        dist.columns = ['Category', 'Count']
    else:
        dist = pd.DataFrame({'Category': ['No data'], 'Count': [0]})

    colors_pie = {
        'Normal GNSS': '#38bdf8',
        'GNSS Jamming': '#f87171',
        'GNSS Spoofing': '#c084fc',
        'Weak GNSS Signal': '#fb923c',
    }

    col1, col2 = st.columns([1.2, 1])

    with col1:
        fig_pie = px.pie(
            dist, values='Count', names='Category',
            color='Category',
            color_discrete_map=colors_pie,
            template='plotly_dark',
            hole=0.45,
        )
        fig_pie.update_traces(
            textinfo='percent+label',
            textfont=dict(size=13),
            pull=[0.04] * len(dist),
        )
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#94a3b8'),
            height=430,
            legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='#1e3a5f'),
            margin=dict(t=10, b=10),
            showlegend=True,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.markdown("### Class Breakdown")
        for _, row in dist.iterrows():
            cat   = row['Category']
            count = row['Count']
            pct   = round(count / dist['Count'].sum() * 100, 2)
            color_map = {
                'Normal GNSS': '#38bdf8',
                'GNSS Jamming': '#f87171',
                'GNSS Spoofing': '#c084fc',
                'Weak GNSS Signal': '#fb923c',
            }
            clr = color_map.get(cat, '#94a3b8')
            st.markdown(f"""
            <div style='
                background: #0f2235;
                border-left: 4px solid {clr};
                border-radius: 8px;
                padding: 12px 16px;
                margin-bottom: 10px;
            '>
                <div style='color:{clr}; font-weight:700; font-size:1rem;'>{cat}</div>
                <div style='color:#94a3b8; font-size:0.85rem;'>{count:,} records &nbsp;|&nbsp; {pct}%</div>
            </div>
            """, unsafe_allow_html=True)

    # Unsupervised comparison
    unspv_col = 'Unsupervised_Category' if not df_unspv.empty and 'Unsupervised_Category' in df_unspv.columns else None
    if unspv_col:
        st.markdown("---")
        st.markdown("### 🔬 Unsupervised (K-Means) Distribution Comparison")
        st.caption("Class distribution derived completely without labels.")
        unspv_dist = df_unspv[unspv_col].value_counts().reset_index()
        unspv_dist.columns = ['Category', 'Count']
        fig_u = px.bar(
            unspv_dist, x='Category', y='Count',
            color='Category',
            color_discrete_map=colors_pie,
            template='plotly_dark',
            text_auto=True
        )
        fig_u.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15,34,53,0.5)',
            showlegend=False,
            font=dict(color='#94a3b8'),
            height=340,
            xaxis=dict(gridcolor='#1e3a5f'),
            yaxis=dict(gridcolor='#1e3a5f'),
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig_u, use_container_width=True)


# ═══════════════════════════════════════════
#  SECTION 6: JAMMER LOCATIONS
# ═══════════════════════════════════════════
elif section == "📍 Jammer Locations":
    st.markdown("## 📍 Jammer Locations")
    st.caption("Estimated jammer source coordinates derived via DBSCAN clustering on jamming anomalies.")
    st.markdown("---")

    if not df_jammers.empty:
        total_jammers = len(df_jammers)
        st.metric("🎯 Total Estimated Jammer Sources", total_jammers)
        st.markdown("<br>", unsafe_allow_html=True)

        # Scatter map of jammer locations
        if 'estimated_lat' in df_jammers.columns and 'estimated_lon' in df_jammers.columns:
            st.markdown("### Jammer Location Map")
            fig_map = px.scatter_mapbox(
                df_jammers,
                lat='estimated_lat',
                lon='estimated_lon',
                size='num_points' if 'num_points' in df_jammers.columns else None,
                color_discrete_sequence=['#f87171'],
                hover_data=df_jammers.columns.tolist(),
                zoom=6,
                template='plotly_dark',
            )
            fig_map.update_layout(
                mapbox_style="carto-darkmatter",
                paper_bgcolor='rgba(0,0,0,0)',
                height=420,
                margin=dict(t=0, b=0, l=0, r=0),
            )
            st.plotly_chart(fig_map, use_container_width=True)

        st.markdown("### Full Jammer Data Table")
        st.dataframe(df_jammers, use_container_width=True, hide_index=True)
    else:
        st.warning("⚠️ jammer_locations.csv not found. Run `python main.py` in NavSec-GNSS-Detection first.")


# ═══════════════════════════════════════════
#  SECTION 7: DATASET EXPLORER
# ═══════════════════════════════════════════
elif section == "🔍 Dataset Explorer":
    st.markdown("## 🔍 Dataset Explorer")
    st.caption("Browse, filter, and explore the full predictions dataset interactively.")
    st.markdown("---")

    if df_pred.empty:
        st.warning("predictions.csv not found.")
    else:
        # Dataset picker
        ds_choice = st.radio(
            "Choose dataset:",
            ["Supervised Predictions", "Unsupervised Category Detection"],
            horizontal=True
        )
        df_view = df_pred if ds_choice == "Supervised Predictions" else df_unspv

        cat_options = []
        filter_col = None
        if ds_choice == "Supervised Predictions":
            filter_col = 'Predicted_Category' if 'Predicted_Category' in df_view.columns else None
        else:
            filter_col = 'Unsupervised_Category' if 'Unsupervised_Category' in df_view.columns else None

        col1, col2 = st.columns([2, 1])
        with col1:
            if filter_col and filter_col in df_view.columns:
                options = ['All'] + sorted(df_view[filter_col].dropna().unique().tolist())
                chosen = st.selectbox(f"Filter by {filter_col}", options)
                if chosen != 'All':
                    df_view = df_view[df_view[filter_col] == chosen]

        with col2:
            max_rows = st.slider("Rows to display", 50, min(5000, len(df_view)), 200, step=50)

        st.caption(f"Showing {min(max_rows, len(df_view)):,} of {len(df_view):,} records")
        st.dataframe(df_view.head(max_rows), use_container_width=True, hide_index=True)

        # Download button
        csv_data = df_view.to_csv(index=False).encode()
        st.download_button(
            label="⬇️ Download Filtered Data as CSV",
            data=csv_data,
            file_name=f"navsec_filtered_{ds_choice.replace(' ', '_').lower()}.csv",
            mime='text/csv'
        )
