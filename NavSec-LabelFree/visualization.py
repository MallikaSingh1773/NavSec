import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster

# Color mapping for interference types
CATEGORY_COLORS = {
    'Normal GNSS':     'blue',
    'GNSS Jamming':    'red',
    'GNSS Spoofing':   'orange',
    'Weak GNSS Signal': 'yellow',
}


def plot_confusion_matrix(cm, labels, output_path='visualization_outputs/confusion_matrix.png'):
    """
    Plot and save a styled confusion matrix heatmap.
    Title clearly indicates this is a Label-Free Model.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=labels, yticklabels=labels, ax=ax,
        linewidths=0.5, linecolor='gray'
    )
    ax.set_title('Confusion Matrix — Label-Free Model (K-Means Pseudo-Labels)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('True Pseudo-Label (K-Means)', fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix to {output_path}")


def plot_feature_importance(model, feature_names,
                            output_path='visualization_outputs/feature_importance.png'):
    """
    Plot feature importances from tree-based models (Random Forest / XGBoost).
    Falls back gracefully for non-tree models.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if not hasattr(model, 'feature_importances_'):
        print("  Model does not support feature_importances_. Skipping plot.")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(sorted_features)))
    ax.bar(range(len(sorted_features)), sorted_importances, color=colors)
    ax.set_xticks(range(len(sorted_features)))
    ax.set_xticklabels(sorted_features, rotation=45, ha='right', fontsize=9)
    ax.set_title('Feature Importances — Label-Free GNSS Interference Model', fontsize=13, fontweight='bold')
    ax.set_ylabel('Importance Score')
    ax.set_xlabel('Feature')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved feature importance to {output_path}")


def create_interference_map(df_raw, pred_labels, jammer_locations,
                             output_path='visualization_outputs/interference_map.html',
                             max_points=5000):
    """
    Create an interactive Folium map showing:
    - Aircraft colored by predicted interference category
    - Estimated jammer locations with star markers

    Color scheme:
        Normal GNSS     → Blue
        GNSS Jamming    → Red
        GNSS Spoofing   → Orange
        Weak GNSS Signal → Yellow
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = df_raw.copy()
    df['Predicted_Category'] = pred_labels
    df = df.dropna(subset=['lat', 'lon'])

    # Subsample for performance
    if len(df) > max_points:
        print(f"  Subsampling {max_points:,} points from {len(df):,} for map performance.")
        # Ensure all categories are represented in sample
        df = (df.groupby('Predicted_Category', group_keys=False)
                .apply(lambda x: x.sample(min(len(x), max_points // 4), random_state=42))
                .reset_index(drop=True))

    center_lat = df['lat'].median()
    center_lon = df['lon'].median()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=5,
                   tiles='CartoDB dark_matter')

    # Draw points per category (Normal last so interference sits on top)
    draw_order = ['Normal GNSS', 'Weak GNSS Signal', 'GNSS Jamming', 'GNSS Spoofing']
    point_sizes = {
        'Normal GNSS': 3,
        'Weak GNSS Signal': 5,
        'GNSS Jamming': 7,
        'GNSS Spoofing': 7,
    }

    for category in draw_order:
        subset = df[df['Predicted_Category'] == category]
        color  = CATEGORY_COLORS.get(category, 'gray')
        radius = point_sizes.get(category, 4)
        for _, row in subset.iterrows():
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=folium.Popup(
                    f"<b>{category}</b><br>Lat: {row['lat']:.4f}<br>Lon: {row['lon']:.4f}",
                    max_width=200
                )
            ).add_to(m)

    # Draw jammer locations with star markers
    if jammer_locations is not None and not jammer_locations.empty:
        for _, jammer in jammer_locations.iterrows():
            folium.Marker(
                location=[jammer['estimated_lat'], jammer['estimated_lon']],
                popup=folium.Popup(
                    f"<b>⭐ Estimated Jammer</b><br>"
                    f"Type: {jammer.get('interference_type', 'Unknown')}<br>"
                    f"Points: {jammer.get('num_points', 'N/A')}",
                    max_width=200
                ),
                icon=folium.Icon(color='red', icon='star', prefix='fa')
            ).add_to(m)
            folium.Circle(
                location=[jammer['estimated_lat'], jammer['estimated_lon']],
                radius=30_000,  # 30 km radius circle
                color='red',
                fill=False,
                weight=1.5,
                dash_array='5'
            ).add_to(m)

    # Legend
    legend_html = """
    <div style="position:fixed; bottom:30px; left:30px; z-index:1000;
                background:rgba(0,0,0,0.75); padding:12px 18px;
                border-radius:8px; color:white; font-size:13px; font-family:Arial;">
        <b>🛰️ NavSec — GNSS Interference Map</b><br><br>
        <span style="color:#4fc3f7;">●</span> Normal GNSS<br>
        <span style="color:#ff5252;">●</span> GNSS Jamming<br>
        <span style="color:#ff9800;">●</span> GNSS Spoofing<br>
        <span style="color:#ffee58;">●</span> Weak GNSS Signal<br>
        <span style="color:#ff5252;">⭐</span> Estimated Jammer Location
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    m.save(output_path)
    print(f"Saved interference map to {output_path}")
