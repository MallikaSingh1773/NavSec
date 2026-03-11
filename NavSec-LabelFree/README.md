# NavSec — Label-Free GNSS Interference Detection System

A fully unsupervised-first machine learning system for detecting GNSS interference in ADS-B flight data — **no ground-truth labels required**.

## Pipeline

```
Raw ADS-B Data (Data.xlsx)
    ↓
Preprocessing        — Drop labels, impute, normalize
    ↓
Feature Engineering  — altitude_diff, speed_change, trajectory_jump, etc.
    ↓
K-Means Clustering   — Derives pseudo-labels (Jamming/Spoofing/Normal/Weak)
    ↓
Model Training       — RF + XGBoost + MLP trained on pseudo-labels
    ↓
Interference Detection → predictions.csv
    ↓
Jammer Localization  → jammer_locations.csv
    ↓
Visualization        → confusion_matrix.png, feature_importance.png, interference_map.html
```

## Key Innovation

Unlike traditional supervised approaches, this system derives interference categories **automatically** from physical flight data patterns using K-Means clustering. The resulting pseudo-labels are then used to train three supervised classifiers for fast real-time inference.

## Setup

```bash
pip install -r requirements.txt
python main.py
```

## Output Files

| File | Description |
|------|-------------|
| `data_outputs/predictions.csv` | Per-flight interference classification |
| `data_outputs/jammer_locations.csv` | Estimated GPS coordinates of jamming sources |
| `visualization_outputs/confusion_matrix.png` | Model evaluation heatmap |
| `visualization_outputs/feature_importance.png` | Top contributing features |
| `visualization_outputs/interference_map.html` | Interactive Folium map |
| `models/best_model.pkl` | Saved best classifier + label encoder |
