# NavSec-GNSS-Detection

NavSec is a complete machine learning project for detecting GNSS interference using ADS-B data and identifying whether the signal is normal, jammed, spoofed, or weak.

## Project Structure
- `dataset/Data.xlsx`: The dataset used for training and inference.
- `preprocessing.py`: Handles data loading, cleaning, missing value imputation, and normalization.
- `feature_engineering.py`: Computes new domain-specific features related to speed, altitude, trajectory, and signal strength.
- `train_model.py`: Trains Random Forest, XGBoost, and Neural Network models, evaluates their performance, and saves the best model.
- `detect_interference.py`: Uses the trained model to predict GNSS interference types (0=Normal, 1=Jamming, 2=Spoofing, 3=Weak).
- `jammer_localization.py`: Applies DBSCAN clustering on anomalies to estimate the jammer's location by calculating cluster centroids.
- `visualization.py`: Generates a Folium map `interference_map.html`, confusion matrix, and feature importance plots.
- `main.py`: The entry point that integrates the entire pipeline from data ingestion to map generation.

## Setup and Execution
1. Install dependencies using your preferred Python environment:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the end-to-end pipeline:
   ```bash
   python main.py
   ```

## Expected Outputs
- `models/best_model.pkl`: The best performing ML model.
- `predictions.csv`: The input dataset appended with predicted interference labels.
- `jammer_locations.csv`: Estimated latitude and longitude of detected jammers based on DBSCAN clustering.
- `interference_map.html`: Interactive Folium map visually representing aircraft positions, categorized interference points, and inferred jammer locations.
- `confusion_matrix.png` & `feature_importance.png`: Evaluation and model interpretability plots.
