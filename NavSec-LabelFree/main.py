import os
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from preprocessing         import load_and_preprocess, normalize_features
from feature_engineering   import engineer_features
from label_free_clustering import run_label_free_clustering
from train_model           import train_and_evaluate
from detect_interference   import predict_interference
from jammer_localization   import estimate_jammer_location
from visualization         import (plot_confusion_matrix,
                                   plot_feature_importance,
                                   create_interference_map)

DATASET_PATH = 'dataset/Data.xlsx'


def main():
    print("=" * 60)
    print("  NavSec — Label-Free GNSS Interference Detection System")
    print("=" * 60)

    # ─────────────────────────────────────────────
    # PHASE 1: PREPROCESSING
    # ─────────────────────────────────────────────
    print("\n--- Phase 1: Preprocessing ---")
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset not found at '{DATASET_PATH}'. Please check the path.")
        return

    df_raw, X_raw, orig_labels = load_and_preprocess(DATASET_PATH)

    # ─────────────────────────────────────────────
    # PHASE 2: FEATURE ENGINEERING
    # ─────────────────────────────────────────────
    print("\n--- Phase 2: Feature Engineering ---")
    df_raw_engineered = engineer_features(df_raw)
    X_engineered      = engineer_features(X_raw)

    # Normalize engineered feature matrix for ML
    # Keep lat/lon out of the model but inside df_raw for mapping
    spatial = ['lat', 'lon']
    X_norm, scaler = normalize_features(
        X_engineered.drop(columns=[c for c in spatial if c in X_engineered.columns])
    )

    # ─────────────────────────────────────────────
    # PHASE 3: K-MEANS LABEL-FREE CLUSTERING
    # ─────────────────────────────────────────────
    print("\n--- Phase 3: K-Means Label-Free Clustering ---")
    pseudo_labels, kmeans_model, cluster_mapping = run_label_free_clustering(
        X_engineered, n_clusters=4
    )

    # ─────────────────────────────────────────────
    # PHASE 4: MODEL TRAINING ON PSEUDO-LABELS
    # ─────────────────────────────────────────────
    print("\n--- Phase 4: Model Training on Pseudo-Labels ---")
    best_model, results, best_model_name, label_encoder = train_and_evaluate(
        X_norm, pseudo_labels
    )

    # ─────────────────────────────────────────────
    # PHASE 5: INTERFERENCE DETECTION
    # ─────────────────────────────────────────────
    print("\n--- Phase 5: Interference Detection ---")
    pred_numeric, pred_labels = predict_interference(best_model, X_norm, label_encoder)

    os.makedirs('data_outputs', exist_ok=True)
    df_output = df_raw.copy()
    df_output['Predicted_Category'] = pred_labels
    df_output['Predicted_Label']    = pred_numeric
    df_output.to_csv('data_outputs/predictions.csv', index=False)
    print("Saved predictions to data_outputs/predictions.csv")

    # ─────────────────────────────────────────────
    # OPTIONAL: VALIDATION AGAINST ORIGINAL LABELS
    # ─────────────────────────────────────────────
    if orig_labels is not None:
        print("\n--- Optional: Validation Against Original Labels ---")
        label_map = {0: 'Normal GNSS', 1: 'GNSS Jamming',
                     2: 'GNSS Spoofing', 3: 'Weak GNSS Signal'}
        orig_mapped = orig_labels.map(label_map).fillna('Unknown')
        match_rate = (pd.Series(pred_labels).values == orig_mapped.values).mean()
        print(f"  Agreement between model predictions and original labels: {match_rate:.2%}")
        print("  (This is a sanity check only — original labels were NOT used for training)")

    # ─────────────────────────────────────────────
    # PHASE 6: JAMMER LOCALIZATION
    # ─────────────────────────────────────────────
    print("\n--- Phase 6: Jammer Localization ---")
    jammer_locations = estimate_jammer_location(df_raw_engineered, pred_labels)
    if not jammer_locations.empty:
        jammer_locations.to_csv('data_outputs/jammer_locations.csv', index=False)
        print(f"Saved {len(jammer_locations)} jammer locations to data_outputs/jammer_locations.csv")

    # ─────────────────────────────────────────────
    # PHASE 7: VISUALIZATION
    # ─────────────────────────────────────────────
    print("\n--- Phase 7: Visualization ---")
    os.makedirs('visualization_outputs', exist_ok=True)

    # Confusion matrix using K-Means pseudo-labels as "true" labels
    best_cm = results[best_model_name]['Confusion Matrix']
    plot_confusion_matrix(
        best_cm,
        labels=list(label_encoder.classes_),
        output_path='visualization_outputs/confusion_matrix.png'
    )

    # Feature importance
    plot_feature_importance(
        best_model,
        feature_names=list(X_norm.columns),
        output_path='visualization_outputs/feature_importance.png'
    )

    # Interactive map
    create_interference_map(
        df_raw_engineered,
        pred_labels,
        jammer_locations,
        output_path='visualization_outputs/interference_map.html'
    )

    print("\n" + "=" * 60)
    print("  Processing Complete — All outputs saved!")
    print("=" * 60)
    print("\nOutput Files:")
    print("  data_outputs/predictions.csv")
    print("  data_outputs/jammer_locations.csv")
    print("  visualization_outputs/confusion_matrix.png")
    print("  visualization_outputs/feature_importance.png")
    print("  visualization_outputs/interference_map.html")
    print("  models/best_model.pkl")


if __name__ == "__main__":
    main()
