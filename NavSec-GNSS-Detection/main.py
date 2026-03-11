import os
import pandas as pd
from preprocessing import load_and_clean_data, normalize_features
from feature_engineering import engineer_features
from train_model import train_and_evaluate
from detect_interference import predict_interference
from jammer_localization import estimate_jammer_location
from unsupervised_anomaly_detection import run_unsupervised_detection
from visualization import plot_confusion_matrix, plot_feature_importance, create_interference_map

def main():
    print("=== NavSec GNSS Interference Detection System ===")
    
    # 1. Configuration
    dataset_path = 'dataset/Data.xlsx'
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}. Please check the path.")
        return
        
    # 2. Data Preprocessing
    print("\n--- Phase 1: Data Preprocessing ---")
    df = load_and_clean_data(dataset_path)
    
    # 3. Feature Engineering
    print("\n--- Phase 2: Feature Engineering ---")
    df_features = engineer_features(df)
    
    # 4. Normalization
    print("\n--- Phase 3: Normalization ---")
    df_normalized, scaler = normalize_features(df_features, target_col='label')
    
    # 5. Machine Learning Model Training
    print("\n--- Phase 4: Model Training & Evaluation ---")
    if 'label' not in df_normalized.columns:
        print("Error: 'label' column not found in dataset!")
        return
        
    X = df_normalized.drop(columns=['label'])
    y = df_normalized['label']
    
    best_model, results, best_model_name = train_and_evaluate(X, y)
    
    # 6. Interference Detection (Inference)
    print("\n--- Phase 5: Supervised Interference Detection ---")
    predictions, prediction_labels = predict_interference(best_model, X)
    
    # Output directories setup
    os.makedirs('data_outputs', exist_ok=True)
    os.makedirs('visualization_outputs', exist_ok=True)

    # Save predictions
    df_output = df.copy() 
    df_output['Predicted_Label'] = predictions
    df_output['Predicted_Category'] = prediction_labels
    df_output.to_csv('data_outputs/predictions.csv', index=False)
    print("Saved supervised predictions to data_outputs/predictions.csv")
    
    # 6.5. Unsupervised Anomaly Detection
    print("\n--- Phase 5.5: Unsupervised Anomaly Detection ---")
    unsupervised_preds, kmeans_model = run_unsupervised_detection(df_features, n_clusters=4)
    df_unsupervised = df.copy()
    df_unsupervised['Unsupervised_Category'] = unsupervised_preds
    df_unsupervised.to_csv('data_outputs/unsupervised_anomalies.csv', index=False)
    print("Saved unsupervised multi-class anomaly predictions to data_outputs/unsupervised_anomalies.csv")
    
    # 7. Jammer Location Estimation
    print("\n--- Phase 6: Jammer Localization ---")
    jammer_locations = estimate_jammer_location(df, predictions)
    if not jammer_locations.empty:
        jammer_locations.to_csv('data_outputs/jammer_locations.csv', index=False)
        print("Saved jammer locations to data_outputs/jammer_locations.csv")
        
    # 8. Visualization
    print("\n--- Phase 7: Visualization ---")
    # Confusion matrix of best model
    best_cm = results[best_model_name]['Confusion Matrix']
    labels = ['Normal', 'Jamming', 'Spoofing', 'Weak']
    plot_confusion_matrix(best_cm, labels=labels, output_path='visualization_outputs/confusion_matrix.png')
    
    # Feature importance
    plot_feature_importance(best_model, X.columns, output_path='visualization_outputs/feature_importance.png')
    
    # Mapping
    create_interference_map(df, predictions, jammer_locations, output_path='visualization_outputs/interference_map.html')
    
    print("\n=== Processing Complete ===")

if __name__ == "__main__":
    main()
