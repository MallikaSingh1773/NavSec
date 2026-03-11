import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def run_unsupervised_detection(df_features, n_clusters=4):
    """
    Train a K-Means clustering algorithm on continuous features, assigning
    points blindly into n_clusters (4 for: Normal, Weak, Spoofing, Jamming).
    Matches cluster characteristics to interference types dynamically based
    on severity parameters like speed_change and trajectory_jump.
    """
    print(f"Running Unsupervised {n_clusters}-Class Detection (K-Means)...")
    
    # Select only the engineered physical features (excluding label completely)
    feature_cols = ['altitude_difference', 'speed_change', 'trajectory_jump', 
                    'signal_strength_variation', 'doppler_shift_anomaly']
    
    # We must ensure robust columns exist, or fallback to any numeric except 'label'
    present_cols = [c for c in feature_cols if c in df_features.columns]
    if not present_cols:
        present_cols = [c for c in df_features.select_dtypes(include=[np.number]).columns if c != 'label']
        
    X_train = df_features[present_cols].copy()
    
    # Re-normalize just these columns to give equal weight to K-Means distance metrics
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    # 1. Fit KMeans clustering
    print("Fitting K-Means model...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    X_train['cluster'] = clusters
    
    # 2. Heuristic Mapping of Clusters to Categories
    # We examine the cluster centers (averages) across features to logically deduce their labels
    cluster_centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=present_cols)
    cluster_centers['cluster'] = range(n_clusters)
    
    # Strategy: 
    # High trajectory_jump + altitude_difference often signifies Spoofing.
    # High speed_change + high signal_strength_variation often means Jamming.
    # The cluster with the lowest overall deviation across all is "Normal".
    
    mapping = {}
    remaining_clusters = list(range(n_clusters))
    
    # A. Normal GNSS: Lowest sum of absolute features (baseline)
    cluster_centers['total_deviation'] = cluster_centers.drop(columns='cluster').abs().sum(axis=1)
    normal_cluster = cluster_centers.loc[cluster_centers['total_deviation'].idxmin(), 'cluster']
    mapping[normal_cluster] = 'Normal GNSS'
    remaining_clusters.remove(normal_cluster)
    
    # B. Spoofing GNSS: Highest trajectory/altitude jumps
    if 'trajectory_jump' in cluster_centers.columns:
        # Find cluster with highest trajectory jump among remaining
        spoof_idx = cluster_centers.loc[cluster_centers['cluster'].isin(remaining_clusters), 'trajectory_jump'].idxmax()
        spoof_cluster = cluster_centers.loc[spoof_idx, 'cluster']
        mapping[spoof_cluster] = 'GNSS Spoofing'
        remaining_clusters.remove(spoof_cluster)
    else:
        spoof_cluster = remaining_clusters.pop(0)
        mapping[spoof_cluster] = 'GNSS Spoofing'
        
    # C. Jamming GNSS: Highest signal severity / speed changes
    if 'speed_change' in cluster_centers.columns:
        jam_idx = cluster_centers.loc[cluster_centers['cluster'].isin(remaining_clusters), 'speed_change'].idxmax()
        jam_cluster = cluster_centers.loc[jam_idx, 'cluster']
        mapping[jam_cluster] = 'GNSS Jamming'
        remaining_clusters.remove(jam_cluster)
    else:
        jam_cluster = remaining_clusters.pop(0)
        mapping[jam_cluster] = 'GNSS Jamming'
        
    # D. Weak Signal: The last remaining cluster
    if remaining_clusters:
        weak_cluster = remaining_clusters.pop(0)
        mapping[weak_cluster] = 'Weak GNSS Signal'
        
    print(f"K-Means Dynamic Mapping Determined: {mapping}")
    
    # Apply mapping to predictions
    mapped_predictions = [mapping[c] for c in clusters]
    
    # Output statistics
    pred_series = pd.Series(mapped_predictions)
    print("Unsupervised Class Distribution Derived Blindly:")
    print(pred_series.value_counts())
    
    return mapped_predictions, kmeans
