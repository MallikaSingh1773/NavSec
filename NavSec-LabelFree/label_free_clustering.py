import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import Counter


# Features used exclusively for clustering-based pseudo-label generation
CLUSTER_FEATURES = [
    'altitude_difference',
    'speed_change',
    'trajectory_jump',
    'signal_strength_variation',
    'doppler_shift_anomaly',
]


def _assign_cluster_names(centers_df, present_cols):
    """
    Heuristically map cluster IDs to interference category names
    based on the physical characteristics of cluster centroids.

    Rules:
        Normal GNSS     -> lowest total anomaly score
        GNSS Spoofing   -> highest (altitude_difference + trajectory_jump)
        GNSS Jamming    -> highest (speed_change + signal_strength_variation)
        Weak GNSS Signal -> remaining cluster
    """
    mapping = {}
    remaining = list(centers_df['cluster_id'].astype(int))

    # Compute total anomaly magnitude for each cluster
    centers_df = centers_df.copy()
    feature_sum_cols = [c for c in present_cols if c in centers_df.columns]
    centers_df['total_anomaly'] = centers_df[feature_sum_cols].abs().sum(axis=1)

    # === Normal GNSS: Lowest total anomaly ===
    normal_idx = centers_df['total_anomaly'].idxmin()
    normal_cluster = int(centers_df.loc[normal_idx, 'cluster_id'])
    mapping[normal_cluster] = 'Normal GNSS'
    remaining.remove(normal_cluster)

    # === GNSS Spoofing: High altitude_difference + trajectory_jump ===
    rem_df = centers_df[centers_df['cluster_id'].isin(remaining)]
    spoof_score = pd.Series(0.0, index=rem_df.index)
    if 'altitude_difference' in rem_df.columns:
        spoof_score += rem_df['altitude_difference'].abs()
    if 'trajectory_jump' in rem_df.columns:
        spoof_score += rem_df['trajectory_jump'].abs()
    spoof_cluster = int(rem_df.loc[spoof_score.idxmax(), 'cluster_id'])
    mapping[spoof_cluster] = 'GNSS Spoofing'
    remaining.remove(spoof_cluster)

    # === GNSS Jamming: High speed_change + signal_strength_variation ===
    rem_df = centers_df[centers_df['cluster_id'].isin(remaining)]
    jam_score = pd.Series(0.0, index=rem_df.index)
    if 'speed_change' in rem_df.columns:
        jam_score += rem_df['speed_change'].abs()
    if 'signal_strength_variation' in rem_df.columns:
        jam_score += rem_df['signal_strength_variation'].abs()
    jam_cluster = int(rem_df.loc[jam_score.idxmax(), 'cluster_id'])
    mapping[jam_cluster] = 'GNSS Jamming'
    remaining.remove(jam_cluster)

    # === Weak GNSS Signal: Remaining cluster ===
    if remaining:
        mapping[remaining[0]] = 'Weak GNSS Signal'

    return mapping


def run_label_free_clustering(df_features, n_clusters=4):
    """
    Perform K-Means clustering on engineered features to derive pseudo-labels
    without using the original label column.

    Steps:
        1. Select clustering features
        2. Scale features
        3. Fit K-Means (k=4)
        4. Analyze cluster centroids to assign category names
        5. Return pseudo-label list for all records

    Returns:
        pseudo_labels (List[str]) - category name for each record
        kmeans_model              - fitted KMeans model
        cluster_mapping           - {cluster_id -> category_name}
    """
    print(f"\nRunning K-Means Label-Free Clustering (k={n_clusters})...")

    # Select available clustering features
    present_cols = [c for c in CLUSTER_FEATURES if c in df_features.columns]
    if not present_cols:
        # Fallback to all numeric columns
        present_cols = list(df_features.select_dtypes(include=[np.number]).columns)

    X = df_features[present_cols].copy()
    print(f"  Using features: {present_cols}")

    # Scale for K-Means distance metric
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=500)
    cluster_ids = kmeans.fit_predict(X_scaled)

    # Analyze cluster centroids in original scale
    centers = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=present_cols
    )
    centers['cluster_id'] = range(n_clusters)

    # Assign category names based on centroid characteristics
    mapping = _assign_cluster_names(centers, present_cols)

    print(f"\n  Cluster -> Category Mapping:")
    for cid, name in mapping.items():
        print(f"    Cluster {cid}  ->  {name}")

    # Generate pseudo-labels
    pseudo_labels = [mapping[int(c)] for c in cluster_ids]

    # Print distribution
    counts = Counter(pseudo_labels)
    print(f"\n  Pseudo-Label Distribution (Derived Without Original Labels):")
    for cat, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        pct = cnt / len(pseudo_labels) * 100
        print(f"    {cat:<22}: {cnt:>6,}  ({pct:.1f}%)")

    return pseudo_labels, kmeans, mapping
