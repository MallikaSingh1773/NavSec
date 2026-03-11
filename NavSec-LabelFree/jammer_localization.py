import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np


INTERFERENCE_TYPES = ['GNSS Jamming', 'GNSS Spoofing']


def estimate_jammer_location(df_raw, pred_labels):
    """
    Estimate jammer coordinates by clustering anomalous aircraft positions
    (those classified as GNSS Jamming or GNSS Spoofing) using DBSCAN.

    Includes confidence_score per cluster:
        confidence_score = cluster_num_points / total_anomaly_points

    Returns:
        DataFrame with columns:
            cluster_id, estimated_lat, estimated_lon,
            num_points, interference_type, confidence_score
    """
    print("Estimating jammer locations via DBSCAN ...")

    df = df_raw.copy()
    df['Predicted_Category'] = pred_labels

    anomaly_mask = df['Predicted_Category'].isin(INTERFERENCE_TYPES)
    df_anomaly = df[anomaly_mask & df['lat'].notna() & df['lon'].notna()]

    if df_anomaly.empty:
        print("  No jamming/spoofing records found for localization.")
        return pd.DataFrame(columns=['cluster_id', 'estimated_lat', 'estimated_lon',
                                     'num_points', 'interference_type', 'confidence_score'])

    total_anomaly_points = len(df_anomaly)
    coords = df_anomaly[['lat', 'lon']].values
    coords_rad = np.radians(coords)
    eps_rad = 50 / 6371.0

    db = DBSCAN(eps=eps_rad, min_samples=5, algorithm='ball_tree', metric='haversine')
    cluster_labels = db.fit_predict(coords_rad)
    df_anomaly = df_anomaly.copy()
    df_anomaly['dbscan_cluster'] = cluster_labels

    jammer_locations = []
    unique_clusters = set(cluster_labels) - {-1}
    print(f"  Identified {len(unique_clusters)} jammer cluster(s).")

    for cid in unique_clusters:
        cluster_pts   = df_anomaly[df_anomaly['dbscan_cluster'] == cid]
        centroid_lat  = cluster_pts['lat'].mean()
        centroid_lon  = cluster_pts['lon'].mean()
        dominant_type = cluster_pts['Predicted_Category'].mode()[0]
        n_pts         = len(cluster_pts)
        confidence    = float(round(n_pts / total_anomaly_points, 4)) if total_anomaly_points > 0 else 0.0

        jammer_locations.append({
            'cluster_id':        int(cid),
            'estimated_lat':     round(centroid_lat, 6),
            'estimated_lon':     round(centroid_lon, 6),
            'num_points':        n_pts,
            'interference_type': dominant_type,
            'confidence_score':  confidence,
        })
        print(f"    Cluster {cid}: {n_pts} pts, confidence = {confidence:.2%}")

    jammer_df = (pd.DataFrame(jammer_locations)
                   .sort_values('confidence_score', ascending=False)
                   .reset_index(drop=True))
    print(f"  Total estimated jammer locations: {len(jammer_df)}")
    return jammer_df
