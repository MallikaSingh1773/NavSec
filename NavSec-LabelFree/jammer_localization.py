import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np


INTERFERENCE_TYPES = ['GNSS Jamming', 'GNSS Spoofing']


def estimate_jammer_location(df_raw, pred_labels):
    """
    Estimate jammer coordinates by clustering anomalous aircraft positions
    (those classified as GNSS Jamming or GNSS Spoofing) using DBSCAN.

    Args:
        df_raw      — Original dataframe with lat/lon
        pred_labels — List of predicted category strings

    Returns:
        DataFrame of estimated jammer locations with lat, lon, and cluster size
    """
    print("Estimating jammer locations via DBSCAN ...")

    df = df_raw.copy()
    df['Predicted_Category'] = pred_labels

    # Filter to anomalous records only
    anomaly_mask = df['Predicted_Category'].isin(INTERFERENCE_TYPES)
    df_anomaly = df[anomaly_mask & df['lat'].notna() & df['lon'].notna()]

    if df_anomaly.empty:
        print("  No jamming/spoofing records found for localization.")
        return pd.DataFrame(columns=['estimated_lat', 'estimated_lon',
                                     'num_points', 'interference_type'])

    coords = df_anomaly[['lat', 'lon']].values
    # DBSCAN in radians for haversine distance
    coords_rad = np.radians(coords)
    eps_km = 50  # 50 km radius
    eps_rad = eps_km / 6371.0

    db = DBSCAN(eps=eps_rad, min_samples=5, algorithm='ball_tree', metric='haversine')
    cluster_labels = db.fit_predict(coords_rad)
    df_anomaly = df_anomaly.copy()
    df_anomaly['dbscan_cluster'] = cluster_labels

    jammer_locations = []
    unique_clusters = set(cluster_labels) - {-1}
    print(f"  Identified {len(unique_clusters)} jammer cluster(s).")

    for cid in unique_clusters:
        cluster_pts = df_anomaly[df_anomaly['dbscan_cluster'] == cid]
        centroid_lat = cluster_pts['lat'].mean()
        centroid_lon = cluster_pts['lon'].mean()
        dominant_type = cluster_pts['Predicted_Category'].mode()[0]
        jammer_locations.append({
            'estimated_lat':    round(centroid_lat, 6),
            'estimated_lon':    round(centroid_lon, 6),
            'num_points':       len(cluster_pts),
            'interference_type': dominant_type,
        })

    jammer_df = pd.DataFrame(jammer_locations).sort_values('num_points', ascending=False)
    print(f"  Total estimated jammer locations: {len(jammer_df)}")
    return jammer_df
