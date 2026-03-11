import pandas as pd
from sklearn.cluster import DBSCAN

def estimate_jammer_location(df, predictions, eps=0.01, min_samples=5):
    """
    Estimate jammer location using DBSCAN on jamming points (label == 1).
    eps: 0.01 degrees is roughly 1.1 km.
    """
    print("Estimating jammer locations based on anomalies...")
    
    df_eval = df.copy()
    if 'predicted_label' not in df_eval.columns:
        df_eval['predicted_label'] = predictions
    
    # Filter for Jamming (Label 1)
    jamming_points = df_eval[df_eval['predicted_label'] == 1].copy()
    
    if jamming_points.empty:
        print("No jamming anomalies detected. Cannot estimate jammer location.")
        return pd.DataFrame()
        
    # Extract coordinates
    coords = jamming_points[['lat', 'lon']].values
    
    # Apply DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    jamming_points['cluster'] = db.labels_
    
    # Calculate centroids of each valid cluster (exclude noise cluster -1)
    centroids = []
    for cluster_id in jamming_points['cluster'].unique():
        if cluster_id != -1:
            cluster_data = jamming_points[jamming_points['cluster'] == cluster_id]
            centroid_lat = cluster_data['lat'].mean()
            centroid_lon = cluster_data['lon'].mean()
            num_points = len(cluster_data)
            centroids.append({
                'cluster_id': cluster_id,
                'estimated_lat': centroid_lat,
                'estimated_lon': centroid_lon,
                'num_points': num_points
            })
            
    centroids_df = pd.DataFrame(centroids)
    if not centroids_df.empty:
        print(f"Identified {len(centroids_df)} potential jammer locations.")
    else:
        print("No dense clusters found for jamming points. (Only noise detected)")
        
    return centroids_df
