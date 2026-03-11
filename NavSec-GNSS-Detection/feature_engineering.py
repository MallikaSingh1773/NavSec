import numpy as np

def engineer_features(df):
    """
    Create additional features for GNSS interference detection.
    
    Required new features:
    - altitude_difference = geoaltitude - baroaltitude
    - speed_change
    - trajectory_jump = change in latitude/longitude
    - signal_strength_variation from rss
    - doppler_shift anomaly
    """
    print("Engineering features...")
    df_new = df.copy()
    
    # 1. altitude_difference
    if 'geoaltitude' in df_new.columns and 'baroaltitude' in df_new.columns:
        df_new['altitude_difference'] = df_new['geoaltitude'] - df_new['baroaltitude']
    
    # We assume sequential data to compute diffs. 
    # 2. speed_change
    if 'velocity' in df_new.columns:
        df_new['speed_change'] = df_new['velocity'].diff().fillna(0)
        
    # 3. trajectory_jump (Approximated as Euclidean distance of lat/lon change)
    if 'lat' in df_new.columns and 'lon' in df_new.columns:
        lat_diff = df_new['lat'].diff().fillna(0)
        lon_diff = df_new['lon'].diff().fillna(0)
        df_new['trajectory_jump'] = np.sqrt(lat_diff**2 + lon_diff**2)
        
    # 4. signal_strength_variation
    if 'rss' in df_new.columns:
        df_new['signal_strength_variation'] = df_new['rss'].diff().fillna(0).abs()
        
    # 5. doppler_shift anomaly
    if 'doppler' in df_new.columns:
        # Define anomaly as deviation from local rolling mean
        rolling_mean = df_new['doppler'].rolling(window=5, min_periods=1).mean()
        df_new['doppler_shift_anomaly'] = df_new['doppler'] - rolling_mean
        df_new['doppler_shift_anomaly'] = df_new['doppler_shift_anomaly'].fillna(0)
        
    print(f"Shape after feature engineering: {df_new.shape}")
    return df_new
