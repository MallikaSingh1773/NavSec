import numpy as np


def engineer_features(df):
    """
    Add physics-based derived features to the dataframe.
    These features capture signal anomaly patterns indicative of interference.

    Features Added:
        altitude_difference       — GPS vs barometer discrepancy (spoofing indicator)
        speed_change              — Sudden velocity changes (jamming indicator)
        trajectory_jump           — Unrealistic lat/lon jumps (spoofing indicator)
        signal_strength_variation — RSS rolling std (jamming / weak signal indicator)
        doppler_shift_anomaly     — Doppler frequency deviation (jamming indicator)
    """
    df = df.copy()

    # 1. Altitude Difference: GPS vs barometric altitude mismatch
    if 'geoaltitude' in df.columns and 'baroaltitude' in df.columns:
        df['altitude_difference'] = (df['geoaltitude'] - df['baroaltitude']).abs()
    else:
        df['altitude_difference'] = 0.0

    # 2. Speed Change: Absolute change in velocity between consecutive records
    if 'velocity' in df.columns:
        df['speed_change'] = df['velocity'].diff().abs().fillna(0)
    else:
        df['speed_change'] = 0.0

    # 3. Trajectory Jump: Sum of lat/lon sudden changes
    if 'lat' in df.columns and 'lon' in df.columns:
        df['trajectory_jump'] = (
            df['lat'].diff().abs() + df['lon'].diff().abs()
        ).fillna(0)
    else:
        df['trajectory_jump'] = 0.0

    # 4. Signal Strength Variation: Rolling standard deviation of RSS
    if 'rss' in df.columns:
        df['signal_strength_variation'] = (
            df['rss'].rolling(window=5, min_periods=1).std().fillna(0)
        )
    else:
        df['signal_strength_variation'] = 0.0

    # 5. Doppler Shift Anomaly: Deviation from rolling mean
    if 'doppler' in df.columns:
        rolling_mean = df['doppler'].rolling(window=10, min_periods=1).mean()
        df['doppler_shift_anomaly'] = (df['doppler'] - rolling_mean).abs().fillna(0)
    else:
        df['doppler_shift_anomaly'] = 0.0

    print(f"Feature engineering complete. Total features: {df.shape[1]}")
    return df
