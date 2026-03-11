import pandas as pd
import numpy as np
import os


# ─────────────────────────────────────────────────────────
#  GNSS Attack Simulation Module
#  Artificially introduces interference signatures into
#  clean flight records to test detection capability.
# ─────────────────────────────────────────────────────────

ATTACK_TYPES = ['jamming', 'spoofing', 'weak']


def simulate_attack(df: pd.DataFrame, attack_type: str,
                    num_samples: int = 200, random_seed: int = 42) -> pd.DataFrame:
    """
    Simulate a real-world GNSS attack by modifying physical signal features
    on a random sample of records.

    Args:
        df          — Original feature dataframe (must have engineered features)
        attack_type — One of: 'jamming', 'spoofing', 'weak'
        num_samples — Number of records to perturb
        random_seed — For reproducibility

    Returns:
        DataFrame with simulated attack records and a 'Simulated_Attack' column
    """
    attack_type = attack_type.lower().strip()
    if attack_type not in ATTACK_TYPES:
        raise ValueError(f"Unknown attack_type '{attack_type}'. Choose from: {ATTACK_TYPES}")

    rng = np.random.default_rng(random_seed)
    n = min(num_samples, len(df))
    sample_idx = rng.choice(df.index, size=n, replace=False)
    sim_df = df.loc[sample_idx].copy()

    if attack_type == 'jamming':
        # GNSS Jamming: large Doppler anomaly + RSS fluctuation
        if 'doppler_shift_anomaly' in sim_df.columns:
            sim_df['doppler_shift_anomaly'] += rng.uniform(150, 300, size=n)
        if 'signal_strength_variation' in sim_df.columns:
            sim_df['signal_strength_variation'] += rng.uniform(80, 150, size=n)
        if 'doppler' in sim_df.columns:
            sim_df['doppler'] += rng.uniform(50, 120, size=n) * rng.choice([-1, 1], size=n)
        label = 'GNSS Jamming (Simulated)'

    elif attack_type == 'spoofing':
        # GNSS Spoofing: trajectory jumps + altitude mismatch
        if 'trajectory_jump' in sim_df.columns:
            sim_df['trajectory_jump'] += rng.uniform(0.5, 2.5, size=n)
        if 'altitude_difference' in sim_df.columns:
            sim_df['altitude_difference'] += rng.uniform(800, 3000, size=n)
        if 'geoaltitude' in sim_df.columns and 'baroaltitude' in sim_df.columns:
            offset = rng.uniform(500, 2000, size=n)
            sim_df['geoaltitude'] = sim_df['geoaltitude'] + offset
        label = 'GNSS Spoofing (Simulated)'

    elif attack_type == 'weak':
        # Weak GNSS Signal: significantly reduced RSS
        if 'rss' in sim_df.columns:
            sim_df['rss'] = sim_df['rss'] - rng.uniform(40, 80, size=n)
        if 'signal_strength_variation' in sim_df.columns:
            sim_df['signal_strength_variation'] += rng.uniform(20, 60, size=n)
        if 'velocity' in sim_df.columns:
            sim_df['velocity'] = sim_df['velocity'] * rng.uniform(0.3, 0.6, size=n)
        label = 'Weak GNSS Signal (Simulated)'

    sim_df['Simulated_Attack'] = label
    print(f"Simulated {n} records with attack type: '{attack_type}' -> {label}")
    return sim_df.reset_index(drop=True)


def run_all_simulations(df: pd.DataFrame, model, label_encoder,
                        feature_cols: list, num_samples: int = 200) -> pd.DataFrame:
    """
    Run all three attack simulations and pass results through the trained model.

    Returns combined simulation results with both simulated attack type
    and model-predicted category.
    """
    from sklearn.preprocessing import StandardScaler

    all_results = []

    for attack in ATTACK_TYPES:
        print(f"\nRunning simulation: {attack.upper()} ...")
        sim = simulate_attack(df, attack, num_samples=num_samples)

        # Select only model features
        sim_features = [c for c in feature_cols if c in sim.columns]
        X_sim = sim[sim_features].copy()

        # Normalize with fresh scaler (same as training normalization)
        scaler = StandardScaler()
        X_sim_scaled = scaler.fit_transform(X_sim)

        # Predict
        pred_numeric = model.predict(X_sim_scaled)
        pred_labels  = label_encoder.inverse_transform(pred_numeric)

        sim['Predicted_Category'] = pred_labels
        all_results.append(sim)

    combined = pd.concat(all_results, ignore_index=True)

    # Save results
    os.makedirs('data_outputs', exist_ok=True)
    combined.to_csv('data_outputs/simulation_results.csv', index=False)
    print(f"\nSaved simulation results to data_outputs/simulation_results.csv "
          f"({len(combined)} total records)")

    # Print detection summary
    print("\nSimulation Detection Summary:")
    for attack in ATTACK_TYPES:
        label_filter = f'{attack.capitalize()} (Simulated)' if attack != 'weak' else 'Weak GNSS Signal (Simulated)'
        subset = combined[combined['Simulated_Attack'].str.lower().str.startswith(attack)]
        if not subset.empty:
            top_pred = subset['Predicted_Category'].value_counts().idxmax()
            pct = (subset['Predicted_Category'] == top_pred).mean() * 100
            print(f"  [{attack.upper():8s}] Most predicted: '{top_pred}' ({pct:.1f}%)")

    return combined

if __name__ == "__main__":
    import joblib
    from preprocessing import load_and_preprocess
    from feature_engineering import engineer_features

    print("Loading data for standalone simulation run...")
    df_raw, X_raw, _ = load_and_preprocess('dataset/Data.xlsx')
    df_eng = engineer_features(df_raw)
    
    print("Loading saved model...")
    saved = joblib.load('models/best_model.pkl')
    
    spatial = ['lat', 'lon']
    feature_cols = [c for c in engineer_features(X_raw).columns if c not in spatial]
    
    run_all_simulations(df_eng, saved['model'], saved['label_encoder'], feature_cols)
