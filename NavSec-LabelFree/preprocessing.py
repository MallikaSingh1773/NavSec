import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# Columns to use as signal/physics features (exclude spatial identifiers)
FEATURE_COLS = ['velocity', 'heading', 'vertrate', 'baroaltitude',
                'geoaltitude', 'rss', 'doppler']

# Columns kept for map/spatial context only
SPATIAL_COLS = ['lat', 'lon']


def load_and_preprocess(dataset_path):
    """
    Load data from Excel/CSV, separate and discard the label column,
    impute missing values, remove duplicates, and return clean features.

    Returns:
        df_raw    — original dataframe (with lat/lon for mapping)
        X_raw     — feature-only dataframe (no label, no spatial cols)
        orig_labels — original label series kept aside for optional comparison
    """
    print(f"Loading dataset from {dataset_path} ...")

    if dataset_path.endswith('.xlsx'):
        df = pd.read_excel(dataset_path)
    else:
        df = pd.read_csv(dataset_path)

    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns.")

    # Keep original labels aside — NEVER used for training
    orig_labels = df['label'].copy() if 'label' in df.columns else None
    if orig_labels is not None:
        print("Original label column stored separately (NOT used for training).")

    # Remove duplicates
    df.drop_duplicates(inplace=True)
    if orig_labels is not None:
        orig_labels = orig_labels.loc[df.index].reset_index(drop=True)
    df.reset_index(drop=True, inplace=True)
    print(f"After deduplication: {len(df):,} rows.")

    # Handle missing values
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mean())

    # Build feature matrix — drop label and spatial columns
    drop_cols = ['label'] + [c for c in SPATIAL_COLS if c in df.columns]
    X_raw = df.drop(columns=[c for c in drop_cols if c in df.columns])

    print(f"Feature matrix shape: {X_raw.shape}")
    return df, X_raw, orig_labels


def normalize_features(df_features, exclude_cols=None):
    """
    Normalize numeric features with StandardScaler.
    Returns normalized dataframe and fitted scaler.
    """
    exclude_cols = exclude_cols or []
    numeric_cols = [c for c in df_features.select_dtypes(include=[np.number]).columns
                    if c not in exclude_cols]

    scaler = StandardScaler()
    df_norm = df_features.copy()
    df_norm[numeric_cols] = scaler.fit_transform(df_features[numeric_cols])
    print(f"Normalized {len(numeric_cols)} numeric columns.")
    return df_norm, scaler
