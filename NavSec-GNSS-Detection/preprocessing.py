import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_clean_data(file_path):
    """
    Load the dataset, handle missing values (mean imputation), and remove duplicates.
    """
    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        # Try to read csv if excel fails, just in case
        df = pd.read_csv(file_path.replace('.xlsx', '.csv'))
        
    print(f"Initial shape: {df.shape}")
    
    # Remove duplicate rows
    df = df.drop_duplicates()
    print(f"Shape after removing duplicates: {df.shape}")
    
    # Handle missing values using mean imputation
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    return df

def normalize_features(df, target_col='label'):
    """
    Normalize numerical features using StandardScaler.
    """
    print("Normalizing features...")
    scaler = StandardScaler()
    
    # Select columns to normalize (exclude target)
    features = [col for col in df.columns if col != target_col]
    
    df_normalized = df.copy()
    
    if len(features) > 0:
        df_normalized[features] = scaler.fit_transform(df[features])
        
    return df_normalized, scaler
