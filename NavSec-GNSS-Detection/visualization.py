import matplotlib.pyplot as plt
import seaborn as sns
import folium

def plot_confusion_matrix(cm, labels, output_path='confusion_matrix.png'):
    """
    Plot and save the confusion matrix.
    """
    print("Generating confusion matrix plot...")
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved confusion matrix to {output_path}")

def plot_feature_importance(model, feature_names, output_path='feature_importance.png'):
    """
    Plot and save feature importance (for models like Random Forest/XGBoost).
    """
    print("Generating feature importance plot...")
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        # Sort feature importances in descending order
        indices = importances.argsort()[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances")
        plt.bar(range(len(importances)), importances[indices], align="center")
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"Saved feature importance to {output_path}")
    else:
        print("Model does not have feature_importances_ attribute.")

def create_interference_map(df, predictions, jammer_locations, output_path='interference_map.html'):
    """
    Create a Folium map showing aircraft positions, interference points, and jammer locations.
    """
    print("Generating Folium map...")
    
    # Check if necessary columns exist
    if 'lat' not in df.columns or 'lon' not in df.columns:
        print("Cannot create map: missing 'lat' or 'lon' columns.")
        return
        
    df_map = df.copy()
    if 'predicted_label' not in df_map.columns:
        df_map['predicted_label'] = predictions
        
    # Initialize map centered at the mean coordinates
    map_center = [df_map['lat'].mean(), df_map['lon'].mean()]
    m = folium.Map(location=map_center, zoom_start=10)
    
    # 0 = Normal, 1 = Jamming, 2 = Spoofing, 3 = Weak
    color_map = {
        0: 'blue',     # Normal
        1: 'crimson',  # Jamming
        2: 'purple',   # Spoofing
        3: 'orange'    # Weak
    }
    
    if len(df_map) > 5000:
        print(f"Subsampling map points from {len(df_map)} to 5000 for performance.")
        df_sample = df_map.sample(5000, random_state=42)
    else:
        df_sample = df_map
        
    # Draw Normal/Weak points first, then Jamming/Spoofing on top
    df_sample['priority'] = df_sample['predicted_label'].apply(lambda x: 1 if x in [1, 2] else 0)
    df_sample = df_sample.sort_values(by='priority')

    for _, row in df_sample.iterrows():
        label = int(row['predicted_label'])
        color = color_map.get(label, 'gray')
        radius = 5 if label in [1, 2] else 2 # Make interference points bigger
        opacity = 0.9 if label in [1, 2] else 0.4
        
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=opacity,
            weight=1,
            popup=f"Label: {label}"
        ).add_to(m)
        
    # Mark estimated jammer locations
    if not jammer_locations.empty:
        for _, row in jammer_locations.iterrows():
            folium.Marker(
                location=[row['estimated_lat'], row['estimated_lon']],
                popup=f"Estimated Jammer (Cluster {row['cluster_id']})",
                icon=folium.Icon(color='red', icon='fire', prefix='fa')
            ).add_to(m)
            
            # Optionally, add a radius circle to represent assumed jammer range
            folium.Circle(
                location=[row['estimated_lat'], row['estimated_lon']],
                radius=3000, # 3 km radius (reduced from 5km to avoid overlapping)
                color='red',
                weight=2,
                fill=False
            ).add_to(m)
            
    m.save(output_path)
    print(f"Saved interference map to {output_path}")
