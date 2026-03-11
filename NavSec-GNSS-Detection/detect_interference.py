def predict_interference(model, new_data):
    """
    Use the trained model to predict GNSS interference types.
    Labels: 0 = Normal, 1 = Jamming, 2 = Spoofing, 3 = Weak
    """
    print("Predicting interference...")
    predictions = model.predict(new_data)
    
    # Map back to string labels
    label_map = {0: 'Normal GNSS', 1: 'GNSS Jamming', 2: 'GNSS Spoofing', 3: 'Weak GNSS Signal'}
    prediction_labels = [label_map.get(pred, 'Unknown') for pred in predictions]
    
    return predictions, prediction_labels
