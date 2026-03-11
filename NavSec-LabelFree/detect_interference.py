import numpy as np


def predict_interference(model, X, label_encoder):
    """
    Use the trained model to classify GNSS signals into interference categories.

    Args:
        model         — Trained classifier (best from training phase)
        X             — Feature matrix
        label_encoder — LabelEncoder to convert numeric predictions back to category names

    Returns:
        pred_numeric  — Integer class predictions
        pred_labels   — Human-readable category strings
    """
    print("Predicting interference categories ...")
    pred_numeric = model.predict(X)
    pred_labels  = label_encoder.inverse_transform(pred_numeric)

    from collections import Counter
    counts = Counter(pred_labels)
    print("  Prediction Distribution:")
    for cat, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        pct = cnt / len(pred_labels) * 100
        print(f"    {cat:<22}: {cnt:>6,}  ({pct:.1f}%)")

    return pred_numeric, list(pred_labels)
