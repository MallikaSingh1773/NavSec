import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, confusion_matrix)
from xgboost import XGBClassifier


def train_and_evaluate(X, pseudo_labels):
    """
    Train Random Forest, XGBoost, and Neural Network classifiers
    on K-Means-derived pseudo_labels (NOT original labels).

    Args:
        X             — Feature matrix (DataFrame)
        pseudo_labels — List of category strings from K-Means clustering

    Returns:
        best_model       — Best trained model
        results          — Dict of performance metrics per model
        best_model_name  — Name of the best model
        label_encoder    — Fitted LabelEncoder for decoding predictions
    """
    print("Encoding pseudo-labels ...")
    le = LabelEncoder()
    y = le.fit_transform(pseudo_labels)
    print(f"  Classes: {list(le.classes_)}")

    print("Splitting data (80% train, 20% test) ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost':       XGBClassifier(eval_metric='mlogloss', random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
    }

    results = {}
    best_model = None
    best_f1 = -1
    best_model_name = ''

    for name, model in models.items():
        print(f"\nTraining {name} on pseudo-labels ...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec  = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1   = f1_score(y_test, y_pred, average='macro', zero_division=0)
        cm   = confusion_matrix(y_test, y_pred)

        results[name] = {
            'Accuracy': acc, 'Precision': prec,
            'Recall': rec,   'F1 Score': f1,
            'Confusion Matrix': cm,
        }

        print(f"  Accuracy : {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall   : {rec:.4f}")
        print(f"  F1 Score : {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_model_name = name

    print(f"\nBest Model: {best_model_name}  (F1 = {best_f1:.4f})")

    # Save best model + label encoder together
    os.makedirs('models', exist_ok=True)
    joblib.dump({'model': best_model, 'label_encoder': le}, 'models/best_model.pkl')
    print("Saved best model and label encoder to models/best_model.pkl")

    return best_model, results, best_model_name, le
