import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def test_model_function():
    print("Loading dataset and model...")
    df = pd.read_csv("data/autism.csv")

    # Identify target column
    if 'result' in df.columns:
        target_column = 'result'
    elif 'Class/ASD' in df.columns:
        target_column = 'Class/ASD'
    else:
        raise ValueError("Target column not found!")

    # One-hot encoding
    df = pd.get_dummies(df, drop_first=True)

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Load trained model
    model = joblib.load("model.pkl")
    preds = model.predict(X)

    assert len(preds) == len(X), "Prediction length mismatch"

    # Accuracy & reports
    acc = accuracy_score(y, preds)
    print("Accuracy:", acc)
    print("Confusion Matrix:\n", confusion_matrix(y, preds))
    print("Classification Report:\n", classification_report(y, preds))

if __name__ == "__main__":
    test_model_function()
