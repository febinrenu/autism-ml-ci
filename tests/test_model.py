import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def test_model_function():
    print("Loading dataset and model...")
    df = pd.read_csv("data/autism.csv")

    # Identify target
    if 'Class/ASD' in df.columns:
        target_col = 'Class/ASD'
    elif 'result' in df.columns:
        target_col = 'result'
    else:
        raise ValueError("Target column not found")

    df = pd.get_dummies(df, drop_first=True)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    model = joblib.load("model.pkl")
    preds = model.predict(X)

    assert len(preds) == len(X), "Mismatch in number of predictions"

    acc = accuracy_score(y, preds)
    print("Accuracy:", acc)
    print("Confusion Matrix:\n", confusion_matrix(y, preds))
    print("Classification Report:\n", classification_report(y, preds))

if __name__ == "__main__":
    test_model_function()
