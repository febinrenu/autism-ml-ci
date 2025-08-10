# autism_model.py
import os
import time
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

DATA_PATH = os.environ.get("AUTISM_CSV", "data/autism.csv")
TARGET = "Class/ASD"

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    return df

def preprocess(df):
    # drop obvious non-feature columns
    df = df.copy()
    if "ID" in df.columns:
        df.drop(columns=["ID"], inplace=True)
    # remove rows with missing target
    df = df.dropna(subset=[TARGET])
    # simple imputation:
    num_cols = df.select_dtypes(include=["float64","int64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    # fill numeric with median
    for c in num_cols:
        df[c].fillna(df[c].median(), inplace=True)
    # fill categorical with mode and one-hot encode
    for c in cat_cols:
        df[c].fillna(df[c].mode()[0] if not df[c].mode().empty else "unknown", inplace=True)
    df = pd.get_dummies(df, drop_first=True)
    return df

def train_and_evaluate(df, target=TARGET, test_size=0.2, random_state=42):
    X = df.drop(columns=[target])
    y = df[target].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size,
                                                        random_state=random_state,
                                                        stratify=y if len(y.unique())>1 else None)
    model = RandomForestClassifier(random_state=random_state, n_estimators=100)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # Build a filename that can include commit sha (in CI) or timestamp locally
    sha = os.environ.get("GITHUB_SHA")
    stamp = sha[:8] if sha else time.strftime("%Y%m%d-%H%M%S")
    model_fname = f"artifacts/autism_model_{stamp}.pkl"
    metrics_fname = f"artifacts/metrics_{stamp}.json"

    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(model, model_fname)
    with open(metrics_fname, "w") as f:
        json.dump({"accuracy": acc}, f)

    return {"model_path": model_fname, "metrics_path": metrics_fname, "accuracy": acc}

if __name__ == "__main__":
    df = load_data()
    df_proc = preprocess(df)
    out = train_and_evaluate(df_proc)
    print(f"Trained. Accuracy={out['accuracy']:.4f}")
    print("Saved model to", out["model_path"])
