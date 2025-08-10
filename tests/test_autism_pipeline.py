# tests/test_autism_pipeline.py
import os
import shutil
import pandas as pd
from autism_model import load_data, preprocess, train_and_evaluate

def test_load_data_exists():
    df = load_data()
    assert df is not None
    assert "Class/ASD" in df.columns

def test_preprocess_basic():
    df = load_data().head(20)  # small slice for speed
    proc = preprocess(df)
    assert not proc.isnull().values.any()

def test_train_and_evaluate_runs(tmp_path):
    # run train on a small slice to keep it quick
    df = load_data().sample(n=80, random_state=1)
    proc = preprocess(df)
    out = train_and_evaluate(proc)
    assert 0.0 <= out["accuracy"] <= 1.0
    # artifacts exist
    assert os.path.exists(out["model_path"])
    assert os.path.exists(out["metrics_path"])
    # cleanup
    shutil.rmtree("artifacts", ignore_errors=True)
