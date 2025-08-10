import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv("data/autism.csv")  # your dataset path

# Drop non-feature columns if needed
if 'id' in df.columns:
    df = df.drop('id', axis=1)

# One-hot encode categorical features (except target)
if 'gender' in df.columns and df['gender'].dtype == object:
    df = pd.get_dummies(df, columns=['gender'], drop_first=True)

# Define X and y
X = df.drop("class", axis=1)
y = df["class"]

# ✅ Ensure target is categorical
if y.dtype != 'int' and y.dtype != 'int64':
    try:
        # If numeric but continuous
        y = y.round().astype(int)
    except:
        # If non-numeric (string labels)
        y = LabelEncoder().fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model.pkl")
