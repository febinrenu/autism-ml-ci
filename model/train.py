import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv("data/autism.csv")
print("Dataset columns:", df.columns.tolist())

# Identify and preprocess target column
if 'Class/ASD' in df.columns:
    target_col = 'Class/ASD'
elif 'result' in df.columns:
    target_col = 'result'
else:
    raise ValueError("Target column not found")

# One-hot encode features (if categorical)
df = pd.get_dummies(df, drop_first=True)

X = df.drop(columns=[target_col])
y = df[target_col]

# Ensure target y is categorical integer
if y.dtype.name != 'int64':
    try:
        y = y.round().astype(int)
    except Exception:
        y = LabelEncoder().fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model.pkl")
print("Model trained and saved as model.pkl")
