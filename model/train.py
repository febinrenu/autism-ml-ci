import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv("data/autism.csv")

print("Dataset shape:", df.shape)
print("Columns:", df.columns)

# Drop non-feature columns if present
if 'result' in df.columns:
    target_column = 'result'
elif 'Class/ASD' in df.columns:
    target_column = 'Class/ASD'
else:
    raise ValueError("Target column not found!")

# Handle categorical data (one-hot encoding)
df = pd.get_dummies(df, drop_first=True)

# Features and target
X = df.drop(target_column, axis=1)
y = df[target_column]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model.pkl")
print("Model training complete. Saved as model.pkl")
