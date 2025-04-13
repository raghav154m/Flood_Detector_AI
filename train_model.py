import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv("floods.csv")

# Rename columns for consistency (optional)
df.columns = ['rainfall', 'river_level', 'soil_moisture', 'humidity', 'temperature', 'wind_speed', 'flood']

# Features and label
X = df.drop("flood", axis=1)
y = df["flood"]

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Save model and scaler
joblib.dump(model, "flood_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model and scaler saved successfully.")