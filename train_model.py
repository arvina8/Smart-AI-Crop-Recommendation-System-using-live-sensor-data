import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# Generate synthetic data
np.random.seed(42)
samples = 500
crops = ['Rice', 'Wheat', 'Maize', 'Sugarcane', 'Cotton', 'Millet', 'Barley']

df = pd.DataFrame({
    'Latitude': np.random.uniform(8.0, 28.0, samples),
    'Longitude': np.random.uniform(72.0, 88.0, samples),
    'Avg_temp': np.random.uniform(20, 40, samples),
    'Avg_humidity': np.random.uniform(30, 90, samples),
    'N': np.random.uniform(50, 150, samples),
    'P': np.random.uniform(20, 100, samples),
    'K': np.random.uniform(20, 120, samples),
    'Crop': np.random.choice(crops, samples)
})

X = df.drop(columns=['Crop'])
y = df['Crop']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

model = GradientBoostingClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, "model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("✅ Model and label encoder saved successfully!")
