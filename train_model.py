import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv("TB_PATIENT ADR DATASET.csv")  # Ensure the dataset is correct

# Convert weight (e.g., "32kg") to a number
df["weight"] = df["weight"].str.replace("kg", "", regex=True).astype(float)

# Handle missing values (Fill with 'Unknown' for categories and median for numbers)
df.fillna({
    "drug_name": "Unknown",
    "sex": "Unknown",
    "disease_status": "Unknown",
    "dosage": "Unknown",
    "dose_duration": "Unknown",
    "comorbidities": "none",
    "lifestyle_factors": "none",
    "pregnancy": "none",
    "pregnancy_month": "0",
    "genetic_factors": "Unknown",
    "concomitant_medicine": "Unknown"
}, inplace=True)

# Fill missing numerical values with the median
for col in ["weight", "ast(10-40)", "alt(5-30)", "alp(150-280)"]:
    df[col].fillna(df[col].median(), inplace=True)

# Target variables (Now includes pharmacokinetics, pharmacodynamics, and drug interactions)
target_columns = ["adr", "symptoms", "suggestions", "pharmacokinetics", "pharmacodynamics", "drug_interactions"]
X = df.drop(columns=target_columns)
y = df[target_columns]

# Convert categorical features to numerical using Label Encoding
label_encoders = {}
for col in X.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le  # Store encoder for later use

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model and encoders
joblib.dump(model, "adr_prediction_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

print("✅ Model training complete! Model and encoders saved.")
