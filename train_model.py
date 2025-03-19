import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv("TB_PATIENT ADR DATASET.csv")  # Ensure the dataset is correct

# Convert weight (e.g., "32kg") to a number
df["weight"] = df["weight"].str.replace("kg", "").astype(float)

# Drop non-predictive columns
X = df.drop(columns=["adr", "symptoms", "suggestions"])
y = df[["adr", "symptoms", "suggestions"]]

# Convert categorical features to numerical using Label Encoding (except weight)
label_encoders = {}
for col in X.select_dtypes(include=["object"]).columns:
    label_encoders[col] = LabelEncoder()
    X[col] = label_encoders[col].fit_transform(X[col])

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(df.isnull().sum())  # Ensure no NaN values exist
# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
# Save the model and encoders
joblib.dump(model, "adr_prediction_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

print("✅ Model training complete! Model and encoders saved.")
