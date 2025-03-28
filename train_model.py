import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Load dataset
df = pd.read_csv("TB_PATIENT ADR DATASET.csv")

# Convert weight (e.g., "32kg") to a number
df["weight"] = df["weight"].str.replace("kg", "", regex=True).astype(float)

# Handle missing values
df.fillna("Unknown", inplace=True)

# Select text-based features
text_columns = ["drug_name", "dosage", "comorbidities", "concomitant_medicine", "disease_status"]

# Convert text columns to a combined string for NLP processing
df["combined_text"] = df[text_columns].apply(lambda x: " ".join(x), axis=1)

# Target variables
target_columns = ["adr", "symptoms", "suggestions", "pharmacokinetics", "pharmacodynamics", "drug_interactions"]
X = df["combined_text"]  # Use combined text as features
y = df[target_columns]

# Convert text into numerical features using TF-IDF
vectorizer = TfidfVectorizer()
X_transformed = vectorizer.fit_transform(X)  # Convert text to numerical representation

# Save the vectorizer for future use
joblib.dump(vectorizer, "text_vectorizer.pkl")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "adr_prediction_model.pkl")

print("✅ Model trained with text-based features and saved successfully!")
